import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure this folder and its parent are importable when running as a script/module
_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))

from pyldpc import make_ldpc

from ldpc_utils import ldpc_encode, add_awgn
from modulation_utils import get_modulation_params, map_binary_to_symbol, calculate_llrs
from problem_construction import build_parity_terms, allocate_parity_slack_ancillas, build_qubo_from_parity_llr
from solvers import SOLVERS, spin_to_binary, _spin_hubo_dict_to_terms
from pysa.spinhubo import SpinHUBOSolver
from pysa.sa import Solver as QuboSolver


def _qubo_best_energy_once(args):
    """Top-level helper for multiprocessing: run a single QUBO read and return best energy."""
    Q_, sweeps, tmin, tmax = args
    solver = QuboSolver(Q_, problem_type='qubo')
    df = solver.metropolis_update(
        num_sweeps=sweeps,
        num_reads=1,
        min_temp=tmin,
        max_temp=tmax,
        return_dataframe=True,
        parallel=False,
        use_pt=False,
        verbose=False,
    )
    return float(df['best_energy'][0])


def _build_slack_assignments_for_qubo(H, x_bits, ancilla_plan):
    """Given original bits x and ancilla plan, assign slack ancillas per check.
    For each check c with neighborhood S: t_c = round(sum_{i in S} x_i / 2),
    clipped to [0, floor(|S|/2)], encoded in binary into ancilla bits.
    Returns a dict {anc_idx: 0/1} for all ancilla bits in the plan.
    """
    H = np.asarray(H)
    m, n = H.shape
    x = np.asarray(x_bits).astype(int).reshape(-1)
    anc_assign = {}
    for c in range(m):
        S = np.where(H[c] == 1)[0]
        d = int(S.size)
        if d == 0:
            continue
        anc_bits = ancilla_plan[c] if c < len(ancilla_plan) else []
        if not anc_bits:
            continue
        t_max = d // 2
        t_val = int(np.rint(x[S].sum() / 2.0))
        t_val = max(0, min(t_val, t_max))
        # encode in binary across anc_bits (LSB first)
        for k, anc_idx in enumerate(anc_bits):
            anc_assign[int(anc_idx)] = (t_val >> k) & 1
    return anc_assign


def generate_ldpc_code(N=120, d_v=2, d_c=3, seed=42):
    np.random.seed(seed)
    H_raw, G_raw = make_ldpc(N, d_v, d_c, systematic=True, sparse=True)
    H = H_raw.toarray() if hasattr(H_raw, 'toarray') else np.asarray(H_raw)
    G = G_raw.toarray() if hasattr(G_raw, 'toarray') else np.asarray(G_raw)
    k = G.shape[1]
    n = H.shape[1]
    if n != N:
        print(f"Warning: requested N={N} but generated code length is {n}. Using actual length.")
    return H, G, k, n


def simulate_ldpc_PySA(snr_db, H, G, k, n, modulation_type, num_trials, lambda_penalty, solver_name, formulation_type):
    ber_total = 0
    bits_per_symbol = get_modulation_params(modulation_type)

    m, original_n = H.shape
    parity_terms = None
    ancilla_plan = None
    if solver_name == 'SA':
        parity_terms = build_parity_terms(H, lambda_penalty)
    elif solver_name == 'QUBO-PAPER':
        ancilla_plan, _ = allocate_parity_slack_ancillas(H)

    for _ in range(num_trials):
        message = np.random.randint(0, 2, k)
        codeword = ldpc_encode(message, G)

        len_codeword = len(codeword)
        padding_needed = (bits_per_symbol - (len_codeword % bits_per_symbol)) % bits_per_symbol
        padded_codeword = np.pad(codeword, (0, padding_needed), 'constant')
        n_vars = original_n

        symbols_binary_blocks = padded_codeword.reshape(-1, bits_per_symbol)
        modulated_symbols = np.array([map_binary_to_symbol(b, modulation_type) for b in symbols_binary_blocks])
        received_symbols, sigma = add_awgn(modulated_symbols, snr_db)

        noise_variance = 2 * sigma**2
        llrs_full = calculate_llrs(received_symbols, noise_variance, modulation_type, padded_codeword.size)
        llr = llrs_full[:original_n]

        if solver_name == 'BP':
            from collections import defaultdict
            hubo_bp = defaultdict(float)
            hubo_bp[('H_matrix',)] = H
            for i in range(n_vars):
                hubo_bp[(i,)] = -llr[i]
            decoded_codeword = SOLVERS['BP'](hubo_bp, n_vars, max_iters=5, damping=0.0)
        elif solver_name == 'QUBO-PAPER':
            from solvers import qubo_sa_paper_from_H_llr
            result_state = qubo_sa_paper_from_H_llr(H, llr, num_sweeps=50, num_reads=5, min_temp=0.1, max_temp=3.0, penalty_strength=lambda_penalty, verbose=False, ancilla_plan=ancilla_plan)
            decoded_codeword = np.asarray(result_state, dtype=int)[:n_vars]
        else:
            from collections import defaultdict
            hubo_sa = defaultdict(float)
            for i in range(n_vars):
                hubo_sa[(i,)] += -llr[i]
            for k_idx, v in parity_terms.items():
                hubo_sa[k_idx] += v
            result_state = SOLVERS['SA'](hubo_sa, n_vars, num_sweeps=50, num_reads=5, min_temp=0.1, max_temp=3.0, verbose=False)
            decoded_codeword = spin_to_binary(result_state)

        decoded_message = decoded_codeword[:k]
        errors = np.sum(decoded_message != message)
        ber_total += errors / k

    return ber_total / num_trials


def plot_energy_distribution(H, G, k, modulation_type, snr_db, solver_name='QUBO-PAPER',
                             lambda_penalty=2.0, num_reads=200, num_sweeps=300,
                             min_temp=0.1, max_temp=3.0, ancilla_plan=None,
                             save_dir=None, parallel_reads=False, max_workers=None):
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(save_dir, exist_ok=True)

    bits_per_symbol = get_modulation_params(modulation_type)
    m, original_n = H.shape

    message = np.random.randint(0, 2, k)
    codeword = ldpc_encode(message, G)
    len_codeword = len(codeword)
    padding_needed = (bits_per_symbol - (len_codeword % bits_per_symbol)) % bits_per_symbol
    padded_codeword = np.pad(codeword, (0, padding_needed), 'constant')
    n_vars = original_n

    symbols_binary_blocks = padded_codeword.reshape(-1, bits_per_symbol)
    modulated_symbols = np.array([map_binary_to_symbol(b, modulation_type) for b in symbols_binary_blocks])
    received_symbols, sigma = add_awgn(modulated_symbols, snr_db)

    noise_variance = 2 * sigma**2
    llrs_full = calculate_llrs(received_symbols, noise_variance, modulation_type, padded_codeword.size)
    llr = llrs_full[:original_n]

    if solver_name == 'QUBO-PAPER':
        if ancilla_plan is None:
            ancilla_plan, _ = allocate_parity_slack_ancillas(H)
        Q, total_n = build_qubo_from_parity_llr(H, llr, penalty_strength=lambda_penalty, ancilla_plan=ancilla_plan)
        # Decode once with BP to get reference solution and its energy under QUBO
        from collections import defaultdict
        hubo_bp = defaultdict(float)
        hubo_bp[('H_matrix',)] = H
        for i in range(n_vars):
            hubo_bp[(i,)] = -llr[i]
        # Increase BP iterations to approach a lower-energy reference
        bp_bits = SOLVERS['BP'](hubo_bp, n_vars, max_iters=500, damping=0.0)
        qubo_solver_ref = QuboSolver(Q, problem_type='qubo')
        # Assign ancilla bits consistent with bp_bits to minimize slack penalties
        anc_assign = _build_slack_assignments_for_qubo(H, bp_bits, ancilla_plan)
        x_full = np.zeros(Q.shape[0], dtype=qubo_solver_ref.float_type)
        x_full[:n_vars] = bp_bits
        for anc_idx, val in anc_assign.items():
            if anc_idx < x_full.shape[0]:
                x_full[anc_idx] = val
        bp_energy = float(qubo_solver_ref.get_energy(x_full))
        if parallel_reads and num_reads > 1:
            # Distribute independent reads across processes
            from concurrent.futures import ProcessPoolExecutor

            args_list = [(Q, num_sweeps, min_temp, max_temp)] * int(num_reads)
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                energies = list(ex.map(_qubo_best_energy_once, args_list))
            energies = np.asarray(energies, dtype=float)
        else:
            qubo_solver = QuboSolver(Q, problem_type='qubo')
            df = qubo_solver.metropolis_update(
                num_sweeps=num_sweeps,
                num_reads=num_reads,
                min_temp=min_temp,
                max_temp=max_temp,
                return_dataframe=True,
                parallel=False,
                use_pt=False,
                verbose=False,
            )
            energies = np.asarray(df['best_energy'], dtype=float)
    elif solver_name == 'SA':
        from problem_construction import build_parity_terms
        hubo_sa = {}
        parity_terms = build_parity_terms(H, lambda_penalty)
        hubo_sa.update(parity_terms)
        for i in range(n_vars):
            hubo_sa[(i,)] = hubo_sa.get((i,), 0.0) + (-llr[i])
        terms = _spin_hubo_dict_to_terms(hubo_sa)
        sh_solver = SpinHUBOSolver(terms, n_vars)
        # BP reference energy under Spin-HUBO energy
        from collections import defaultdict
        hubo_bp = defaultdict(float)
        hubo_bp[('H_matrix',)] = H
        for i in range(n_vars):
            hubo_bp[(i,)] = -llr[i]
        # Increase BP iterations to approach a lower-energy reference
        bp_bits = SOLVERS['BP'](hubo_bp, n_vars, max_iters=500, damping=0.0)
        bp_spin = 1 - 2 * np.asarray(bp_bits, dtype=int)
        bp_energy = float(sh_solver.get_energy(bp_spin))
        result_df = sh_solver.simulated_annealing(
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            min_temp=min_temp,
            max_temp=max_temp,
            return_dataframe=True,
            verbose=False,
            use_pt=False,
            num_replicas=1,  # PT disabled -> single replica suffices
            parallel=bool(parallel_reads),
        )
        energies = np.asarray(result_df['best_energy'], dtype=float)
    else:
        raise ValueError("Energy distribution plot supported only for 'SA' and 'QUBO-PAPER'.")

    # Round energies for stable binning and build a probability histogram
    energies_r = np.round(np.asarray(energies, dtype=float), 10)
    unique_E, counts = np.unique(energies_r, return_counts=True)
    probs = counts / counts.sum()

    # Construct bin edges so each unique energy has its own bin
    if unique_E.size == 1:
        gap = 1.0
        edges = np.array([unique_E[0] - gap/2, unique_E[0] + gap/2])
    else:
        gaps = np.diff(unique_E)
        left = unique_E[0] - gaps[0] / 2
        right = unique_E[-1] + gaps[-1] / 2
        mids = (unique_E[:-1] + unique_E[1:]) / 2
        edges = np.concatenate([[left], mids, [right]])

    plt.figure(figsize=(8, 5))
    plt.hist(energies_r, bins=edges, weights=np.ones_like(energies_r) / energies_r.size,
             edgecolor='black', color='C0', label='SA/QUBO reads')
    # Overlay BP reference energy
    try:
        plt.axvline(bp_energy, color='r', linestyle='--', linewidth=1.5, label='BP energy')
    except Exception:
        pass
    plt.xlabel("Best Energy")
    plt.ylabel("Probability")
    plt.title(f"P(best energy) — {solver_name} @ {snr_db} dB")
    plt.legend()
    plt.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"energy_prob_{solver_name}_snr{snr_db}.png")
    plt.savefig(out_path, dpi=300)

    # Probability that best energy matches BP reference within tolerance
    # "Very close" interpreted as absolute tolerance to absorb FP noise
    eps = 1e-8
    diffs = np.abs(np.asarray(energies, dtype=float) - float(bp_energy))
    hits = int(np.sum(diffs <= eps))
    prob_bp = float(hits / max(1, len(energies)))

    return out_path, prob_bp, float(bp_energy), hits, int(len(energies))
