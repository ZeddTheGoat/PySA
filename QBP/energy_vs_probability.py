import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Make parent importable if run as a script path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiment_utils import generate_ldpc_code
from ldpc_utils import ldpc_encode, add_awgn
from modulation_utils import get_modulation_params, map_binary_to_symbol, calculate_llrs
from problem_construction import build_parity_terms, hubo_dict_to_terms
from solvers import SOLVERS
from pysa.spinhubo import SpinHUBOSolver


def compute_llr_once(H, G, k, modulation_type, snr_db, seed=1234):
    """Generate one random message/codeword, transmit over AWGN, and compute LLRs for the first n variables.

    This keeps the instance fixed across hyperparameter sweeps.
    """
    rng = np.random.default_rng(seed)
    m, n = H.shape
    bits_per_symbol = get_modulation_params(modulation_type)

    message = rng.integers(0, 2, size=k)
    codeword = ldpc_encode(message, G)
    padding_needed = (bits_per_symbol - (len(codeword) % bits_per_symbol)) % bits_per_symbol
    padded_codeword = np.pad(codeword, (0, padding_needed), 'constant')

    symbols_binary_blocks = padded_codeword.reshape(-1, bits_per_symbol)
    modulated_symbols = np.array([map_binary_to_symbol(b, modulation_type) for b in symbols_binary_blocks])
    # Fix AWGN with deterministic RNG by seeding numpy's legacy RNG used in add_awgn
    np.random.seed(seed)
    received_symbols, sigma = add_awgn(modulated_symbols, snr_db)

    noise_variance = 2 * sigma**2
    llrs_full = calculate_llrs(received_symbols, noise_variance, modulation_type, padded_codeword.size)
    llr = llrs_full[:n]
    return llr


def build_spin_hubo_solver(H, llr, lambda_penalty):
    """Construct a Spin-HUBO solver for the LDPC instance H + LLR with penalty lambda."""
    n_vars = H.shape[1]
    hubo_sa = {}
    parity_terms = build_parity_terms(H, lambda_penalty)
    hubo_sa.update(parity_terms)
    for i in range(n_vars):
        hubo_sa[(i,)] = hubo_sa.get((i,), 0.0) + (-float(llr[i]))
    terms = hubo_dict_to_terms(hubo_sa)
    solver = SpinHUBOSolver(terms, n_vars)
    return solver


def compute_bp_reference_energy(H, llr, solver, max_iters=500, damping=0.0):
    """Decode with BP to get bits, map to spins, and evaluate energy under the given Spin-HUBO solver."""
    n_vars = H.shape[1]
    from collections import defaultdict
    hubo_bp = defaultdict(float)
    hubo_bp[('H_matrix',)] = H
    for i in range(n_vars):
        hubo_bp[(i,)] += -float(llr[i])
    bp_bits = SOLVERS['BP'](hubo_bp, n_vars, max_iters=max_iters, damping=damping)
    bp_spin = 1 - 2 * np.asarray(bp_bits, dtype=int)
    bp_energy = float(solver.get_energy(bp_spin))
    return bp_energy


def sa_sample_energies(
    solver,
    num_reads,
    num_sweeps,
    min_temp,
    max_temp,
    parallel_reads,
    use_pt=True,
    num_replicas=2,
):
    """Run SA/PT on the provided Spin-HUBO solver and return per-read best energies."""
    df = solver.simulated_annealing(
        num_sweeps=num_sweeps,
        num_reads=num_reads,
        min_temp=min_temp,
        max_temp=max_temp,
        return_dataframe=True,
        verbose=False,
        use_pt=use_pt,
        num_replicas=num_replicas,
        parallel=bool(parallel_reads),
    )
    return np.asarray(df['best_energy'], dtype=float)


def probability_leq_reference(energies, ref_energy, tol=1e-8):
    """Probability that energy is <= reference + tol (allows better-than-BP solutions)."""
    energies = np.asarray(energies, dtype=float)
    if energies.size == 0:
        return 0.0, 0
    thresh = float(ref_energy) + float(tol)
    hits = int(np.sum(energies <= thresh))
    prob = float(hits / energies.size)
    return prob, hits


def main():
    # Parameters
    N = 420
    d_v, d_c = 2, 3
    snr_list = [1, 2, 3, 4, 5]
    lambda_penalty = 2
    modulation_type = 'BPSK'
    num_reads = 2000
    num_sweeps = 5  # tuned: 5 sweeps appear sufficient
    min_temp = 0.1
    max_temp = 3.0
    parallel_reads = True  # run reads concurrently when supported
    fixed_instance_seed = 2025  # controls message/noise used to compute LLRs

    qbp_plots = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(qbp_plots, exist_ok=True)
    H, G, k, n = generate_ldpc_code(N, d_v, d_c, seed=42)

    # 2) Grid over (min_temp, max_temp) at fixed num_sweeps for SNRs 1..5
    # Temperature grids
    min_grid = np.linspace(0.01, 0.50, 11)
    max_grid = np.linspace(0.10, 5.00, 11)
    energy_tol = 1e-8

    for snr_db in snr_list:
        # Fix the decoding instance once per SNR (reproducible across runs)
        llr = compute_llr_once(H, G, k, modulation_type, snr_db, seed=fixed_instance_seed)

        # Build solver and BP reference energy
        solver = build_spin_hubo_solver(H, llr, lambda_penalty)
        bp_energy = compute_bp_reference_energy(H, llr, solver, max_iters=500, damping=0.0)

        # Quick single-run summary with default temps
        energies = sa_sample_energies(
            solver,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            min_temp=min_temp,
            max_temp=max_temp,
            parallel_reads=parallel_reads,
            use_pt=True,
            num_replicas=2,
        )
        p_bp, hits = probability_leq_reference(energies, bp_energy, tol=energy_tol)
        print(f"SNR={snr_db} | SA(PT-2) vs BP: hits={hits}/{energies.size} | P(E <= E_BP+tol) = {p_bp:.4f} | E_BP = {bp_energy:.6f} | tol={energy_tol}")

        prob_mat = np.zeros((len(min_grid), len(max_grid)), dtype=float)
        for i, tmin in enumerate(min_grid):
            for j, tmax in enumerate(max_grid):
                if tmax <= tmin:
                    prob_mat[i, j] = np.nan
                    continue
                en = sa_sample_energies(
                    solver,
                    num_reads=num_reads,
                    num_sweeps=num_sweeps,
                    min_temp=float(tmin),
                    max_temp=float(tmax),
                    parallel_reads=parallel_reads,
                    use_pt=True,
                    num_replicas=2,
                )
                p, _ = probability_leq_reference(en, bp_energy, tol=energy_tol)
                prob_mat[i, j] = p

        # Find best (min_temp, max_temp) pair by probability (ignore NaNs)
        if np.all(np.isnan(prob_mat)):
            best_info = None
            print(f"SNR={snr_db} | Warning: all (min,max) pairs invalid; no best temperature found.")
        else:
            best_flat = int(np.nanargmax(prob_mat))
            bi, bj = np.unravel_index(best_flat, prob_mat.shape)
            best_min = float(min_grid[bi])
            best_max = float(max_grid[bj])
            best_prob = float(prob_mat[bi, bj])
            best_info = (best_min, best_max, best_prob)
            print(f"SNR={snr_db} | Best temps: min={best_min}, max={best_max} with P(E<=E_BP+tol)={best_prob:.4f}")

        # 3D surface plot of probability vs (min_temp, max_temp)
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(max_grid, min_grid)
        Z = np.ma.masked_invalid(prob_mat)
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', vmin=0.0, vmax=1.0,
                               edgecolor='k', linewidth=0.3, antialiased=True, alpha=0.9)
        # Highlight the best point if available
        if best_info is not None:
            ax.scatter([best_max], [best_min], [best_prob], color='red', s=40, label='best')
            ax.legend(loc='upper left')
        fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label='P(E <= E_BP + tol)')
        ax.set_xlabel('max_temp (start T)')
        ax.set_ylabel('min_temp (end T)')
        ax.set_zlabel('P(E <= E_BP + tol)')
        ax.set_zlim(0.0, 1.0)
        ax.set_title(f'P(E <= E_BP+tol) 3D surface @ {snr_db} dB, sweeps={num_sweeps}')
        fig.tight_layout()
        out_curve = os.path.join(qbp_plots, f"prob_best_vs_minmax_snr{snr_db}.png")
        fig.savefig(out_curve, dpi=300)
        print(f"Saved 3D grid plot to {out_curve}")


if __name__ == "__main__":
    main()
