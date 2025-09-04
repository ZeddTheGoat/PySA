import os
import sys
# Now import as:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from pyldpc import make_ldpc
from tqdm import tqdm

# Import functions from our modularized files
from ldpc_utils import ldpc_encode, add_awgn
from modulation_utils import get_modulation_params, map_binary_to_symbol, calculate_llrs
from problem_construction import build_parity_terms, allocate_parity_slack_ancillas
from solvers import SOLVERS, spin_to_binary


def simulate_ldpc_PySA(snr_db, H, G, k, n, modulation_type, num_trials, lambda_penalty, solver_name, formulation_type):
    """
    Main simulation loop for a single SNR point using a specified solver and formulation.

    Supported PySA-backed solver flows used for comparison:
    - BP -> classic belief propagation baseline (key: 'BP').
    - Spin HUBO SA -> pysa.spinhubo.SpinHUBOSolver (key: 'SA').
    - QUBO (paper) -> PySA QUBO solver from parity-slack formulation (key: 'QUBO-PAPER').

    Remains compatible with the existing SOLVERS dispatcher for legacy spin-based functions.
    """
    ber_total = 0
    bits_per_symbol = get_modulation_params(modulation_type)

    # Dimensions
    m, original_n = H.shape
    # Precompute only what's needed per solver
    parity_terms = None
    ancilla_plan = None
    if solver_name == 'SA':
        # Spin-HUBO SA needs parity penalty terms
        parity_terms = build_parity_terms(H, lambda_penalty)
    elif solver_name == 'QUBO-PAPER':
        # Paper QUBO uses ancilla plan for slack bits
        ancilla_plan, _ = allocate_parity_slack_ancillas(H)

    desc = f"{solver_name}({formulation_type[:3]}) {modulation_type} @ {snr_db} dB"
    for _ in tqdm(range(num_trials), desc=desc, leave=False):
        message = np.random.randint(0, 2, k)
        codeword = ldpc_encode(message, G)

        # Compute padding based on actual codeword length (len_codeword)
        len_codeword = len(codeword)
        padding_needed = (bits_per_symbol - (len_codeword % bits_per_symbol)) % bits_per_symbol
        padded_codeword = np.pad(codeword, (0, padding_needed), 'constant')
        # n_vars_padded is the number of bits after padding; solver variables correspond to original code length
        n_vars_padded = padded_codeword.size
        n_vars = original_n

        symbols_binary_blocks = padded_codeword.reshape(-1, bits_per_symbol)
        modulated_symbols = np.array([map_binary_to_symbol(b, modulation_type) for b in symbols_binary_blocks])

        received_symbols, sigma = add_awgn(modulated_symbols, snr_db)

        # Compute LLRs (vector) once per trial for BP and QUBO; also used to build SA HUBO linear terms if needed
        noise_variance = 2 * sigma**2
        llrs_full = calculate_llrs(received_symbols, noise_variance, modulation_type, n_vars_padded)
        llr = llrs_full[:original_n]

        if solver_name == 'BP':
            # Minimal hubo_dict for BP: H and linear terms only (stored as -LLR)
            from collections import defaultdict
            hubo_bp = defaultdict(float)
            hubo_bp[('H_matrix',)] = H
            for i in range(n_vars):
                hubo_bp[(i,)] = -llr[i]
            decoded_codeword = SOLVERS['BP'](hubo_bp, n_vars, max_iters=5, damping=0.0)
        elif solver_name == 'QUBO-PAPER':
            # Directly build Q from H and LLR; skip HUBO construction entirely
            # Call dedicated helper that takes H, llr
            from solvers import qubo_sa_paper_from_H_llr
            result_state = qubo_sa_paper_from_H_llr(H, llr, num_sweeps=50, num_reads=5, min_temp=0.1, max_temp=3.0, penalty_strength=lambda_penalty, verbose=False, ancilla_plan=ancilla_plan)
            decoded_codeword = np.asarray(result_state, dtype=int)[:n_vars]
        else:
            # Spin-HUBO SA: build parity + linear terms as HUBO dict
            from collections import defaultdict
            hubo_sa = defaultdict(float)
            # build single-variable dict from llr vector: store as -LLR
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


# --- Main Execution ---
if __name__ == "__main__":
    # Parameters
    N = 120
    d_v, d_c = 2, 3
    snr_range = np.arange(0, 7, 1)
    num_trials = 100
    lambda_penalty = 2
    modulation_type = 'BPSK'
    # Compare BP, Spin-HUBO SA, and paper QUBO by default
    solvers_to_compare = ['BP', 'SA', 'QUBO-PAPER']
    formulations_to_compare = ['llr']

    # Setup
    np.random.seed(42)
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Generate H and G (pyldpc may return sparse matrices). Convert to dense numpy arrays when necessary.
    H_raw, G_raw = make_ldpc(N, d_v, d_c, systematic=True, sparse=True)
    if hasattr(H_raw, 'toarray'):
        H_array = H_raw.toarray()
    else:
        H_array = np.asarray(H_raw)
    if hasattr(G_raw, 'toarray'):
        G_array = G_raw.toarray()
    else:
        G_array = np.asarray(G_raw)
    k = G_array.shape[1]
    print(H_array.shape)
    print(G_array.shape)

    # Verify actual code length matches requested N
    actual_n = H_array.shape[1]
    if actual_n != N:
        print(f"Warning: requested N={N} but generated code length is {actual_n}. Using actual length.")
        N = actual_n
    rate = k / N

    print(f"LDPC Code: N={N}, k={k}, Rate={rate:.2f}")
    print(f"Modulation: {modulation_type}, HUBO Penalty (lambda): {lambda_penalty}")

    # Run Simulation
    all_bers = {}
    for solver in solvers_to_compare:
        for formulation in formulations_to_compare:
            run_name = f"{solver}-{formulation}"
            print(f"\n--- Running: {run_name} ---")
            bers = []
            for snr in snr_range:
                actual_n = H_array.shape[1]
                ber = simulate_ldpc_PySA(snr, H_array, G_array, k, actual_n, modulation_type, num_trials, lambda_penalty, solver, formulation)
                print(f"  SNR = {snr:2d} dB -> BER = {ber:.6f}")
                bers.append(ber)
            all_bers[run_name] = bers

    # Plotting Results
    plt.figure(figsize=(10, 7))
    for run_name, bers_list in all_bers.items():
        plot_snr = [snr for snr, ber in zip(snr_range, bers_list) if ber > 0]
        plot_ber = [ber for ber in bers_list if ber > 0]
        if plot_ber:
            plt.semilogy(plot_snr, plot_ber, marker='o', linestyle='-', label=f'{run_name}')

    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"Solver & Formulation Comparison for LDPC HUBO Decoding (N={N})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.ylim(bottom=1e-5, top=1.0)
    plt.tight_layout()

    # Include code parameters in filename for easier identification
    plot_filename = f"plots/formulation_comparison_dv{d_v}_dc{d_c}_N{N}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\nPlot saved to {plot_filename}")
