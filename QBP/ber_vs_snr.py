import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Make parent importable if run as a script path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiment_utils import generate_ldpc_code, simulate_ldpc_PySA


def main():
    # Parameters
    N = 120
    d_v, d_c = 2, 3
    snr_range = np.arange(0, 7, 1)
    num_trials = 100
    lambda_penalty = 2
    modulation_type = 'BPSK'
    solvers_to_compare = ['BP', 'SA', 'QUBO-PAPER']
    formulations_to_compare = ['llr']

    # Setup
    qbp_plots = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(qbp_plots, exist_ok=True)
    H, G, k, n = generate_ldpc_code(N, d_v, d_c, seed=42)
    rate = k / n
    print(f"LDPC Code: N={n}, k={k}, Rate={rate:.2f}")
    print(f"Modulation: {modulation_type}, Penalty (lambda): {lambda_penalty}")

    all_bers = {}
    for solver in solvers_to_compare:
        for formulation in formulations_to_compare:
            run_name = f"{solver}-{formulation}"
            print(f"\n--- Running: {run_name} ---")
            bers = []
            for snr in snr_range:
                ber = simulate_ldpc_PySA(snr, H, G, k, n, modulation_type, num_trials, lambda_penalty, solver, formulation)
                print(f"  SNR = {snr:2d} dB -> BER = {ber:.6f}")
                bers.append(ber)
            all_bers[run_name] = bers

    # Plotting
    plt.figure(figsize=(10, 7))
    for run_name, bers_list in all_bers.items():
        plot_snr = [snr for snr, ber in zip(snr_range, bers_list) if ber > 0]
        plot_ber = [ber for ber in bers_list if ber > 0]
        if plot_ber:
            plt.semilogy(plot_snr, plot_ber, marker='o', linestyle='-', label=f'{run_name}')

    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"BER vs SNR for LDPC (N={n}, dv={d_v}, dc={d_c})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.ylim(bottom=1e-5, top=1.0)
    plt.tight_layout()

    plot_filename = os.path.join(qbp_plots, f"ber_vs_snr_dv{d_v}_dc{d_c}_N{n}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"\nBER plot saved to {plot_filename}")


if __name__ == "__main__":
    main()
