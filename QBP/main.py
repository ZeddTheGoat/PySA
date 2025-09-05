import os
import sys
import argparse

# Ensure both this folder and its parent are on sys.path for relative imports
_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))


def main():
    """
    Thin launcher for QBP experiments. Prefer running dedicated scripts:
      - BER vs SNR:            python PySA/QBP/ber_vs_snr.py
      - Energy vs Probability: python PySA/QBP/energy_vs_probability.py

    Or use this launcher with subcommands:
      python PySA/QBP/main.py ber
      python PySA/QBP/main.py energy
    """
    parser = argparse.ArgumentParser(description="QBP experiment launcher (deprecated). Use ber_vs_snr.py or energy_vs_probability.py.")
    subparsers = parser.add_subparsers(dest="cmd")

    subparsers.add_parser("ber", help="Run BER vs SNR experiment")
    subparsers.add_parser("energy", help="Run Probability vs Best Energy experiment")

    args = parser.parse_args()
    if args.cmd == "ber":
        try:
            import ber_vs_snr as ber
        except ImportError:
            from QBP import ber_vs_snr as ber  # fallback if run from package root
        ber.main()
    elif args.cmd == "energy":
        try:
            import energy_vs_probability as en
        except ImportError:
            from QBP import energy_vs_probability as en
        en.main()
    else:
        print("Main launcher is deprecated. Use one of:")
        print("  python PySA/QBP/ber_vs_snr.py")
        print("  python PySA/QBP/energy_vs_probability.py")
        print("Or:")
        print("  python PySA/QBP/main.py ber")
        print("  python PySA/QBP/main.py energy")


if __name__ == "__main__":
    main()
