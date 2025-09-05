# QBP LDPC testing folder

This folder contains LDPC testing utilities and the LDPC pipeline implementation.

Key files:
- main.py: deprecated thin launcher; prefer direct scripts below
- ber_vs_snr.py: BER vs SNR experiment entrypoint
- energy_vs_probability.py: probability vs best energy experiment entrypoint
- experiment_utils.py: shared helpers for experiments
- ldpc_utils.py: LDPC encode / noise functions
- modulation_utils.py: modulation helpers
- problem_construction.py: build HUBO/QUBO from parity/LLR

Removed/archived scripts:
- Temporary test and smoke scripts were archived and/or moved to the repository-level `tests/` directory. See `QBP/tools/archived_files.md` for original content references.

Notes:
- The authoritative solver implementations are in `pysa/` (`h ubo.py`, `spinhubo.py`).
- Canonical tests: `tests/test_hubo_integration.py`.

Usage:
- BER curves: `python PySA/QBP/ber_vs_snr.py`
- Energy distribution: `python PySA/QBP/energy_vs_probability.py`
- Launcher (deprecated):
  - `python PySA/QBP/main.py ber`
  - `python PySA/QBP/main.py energy`
