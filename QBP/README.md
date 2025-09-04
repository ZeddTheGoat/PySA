# QBP LDPC testing folder

This folder contains LDPC testing utilities and the LDPC pipeline implementation.

Key files:
- main.py: main LDPC experiment runner (uses pysa solvers in pysa/)
- ldpc_utils.py: LDPC encode / noise functions
- modulation_utils.py: modulation helpers
- problem_construction.py: build HUBO from received symbols

Removed/archived scripts:
- Temporary test and smoke scripts were archived and/or moved to the repository-level `tests/` directory. See `QBP/tools/archived_files.md` for original content references.

Notes:
- The authoritative solver implementations are in `pysa/` (`h ubo.py`, `spinhubo.py`).
- Canonical tests: `tests/test_hubo_integration.py`.
