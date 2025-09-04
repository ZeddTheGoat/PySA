Archived QBP temporary scripts

The following items were archived/relocated from QBP/ to keep the directory focused on the LDPC pipeline implementation:

- test_hubo_integration.py -> moved to tests/test_hubo_integration.py
- check_hubo_core.py -> archived (checks remain in tests/ and QBP/check_hubo_core.py can be re-added if needed)

Solver wrappers reorganization:
- Parallel tempering (spin), binary-HUBO SA, and Rosenberg QUBO wrappers were moved out of the primary `solvers.py` into `QBP/experimental_solvers.py`.
  These are not used by the main simulation defaults (`BP`, `SA`, `QUBO-PAPER`) but are available for direct import.
- run_smoketest.py -> archived (if needed, a lightweight smoke test exists elsewhere)

These files were removed from QBP/ in the cleanup to keep the directory focused on the LDPC pipeline implementation. If you need any archived content restored, please check git history or contact the maintainer.
