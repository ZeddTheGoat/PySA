import os
import sys
import traceback
# ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyldpc import make_ldpc

from main import simulate_ldpc_PySA


# Archived: lightweight smoke test moved or consolidated elsewhere.
# If you need to run a smoke test, use tests or QBP/tools/archived_files.md for original script.

if __name__ == '__main__':
    print('This smoke test has been archived. See QBP/tools/archived_files.md or tests/ for alternatives.')
