"""
Copyright © 2023, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The PySA, a powerful tool for solving optimization problems is licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from typing import List, NoReturn, Tuple, Any, Iterable
import numpy as np
import scipy
import numba
from warnings import warn

Vector = List[float]
Matrix = List[List[float]]
State = List[float]


@numba.njit(cache=True, fastmath=True, nogil=True, parallel=False)
def pt(states: List[State], energies: List[float], beta_idx: List[int],
       betas: List[float]) -> NoReturn:
    """
  Parallel tempering move.
    states: [n_replicas, ...]  Array of replicas
    energies: [n_replicas] Array of energies of each replica
    beta_idx: [n_replicas] The replica index currently assigned to each beta,
        i.e. inverse temperature K is currently used for simulating replica beta_idx[K]
    betas: [n_replicas] Sequential array of inverse temperatures.

    This function only modifies the order of beta_idx.
  """

    # Get number of replicas
    n_replicas = len(states)

    # Apply PT for each pair of replicas
    for k in range(n_replicas - 1):

        # Get first index
        k1 = n_replicas - k - 1

        # Get second index
        k2 = n_replicas - k - 2

        # Compute delta energy
        de = (energies[beta_idx[k1]] - energies[beta_idx[k2]]) * (betas[k1] -
                                                                  betas[k2])

        # Accept/reject following Metropolis
        if de >= 0 or np.random.random() < np.exp(de):
            beta_idx[k1], beta_idx[k2] = beta_idx[k2], beta_idx[k1]


def get_problem_matrix(instance: List[Tuple[int, int, float]]) -> Matrix:
    """
    Generate problem matrix from a list of interactions.
    """

    from more_itertools import flatten
    from collections import Counter

    # Check that couplings are not repeated
    if set(Counter(tuple(sorted(x[:2])) for x in instance).values()) != {1}:
        warn("Duplicated couplings are ignored.")

    # Get list of variables
    _vars = sorted(set(flatten(x[:2] for x in instance)))

    # Get number of variables
    _n_vars = len(_vars)

    # Get map
    _map_vars = {x: i for i, x in enumerate(_vars)}

    # Get problem
    problem = np.zeros((_n_vars, _n_vars))
    for x, y, J in instance:
        x = _map_vars[x]
        y = _map_vars[y]
        problem[y][x] = problem[x][y] = J

    return problem


def normalize_higher_order_terms(
    terms: List[Tuple[float, List[int]]],
    n_vars: int,
    *,
    zero_tol: float = 0.0,
    sort_indices: bool = True,
    allow_constant: bool = True,
    variable_domain: str = 'binary',
) -> List[Tuple[float, List[int]]]:
    """
    Validate and coalesce higher-order terms for HUBO/Spin-HUBO.

    - Ensures indices are ints within [0, n_vars).
    - Collapses repeated indices inside a term (x_i^k -> x_i, s_i^k -> s_i).
    - Sorts indices and merges duplicate index-sets by summing coefficients.
    - Drops near-zero coefficients (|c| <= zero_tol).
    - Keeps constant terms ([]) only if allow_constant=True.

    Args:
        terms: list of (coef, [indices])
        n_vars: number of variables
        zero_tol: coefficients with abs <= zero_tol are removed
        sort_indices: sort indices in each term for canonicalization
        allow_constant: keep [] terms if present
        variable_domain: 'binary' or 'spin' (duplicates are collapsed in both)
    Returns:
        normalized list of (coef, [indices])
    """
    from collections import defaultdict

    if variable_domain not in ('binary', 'spin'):
        raise ValueError("variable_domain must be 'binary' or 'spin'")

    acc = defaultdict(float)
    for coef, idxs in terms:
        # Validate coefficient
        c = float(coef)
        # Validate indices
        clean = []
        for i in idxs:
            if not isinstance(i, (int, np.integer)):
                raise ValueError(f"Index {i} is not an integer")
            if i < 0 or i >= n_vars:
                raise ValueError(f"Index {i} out of bounds for n_vars={n_vars}")
            clean.append(int(i))

        # Collapse duplicates within a term, domain-aware
        if len(clean) >= 1:
            from collections import Counter
            cnt = Counter(clean)
            if variable_domain == 'spin':
                # keep variables with odd multiplicity
                clean = [i for i, k in cnt.items() if (k % 2) == 1]
            else:  # binary
                clean = list(cnt.keys())

        if sort_indices:
            clean.sort()

        # Handle constant term
        if len(clean) == 0 and not allow_constant:
            continue

        acc[tuple(clean)] += c

    out = []
    for key, c in acc.items():
        if abs(c) <= zero_tol:
            continue
        out.append((float(c), list(key)))

    return out
