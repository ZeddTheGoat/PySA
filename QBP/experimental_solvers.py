"""
Experimental and supplemental solver wrappers for QBP.

These are not referenced by the main simulation flow and are kept separate
to keep the primary SOLVERS registry minimal. Import directly if needed.
"""

import numpy as np
from pysa.spinhubo import SpinHUBOSolver
from pysa.hubo import HUBOSolver
from pysa.sa import Solver as QuboSolver


def _spin_hubo_dict_to_terms(hubo_dict):
    terms = []
    for key in hubo_dict:
        if not (isinstance(key, tuple) and all(isinstance(k, (int, np.integer)) for k in key)):
            continue
        terms.append((float(hubo_dict[key]), list(key)))
    return terms


def parallel_tempering_hubo(hubo_dict, n_spins, num_sweeps=100, num_reads=1, num_replicas=8, min_temp=0.05, max_temp=10.0, verbose=False):
    """Spin-HUBO with Parallel Tempering via PySA SpinHUBOSolver."""
    terms = _spin_hubo_dict_to_terms(hubo_dict)
    solver = SpinHUBOSolver(terms, n_spins)
    result_df = solver.simulated_annealing(
        num_sweeps=num_sweeps,
        num_reads=num_reads,
        use_pt=True,
        num_replicas=num_replicas,
        min_temp=min_temp,
        max_temp=max_temp,
        verbose=verbose,
    )
    return result_df['best_state'][0]


def adaptive_pt_from_paper(hubo_dict, n_spins, **kwargs):
    """Placeholder for APT schedule; currently aliases to PT."""
    return parallel_tempering_hubo(hubo_dict, n_spins, **kwargs)


def _expand_spin_monomial_to_binary_terms(indices, coeff):
    idx = list(indices)
    out = []
    L = len(idx)
    import itertools
    for k in range(1, L + 1):
        factor = (-2.0) ** k
        for combo in itertools.combinations(idx, k):
            out.append((float(coeff) * factor, list(combo)))
    return out


def _spin_hubo_dict_to_binary_terms(hubo_dict):
    from collections import defaultdict
    acc = defaultdict(float)
    for key, coeff in hubo_dict.items():
        if not (isinstance(key, tuple) and all(isinstance(k, (int, np.integer)) for k in key)):
            continue
        for c, inds in _expand_spin_monomial_to_binary_terms(key, coeff):
            acc[tuple(sorted(int(i) for i in inds))] += c
    return [(float(c), list(k)) for k, c in acc.items()]


def simulated_annealing_hubo_binary(hubo_dict, n_vars, num_sweeps=100, num_reads=1, min_temp=0.3, max_temp=3.0, verbose=False):
    """Binary-HUBO SA via PySA HUBOSolver (requires spin→binary expansion)."""
    terms = _spin_hubo_dict_to_binary_terms(hubo_dict)
    solver = HUBOSolver(terms, n_vars)
    result_df = solver.simulated_annealing(
        num_sweeps=num_sweeps,
        num_reads=num_reads,
        min_temp=min_temp,
        max_temp=max_temp,
        verbose=verbose,
    )
    return result_df['best_state'][0]


def qubo_sa_from_hubo(hubo_dict, n_vars, num_sweeps=100, num_reads=1, min_temp=0.3, max_temp=3.0, penalty_strength=None, verbose=False):
    """
    Spin→Binary expansion + Rosenberg quadratization to QUBO, then PySA QUBO SA.
    """
    try:
        from QBP.problem_construction import (
            spin_hubo_dict_to_binary_terms,
            quadratize_binary_terms_to_qubo,
        )
    except Exception:
        from problem_construction import (
            spin_hubo_dict_to_binary_terms,
            quadratize_binary_terms_to_qubo,
        )

    terms = spin_hubo_dict_to_binary_terms(hubo_dict)
    Q, total_n = quadratize_binary_terms_to_qubo(terms, n_vars, penalty_strength=penalty_strength)
    solver = QuboSolver(Q, problem_type='qubo')
    df = solver.metropolis_update(
        num_sweeps=num_sweeps,
        num_reads=num_reads,
        min_temp=min_temp,
        max_temp=max_temp,
        return_dataframe=True,
        parallel=False,
        use_pt=False,
        verbose=verbose,
    )
    return df['best_state'][0]

