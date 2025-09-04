import numpy as np
from pysa.hubo import HUBOSolver
from pysa.spinhubo import SpinHUBOSolver


def enum_binary_min_energy(terms, n_vars):
    best = None
    best_state = None
    for x in range(1 << n_vars):
        state = [(x >> i) & 1 for i in range(n_vars)]
        energy = 0.0
        for coef, idxs in terms:
            prod = 1
            for i in idxs:
                prod *= state[i]
            energy += coef * prod
        if best is None or energy < best:
            best = energy
            best_state = state[:]
    return best, np.array(best_state, dtype=int)


def enum_spin_min_energy(terms, n_vars):
    best = None
    best_state = None
    for mask in range(1 << n_vars):
        state = [1 if (mask >> i) & 1 else -1 for i in range(n_vars)]
        energy = 0.0
        for coef, idxs in terms:
            prod = 1
            for i in idxs:
                prod *= state[i]
            energy += coef * prod
        if best is None or energy < best:
            best = energy
            best_state = state[:]
    return best, np.array(best_state, dtype=int)


def make_random_terms(n_vars, max_order=3, density=0.4, scale=2.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    terms = []
    for order in range(1, max_order+1):
        # try many combinations
        from itertools import combinations
        for comb in combinations(range(n_vars), order):
            if rng.random() < density:
                coef = float(rng.normal(0, scale))
                terms.append((coef, list(comb)))
    return terms


def test_hubo_solver():
    rng = np.random.default_rng(123)
    n_vars = 6
    terms = make_random_terms(n_vars, max_order=3, density=0.4, scale=3.0, rng=rng)
    true_energy, true_state = enum_binary_min_energy(terms, n_vars)

    solver = HUBOSolver(terms, n_vars)
    res = solver.simulated_annealing(num_sweeps=200, num_reads=4, min_temp=0.1, max_temp=3.0, verbose=False)
    found_energy = float(res['best_energy'][0])
    assert np.isclose(found_energy, true_energy, atol=1e-6)


def test_spinhubo_solver():
    rng = np.random.default_rng(456)
    n_vars = 6
    terms = make_random_terms(n_vars, max_order=3, density=0.4, scale=3.0, rng=rng)
    true_energy, true_state = enum_spin_min_energy(terms, n_vars)

    solver = SpinHUBOSolver(terms, n_vars)
    res = solver.simulated_annealing(num_sweeps=200, num_reads=4, min_temp=0.1, max_temp=3.0, verbose=False)
    found_energy = float(res['best_energy'][0])
    assert np.isclose(found_energy, true_energy, atol=1e-6)
