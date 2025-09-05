import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from time import time
from typing import List, Tuple, Any
import numba
from concurrent.futures import ProcessPoolExecutor
from .utils import normalize_higher_order_terms

def preprocess_spinhubo_terms(
    terms: List[Tuple[float, List[int]]], 
    n_vars: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess Spin-HUBO terms into Numba-friendly structures.
    """
    n_terms = len(terms)
    max_order = max((len(idxs) for _, idxs in terms), default=0)
    coefs = np.array([coef for coef, idxs in terms], dtype=np.float64)
    term_lens = np.array([len(idxs) for _, idxs in terms], dtype=np.int32)
    term_indices = np.full((n_terms, max_order), -1, dtype=np.int32)
    for i, (_, idxs) in enumerate(terms):
        term_indices[i, :len(idxs)] = idxs

    terms_by_var_lists: List[List[int]] = [[] for _ in range(n_vars)]
    for t_idx, (_, idxs) in enumerate(terms):
        for v in idxs:
            terms_by_var_lists[v].append(t_idx)
    max_terms_per_var = max((len(lst) for lst in terms_by_var_lists), default=0) if n_vars > 0 else 0
    terms_by_var = np.full((n_vars, max_terms_per_var), -1, dtype=np.int32)
    for v in range(n_vars):
        terms_by_var[v, :len(terms_by_var_lists[v])] = terms_by_var_lists[v]
    return coefs, term_indices, term_lens, terms_by_var

@numba.njit
def get_energy_spinhubo_numba(
    state: np.ndarray,
    coefs: np.ndarray,
    term_indices: np.ndarray,
    term_lens: np.ndarray
) -> float:
    energy = 0.0
    n_terms = coefs.shape[0]
    for t in range(n_terms):
        prod = 1
        for j in range(term_lens[t]):
            idx = term_indices[t, j]
            prod *= state[idx]
        energy += coefs[t] * prod
    return energy

@numba.njit
def delta_energy_spinhubo_numba(
    state: np.ndarray,
    i: int,
    coefs: np.ndarray,
    term_indices: np.ndarray,
    term_lens: np.ndarray,
    terms_by_var: np.ndarray
) -> float:
    dE = 0.0
    for k in range(terms_by_var.shape[1]):
        t = terms_by_var[i, k]
        if t == -1:
            break
        prod = 1
        for j in range(term_lens[t]):
            idx = term_indices[t, j]
            prod *= state[idx]
        dE += -2 * coefs[t] * prod
    return dE

class SpinHUBOSolver:
    """
    Numba-accelerated Simulated Annealing and Parallel Tempering solver for Higher-Order Unconstrained Spin Optimization (Spin-HUBO).
    Spin variables: s_i ∈ {-1, +1}
    """

    def __init__(
        self,
        terms: List[Tuple[float, List[int]]],
        n_vars: int,
        float_type: Any = 'float32'
    ):
        """
        Args:
            terms: list of (coefficient, [variable indices])
            n_vars: int, number of spin variables
        """
        # Validate and coalesce terms
        self.terms = normalize_higher_order_terms(terms, n_vars, zero_tol=0.0, sort_indices=True, allow_constant=True, variable_domain='spin')
        self.n_vars = n_vars
        self.float_type = float_type

        # Preprocess terms for numba
        self.coefs, self.term_indices, self.term_lens, self.terms_by_var = preprocess_spinhubo_terms(terms, n_vars)

    def get_energy(self, state: np.ndarray) -> float:
        state = np.asarray(state, dtype=np.int8)
        return get_energy_spinhubo_numba(state, self.coefs, self.term_indices, self.term_lens)

    def delta_energy(self, state: np.ndarray, i: int) -> float:
        state = np.asarray(state, dtype=np.int8)
        return delta_energy_spinhubo_numba(state, i, self.coefs, self.term_indices, self.term_lens, self.terms_by_var)

    def simulated_annealing(
        self,
        num_sweeps: int,
        num_reads: int = 1,
        min_temp: float = 0.3,
        max_temp: float = 3.0,
        schedule: Any = None,
        return_dataframe: bool = True,
        verbose: bool = False,
        seed: Any = None,
        use_pt: bool = False,
        num_replicas: int = 4,
        record_history: bool = False,
        parallel: bool = False,
    ):
        """
        Simulated annealing for Spin-HUBO, with optional Parallel Tempering (PT).

        Args:
            num_sweeps: number of sweeps per read
            num_reads: independent restarts
            min_temp, max_temp: temperature endpoints
            schedule: custom schedule (disabled if use_pt=True)
            return_dataframe: return pandas DataFrame if True
            verbose: tqdm progress
            seed: RNG seed
            use_pt: enable parallel tempering
            num_replicas: number of PT replicas
            record_history: if True, store per-sweep states/energies (memory heavy)
            parallel: if True, parallelize independent reads across processes
        """
        rng = np.random.default_rng(seed)
        results = []

        if schedule is not None and use_pt:
            raise ValueError("Custom schedules not supported with PT.")

        if parallel and record_history:
            raise ValueError("record_history is not supported with parallel=True.")

        # Parallel path: distribute independent reads across processes
        if parallel and num_reads > 1:
            seeds = rng.integers(0, 2**32 - 1, size=num_reads, dtype=np.uint32)
            args_list = [
                (
                    int(seeds[i]), int(num_sweeps), float(min_temp), float(max_temp),
                    bool(use_pt), int(num_replicas),
                    int(self.n_vars), self.coefs, self.term_indices, self.term_lens, self.terms_by_var
                )
                for i in range(num_reads)
            ]
            with ProcessPoolExecutor() as ex:
                for rec in tqdm(ex.map(_run_single_read_spinhubo, args_list), total=num_reads, disable=not verbose):
                    results.append(rec)
            return pd.DataFrame(results) if return_dataframe else results

        for run in tqdm(range(num_reads), disable=not verbose):
            if use_pt:
                # --- Parallel Tempering ---
                temps = np.linspace(max_temp, min_temp, num_replicas)
                betas = 1.0 / temps

                # Initialize replicas randomly (-1/+1)
                states = rng.choice([-1, 1], size=(num_replicas, self.n_vars)).astype(np.int8)
                energies = np.array([self.get_energy(states[r]) for r in range(num_replicas)])
                best_energy = energies.min()
                best_state = states[energies.argmin()].copy()

                if record_history:
                    states_history = [states.copy()]
                    energies_history = [energies.copy()]

                t0 = time()
                for sweep in range(num_sweeps):
                    # Local moves in each replica
                    for r in range(num_replicas):
                        for i in range(self.n_vars):
                            dE = delta_energy_spinhubo_numba(states[r], i, self.coefs, self.term_indices, self.term_lens, self.terms_by_var)
                            if dE <= 0 or rng.random() < np.exp(-dE / temps[r]):
                                states[r, i] *= -1
                                energies[r] += dE
                                if energies[r] < best_energy:
                                    best_energy = energies[r]
                                    best_state = states[r].copy()

                    # Attempt swaps between adjacent replicas
                    for r in range(num_replicas - 1):
                        d_beta = betas[r+1] - betas[r]
                        d_energy = energies[r+1] - energies[r]
                        delta = d_beta * d_energy
                        # Accept always if delta >= 0; otherwise accept with exp(delta)
                        if delta >= 0 or rng.random() < np.exp(delta):
                            # Swap states and energies
                            states[[r, r+1]] = states[[r+1, r]]
                            energies[[r, r+1]] = energies[[r+1, r]]

                    if record_history:
                        states_history.append(states.copy())
                        energies_history.append(energies.copy())
                t1 = time()
                rec = {
                    'best_state': best_state.copy(),
                    'best_energy': best_energy,
                    'runtime (us)': int((t1 - t0) * 1e6)
                }
                if record_history:
                    rec['states'] = np.array(states_history)
                    rec['energies'] = np.array(energies_history)
                results.append(rec)
            else:
                # --- Standard SA ---
                if schedule is None:
                    schedule = np.linspace(max_temp, min_temp, num_sweeps)
                else:
                    schedule = np.asarray(schedule)
                state = rng.choice([-1, 1], size=self.n_vars).astype(np.int8)
                energy = self.get_energy(state)
                best_state = state.copy()
                best_energy = energy
                if record_history:
                    energies = [energy]
                    states_list = [state.copy()]
                t0 = time()
                for sweep_idx, T in enumerate(schedule):
                    for i in range(self.n_vars):
                        dE = delta_energy_spinhubo_numba(state, i, self.coefs, self.term_indices, self.term_lens, self.terms_by_var)
                        if dE <= 0 or rng.random() < np.exp(-dE / T):
                            state[i] *= -1
                            energy += dE
                            if energy < best_energy:
                                best_energy = energy
                                best_state = state.copy()
                    if record_history:
                        energies.append(energy)
                        states_list.append(state.copy())
                t1 = time()
                rec = {
                    'best_state': best_state.copy(),
                    'best_energy': best_energy,
                    'runtime (us)': int((t1 - t0) * 1e6)
                }
                if record_history:
                    rec['states'] = np.array(states_list)
                    rec['energies'] = np.array(energies)
                results.append(rec)

        if return_dataframe:
            return pd.DataFrame(results)
        else:
            return results


def _run_single_read_spinhubo(args_tuple):
    """Run a single SA/PT read for Spin-HUBO in a separate process."""
    (seed, num_sweeps, min_temp, max_temp, use_pt, num_replicas,
     n_vars, coefs, term_indices, term_lens, terms_by_var) = args_tuple
    rng = np.random.default_rng(int(seed))
    t0 = time()
    if use_pt:
        temps = np.linspace(max_temp, min_temp, num_replicas)
        betas = 1.0 / temps
        states = rng.choice([-1, 1], size=(num_replicas, n_vars)).astype(np.int8)
        energies = np.array([get_energy_spinhubo_numba(states[r], coefs, term_indices, term_lens) for r in range(num_replicas)])
        best_energy = energies.min()
        best_state = states[energies.argmin()].copy()
        for _ in range(num_sweeps):
            for r in range(num_replicas):
                for i in range(n_vars):
                    dE = delta_energy_spinhubo_numba(states[r], i, coefs, term_indices, term_lens, terms_by_var)
                    if dE <= 0 or rng.random() < np.exp(-dE / temps[r]):
                        states[r, i] *= -1
                        energies[r] += dE
                        if energies[r] < best_energy:
                            best_energy = energies[r]
                            best_state = states[r].copy()
            for r in range(num_replicas - 1):
                d_beta = betas[r+1] - betas[r]
                d_energy = energies[r+1] - energies[r]
                delta = d_beta * d_energy
                if delta >= 0 or rng.random() < np.exp(delta):
                    states[[r, r+1]] = states[[r+1, r]]
                    energies[[r, r+1]] = energies[[r+1, r]]
    else:
        schedule = np.linspace(max_temp, min_temp, num_sweeps)
        state = rng.choice([-1, 1], size=n_vars).astype(np.int8)
        energy = get_energy_spinhubo_numba(state, coefs, term_indices, term_lens)
        best_state = state.copy()
        best_energy = energy
        for T in schedule:
            for i in range(n_vars):
                dE = delta_energy_spinhubo_numba(state, i, coefs, term_indices, term_lens, terms_by_var)
                if dE <= 0 or rng.random() < np.exp(-dE / T):
                    state[i] *= -1
                    energy += dE
                    if energy < best_energy:
                        best_energy = energy
                        best_state = state.copy()
    t1 = time()
    return {
        'best_state': best_state.copy(),
        'best_energy': float(best_energy),
        'runtime (us)': int((t1 - t0) * 1e6),
    }

# Example usage:
# terms = [
#     (3.0, [0, 1]),
#     (2.0, [1, 2, 3]),
#     (-4.0, [0]),
# ]
# solver = SpinHUBOSolver(terms, n_vars=4)
# result_df = solver.simulated_annealing(num_sweeps=100, num_reads=5, verbose=True)
# pt_df = solver.simulated_annealing(num_sweeps=100, num_reads=5, use_pt=True, num_replicas=6, verbose=True)
# print(result_df[['best_state', 'best_energy']])
