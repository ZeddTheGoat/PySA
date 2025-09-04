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

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Any, Optional, Dict
from tqdm.auto import tqdm
from time import time
import numpy as np
import pandas as pd
import pysa.qubo
import pysa.ising
import pysa.simulation
import pysa.ais
import pysa.hubo
import pysa.spinhubo

Vector = List[float]

class Solver:
    """
    Unified solver interface for QUBO, Ising, HUBO, and Spin-HUBO problems.
    For HUBO and Spin-HUBO, uses Numba-accelerated solvers.
    """

    def __init__(
        self,
        problem: Any,
        problem_type: str,
        float_type: Any = 'float32'
    ):
        """
        Args:
            problem: 
                - For QUBO or Ising: square numpy.ndarray
                - For HUBO: dict with 'terms' (list of (coef, [idxs])) and 'n_vars'
                - For Spin-HUBO: dict with 'terms' (list of (coef, [idxs])) and 'n_vars'
            problem_type: 'qubo', 'ising', 'hubo', or 'spinhubo'
            float_type: dtype for computation
        """
        self.problem_type = problem_type
        self.float_type = float_type

        if problem_type == 'qubo':
            self._module = pysa.qubo
            self.n_vars = len(problem)
            if problem.shape != (self.n_vars, self.n_vars):
                raise ValueError("Problem must be a square matrix.")
            if not np.allclose(problem, problem.T):
                raise ValueError("Problem must be a symmetric matrix.")
            self.local_fields = np.copy(np.diag(problem))
            self.couplings = np.copy(problem)
            np.fill_diagonal(self.couplings, 0)
        elif problem_type == 'ising':
            self._module = pysa.ising
            self.n_vars = len(problem)
            if problem.shape != (self.n_vars, self.n_vars):
                raise ValueError("Problem must be a square matrix.")
            if not np.allclose(problem, problem.T):
                raise ValueError("Problem must be a symmetric matrix.")
            self.local_fields = np.copy(np.diag(problem))
            self.couplings = np.copy(problem)
            np.fill_diagonal(self.couplings, 0)
        elif problem_type == 'hubo':
            if isinstance(problem, dict) and "terms" in problem and "n_vars" in problem:
                self.hubo_terms = problem["terms"]
                self.n_vars = problem["n_vars"]
            else:
                raise ValueError("For HUBO, provide problem as {'terms': [...], 'n_vars': N}")
            self._hubo_solver = pysa.hubo.HUBOSolver(self.hubo_terms, self.n_vars, float_type=self.float_type)
        elif problem_type == 'spinhubo':
            if isinstance(problem, dict) and "terms" in problem and "n_vars" in problem:
                self.spinhubo_terms = problem["terms"]
                self.n_vars = problem["n_vars"]
            else:
                raise ValueError("For Spin-HUBO, provide problem as {'terms': [...], 'n_vars': N}")
            self._spinhubo_solver = pysa.spinhubo.SpinHUBOSolver(self.spinhubo_terms, self.n_vars, float_type=self.float_type)
        else:
            raise ValueError(f"problem_type='{problem_type}' not supported.")

    def get_energy(self, conf: np.ndarray, dtype: Optional[Any] = None) -> float:
        """
        Return the energy of a given configuration.
        """
        if self.problem_type == "hubo":
            return self._hubo_solver.get_energy(np.asarray(conf, dtype=int))
        if self.problem_type == "spinhubo":
            return self._spinhubo_solver.get_energy(np.asarray(conf, dtype=int))
        # QUBO or Ising
        if dtype is None:
            dtype = self.couplings.dtype
        return self._module.get_energy(self.couplings.astype(dtype), self.local_fields.astype(dtype), conf.astype(dtype))

    def metropolis_update(
        self,
        num_sweeps: int,
        num_reads: int = 1,
        num_replicas: Optional[int] = None,
        temps: Optional[np.ndarray] = None,
        min_temp: float = 0.3,
        max_temp: float = 1.5,
        update_strategy: str = 'random',
        initialize_strategy: str = 'random',
        init_energies: Optional[List[float]] = None,
        recompute_energy: bool = False,
        sort_output_temps: bool = False,
        return_dataframe: bool = True,
        parallel: bool = True,
        use_pt: bool = True,
        send_background: bool = False,
        verbose: bool = False,
        get_part_fun: bool = False,
        beta0: bool = False,
        schedule: Optional[np.ndarray] = None,
        seed: Optional[Any] = None,
        **kwargs
    ) -> pd.DataFrame:
        '''
        This function runs a full simulated annealing importance sampling.
        For HUBO, it delegates to HUBOSolver. For Spin-HUBO, it delegates to SpinHUBOSolver.
        For QUBO/Ising, uses legacy logic.
        '''
        if self.problem_type == "hubo":
            return self._hubo_solver.simulated_annealing(
                num_sweeps=num_sweeps,
                num_reads=num_reads,
                min_temp=min_temp,
                max_temp=max_temp,
                schedule=schedule,
                return_dataframe=return_dataframe,
                verbose=verbose,
                seed=seed,
                use_pt=kwargs.get("use_pt", False),
                num_replicas=kwargs.get("num_replicas", 4),
                record_history=kwargs.get("record_history", False),
            )
        if self.problem_type == "spinhubo":
            return self._spinhubo_solver.simulated_annealing(
                num_sweeps=num_sweeps,
                num_reads=num_reads,
                min_temp=min_temp,
                max_temp=max_temp,
                schedule=schedule,
                return_dataframe=return_dataframe,
                verbose=verbose,
                seed=seed,
                use_pt=kwargs.get("use_pt", False),
                num_replicas=kwargs.get("num_replicas", 4),
                record_history=kwargs.get("record_history", False),
                parallel=kwargs.get("parallel", False),
            )

        # --- QUBO and Ising logic below (original code) ---
        if send_background:
            return ThreadPoolExecutor(max_workers=1).submit(
                Solver.metropolis_update, **locals())

        if not temps is None:
            if num_replicas and len(temps) != num_replicas:
                raise ValueError(f"len('temps') != 'num_replicas'.")
            else:
                num_replicas = len(temps)

        if type(initialize_strategy) != str:
            if num_replicas and len(initialize_strategy) != num_replicas:
                raise ValueError(
                    f"len('initialize_strategy') != 'num_replicas'.")
            else:
                num_replicas = len(initialize_strategy)

            if not init_energies is None and len(init_energies) != num_replicas:
                raise ValueError(f"len('init_energies') != 'num_replicas'.")

        if get_part_fun:
            beta0 = True
        if temps is None:
            if not num_replicas:
                num_replicas = 4
            if num_replicas == 1:
                betas = np.array([1 / min_temp], dtype=self.float_type)
            elif num_replicas == 2:
                betas = np.array([1 / min_temp, 1 / max_temp], dtype=self.float_type)
            elif num_replicas > 2:
                if beta0:
                    _ratio = np.exp(1 / (num_replicas - 2) * np.log(max_temp / min_temp))
                    betas = np.array([
                        1 / (min_temp * _ratio**k)
                        for k in range(num_replicas - 1)
                    ], dtype=self.float_type)
                    betas = np.append(betas, 0.0)
                else:
                    _ratio = np.exp(1 / (num_replicas - 1) * np.log(max_temp / min_temp))
                    betas = np.array([
                        1 / (min_temp * _ratio**k) for k in range(num_replicas)
                    ], dtype=self.float_type)
        else:
            if beta0:
                if np.inf not in temps:
                    raise ValueError("get_part_fun = True or beta0 = True require the temps array to include np.inf")
            betas = 1 / np.array(temps, dtype=self.float_type)

        couplings = self.couplings.astype(self.float_type)
        local_fields = self.local_fields.astype(self.float_type)

        if parallel:
            simulation = pysa.simulation.simulation_parallel
        else:
            simulation = pysa.simulation.simulation_sequential

        if type(initialize_strategy) == str:
            if initialize_strategy == 'random':
                if self.problem_type == 'qubo':
                    def _init_strategy():
                        states = np.random.randint(2, size=(num_replicas, self.n_vars)).astype(self.float_type)
                        energies = np.array([self.get_energy(state, dtype=self.float_type) for state in states])
                        return states, energies
                elif self.problem_type == 'ising':
                    def _init_strategy():
                        states = 2 * np.random.randint(2, size=(num_replicas, self.n_vars)).astype(self.float_type) - 1
                        energies = np.array([self.get_energy(state, dtype=self.float_type) for state in states])
                        return states, energies
                else:
                    raise ValueError(f"self.problem_type=='{self.problem_type}' not supported.")
            elif initialize_strategy == 'zeros' and self.problem_type == 'qubo':
                def _init_strategy():
                    states = np.zeros((num_replicas, self.n_vars), dtype=self.float_type)
                    energies = np.zeros(num_replicas, dtype=self.float_type)
                    return states, energies
            elif initialize_strategy == 'ones':
                def _init_strategy():
                    states = np.ones((num_replicas, self.n_vars), dtype=self.float_type)
                    energies = np.ones(num_replicas, dtype=self.float_type) * (np.sum(couplings) / 2 + np.sum(local_fields))
                    return states, energies
            else:
                raise ValueError(f"initialize_strategy='{initialize_strategy}' not recognized.")
        else:
            try:
                if init_energies is None:
                    init_energies = np.array([
                        self.get_energy(state, dtype=self.float_type)
                        for state in initialize_strategy
                    ])
                def _init_strategy():
                    states = np.copy(initialize_strategy).astype(self.float_type)
                    energies = np.copy(init_energies).astype(self.float_type)
                    return states, energies
            except:
                raise ValueError("Cannot initialize system.")

        if update_strategy == 'random':
            def _simulate_core():
                t_ini = time()
                w = _init_strategy()
                beta_idx = np.arange(num_replicas)
                t_end = time()
                return simulation(self._module.update_spin, pysa.simulation.random_sweep, couplings,
                                  local_fields, *w, beta_idx, betas, num_sweeps, get_part_fun, use_pt), t_end - t_ini
        elif update_strategy == 'sequential':
            def _simulate_core():
                t_ini = time()
                w = _init_strategy()
                beta_idx = np.arange(num_replicas)
                t_end = time()
                return simulation(self._module.update_spin, pysa.simulation.sequential_sweep, couplings,
                                  local_fields, *w, beta_idx, betas, num_sweeps, get_part_fun, use_pt), t_end - t_ini
        else:
            raise ValueError(f"update_strategy='{update_strategy}' not recognized.")

        def _simulate():
            t_ini = time()
            ((out_states, out_energies, out_beta_idx, out_log_omegas),
             (best_state, best_energy, best_sweeps, ns)), init_time = _simulate_core()
            t_end = time()
            if recompute_energy:
                best_energy = self.get_energy(best_state)
                out_energies = np.array([self.get_energy(state) for state in out_states])
            if sort_output_temps:
                out_betas = betas
                out_states = out_states[out_beta_idx]
                out_energies = out_energies[out_beta_idx]
            else:
                out_betas = betas[out_beta_idx]
            with np.errstate(divide='ignore'):
                out_temps = np.divide(1, out_betas)
            if get_part_fun:
                logZf = pysa.ais.omegas_to_partition(out_log_omegas, self.n_vars * np.log(2))
                return {
                    'states': out_states,
                    'energies': out_energies,
                    'best_state': best_state,
                    'best_energy': best_energy,
                    'temps': out_temps,
                    'log_Zf': logZf,
                    'num_sweeps': ns,
                    'min_sweeps': best_sweeps,
                    'init_time (us)': int(init_time * 1e6),
                    'runtime (us)': int((t_end - t_ini - init_time) * 1e6),
                    'problem_type': self.problem_type,
                    'float_type': self.float_type
                }
            else:
                return {
                    'states': out_states,
                    'energies': out_energies,
                    'best_state': best_state,
                    'best_energy': best_energy,
                    'temps': out_temps,
                    'num_sweeps': ns,
                    'min_sweeps': best_sweeps,
                    'init_time (us)': int(init_time * 1e6),
                    'runtime (us)': int((t_end - t_ini - init_time) * 1e6),
                    'problem_type': self.problem_type,
                    'float_type': self.float_type
                }

        _res = [
            _simulate() for _ in tqdm(np.arange(num_reads), disable=not verbose)
        ]

        return pd.DataFrame(_res) if return_dataframe else _res
