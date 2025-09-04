import numpy as np
import pandas as pd
from pysa.sa import Solver

def print_result(df):
    print(df[['best_state', 'best_energy']])

def test_qubo():
    # QUBO: Minimize x0 + x1 + x0*x1 for x in {0,1}^2
    qubo_problem = np.array([[1.0, 1.0],
                             [1.0, 1.0]])
    solver = Solver(qubo_problem, problem_type='qubo')
    df = solver.metropolis_update(num_sweeps=50, num_reads=3, verbose=True)
    print("QUBO Results:")
    print_result(df)

def test_ising():
    # Ising: Minimize -s0*s1 for s in {-1,1}^2
    ising_problem = np.array([[0.0, -1.0],
                              [-1.0, 0.0]])
    solver = Solver(ising_problem, problem_type='ising')
    df = solver.metropolis_update(num_sweeps=50, num_reads=3, verbose=True)
    print("Ising Results:")
    print_result(df)

def test_hubo():
    # HUBO: Minimize x0*x1 + 2*x1*x2*x3 + x2 for x in {0,1}^4
    hubo_terms = [
        (1.0, [0, 1]),
        (2.0, [1, 2, 3]),
        (1.0, [2]),
    ]
    hubo_problem = {"terms": hubo_terms, "n_vars": 4}
    solver = Solver(hubo_problem, problem_type="hubo")
    df = solver.metropolis_update(num_sweeps=50, num_reads=3, verbose=True)
    print("HUBO Results:")
    print_result(df)

def test_spinhubo():
    # Spin-HUBO: Minimize 1.2*s0*s1 + 0.5*s2*s3*s4 - 2.0*s1 for s in {-1,1}^5
    spinhubo_terms = [
        (1.2, [0, 1]),
        (0.5, [2, 3, 4]),
        (-2.0, [1]),
    ]
    spin_hubo_problem = {"terms": spinhubo_terms, "n_vars": 5}
    solver = Solver(spin_hubo_problem, problem_type="spinhubo")
    df = solver.metropolis_update(num_sweeps=50, num_reads=3, verbose=True)
    print("Spin-HUBO Results:")
    print_result(df)

if __name__ == "__main__":
    test_qubo()
    print()
    test_ising()
    print()
    test_hubo()
    print()
    test_spinhubo()