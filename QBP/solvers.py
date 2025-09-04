import numpy as np

# Minimal, formulation-agnostic wrappers that call PySA implementations.
# Keep only actively used decoders in SOLVERS; move extras to experimental_solvers.py
from pysa.spinhubo import SpinHUBOSolver
from pysa.sa import Solver as QuboSolver
from collections import defaultdict


def spin_to_binary(spin_solution):
    """Converts a spin vector {-1, +1} to a binary vector {0, 1} using mapping: +1 -> 0, -1 -> 1."""
    return ((1 - spin_solution) / 2).astype(int)


# --- Helpers to build term lists ---
def _spin_hubo_dict_to_terms(hubo_dict):
    """
    Convert a spin-HUBO dictionary { (i,...): coeff } into pysa spin-HUBO term list.
    Skips non-index metadata keys like ('H_matrix',).
    """
    terms = []
    for key in hubo_dict:
        if not (isinstance(key, tuple) and all(isinstance(k, (int, np.integer)) for k in key)):
            continue
        terms.append((float(hubo_dict[key]), list(key)))
    return terms


def _expand_spin_monomial_to_binary_terms(indices, coeff):
    """
    Expand a single spin monomial: coeff * prod_{i in I} s_i with s_i = 1 - 2 x_i
    into binary HUBO terms sum_{S ⊆ I} coeff * (-2)^{|S|} ∏_{i∈S} x_i.

    Returns list of (coeff_S, [indices in S]) excluding the empty subset (constant term),
    since constant offsets do not affect argmin.
    """
    idx = list(indices)
    out = []
    L = len(idx)
    # For each subset size k >= 1, add combinations
    for k in range(1, L + 1):
        factor = (-2.0) ** k
        for combo in __import__('itertools').combinations(idx, k):
            out.append((float(coeff) * factor, list(combo)))
    return out


def _spin_hubo_dict_to_binary_terms(hubo_dict):
    """
    Convert a spin-HUBO dictionary to a binary-HUBO term list by expanding each monomial
    using s_i = 1 - 2 x_i. Metadata keys are ignored. Constant offsets are dropped.
    """
    acc = defaultdict(float)
    for key, coeff in hubo_dict.items():
        if not (isinstance(key, tuple) and all(isinstance(k, (int, np.integer)) for k in key)):
            continue
        for c, inds in _expand_spin_monomial_to_binary_terms(key, coeff):
            acc[tuple(sorted(int(i) for i in inds))] += c
    # Pack to list
    terms = [(float(c), list(k)) for k, c in acc.items()]
    return terms


# --- Thin wrapper solvers delegating to pysa implementations ---
def simulated_annealing_hubo(hubo_dict, n_spins, num_sweeps=100, num_reads=1, min_temp=0.3, max_temp=3.0, verbose=False, use_pt=False, num_replicas=4, parallel=False):
    """Wrapper: build spin terms and call pysa.spinhubo.SpinHUBOSolver for SA.

    PT is off by default; enable by passing use_pt=True when running ablations.
    """
    terms = _spin_hubo_dict_to_terms(hubo_dict)
    solver = SpinHUBOSolver(terms, n_spins)
    result_df = solver.simulated_annealing(
        num_sweeps=num_sweeps,
        num_reads=num_reads,
        min_temp=min_temp,
        max_temp=max_temp,
        verbose=verbose,
        use_pt=use_pt,
        num_replicas=num_replicas,
        parallel=parallel,
    )
    return result_df['best_state'][0]


# Note: additional wrappers (PT, binary-HUBO, Rosenberg) moved to experimental_solvers.py


def qubo_sa_paper(hubo_dict, n_vars, num_sweeps=100, num_reads=1, min_temp=0.3, max_temp=3.0, penalty_strength=None, verbose=False, ancilla_plan=None):
    """
    QUBO formulation per paper-style parity slack: build Q directly from H and LLR using
    (sum x - 2 t)^2 per check with binary-encoded integer slack t, then solve with QUBO SA.
    Returns best binary state (includes ancillas); first n_vars are the code bits.
    """
    try:
        from QBP.problem_construction import build_qubo_from_hubo_dict_paper
    except Exception:
        from problem_construction import build_qubo_from_hubo_dict_paper

    Q, total_n = build_qubo_from_hubo_dict_paper(hubo_dict, n_vars, penalty_strength=penalty_strength, ancilla_plan=ancilla_plan)
    solver = QuboSolver(Q, problem_type='qubo')
    df = solver.metropolis_update(num_sweeps=num_sweeps, num_reads=num_reads, min_temp=min_temp, max_temp=max_temp, return_dataframe=True, parallel=False, use_pt=False, verbose=verbose)
    return df['best_state'][0]


def qubo_sa_paper_from_H_llr(H, llr, num_sweeps=100, num_reads=1, min_temp=0.3, max_temp=3.0, penalty_strength=None, verbose=False, ancilla_plan=None):
    """
    Paper-style QUBO directly from H and LLR vector; avoids building HUBO.
    Returns best binary state (includes ancillas); original code bits are the first n entries.
    """
    try:
        from QBP.problem_construction import build_qubo_from_parity_llr
    except Exception:
        from problem_construction import build_qubo_from_parity_llr

    Q, total_n = build_qubo_from_parity_llr(H, llr, penalty_strength=penalty_strength, ancilla_plan=ancilla_plan)
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


# --- Solver Dispatcher kept for backward compatibility ---
def belief_propagation_bp(hubo_dict, n_spins, max_iters=50, damping=0.0, verbose=False):
    """
    Wrapper for classic belief propagation (sum-product) decoder implemented in QBP/bp_decoder.py.

    The wrapper extracts single-variable linear coefficients from hubo_dict as per-bit LLRs
    and expects the parity-check matrix H to be provided under the special key ('H_matrix',).
    This keeps the adapter simple and avoids changing other call sites.
    """
    try:
        from QBP import bp_decoder as bp
    except Exception:
        import bp_decoder as bp

    # Extract H matrix
    if ('H_matrix',) not in hubo_dict:
        raise ValueError("BP wrapper requires H matrix attached to hubo_dict under key ('H_matrix',)")
    H = hubo_dict[('H_matrix',)]

    # Build LLR vector from single-variable terms (hubo stores -LLR from build_llr_terms)
    llr = np.zeros(n_spins)
    for key, coeff in hubo_dict.items():
        if isinstance(key, tuple) and len(key) == 1:
            idx = key[0]
            # only accept integer indices as variable keys
            if isinstance(idx, (int, np.integer)):
                llr[int(idx)] += coeff

    # The pipeline stores linear terms as -LLR (see build_llr_terms). BP expects LLR where
    # positive => bit 0 more likely, so invert the sign here to recover true LLRs.
    llr = -llr

    # Sanity check: H should have shape (m, n) where n == n_spins
    if hasattr(H, 'shape'):
        if H.shape[1] != n_spins:
            raise ValueError(f"H.shape[1] ({H.shape[1]}) does not match n_spins ({n_spins})")

    decoded_bits = bp.decode(H, llr, max_iters=max_iters, damping=damping)[0]
    return decoded_bits


SOLVERS = {
    # Active decoders
    "SA": simulated_annealing_hubo,
    "QUBO-PAPER": qubo_sa_paper,
    "BP": belief_propagation_bp,
}

# Experimental/extra solvers have been moved to QBP/experimental_solvers.py to
# keep SOLVERS focused on actively used options in main.py.
