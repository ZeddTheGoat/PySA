import numpy as np
from collections import defaultdict
from modulation_utils import calculate_llrs
from itertools import combinations


def build_parity_terms(H, lambda_penalty):
    """
    Build parity penalty terms that depend only on the parity-check matrix H and lambda_penalty.

    Returns a dict mapping tuples of variable indices to coefficients: { tuple(indices): coeff }
    """
    m, original_n = H.shape

    hubo_dict = defaultdict(float)
    checks = [np.where(H[i] == 1)[0].tolist() for i in range(m)]
    for check in checks:
        check_filtered = [idx for idx in check if idx < original_n]
        if len(check_filtered) > 0:
            hubo_dict[tuple(sorted(check_filtered))] -= lambda_penalty

    return hubo_dict


def build_llr_terms(received_symbols, sigma, modulation_type, n_vars):
    """
    Build single-variable LLR terms based on the received symbols and noise level.

    Returns a dict where each key is a single-index tuple and the value is the coefficient: { (i,): coeff }
    """
    hubo_dict = defaultdict(float)
    noise_variance = 2 * sigma**2
    llrs = calculate_llrs(received_symbols, noise_variance, modulation_type, n_vars)
    for i in range(n_vars):
        hubo_dict[(i,)] = -llrs[i]
    return hubo_dict


def build_hubo(H, received_symbols, sigma, modulation_type, lambda_penalty, formulation_type='llr', n_vars=None):
    """
    Compatibility wrapper: merge precomputed parity terms with per-trial LLR terms
    and return the complete HUBO dictionary.
    """
    if formulation_type != 'llr':
        raise ValueError("Only 'llr' formulation is supported in this simplified implementation.")

    m, original_n = H.shape
    if n_vars is None:
        n_vars = original_n

    parity = build_parity_terms(H, lambda_penalty)
    llr = build_llr_terms(received_symbols, sigma, modulation_type, n_vars)

    # Merge dictionaries (parity + llr) and return
    hubo_dict = defaultdict(float)
    for k, v in parity.items():
        hubo_dict[k] += v
    for k, v in llr.items():
        hubo_dict[k] += v

    return hubo_dict


def hubo_dict_to_terms(hubo_dict):
    """
    Convert a HUBO dictionary { (i,...): coeff } into a list of (coeff, [i,...]) terms.
    Ensures indices are ints and terms are sorted for deterministic ordering.
    """
    terms = []
    for indices, coeff in hubo_dict.items():
        idx_list = [int(i) for i in indices]
        idx_list = sorted(idx_list)
        terms.append((float(coeff), idx_list))
    return terms


# --- QUBO formulation helpers ---
def spin_hubo_dict_to_binary_terms(hubo_dict):
    """
    Expand spin-HUBO dict {(i1,...,ik): coeff} into binary higher-order terms via s=1-2x.
    Returns list of (coeff, [indices]) for binary variables x, dropping constant offsets.
    """
    terms = defaultdict(float)
    for key, coeff in hubo_dict.items():
        # skip metadata keys like ('H_matrix',)
        if not (isinstance(key, tuple) and all(isinstance(k, (int, np.integer)) for k in key)):
            continue
        idx = list(key)
        L = len(idx)
        for k in range(1, L + 1):
            factor = (-2.0) ** k
            for combo in combinations(idx, k):
                terms[tuple(sorted(int(i) for i in combo))] += float(coeff) * factor
    return [(c, list(k)) for k, c in terms.items()]


def quadratize_binary_terms_to_qubo(terms, n_vars, penalty_strength=None):
    """
    Convert higher-order binary terms to a QUBO matrix using Rosenberg quadratization.

    Args:
        terms: list of (coef, [indices]) on x in {0,1}
        n_vars: number of original codeword variables
        penalty_strength: large M to enforce ancilla constraints; if None, auto-set.
    Returns:
        Q: numpy array (N_total x N_total) QUBO matrix (diag=linear, off-diag=quadratic)
        N_total: total variables including ancillas
    """
    # Estimate a reasonable M if not provided
    if penalty_strength is None:
        total_abs = sum(abs(c) for c, _ in terms) + 1.0
        penalty_strength = 10.0 * total_abs

    # Start with original variables
    Q = np.zeros((n_vars, n_vars), dtype=float)

    def ensure_var(idx):
        nonlocal Q
        N = Q.shape[0]
        if idx >= N:
            newN = idx + 1
            Q_new = np.zeros((newN, newN), dtype=float)
            Q_new[:N, :N] = Q
            Q = Q_new

    def add_linear(i, c):
        ensure_var(i)
        Q[i, i] += c

    def add_quadratic(i, j, c):
        if i == j:
            add_linear(i, c)
            return
        ensure_var(max(i, j))
        Q[i, j] += c
        Q[j, i] += c

    M = float(penalty_strength)

    for c, idxs in terms:
        coeff = float(c)
        idxs = list(int(i) for i in idxs)
        if len(idxs) == 0:
            # constant term: ignore for argmin
            continue
        elif len(idxs) == 1:
            add_linear(idxs[0], coeff)
        elif len(idxs) == 2:
            add_quadratic(idxs[0], idxs[1], coeff)
        else:
            # Reduce order iteratively using ancillas enforcing y = a*b via M (2y - a - b)^2
            cur = idxs[:]
            while len(cur) > 2:
                a = cur.pop()
                b = cur.pop()
                y = Q.shape[0]
                ensure_var(y)
                # Penalty: M(2y - a - b)^2 = 4M y + M a + M b - 4M y a - 4M y b + 2M a b
                add_linear(y, 4*M)
                add_linear(a, M)
                add_linear(b, M)
                add_quadratic(y, a, -4*M)
                add_quadratic(y, b, -4*M)
                add_quadratic(a, b, 2*M)
                # replace a*b with y
                cur.append(y)
            if len(cur) == 2:
                add_quadratic(cur[0], cur[1], coeff)
            else:
                add_linear(cur[0], coeff)

    return Q, Q.shape[0]


def allocate_parity_slack_ancillas(H, base_n=None):
    """
    Precompute and assign ancilla bit indices for each parity check in H using
    the integer-slack encoding of t in [0, floor(deg/2)].

    Args:
        H: parity-check matrix (m x n), numpy array of 0/1
        base_n: starting index for ancillas (defaults to n)
    Returns:
        ancilla_indices: list of lists; ancilla_indices[c] = [indices of ancillas for check c]
        N_total: total number of variables = n + total_ancillas
    """
    H = np.asarray(H)
    m, n = H.shape
    if base_n is None:
        base_n = n
    next_idx = int(base_n)
    ancilla_indices = []
    for c in range(m):
        S = np.where(H[c] == 1)[0]
        d = int(S.size)
        t_max = d // 2
        if t_max <= 0:
            ancilla_indices.append([])
            continue
        bbits = int(np.ceil(np.log2(t_max + 1)))
        inds = list(range(next_idx, next_idx + bbits))
        ancilla_indices.append(inds)
        next_idx += bbits
    return ancilla_indices, next_idx


def build_qubo_from_parity_llr(H, llr, penalty_strength=None, ancilla_plan=None):
    """
    Build a pure QUBO matrix for LDPC decoding using a parity slack formulation:
      For each check c with neighborhood S: add M (sum_{i in S} x_i - 2 t_c)^2,
      where t_c is an integer slack encoded in binary with enough bits to represent 0..floor(|S|/2).

    Channel term: add 2*LLR_i to Q[i,i] (since -LLR*s translates to +2*LLR*x up to a constant).

    Args:
        H: parity-check matrix (m x n), numpy array of 0/1
        llr: numpy array length n (positive favors 0)
        penalty_strength: M; if None, auto choose based on LLR scale
    Returns:
        Q: QUBO matrix (N_total x N_total)
        N_total: total variable count including ancillas
    """
    H = np.asarray(H)
    m, n = H.shape
    llr = np.asarray(llr).reshape(-1)
    if llr.shape[0] != n:
        raise ValueError("llr length must match number of variables")

    # Heuristic M if not provided
    if penalty_strength is None:
        # Scale relative to typical LLR magnitude
        scale = max(1.0, float(np.median(np.abs(llr)) + np.mean(np.abs(llr))))
        penalty_strength = 10.0 * scale
    M = float(penalty_strength)

    # Pre-assign ancilla indices (plan can be reused across trials for a fixed H)
    if ancilla_plan is None:
        ancilla_indices, N_total = allocate_parity_slack_ancillas(H, base_n=n)
    else:
        ancilla_indices = ancilla_plan
        # infer N_total from plan
        max_anc = max((max(a) if a else -1) for a in ancilla_indices)
        N_total = max(n, max_anc + 1)

    # Preallocate full Q
    Q = np.zeros((N_total, N_total), dtype=float)

    def add_linear(i, c):
        Q[i, i] += c

    def add_quadratic(i, j, c):
        if i == j:
            Q[i, i] += c
            return
        Q[i, j] += c
        Q[j, i] += c

    # Channel linear terms: +2*LLR*x
    for i in range(n):
        add_linear(i, 2.0 * float(llr[i]))

    # Parity checks with binary-encoded slack
    for i_check in range(m):
        S = np.where(H[i_check] == 1)[0]
        d = int(S.size)
        if d == 0:
            continue
        # (sum x)^2 part: M * sum x + 2M * sum_{i<j} x_i x_j
        for idx in S:
            add_linear(int(idx), M)
        for a in range(d):
            ia = int(S[a])
            for b in range(a + 1, d):
                ib = int(S[b])
                add_quadratic(ia, ib, 2.0 * M)

        # slack terms if present
        anc_bits = ancilla_indices[i_check] if i_check < len(ancilla_indices) else []
        if not anc_bits:
            continue
        bbits = len(anc_bits)
        pow2 = [1 << k for k in range(bbits)]

        # Cross term -4 sum_i sum_k pow2[k] x_i b_k
        for idx in S:
            ii = int(idx)
            for k, p in enumerate(pow2):
                add_quadratic(ii, anc_bits[k], -4.0 * M * p)
        # 4 (sum_k p b_k)^2 = 4( sum_k p^2 b_k + 2 sum_{k<l} p_k p_l b_k b_l )
        for k, p in enumerate(pow2):
            add_linear(anc_bits[k], 4.0 * M * (p * p))
        for k in range(bbits):
            for l in range(k + 1, bbits):
                add_quadratic(anc_bits[k], anc_bits[l], 8.0 * M * pow2[k] * pow2[l])

    return Q, Q.shape[0]


def build_qubo_from_hubo_dict_paper(hubo_dict, n_vars, penalty_strength=None, ancilla_plan=None):
    """
    Convenience wrapper: extract H and LLR from hubo_dict and build QUBO via parity‑slack method.
    hubo_dict contains linear spin terms as -(LLR_i) at keys (i,), and H at ('H_matrix',).
    """
    if ('H_matrix',) not in hubo_dict:
        raise ValueError("H matrix missing from hubo_dict under key ('H_matrix',)")
    H = hubo_dict[('H_matrix',)]
    llr = np.zeros(n_vars)
    for key, coeff in hubo_dict.items():
        if isinstance(key, tuple) and len(key) == 1 and isinstance(key[0], (int, np.integer)):
            llr[int(key[0])] += -float(coeff)  # invert sign to recover LLR
    return build_qubo_from_parity_llr(H, llr, penalty_strength=penalty_strength, ancilla_plan=ancilla_plan)
