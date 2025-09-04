import numpy as np

"""
Simple Belief Propagation (sum-product) decoder for binary LDPC codes.
This implementation expects parity-check matrix H (m x n) as numpy array with entries in {0,1}.
It decodes soft LLR inputs for codeword bits and returns a hard decision codeword.

API:
  decode(H, llr, max_iters=50, damping=0.0)

Inputs:
  H: numpy array shape (m, n)
  llr: numpy array length n with log-likelihood ratios (LLR) for each bit (positive -> 0 more likely)
  max_iters: maximum number of BP iterations
  damping: damping factor in [0,1) applied to message updates (0 = no damping)

Returns:
  decoded_bits: numpy array length n with values {0,1}
  success: boolean indicating whether parity checks are satisfied

Notes:
  - This is a straightforward implementation of the sum-product algorithm (log-domain message passing)
    using tanh/atanh updates on check nodes. It is not highly optimized but sufficient for algorithmic
    comparison against the spin-HUBO solver in experiments.
"""


def decode(H, llr, max_iters=50, damping=0.0):
    H = np.asarray(H, dtype=np.int8)
    m, n = H.shape
    assert llr.shape[0] == n, "LLR length must match number of code bits"

    # Build neighbor lists
    var_neighbors = [np.where(H[:, j])[0].tolist() for j in range(n)]  # list of check node indices per variable
    check_neighbors = [np.where(H[i, :])[0].tolist() for i in range(m)]  # list of variable indices per check

    # Initialize messages: from var -> check (log-likelihood ratio messages)
    # messages_vc[j][i_idx] is message from variable j to check check_neighbors[j][i_idx]
    # We'll store as dict-of-dicts to keep sparse structure simple
    msgs_vc = {j: {i: 0.0 for i in var_neighbors[j]} for j in range(n)}
    msgs_cv = {i: {j: 0.0 for j in check_neighbors[i]} for i in range(m)}

    for iteration in range(max_iters):
        # Variable to check updates
        for j in range(n):
            for i in var_neighbors[j]:
                # sum of incoming check->var messages excluding target check
                incoming = 0.0
                for ii in var_neighbors[j]:
                    if ii == i:
                        continue
                    incoming += msgs_cv[ii][j]
                new_msg = llr[j] + incoming
                if damping > 0.0:
                    msgs_vc[j][i] = (1 - damping) * new_msg + damping * msgs_vc[j][i]
                else:
                    msgs_vc[j][i] = new_msg

        # Check to variable updates (log-domain using tanh rule)
        for i in range(m):
            for j in check_neighbors[i]:
                prod = 1.0
                for jj in check_neighbors[i]:
                    if jj == j:
                        continue
                    val = np.tanh(0.5 * msgs_vc[jj][i])
                    prod *= val
                # prevent numerical issues
                prod = np.clip(prod, -0.999999, 0.999999)
                msg = 2.0 * np.arctanh(prod)
                if damping > 0.0:
                    msgs_cv[i][j] = (1 - damping) * msg + damping * msgs_cv[i][j]
                else:
                    msgs_cv[i][j] = msg

        # Compute posterior LLRs and tentative decisions
        posterior = np.zeros(n)
        for j in range(n):
            total = llr[j]
            for i in var_neighbors[j]:
                total += msgs_cv[i][j]
            posterior[j] = total
        bits_hat = (posterior < 0).astype(int)  # negative LLR -> bit=1

        # Check parity
        syndrome = (H.dot(bits_hat) % 2)
        if np.all(syndrome == 0):
            return bits_hat, True

    # After max_iters, return best guess and failure flag
    return bits_hat, False


if __name__ == '__main__':
    # quick smoke test
    from pyldpc import make_ldpc
    H, G = make_ldpc(48, 2, 3, systematic=True, sparse=False)
    H = np.asarray(H)
    k = G.shape[1]
    msg = np.random.randint(0,2,k)
    cw = (G.dot(msg) % 2).astype(int)
    # BPSK mapping
    symbols = 1 - 2 * cw
    noisy, sigma = (symbols + 0.1 * np.random.randn(*symbols.shape), 0.1)
    # simple LLR for BPSK
    llr = 2 * noisy / (2 * sigma**2)
    dec, ok = decode(H, llr, max_iters=10)
    print('decoded ok?', ok)
