import numpy as np
import scipy.linalg


from fractions import Fraction
def isclose(a, b, rtol=None):
    return np.isclose(np.float64(a), np.float64(b), rtol)

def allclose(a, b, rtol=None):
    return np.allclose(np.float64(a), np.float64(b), rtol)

tol = 1e-7

def to_frac(v):
    return Fraction(v).limit_denominator(int(1/tol))



def is_rref(B):
    """
    Check if a matrix B is in Reduced Row Echelon Form (RREF).

    From wikipedia [https://en.wikipedia.org/wiki/Row_echelon_form]:
    > A matrix is in reduced row echelon form if it is in row echelon form, the first nonzero entry of each row is equal to 1 and the ones above it within the same column equal 0.
    """
    row_count, col_count = B.shape
    last_pivot_col = -1

    for i in range(row_count):
        row = B[i, :]
        non_zero_indices = np.where(np.abs(row) >= tol )[0]

        
        if len(non_zero_indices) == 0:
            # If a non-zero row is found below a zero row, it's not RREF
            if i < row_count - 1 and len(np.where(np.abs(B[i + 1:, :]) >= tol)[0]>=1):
                print("bad swapping")
                return False
            continue
        
        first_non_zero_idx = non_zero_indices[0]
        
        # Check if the leading entry in the row is 1
        if not isclose(row[first_non_zero_idx], 1, rtol=tol):
            print("bad 1", row[first_non_zero_idx])
            print(row)
            return False

        # Check if this leading 1 is to the right of the last pivot
        if first_non_zero_idx <= last_pivot_col:
            print("swapping in non zero col")
            return False

        # Check if the column of this leading 1 has zeros everywhere else
        for k in range(row_count):
            if k != i and not isclose(B[k, first_non_zero_idx], 0, rtol=tol):
                print("non zero coef", B[k, first_non_zero_idx])
                return False

        last_pivot_col = first_non_zero_idx

    return True

def compute_rref(K, use_fractions=False):
    """
    Compute the Reduced Row Echelon Form (RREF) using Gauss-Jordan Pivot algorithm.

    Input:
    - K: M(m, n)
    - use_fractions: if true, will do the exact computation using fractions.

    Outputs:
    - T: M(m, m), transformation matrix such that [T @ K = Q].
    - R: M(m, m), such that [K = R @ Q]
    - Q: M(m, n), in RREF

    Complexity: O(max(n, m)^3).
    """

    m, n = K.shape

    R = np.eye(m, m)
    Q = np.copy(K)

    # We keep track of the transformations using T. Each time we do an transformation on Q, we do the same transformation on T. This forces [T @ K = Q] as an invariant.
    T = np.eye(m, m)

    if use_fractions:
        R = np.vectorize(to_frac)(R)
        T = np.vectorize(to_frac)(T)
        Q = np.vectorize(to_frac)(Q)

    r = 0
    for j in range(n):
        i = r
        while i < m and isclose(Q[i, j], 0, rtol=tol):
            i += 1
        if i >= m:
            continue

        if i != r:
            # Swap rows i and r of Q and corresponding columns in R
            Q[[r, i]] = Q[[i, r]]
            T[[r, i]] = T[[i, r]]
            R[:, [r, i]] = R[:, [i, r]]

        # Scale Q[r, j] to a leading 1 and update R
        factor = Q[r, j]
        if not isclose(factor,0, rtol=tol):
            Q[r] /= factor
            T[r] /= factor
            R[:, r] *= factor
        
        for k in range(m):
            if k != r:
                # Make Q[k, j] zero using row r and update R
                factor = Q[k, j] / Q[r, j]
                if not isclose(factor, 0, rtol=tol):
                    Q[k] -= factor * Q[r]
                    T[k] -= factor * T[r]
                    R[:, r] += factor * R[:, k]

        r += 1

    return T, R, Q

def _compute_rref_and_check(K, use_fractions=False):
    T, R, Q = compute_rref(K, use_fractions)
    assert allclose(T @ K, Q, rtol=tol)
    assert allclose(R @ Q, K, rtol=tol)
    assert is_rref(Q) == True

def test_compute_rref1():
    K1 = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=float)
    K2 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0]], dtype=float)
    K3 = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 1]], dtype=float)
    K4 = np.eye(3, dtype=float)
    K5 = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    K6 = np.random.random((50,100))
    # Test the matrices
    matrices = [K1, K2, K4, K5, K3, K6]
    results = [_compute_rref_and_check(X, use_fractions=False) for X in matrices]

def test_compute_rref2():
    import sklearn.datasets
    for i in range(10):
        X, _ = sklearn.datasets.make_regression(noise=0)

        _compute_rref_and_check(X, use_fractions=False)

def test_compute_rref3():
    import networkx as nx
    X = nx.to_numpy_array(nx.balanced_tree(3, 3))
    _compute_rref_and_check(X, use_fractions=False)

    X = nx.to_numpy_array(nx.fast_gnp_random_graph(100, 0.3))
    _compute_rref_and_check(X, use_fractions=False)

    X = nx.to_numpy_array(nx.barabasi_albert_graph(100, 4))
    _compute_rref_and_check(X, use_fractions=False)

    X = nx.to_numpy_array(nx.powerlaw_cluster_graph(100, 5, 0.3))
    _compute_rref_and_check(X, use_fractions=False)

    X = nx.to_numpy_array(nx.random_geometric_graph(100, 0.3))
    _compute_rref_and_check(X, use_fractions=False)

    X = nx.to_numpy_array(nx.uniform_random_intersection_graph(100, 100, 0.1))
    _compute_rref_and_check(X, use_fractions=False)

def test_compute_rref4():
    import networkx as nx
    X = nx.to_numpy_array(nx.random_geometric_graph(1000, 0.1))
    _compute_rref_and_check(X, use_fractions=False)