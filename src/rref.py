import numpy as np
import scipy.linalg


def is_rref(B):
    """
    Check if a matrix B is in Reduced Row Echelon Form (RREF).
    """
    row_count, col_count = B.shape
    last_pivot_col = -1

    for i in range(row_count):
        row = B[i, :]
        non_zero_indices = np.nonzero(row)[0]

        
        if len(non_zero_indices) == 0:
            # If a non-zero row is found below a zero row, it's not RREF
            if i < row_count - 1 and np.any(B[i + 1:, :]):
                return False
            continue
        
        first_non_zero_idx = non_zero_indices[0]
        
        # Check if the leading entry in the row is 1
        if row[first_non_zero_idx] != 1:
            return False

        # Check if this leading 1 is to the right of the last pivot
        if first_non_zero_idx <= last_pivot_col:
            return False

        # Check if the column of this leading 1 has zeros everywhere else
        for k in range(row_count):
            if k != i and B[k, first_non_zero_idx] != 0:
                return False

        last_pivot_col = first_non_zero_idx

    return True





def scale_row_to_leading_one(U, L, row, col):
    
    factor = U[row, col]
    if not np.isclose(factor,0):
        U[row] /= factor
        L[:, row] *= factor


def scale_diagonal_to_one(U, L):
    """
    Scale the diagonal entries of U to one and update L accordingly.
    """
    n = U.shape[0]
    for i in range(n):
        factor = U[i, i]
        if not np.isclose(factor,0):  # Here is the only key change
            U[i] /= factor  # Scale the row in U
            L[:, i] *= factor  # Corresponding scaling in L


def eliminate_upper(U, L, j, i):
    """
    # Eliminate the entry U[i, j] by row operations and update L accordingly.
    """
    if np.isclose(U[i, j],0):  # Check to avoid divide-by-zero
        pass #U[i,j] =0
    else:

        if not np.isclose(U[j,j], 0):
            factor = U[i, j] / U[j, j] # factor cannot be close to zero

            U[i] -= factor * U[j]
            L[:, j] += factor * L[:, i]


def make_column_entry_zero(U, L, row_to_use, row_to_change, col):
    factor = U[row_to_change, col] / U[row_to_use, col]
    if not np.isclose(factor, 0):
        U[row_to_change] -= factor * U[row_to_use]
        L[:, row_to_use] += factor * L[:, row_to_change]

def swap_rows(U, L, row1, row2):
    """
    Swap rows in U and perform the corresponding operations on A to maintain AB = LU.

    Args:
    U (numpy.ndarray): The upper triangular matrix from LU decomposition.
    A (numpy.ndarray): The modified lower triangular matrix.
    row1 (int): The first row to be swapped.
    row2 (int): The second row to be swapped.
    """
    # Swap rows in U
    U[[row1, row2]] = U[[row2, row1]]
    L[:, [row1, row2]] = L[:, [row2, row1]]

def transform_to_rref(P, L, U):
    """
    Transform PLU to PAB where B is RREF of U and A is the corresponding L.
    """
    n = U.shape[0]
    A = np.copy(L)
    B = np.copy(U)

    A[np.abs(A) < 1e-10] = 0
    B[np.abs(B) < 1e-10] = 0


    print("debug rank 1",np.linalg.matrix_rank(A))

    # Scale diagonal entries of U to one and update L
    scale_diagonal_to_one(B, A)
    assert np.allclose(L @ U, A @ B), "LU does not equal AB after scaling"
    print("debug rank 2",np.linalg.matrix_rank(A))


    # Eliminate above-diagonal entries in U (B)
    for j in range(n-1, 0, -1):
        print(np.round(B,2), j)
        for i in range(j-1, -1, -1):
            eliminate_upper(B, A, j, i)

            assert np.allclose(L @ U, A @ B), "LU does not equal AB after elimination"
    print("debug rank 3",np.linalg.matrix_rank(A))

    A[np.abs(A) < 1e-10] = 0
    B[np.abs(B) < 1e-10] = 0

    assert is_rref(B)==True, print("probleme", B)


    return P, A, B

def rref_transform(P,L, U):
    A = np.copy(L)
    B = np.copy(U)
    m, n = B.shape
    r = 0
    for j in range(n):
        i = r
        while i < m and np.isclose(B[i, j], 0):
            i += 1
        if i < m:
            if i != r:
                swap_rows(B, A, r, i)  # Swap rows i and r of B and corresponding columns in A
                assert np.allclose(A @ B, L @ U), "Matrix equality failed after row swap"
            scale_row_to_leading_one(B, A, r, j)  # Scale B[r, j] to a leading 1 and update A
            assert np.allclose(A @ B, L @ U), "Matrix equality failed after scaling to leading 1"
            for k in range(m):
                if k != r:
                    make_column_entry_zero(B, A, r, k, j)  # Make B[k, j] zero using row r and update A
                    assert np.allclose(A @ B, L @ U), "Matrix equality failed after making column entry zero"
            r += 1

    A[np.abs(A) < 1e-10] = 0
    B[np.abs(B) < 1e-10] = 0
    assert is_rref(B)==True, print("probleme", B)
    assert np.allclose(A @ B, L @ U), "Matrix equality failed"
    return P,A, B


def main():
    # Example matrices P, L, U
    P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)  # Convert to float
    L = np.array([[1, 0, 0], [0.5, 1, 0], [0.2, 0.3, 1]], dtype=float)  # Convert to float
    U = np.array([[2, 3, 1], [0, 1, 7], [0, 0, 5]], dtype=float)  # Convert to float

    P, A, B = transform_to_rref(P, L, U)
    print("P:\n", P)
    print("A:\n", A)
    print("B (RREF):\n", B)
    print(is_rref(B))

def test_rref_transformation(matrix):
    print("testing ")
    print(matrix)
    P, L, U = scipy.linalg.lu(matrix)
    _, A, B = rref_transform(P, L, U)
    print(B)
    return is_rref(B)


def main2():
    K1 = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=float)
    K2 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0]], dtype=float)
    K3 = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 1]], dtype=float)
    K4 = np.eye(3, dtype=float)
    K5 = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

    K6 = np.random.random((100,50))

    # Test the matrices
    matrices = [K1, K2, K4, K5, K3, K6]
    results = [test_rref_transformation(K) for K in matrices]

    for i, result in enumerate(results):
        print(f"Matrix {i+1} is in RREF: {result}")


if __name__ == "__main__":
    main2()