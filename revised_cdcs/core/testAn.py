import numpy as np
from sklearn.preprocessing import SplineTransformer

def scale(x):
    """Standardize a 1D array to mean 0 and std 1."""
    return (x - np.mean(x)) / np.std(x)

def compute_test_tensor_G(Y: np.ndarray) -> np.ndarray:
    """
    Compute test tensor G (n x k x p), where:
    - n = number of samples
    - p = number of variables
    - k = 7 test functions per variable
    """
    n, p = Y.shape
    J = 2
    k = 2 * J + 3  # sin, cos (J times) + poly2, poly3, sign25

    G = np.zeros((n, k, p))

    for u in range(p):
        for j in range(1, J + 1):
            G[:, 2 * j - 2, u] = np.sin(j * Y[:, u])
            G[:, 2 * j - 1, u] = np.cos(j * Y[:, u])
        
        G[:, 4, u] = scale(Y[:, u]**2)
        G[:, 5, u] = scale(Y[:, u]**3)
        G[:, 6, u] = scale(np.sign(Y[:, u]) * np.abs(Y[:, u])**2.5)
    return G

# def compute_test_tensor_G(Y, parent_set, H):
#     """
#     Compute G[i, j, u] = h_j(Y[i, parent_set[u]])

#     Args:
#         Y: (n, p) matrix of observed data
#         parent_set: list of variable indices to include as ancestors
#         H: list of test function names, e.g., ['poly2', 'cos1', ...]

#     Returns:
#         G: array of shape (n, k, q) where:
#             - n: number of samples
#             - k: number of test functions (len(H))
#             - q: number of parent variables (len(parent_set))
#     """
#     n = Y.shape[0]
#     k = len(H)
#     q = len(parent_set)
#     G = np.zeros((n, k, q))

#     for u_index, u in enumerate(parent_set):  # ensure axis-2 indexing is consistent
#         col = Y[:, u]
#         for j, h in enumerate(H):
#             if h == 'poly2':
#                 transformed = col ** 2
#             elif h == 'poly3':
#                 transformed = col ** 3
#             elif h == 'sign25':
#                 transformed = np.sign(col) * np.abs(col) ** 2.5
#             elif h == 'sin1':
#                 transformed = np.sin(col)
#             elif h == 'cos1':
#                 transformed = np.cos(col)
#             elif h == 'sin2':
#                 transformed = np.sin(2 * col)
#             elif h == 'cos2':
#                 transformed = np.cos(2 * col)
#             else:
#                 raise ValueError(f"Unknown test function: {h}")

#             # Standardize to zero mean, unit variance
#             transformed = (transformed - np.mean(transformed)) / np.std(transformed)
#             G[:, j, u_index] = transformed

#     return G

def generate_basis(x, bs, intercept=True, method='poly'):
    """
    Generate basis matrix for a single variable:
    - x: 1D array
    - bs: number of basis functions
    - method: 'poly' or 'bspline'
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.shape[1] != 1:
        raise ValueError("Input to generate_basis must be a 1D array or a column vector.")

    if method == 'poly':
        X = np.hstack([x ** i for i in range(1, bs+1)])
    elif method == 'bspline':
        transformer = SplineTransformer(degree=3, n_knots=bs+1, include_bias=intercept)
        X = transformer.fit_transform(x)
    else:
        raise ValueError(f"Unknown basis method: {method}")

    if intercept:
        return np.column_stack((np.ones((x.shape[0], 1)), X))
    return X
