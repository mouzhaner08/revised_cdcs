import numpy as np
import pandas as pd

def scale(x):
    """Standardize a 1D array to mean 0 and std 1."""
    return (x - np.mean(x)) / np.std(x)

def compute_test_tensor_G(Y: np.ndarray) -> np.ndarray:
    """
    Compute test tensor G (n x k x p), where:
    - n = number of samples
    - p = number of variables (here: 1)
    - k = 7 test functions: sin/cos (2), poly2, poly3, sign * |x|^2.5
    """
    n, p = Y.shape
    J = 2
    k = 2 * J + 3
    G = np.zeros((n, k, p))

    for u in range(p):
        for j in range(1, J + 1):
            G[:, 2 * j - 2, u] = np.sin(j * Y[:, u])
            G[:, 2 * j - 1, u] = np.cos(j * Y[:, u])
        G[:, 4, u] = scale(Y[:, u] ** 2)
        G[:, 5, u] = scale(Y[:, u] ** 3)
        G[:, 6, u] = scale(np.sign(Y[:, u]) * np.abs(Y[:, u]) ** 2.5)
    return G

def compute_statistic(eta: np.ndarray, G: np.ndarray, norm: str = 'inf') -> float:
    """
    Compute test statistic || (1/n) sum eta_i * G[i, j, u] ||_norm
    Requires G to be (n, k, 1).
    """
    means = (eta[:, None] * G[:, :, 0]).mean(axis=0)  # shape (k,)
    if norm == 'l2':
        return np.linalg.norm(means, ord=2)
    elif norm == 'l1':
        return np.linalg.norm(means, ord=1)
    elif norm == 'inf':
        return np.linalg.norm(means, ord=np.inf)
    else:
        raise ValueError("Invalid norm")

def test_independence_pval(X: np.ndarray, Y: np.ndarray, B: int = 400, norm: str = 'inf') -> float:
    """
    Perform nonparametric independence test using Wang et al.'s method.
    Inputs:
        X, Y: shape (n,) arrays
        B: number of bootstrap resamples
        norm: which norm to aggregate over test functions
    Returns:
        p-value
    """
    n = len(X)
    
    eta = X.copy()
    G = compute_test_tensor_G(Y.reshape(-1, 1))
    T_obs = compute_statistic(eta, G, norm=norm)

    null_dist = np.zeros(B)
    for b in range(B):
        eta_b = eta[np.random.choice(n, n, replace=True)]
        null_dist[b] = compute_statistic(eta_b, G, norm=norm)

    pval = (np.sum(null_dist >= T_obs) + 1) / (B + 1)
    return pval

def evaluate_test_performance(n: int = 500, reps: int = 1000, B: int = 400,
                              alpha: float = 0.05, norm: str = 'inf') -> float:
    """
    Evaluate Type I error and power across 4 defined scenarios.
    Saves results to 'typeI_error_results.csv'.
    """
    results = []

    scenarios = [
        ("X~Normal, Y~Laplace", lambda n: np.random.normal(0, 1, n), lambda n: np.random.laplace(0, 1, n)),
        ("X~Normal, Y~Uniform", lambda n: np.random.normal(0, 1, n), lambda n: np.random.uniform(-2, 2, n)),
        ("X~Laplace, Y~Uniform", lambda n: np.random.laplace(0, 1, n), lambda n: np.random.uniform(-2, 2, n)),
        ("X~Laplace, Y=|X|^2.5+e", 
         lambda n: np.random.laplace(0, 1, n),
         lambda n, X=None: np.sign(X) * np.abs(X)**2.5 + np.random.normal(0, 0.5, n),
         True)
    ]

    for name, X_gen, Y_gen, *uses_X in scenarios:
        pvals = []
        for i in range(reps):
            X = X_gen(n)
            if uses_X and uses_X[0]:
                Y = Y_gen(n, X=X)
            else:
                Y = Y_gen(n)
            p = test_independence_pval(X, Y, B=B, norm=norm)
            pvals.append(p)
        type_I_error = np.mean(np.array(pvals) < alpha)
        results.append({"Scenario": name, "TypeIError": type_I_error})

    df = pd.DataFrame(results)
    df.to_csv("typeI_error_results.csv", index=False)
    return df

# Main execution
if __name__ == "__main__":
    df_results = evaluate_test_performance(n=500, reps=1000, B=400, alpha=0.05, norm='inf')
    print(df_results)
