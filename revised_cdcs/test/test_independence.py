import os
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.linear_model import LinearRegression
from revised_cdcs.core import compute_test_tensor_G
from revised_cdcs.test import generate_scenarios, get_scenario_description

def compute_statistic(eta: np.ndarray, G: np.ndarray, norm: str = 'inf'):
    """
    Compute test statistic that which quantifies the dependence between a variable eta 
    and a fixed set of test functions stored in G.
    """
    n, k, p = G.shape
    means = (eta[:, None, None] * G).mean(axis=0) * np.sqrt(n)  # shape (k, p)
    
    if norm == 'l2':
       return np.linalg.norm(means, ord=2)
    elif norm == 'l1':
         return np.linalg.norm(means, ord=1)
    elif norm == 'inf':
         return np.linalg.norm(means, ord=np.inf)
    else:
        raise ValueError("Invalid norm")

def test_independence_pval(X: np.ndarray, Y: np.ndarray, B: int = 400, norm: str = 'inf'):
    """
    Test whether Y is conditionally independent of X using residual-based test statistic.
    
    Parameters:
    - X: (n, dx) predictor variables
    - Y: (n, dy) response variables
    - B: number of bootstrap resamples
    - norm: norm used in test statistic ('inf', 'l1', 'l2')
    
    Returns:
    - pvals: array of p-values for each child variable
    """
    # Ensure X and Y are 2D arrays
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
        
    n, dx = X.shape
    _, dy = Y.shape

    # Step 1: Fit linear regression of Y on X (with intercept)
    model = LinearRegression().fit(X, Y)
    Y_hat = model.predict(X)
    eta = (Y - Y_hat).flatten()  # shape (n,)
    
    # Step 2: Compute test functions on regressors
    G = compute_test_tensor_G(X) # shape (n, k, p)

    # Step 3: Compute observed statistic and null distribution
    T_obs = compute_statistic(eta, G, norm=norm)
    null_dist = np.zeros(B)

    for b in range(B):
        eta_b = eta[np.random.choice(n, n, replace=True)]
        null_dist[b] = compute_statistic(eta_b, G, norm=norm)

    # Step 4: Compute empirical p-value
    pval = (np.sum(null_dist >= T_obs) + 1) / (B + 1)
    
    return pval

def evaluate_test_performance(X, Y, reps: int=100, alpha: float=0.05, B: int=400,
                              norm: str='inf'):
    """
    Evaluates the power or type I error of the independence test on fixed (X, Y) data.
    
    Parameters
    ----------
    X: np.ndarray
        A 1D or 2D array of shape (n,) or (n, d1) representing variable X.
    Y: np.ndarray
        A 1D or 2D array of shape (n,) or (n, d2) representing variable Y.
    reps: int, optional
        Number of repetitions to simulate the test, by default 100.
    alpha: float, optional
        Significance level for hypothesis testing, by default 0.05.
    B: int, optional
        number of bootstrap resamples, by default 400.
    norm: str, optional
        which norm to aggregate over test functions, by default 'inf'.
    
    Returns
    -------
    float
        Estimated type I error rate,
        computed as the proportion of times the null hypothesis is rejected.
    """
    n = X.shape[0]
    pvals = []
    for _ in trange(reps, desc="Running tests"):
        idx = np.random.choice(n, n, replace=True)
        pval = test_independence_pval(X=X[idx], Y=Y[idx], B=B, norm=norm)
        pvals.append(pval)
    
    pvals = np.array(pvals)
    rate = np.mean(pvals < alpha)

    print(f"Rejection rate at alpha={alpha:.2f}: {rate:.3f}")
    
    return rate

def main(n=2500, reps=100, alpha=0.05, B=400, norm='inf'):
    scenarios = generate_scenarios()
    results = []
    
    # Classify scenarios for the output
    scenario_types = {
        **{f"Independent GG {i}": "Independent (G-G)" for i in range(1, 11)},
        **{f"Independent GN {i}": "Independent (G-NG)" for i in range(1, 11)},
        **{f"Independent NN {i}": "Independent (NG-NG)" for i in range(1, 11)},
        **{f"Subtle Weak {i}": "Subtle Dependent" for i in range(1, 6)},
        **{f"Subtle Cond {i}": "Subtle Dependent" for i in range(1, 6)},
        **{f"Subtle Higher {i}": "Subtle Dependent" for i in range(1, 6)},
        **{f"Dependent Strong {i}": "Clearly Dependent" for i in range(1, 6)},
        **{f"Dependent Mixed {i}": "Clearly Dependent" for i in range(1, 6)},
    }
    
    for name, gen_func in scenarios.items():
        print(f"Testing scenario: {name}")
        X, Y = gen_func(n)
        rejection_rate = evaluate_test_performance(X, Y, reps=reps, alpha=alpha, B=B, norm=norm)
        print(f"Rejection rate for '{name}': {rejection_rate:.3f}\n")
        
        results.append({
            "Scenario": name,
            "Type": scenario_types[name],
            "RejectionRate": rejection_rate,
            "ExpectedIndependent": "Independent" in scenario_types[name],
            "Description": get_scenario_description(name)
        })
    
    # Define the output directory and file path
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir,f"independence_test_results_n={n}.csv")
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return df_results

if __name__ == "__main__":
    df_results = main()