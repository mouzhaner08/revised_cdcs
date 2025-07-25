import numpy as np
import warnings
from typing import List, Tuple, Literal
from scipy.linalg import solve

def bnb_helper_anm(ancest: np.ndarray, 
                   children: np.ndarray, 
                   G: np.ndarray, 
                   withinAgg: int = 3, 
                   aggType: int = 3, 
                   bs: int = 400, 
                   bootstrap_method: int = 2, 
                   bootstrap_indices = None):
    """
    Python equivalent of bnbHelperanm.cpp. This function performs hypothesis testing 
    using bootstrap resampling for what appears to be a variable selection problem.

    Parameters:
    -----------
    ancest : numpy.ndarray, shape (n, p_ancest)
        Matrix of ancestor basis functions (predictor variables)
    children : numpy.ndarray, shape (n, num_children) 
        Matrix of observed children (response variables to test)
    G : numpy.ndarray, shape (n, k, num_variables)
        3D array where each "slice" (G[:,:,i]) represents k test functions 
        for the i-th variable. This is used to construct test statistics.
    withinAgg : int
        Aggregation method within variables across test functions:
        1 = L1 norm (sum of absolute values)
        2 = L2 norm (Frobenius/Euclidean norm) 
        3 = L∞ norm (maximum absolute value)
    aggType : int  
        Aggregation method across variables:
        1 = L1 norm, 2 = L2 norm, 3 = L∞ norm
    bs : int
        Number of bootstrap samples for null distribution
    bootstrap_method : int, optional (default=2)
        Bootstrap method to use:
        1 = Rademacher bootstrap (sign-flip): multiply residuals by ±1
        2 = Residual resampling bootstrap: resample residuals with replacement
    bootstrap_indices : numpy.ndarray, optional (default=None)
        Pre-computed bootstrap indices of shape (bs, n). Only used when 
        bootstrap_method=2. If None, indices are generated randomly.
    
    Returns:
    --------
        pval : numpy.ndarray
        array of p-values (length m)
    """
    # Get dimensions
    n = ancest.shape[0]  # number of observations
    k = G.shape[1]       # number of test functions per variable
    num_vars = G.shape[2]  # number of variables in G

    # Set up design matrix X (predictors)
    X = ancest.copy()
    ones_column = np.ones((n, 1))
    X = np.column_stack([ones_column, X])

    num_predictors = X.shape[1]

    # Initialize outputs
    num_children = children.shape[1]
    pval = np.zeros(num_children)

    # Hat matrix
    try:
        XtX_inv = solve(X.T @ X, np.eye(num_predictors)) 
        hat_mat = np.eye(n) - X @ XtX_inv @ X.T
    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix encountered in hat matrix computation")
        # Fallback to pseudoinverse
        hat_mat = np.eye(n) - X @ np.linalg.pinv(X.T @ X) @ X.T

    # Pre-compute Q matrices for efficiency
    Q = np.zeros((k, n, num_vars))
    for j in range(num_vars):
        Q[:, :, j] = G[:, :, j].T @ hat_mat

    # Test each child variable
    for j in range(num_children):
        residuals = hat_mat @ children[:, j]

        # Compute test statistics for observed data
        stat_mat = np.zeros((k, num_vars))
        for var_idx in range(num_vars):
            stat_mat[:, var_idx] = Q[:, :, var_idx] @ residuals
        stat_mat = np.abs(stat_mat / np.sqrt(n))

        # Aggregate within each variable (across the k test functions)
        test_statistic_vec = np.zeros(num_vars)
        for z in range(num_vars):
            if withinAgg == 1:
                test_statistic_vec[z] = np.linalg.norm(stat_mat[:, z], ord=1)
            elif withinAgg == 2:
                test_statistic_vec[z] = np.linalg.norm(stat_mat[:, z], ord=2)
            elif withinAgg == 3:
                test_statistic_vec[z] = np.linalg.norm(stat_mat[:, z], ord=np.inf)

        # Aggregate across variables        
        if aggType == 1:
            test_stat = np.linalg.norm(test_statistic_vec, ord=1)
        elif aggType == 2:
            test_stat = np.linalg.norm(test_statistic_vec, ord=2)  
        else:
            test_stat = np.linalg.norm(test_statistic_vec, ord=np.inf)

        # Generate null distribution via bootstrap
        null_dist = np.zeros(bs)

        if bootstrap_method == 1:  # Rademacher bootstrap
            for b in range(bs):
                # Multiply residuals by random signs (+1 or -1)
                signs = np.random.choice([-1, 1], size=n)
                bootstrap_residuals = residuals * signs

                # Compute bootstrap test statistic matrix
                stat_mat_bootstrap = np.zeros((k, num_vars))
                for var_idx in range(num_vars):
                    stat_mat_bootstrap[:, var_idx] = Q[:, :, var_idx] @ bootstrap_residuals
                stat_mat_bootstrap = np.abs(stat_mat_bootstrap / np.sqrt(n - num_predictors))

                # Apply within-variable aggregation (across k test functions)
                null_stat_vec = np.zeros(num_vars)
                for z in range(num_vars):
                    if withinAgg == 1:
                        null_stat_vec[z] = np.linalg.norm(stat_mat_bootstrap[:, z], ord=1)
                    elif withinAgg == 2:
                        null_stat_vec[z] = np.linalg.norm(stat_mat_bootstrap[:, z], ord=2)
                    elif withinAgg == 3:
                        null_stat_vec[z] = np.linalg.norm(stat_mat_bootstrap[:, z], ord=np.inf)
                
                # Apply across-variable aggregation and store in null distribution
                if aggType == 1:
                    null_dist[b] = np.linalg.norm(null_stat_vec, ord=1)
                elif aggType == 2:
                    null_dist[b] = np.linalg.norm(null_stat_vec, ord=2)
                else:
                    null_dist[b] = np.linalg.norm(null_stat_vec, ord=np.inf)

                
        elif bootstrap_method == 2:  # Residual resampling bootstrap
            for b in range(bs):
                # Resample residuals with replacement
                if bootstrap_indices is not None:
                    rand_idx = bootstrap_indices[b]
                else:
                    rand_idx = np.random.randint(0, n, n)
                bootstrap_residuals = residuals[rand_idx]

                # Compute bootstrap test statistic matrix
                stat_mat_bootstrap = np.zeros((k, num_vars))
                for var_idx in range(num_vars):
                    stat_mat_bootstrap[:, var_idx] = Q[:, :, var_idx] @ bootstrap_residuals
                stat_mat_bootstrap = np.abs(stat_mat_bootstrap / np.sqrt(n - num_predictors))
                
                # Apply within-variable aggregation (across k test functions)
                null_stat_vec = np.zeros(num_vars)
                for z in range(num_vars):
                    if withinAgg == 1:
                        null_stat_vec[z] = np.linalg.norm(stat_mat_bootstrap[:, z], ord=1)
                    elif withinAgg == 2:
                        null_stat_vec[z] = np.linalg.norm(stat_mat_bootstrap[:, z], ord=2)
                    elif withinAgg == 3:
                        null_stat_vec[z] = np.linalg.norm(stat_mat_bootstrap[:, z], ord=np.inf)
                
                # Apply across-variable aggregation and store in null distribution
                if aggType == 1:
                    null_dist[b] = np.linalg.norm(null_stat_vec, ord=1)
                elif aggType == 2:
                    null_dist[b] = np.linalg.norm(null_stat_vec, ord=2)
                else:
                    null_dist[b] = np.linalg.norm(null_stat_vec, ord=np.inf)
        
        else:
            raise ValueError(f"Invalid bootstrap_method: {bootstrap_method}. Use 1 (Rademacher) or 2 (Residual resampling)")

        # Compute p-value: proportion of bootstrap stats >= observed stat
        # +1 in numerator and denominator accounts for the observed statistic
        pval[j] = (np.sum(null_dist >= test_stat) + 1) / (bs + 1)

    return pval