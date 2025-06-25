import numpy as np

def bnb_helper_anm(ancest, children, G, withinAgg, aggType, bs, intercept, bootstrap_indices=None):
    """
    Python equivalent of bnbHelperanm.cpp

    Parameters:
        ancest: (n, q) matrix of ancestor features (e.g., Hermite basis)
        children: (n, m) matrix of child variables
        G: (n, k, q) array of test function values: G[i, j, u] = h_j(Y[i, parent_u])
        withinAgg: which norm to use within each variable (1: L1, 2: L2, 3: inf)
        aggType: which norm to use across variables (1: L1, 2: L2, 3: inf)
        bs: number of bootstrap samples
        intercept: 1 or 0 for whether to add intercept

    Returns:
        pvals: array of p-values (length m)
    """
    n = ancest.shape[0]
    m = children.shape[1]
    k = G.shape[1]
    q = G.shape[2]  # number of parent variables

    # Safety check: G and ancest must align
    assert ancest.shape[1] == q, (
        f"[SHAPE ERROR] ancest.shape={ancest.shape}, G.shape={G.shape}"
    )

    # Add intercept if needed
    X = ancest
    if intercept:
        X = np.hstack((np.ones((n, 1)), X))
    p = X.shape[1]

    # Hat matrix
    hat = np.eye(n) - X @ np.linalg.pinv(X.T @ X) @ X.T

    # Precompute Q: shape (k, n, q)
    Q = np.zeros((k, n, q))
    for j in range(q):
        Q[:, :, j] = G[:, :, j].T @ hat

    pvals = np.zeros(m)

    for child_idx in range(m):
        res = hat @ children[:, child_idx]

        # Test statistic: shape (k, q)
        statMat = np.abs(np.einsum('knq,n->kq', Q, res) / np.sqrt(n))

        if withinAgg == 1:
            test_stat_vec = np.linalg.norm(statMat, ord=1, axis=0)
        elif withinAgg == 2:
            test_stat_vec = np.linalg.norm(statMat, ord=2, axis=0)
        elif withinAgg == 3:
            test_stat_vec = np.linalg.norm(statMat, ord=np.inf, axis=0)
        else:
            raise ValueError("Invalid withinAgg")

        if aggType == 1:
            test_stat = np.linalg.norm(test_stat_vec, ord=1)
        elif aggType == 2:
            test_stat = np.linalg.norm(test_stat_vec, ord=2)
        elif aggType == 3:
            test_stat = np.linalg.norm(test_stat_vec, ord=np.inf)
        else:
            raise ValueError("Invalid aggType")

        # Bootstrap null distribution
        null_dist = np.zeros(bs)
        for b in range(bs):
            if bootstrap_indices is not None:
                rand_idx = bootstrap_indices[b]
            else:
                rand_idx = np.random.randint(0, n, n)
            res_b = res[rand_idx]

            statMat_b = np.abs(np.einsum('knq,n->kq', Q, res_b) / np.sqrt(n - p))

            if withinAgg == 1:
                null_vec = np.linalg.norm(statMat_b, ord=1, axis=0)
            elif withinAgg == 2:
                null_vec = np.linalg.norm(statMat_b, ord=2, axis=0)
            elif withinAgg == 3:
                null_vec = np.linalg.norm(statMat_b, ord=np.inf, axis=0)

            if aggType == 1:
                null_dist[b] = np.linalg.norm(null_vec, ord=1)
            elif aggType == 2:
                null_dist[b] = np.linalg.norm(null_vec, ord=2)
            elif aggType == 3:
                null_dist[b] = np.linalg.norm(null_vec, ord=np.inf)

        # Compute p-value
        pvals[child_idx] = (np.sum(null_dist >= test_stat) + 1) / (bs + 1)

    return pvals
