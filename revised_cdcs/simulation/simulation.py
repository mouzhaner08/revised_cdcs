import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from math import factorial
from revised_cdcs.core.testAn import compute_test_tensor_G
from revised_cdcs.core.bnb import ConfidenceSet
from revised_cdcs.core.rDAG import GenerateCausalGraph


def run_onceBnb(D, B, i, size, prop_true_orderings_covered, sim_coverage, num_true_orderings, bs, alpha, K):
    """
    Run a single simulation instance using a fixed DAG and collect coverage metrics.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    D: GenerateCausalGraph
        Instance used to generate simulated data from the DAG.
    B: np.ndarray
        DAG coefficient matrix (adjacency matrix)
    i: int
        Index of the current simulation
    size: np.ndarray
        Array to record the size of the confidence set for each simulation
    prop_true_orderings_covered: np.ndarray
        Array to store marginal coverage (fraction of true orderings covered)
    sim_coverage: np.ndarray
        Array to store whether all true orderings were covered (1 or 0)
    num_true_orderings: np.ndarray
        Array to record the number of valid topological orderings per simulation
        
    ---------------------------------------------------------------------------
    OUTPUT:
    None. All results are stored in the provided arrays.
    """
    B, Y, errors, scale = D.generate_data(B)
    true_orderings = D.get_all_orderings(B)

    cs = ConfidenceSet(Y=Y, bs=bs, alpha=alpha, basis='poly', K=K,
                       agg_type=3, p_value_agg="tippett", intercept=True, verbose=False)
    conf_set_df = cs.branchAndBound()
    conf_set = [tuple(row[1:].astype(int)) for _, row in conf_set_df.iterrows()]
    conf_set = [tuple(x - 1 for x in ordering) for ordering in conf_set]

    size[i] = len(conf_set)
    num_true_orderings[i] = len(true_orderings)

    covered = set(conf_set)
    matches = [tuple(ord) in covered for ord in true_orderings]

    # marginal coverage (fraction of true orderings covered)
    prop_true_orderings_covered[i] = np.mean(matches)

    # simultaneous coverage (all true orderings covered)
    sim_coverage[i] = int(all(matches))


def run_simulation(random_DAG:bool, n_simulations:int, p:int, n:int, dist, coef, 
                   low_scale, high_scale, uniqueTop:bool, parent_prob, bs:int, K:int, alpha:float):
    """
    Run multiple simulations to evaluate confidence set performance under a given error distribution.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    n_simulations : int
        Number of simulations to run.
    p : int
        Number of variables in the DAG.
    n : int
        Number of samples per dataset.
    dist : str
        Error distribution type (e.g., 'unif', 'gauss', etc.).
    K : int
        Either the df for bsplines or the number of polynomial terms.

    ---------------------------------------------------------------------------
    OUTPUT:
    size : np.ndarray
        Sizes of the confidence sets across simulations.
    num_true_orderings : np.ndarray
        Number of valid topological orderings for each simulation.
    prop_true_orderings_covered : np.ndarray
        Proportion of true orderings covered in each simulation (marginal coverage).
    sim_coverage : np.ndarray
        Whether all true orderings were simultaneously covered in each simulation.
    """
    D = GenerateCausalGraph(p=p, n=n, error_dist=dist, coef=coef,
                            low_scale=low_scale, high_scale=high_scale,
                            uniqueTop=uniqueTop, parent_prob=parent_prob)
    num_true_orderings = np.zeros(n_simulations)
    prop_true_orderings_covered = np.zeros(n_simulations, dtype=float)
    size = np.zeros(n_simulations, dtype=int)
    sim_coverage = np.zeros(n_simulations, dtype=int)

    if random_DAG == False:
        # Generate the adjancy matrix B once and reuse it for all simulations
        B = D.generate_B()
        for i in tqdm(range(n_simulations), desc=f"Simulating dist={dist}"):
            run_onceBnb(D, B, i, size, prop_true_orderings_covered, sim_coverage, num_true_orderings, bs, alpha, K)
    else:
        # Generate a new adjancy matrix B for each simulation
        for i in tqdm(range(n_simulations), desc=f"Simulating dist={dist}"):
            B = D.generate_B()
            run_onceBnb(D, B, i, size, prop_true_orderings_covered, sim_coverage, num_true_orderings, bs, alpha, K)

    return size, num_true_orderings, prop_true_orderings_covered, sim_coverage

# -----------------------------
# Run all settings
def simulation(p = 5, 
               dist_list = ['unif', 'lognormal', 'gamma', 'weibull', 'laplace'], 
               n_list = [500, 1000, 2500, 5000],
               n_simulations = 100,
               alpha = 0.1,
               bs = 400,
               K = 5,
               parent_prob = 1/3,
               low_scale = 0.8,
               high_scale = 1.0,
               coef = 1.0,
               uniqueTop = False,
               random_DAG = True
):
    results = []

    for dist, n in product(dist_list, n_list):
        print(f"\nRunning p={p}, n={n}, dist={dist}")
        start_time = time.time()

        size, num_true_orderings, prop_true_orderings_covered, sim_coverage = run_simulation(
            random_DAG=random_DAG, n_simulations=n_simulations, p=p, n=n, dist=dist, coef=coef, 
            low_scale=low_scale, high_scale=high_scale, uniqueTop=uniqueTop, parent_prob=parent_prob,
            bs=bs, K=K, alpha=alpha)
        
        end_time = time.time()
        total_seconds = end_time - start_time
        
        results.append({
            'dist': dist,
            'p': p,
            'n': n,
            'avg_set_size': size.mean(),
            'std_set_size': size.std(),
            'avg_true_orderings': num_true_orderings.mean(),
            'std_true_orderings': num_true_orderings.std(),
            'sim_coverage': sim_coverage.mean(),
            'marginal_coverage': prop_true_orderings_covered.mean(),
            'runtime_sec': total_seconds
        })

    # Create the summary DataFrame
    df_summary = pd.DataFrame(results)
    df_summary['permutation'] = factorial(p)
    df_summary = df_summary[['dist', 'p', 'n', 'permutation', 'avg_true_orderings', 'std_true_orderings',
                             'avg_set_size', 'std_set_size', 'sim_coverage', 'marginal_coverage',
                             'runtime_sec']]

    # Define the output directory and file path
    if random_DAG:
        output_dir = os.path.join(os.path.dirname(__file__), 'results', 'random_DAG')
    else:
        output_dir = os.path.join(os.path.dirname(__file__), 'results', 'fixed_DAG')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"coverage_summary_p={p}_n={max(n_list)}_sim={n_simulations}.csv"
    )

    # Save the results to a CSV file
    df_summary.to_csv(output_path, index=False)
    print(df_summary)

if __name__ == "__main__":
    simulation(p=5, dist_list=['unif'], n_list=[1000])