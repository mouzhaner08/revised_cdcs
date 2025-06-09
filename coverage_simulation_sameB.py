import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from math import factorial
from UncertaintyQuantification.testAn import compute_test_tensor_G
from UncertaintyQuantification.bnb import ConfidenceSet
from rDAG import GenerateCausalGraph

# global parameters
p = 5
dist_list = ['unif']
n_list = [1000]
n_simulations = 100
H = ['poly2', 'poly3', 'sign25', 'sin1', 'cos1', 'sin2', 'cos2']
alpha = 0.1
bs = 400
K = 5

# DAG generation config
parent_prob = 1/3
low_scale = 0.8
high_scale = 1.0
coef = 1.0
uniqueTop = 'T'


# Single simulation run for fixed DAG B
def run_onceBnb(D, B, i, size, prop_true_orderings_covered, sim_coverage, num_true_orderings):
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


# Run simulation across many datasets
def run_simulation(n_simulations, p, n, dist):
    D = GenerateCausalGraph(p=p, n=n, error_dist=dist, coef=coef,
                            low_scale=low_scale, high_scale=high_scale,
                            uniqueTop=uniqueTop, parent_prob=parent_prob)
    B = D.generate_B()

    num_true_orderings = np.zeros(n_simulations)
    prop_true_orderings_covered = np.zeros(n_simulations, dtype=float)
    size = np.zeros(n_simulations, dtype=int)
    sim_coverage = np.zeros(n_simulations, dtype=int)

    for i in tqdm(range(n_simulations), desc=f"Simulating dist={dist}"):
        run_onceBnb(D, B, i, size, prop_true_orderings_covered, sim_coverage, num_true_orderings)

    return size, num_true_orderings, prop_true_orderings_covered, sim_coverage


def summarize_and_save(results, p, n, n_simulations):
    df = pd.DataFrame(results)
    df['permutation'] = factorial(p)
    df = df[['dist', 'p', 'n', 'permutation', 'avg_true_orderings', 'std_true_orderings',
             'avg_set_size', 'std_set_size', 'sim_coverage', 'marginal_coverage']]

    df.to_csv(f"coverage_summary_p={p}_n={n}_sim={n_simulations}_uniqueTop={uniqueTop}_sameB.csv", index=False)
    return df

# -----------------------------
# Run all settings
results = []

for dist, n in product(dist_list, n_list):
    print(f"\nRunning p={p}, n={n}, dist={dist}")
    size, num_true_orderings, prop_true_orderings_covered, sim_coverage = run_simulation(n_simulations, p, n, dist)

    results.append({
        'dist': dist,
        'p': p,
        'n': n,
        'avg_set_size': size.mean(),
        'std_set_size': size.std(),
        'avg_true_orderings': num_true_orderings.mean(),
        'std_true_orderings': num_true_orderings.std(),
        'sim_coverage': sim_coverage.mean(),
        'marginal_coverage': prop_true_orderings_covered.mean()
    })

# Save results
df_summary = summarize_and_save(results, p, n, n_simulations)
print(df_summary)
