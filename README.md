# Replication of Branch-and-Bound Algorithm for Causal Orderings

This repository contains a Python-based replication of the algorithm and simulation setup described in:

> Y. S. Wang, M. Kolar, and M. Drton, “Confidence Sets for Causal Orderings,” arXiv:2305.14506, Oct. 2024.  
> [Original code repository](https://github.com/ysamwang/cdcs)

The goal is to reproduce the simulation results and confidence sets for causal orderings using the branch-and-bound (BnB) algorithm with additive noise models (ANMs).

## Repository Structure

- `bnb.py`: Core implementation of the branch-and-bound algorithm for constructing confidence sets.
- `bnb_helper_anm.py`: Helper functions to evaluate p-values for candidate parent sets under the ANM framework.
- `testAn.py`: Defines a library of test functions used in hypothesis testing within the ANM model.
- `rDAG.py`: Functions to generate random DAGs with additive noise and configurable edge weights and noise distributions.
- `utils.py`: Utility functions used across modules, including statistical tests and graph operations.
- `coverage_simulation.py`: Runs simulation studies varying data distributions, sample sizes, and number of variables. Uses a different DAG for each replicate.
- `coverage_simulation_sameB.py`: Same as above but uses the same DAG across all replicates for a fixed benchmark comparison.

## Installation

This code requires the following Python packages:

- `numpy`
- `scipy`
- `pandas`
- `tqdm`


## How to Run Simulations

To run a coverage simulation using different DAGs per replicate:

```bash
python coverage_simulation.py
```

To run a simulation using a fixed DAG across replicates:

```bash
python coverage_simulation_sameB.py
```

Simulation parameters such as number of simulations (`n_simulations`), sample size (`n`), number of variables (`p`), noise distribution, and bootstrap size (`B`) can be edited directly in the corresponding `.py` files.

### Example Output

Simulation results include:

- Size of the confidence set
- Number of true orderings
- Marginal and simultaneous coverage rates

These are printed during the run and can be modified to save to a `.csv` for downstream analysis.

---

## Acknowledgment

This project replicates and builds upon the methodology introduced by Wang et al. in the paper *“Confidence Sets for Causal Orderings.”*  
The original R/C++ codebase is available at: https://github.com/ysamwang/cdcs

For questions or collaborations, please contact the repository author or contributors.