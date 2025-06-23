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
- `config.py`: Utility functions used across modules, including statistical tests and graph operations.
- `coverage_simulation_randomDAG.py`: Runs simulation studies varying data distributions, sample sizes, and number of variables. Uses a different DAG for each replicate.
- `coverage_simulation_fixedDAG.py`: Same as above but uses the same DAG across all replicates for a fixed benchmark comparison.

## Installation

Clone the repository and install the package in **editable mode**:

```bash
git clone https://github.com/mouzhaner08/revised_cdcs.git
cd revised_cdcs
pip install -e .


## How to Run Simulations

This repository includes two simulation scripts to evaluate coverage properties of confidence sets for causal orderings. Each reflects a different design philosophy:



### 1. `coverage_simulation_randomDAG.py` (Follows the original paper)

- **DAG generation**: A new DAG is generated for each simulation iteration (`B = D.generate_B()` inside the loop).
- **Coverage** is computed with respect to the **true orderings of each newly generated DAG**, meaning:
  - Each simulation may have a different valid ordering set.
  - Results reflect the **average coverage behavior across random DAGs**.
- This setup replicates the design in the original paper:  
  _Wang et al., "Confidence Sets for Causal Orderings" (2024)_

To run a coverage simulation using different DAGs per replicate:

```bash
python -m simulation.coverage_simulation_randomDAG
```

### 2. `coverage_simulation_fixedDAG.py` (Controlled single-DAG evaluation)

- **DAG generation**: A single DAG is generated **once**, and reused across all simulations.
- **Coverage** is computed with respect to the **same set of true orderings**, across datasets simulated from that fixed DAG.
- This evaluates how **sampling variability alone** affects the confidence set construction.
- Useful for controlled studies, debugging, or stress-testing inference under fixed structure.

To run a simulation using a fixed DAG across replicates:

```bash
python -m simulation.coverage_simulation_fixedDAG
```

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