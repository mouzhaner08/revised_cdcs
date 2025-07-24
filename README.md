# Replication of Branch-and-Bound Algorithm for Causal Orderings

This repository contains a Python-based replication of the algorithm and simulation setup described in:

> Y. S. Wang, M. Kolar, and M. Drton, "Confidence Sets for Causal Orderings," arXiv:2305.14506, Oct. 2024.  
> [Original code repository](https://github.com/ysamwang/cdcs)

The goal is to reproduce the simulation results and confidence sets for causal orderings using the branch-and-bound (BnB) algorithm with additive noise models (ANMs).

## Repository Structure

- `core/`: Core implementation modules
  - `bnb.py`: Core implementation of the branch-and-bound algorithm for constructing confidence sets
  - `bnb_helper_anm.py`: Helper functions to evaluate p-values for candidate parent sets under the ANM framework
  - `testAn.py`: Defines a library of test functions used in hypothesis testing within the ANM model
  - `rDAG.py`: Functions to generate random DAGs with additive noise and configurable edge weights and noise distributions
- `simulation/`: Simulation modules
  - `simulation.py`: Main simulation interface that runs coverage studies for confidence sets
- `test/`: Test of independence modules
  - `test_independence.py`: Separate modules for testing of independence. Automatic example cases are stored in `get_scenarios.py` for reference


## Installation

Clone the repository and install the package in **editable mode**:

### Option 1: Conda

```bash
# Clone the repository
git clone https://github.com/mouzhaner08/revised_cdcs.git
cd revised_cdcs

# Create and activate conda environment
conda create -n cdcs_env --file conda-requirements.txt
conda activate cdcs_env

# Install package in editable mode
pip install -e .
```

### Option 2: Pip
```bash
# Clone the repository
git clone https://github.com/mouzhaner08/revised_cdcs.git
cd revised_cdcs

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install with pip
pip install -e .
```

## How to Run Simulations

This package provides a simple interface to run coverage simulations for evaluating confidence sets for causal orderings. 

### Quick Start
```python
from revised_cdcs import simulation

# Comprehensive evaluation
results = simulation(
    random_DAG=True,
    p=3,
    dist_list=['unif', 'lognormal', 'laplace'],
    n_list=[500, 1000, 2000],
    n_simulations=100,
    bs = 400,
    alpha=0.1
)
```

### Simulation Parameters

- **`p`** (int, default=5): Number of variables in the DAG
- **`dist_list`** (list, default=['unif', 'lognormal', 'gamma', 'weibull', 'laplace']): List of error distributions
- **`n_list`** (list, default=[500, 1000, 2500, 5000]): List of sample sizes
- **`n_simulations`** (int, default=100): Number of simulation replicates
- **`alpha`** (float, default=0.1): Significance level for confidence sets
- **`bs`** (int, default=400): Number of bootstrap replicates
- **`parent_prob`** (float, default=1/3): The Prob of an edge between any two nodes
- **`low_scale`** (float, default=0.8): Lower bound on variance of error terms
- **`high_scale`** (float, default=1.0): Upper bound on variance of error terms
- **`uniqueTop`** (str, default='F'): Whether to enforce a unique topological ordering so that u -> u+1 for all u 
- **`random_DAG`** (bool, default=True): Controls DAG generation strategy
  - `True`: Generate a new DAG for each simulation replicate (follows original paper)
  - `False`: Use the same DAG across all replicates (controlled evaluation)

### Example Output

Simulation results include:

- Size of the confidence set
- Number of true orderings
- Marginal and simultaneous coverage rates

These are printed during the run and returned as structured results for further analysis.


## Requirements
Requirements are listed in `conda-requirements.txt` and `setup.py`

## Acknowledgment

This project replicates and builds upon the methodology introduced by Wang et al. in the paper *"Confidence Sets for Causal Orderings."*  
The original R/C++ codebase is available at: https://github.com/ysamwang/cdcs

For questions or collaborations, please contact the repository author or contributors.