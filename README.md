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
  - `config.py`: Utility functions used across modules, including statistical tests and graph operations
- `simulation/`: Simulation modules
  - `simulation.py`: Main simulation interface that runs coverage studies for confidence sets

## Installation

Clone the repository and install the package in **editable mode**:

```bash
git clone https://github.com/mouzhaner08/revised_cdcs.git
cd revised_cdcs
pip install -e .
```

## How to Run Simulations

This package provides a simple interface to run coverage simulations for evaluating confidence sets for causal orderings. The main simulation function supports two different design philosophies:

### Quick Start

```python
from revised_cdcs import simulation

# Run simulation with default parameters
results = simulation()

# Run simulation with custom parameters
results = simulation(
    random_DAG=False, 
    p=3, 
    dist_list=['unif'], 
    n_list=[500, 1000]
)
```

### Simulation Parameters

- **`random_DAG`** (bool, default=True): Controls DAG generation strategy
  - `True`: Generate a new DAG for each simulation replicate (follows original paper)
  - `False`: Use the same DAG across all replicates (controlled evaluation)
- **`p`** (int, default=4): Number of variables in the DAG
- **`dist_list`** (list, default=['unif', 'normal']): List of data distributions to test
- **`n_list`** (list, default=[100, 500, 1000]): List of sample sizes to evaluate
- **`B`** (int, default=100): Number of simulation replicates
- **`alpha`** (float, default=0.05): Significance level for confidence sets

### Simulation Design Options

#### 1. Random DAG per Replicate (`random_DAG=True`)

- **DAG generation**: A new DAG is generated for each simulation iteration
- **Coverage** is computed with respect to the **true orderings of each newly generated DAG**
- Results reflect the **average coverage behavior across random DAGs**
- This setup replicates the design in the original paper: _Wang et al., "Confidence Sets for Causal Orderings" (2024)_

```python
from revised_cdcs import simulation

# Follows original paper methodology
results = simulation(random_DAG=True, p=4, dist_list=['normal'], n_list=[500])
```

#### 2. Fixed DAG across Replicates (`random_DAG=False`)

- **DAG generation**: A single DAG is generated once and reused across all simulations
- **Coverage** is computed with respect to the **same set of true orderings**
- Evaluates how **sampling variability alone** affects confidence set construction
- Useful for controlled studies, debugging, or stress-testing inference under fixed structure

```python
from revised_cdcs import simulation

# Controlled single-DAG evaluation
results = simulation(random_DAG=False, p=3, dist_list=['unif'], n_list=[500, 1000])
```

### Example Output

Simulation results include:

- Size of the confidence set
- Number of true orderings
- Marginal and simultaneous coverage rates

These are printed during the run and returned as structured results for further analysis.

### Advanced Usage

For more complex simulation studies:

```python
from revised_cdcs import simulation

# Comprehensive evaluation
results = simulation(
    random_DAG=True,
    p=5,
    dist_list=['unif', 'normal', 'laplace'],
    n_list=[100, 250, 500, 1000, 2000],
    B=200,
    alpha=0.1
)
```

## Requirements

- Python >= 3.7
- numpy >= 1.19.0
- pandas >= 1.3.0
- scipy >= 1.7.0

## Acknowledgment

This project replicates and builds upon the methodology introduced by Wang et al. in the paper *"Confidence Sets for Causal Orderings."*  
The original R/C++ codebase is available at: https://github.com/ysamwang/cdcs

For questions or collaborations, please contact the repository author or contributors.