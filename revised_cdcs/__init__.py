"""
revised_cdcs: A package for causal discovery and simulation
"""

from .simulation.simulation import simulation

__version__ = "0.1.0"
__author__ = "Zhaner Mou"
__email__ = "zhmou@ucsd.edu"

# This allows users to do: from revised_cdcs import simulation
__all__ = ["simulation"]