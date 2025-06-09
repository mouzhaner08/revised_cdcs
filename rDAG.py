"""
Generate causal graph DAG
function:
    - generate_data(): simuate causal graph for GOF comparison
    - generate_data_confidence_sets(): simulate causal graph for constructing confidence sets
--------------
param:
    - p: the number of variables
    - n: the number of samples
    - error_dist: the distribution of errors 
        ('gauss', 'unif', 'lognorm', 'gamma', 'weibull', 'laplace', 'mixed', 't')
    - low_scale: the smallest possible error std
    - high_scale: the highest possible error std
    - uniqueTop: whether guarantee edge v-1->v for all v<=p
    - parent_prob: the probability of having an edge u->v for any u<v-1

return:
    - B: true linear coefficients (pxp)
    - Y: realized values (nxp)
    - errors: the idiosyncratic errors (nxp)
    - scale: std of each error variable 
"""

import numpy as np
from typing import List
from utils import *

class GenerateCausalGraph:
    def __init__(self, p:int, n:int, error_dist:str, coef:list, 
                 low_scale:int, high_scale:int, uniqueTop:str,
                 parent_prob:int):
        self.p = p
        self.n = n
        self.error_dist = error_dist
        self.coef = coef
        self.low_scale = low_scale
        self.high_scale = high_scale
        self.uniqueTop = uniqueTop
        self.parent_prob = parent_prob

    def get_scale(self):
        return np.random.uniform(low=self.low_scale, high=self.high_scale, size=self.p)

    def get_error(self, scale):
        if self.error_dist == 'gauss':
            errors = np.random.normal(loc=0, scale=1, size=(self.n, self.p))
        if self.error_dist == 'unif':
            errors = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(self.n, self.p))
        elif self.error_dist == 'lognormal':
            errors = np.random.lognormal(mean=0, sigma=1, size=(self.n, self.p))
        elif self.error_dist == 'gamma':
            errors = np.random.gamma(shape=2, scale=1/(np.sqrt(2)), size=(self.n, self.p))
            errors = errors - np.sqrt(2)
        elif self.error_dist == 'laplace':
            errors = np.random.laplace(loc=0, scale=1, size=(self.n, self.p))    
        elif self.error_dist == 'weibull':
            errors = np.random.weibull(a=1, size=(self.n, self.p)) 
        else:
            raise TypeError('Consider error terms either uniform, lognormal, gamma, Weibull, or Laplace.')
        return np.multiply(errors, scale)
    
    def generate_B(self):
        B = np.zeros((self.p, self.p))
        for v in range(2, self.p+1):
            # add edges u -> v with pre-assigned prob for any u < v-1
            if self.uniqueTop == 'T':           
                parents = np.random.choice(a=2, size=v-2, p=np.array([1-self.parent_prob, self.parent_prob]))
                parents = np.append(parents, 1)
            else:
                parents = np.random.choice(a=2, size=v-1, p=np.array([1-self.parent_prob, self.parent_prob]))
            # each linear coefficient is drawn from beta = z x g
            z = np.random.choice([-1, 1], size=v-1, p=np.array([1/2, 1/2]))
            g = np.random.gamma(shape=self.n**(-1/10), scale=1, size=v-1)
            parents = parents * z * g
            B[v-1, :v-1] = parents
        return B
    
    def generate_data(self, B):
        scale = self.get_scale()
        errors = self.get_error(scale)
        Y = np.linalg.solve(np.diag(np.ones(self.p))-B, np.transpose(errors))
        Y = np.transpose(Y)
        Y = (Y-np.mean(Y, axis=0)) / np.std(Y, axis=0)    # standardize Y
        return B, Y, errors, scale
    
    # generate the parent list for each variable
    def get_corret_parents_list_idx(self, B, v):
        return np.nonzero(B[v, :])[0].tolist()
    
    def get_all_orderings(self, B: np.ndarray) -> List[tuple]:
        """
        Find all valid causal orderings (topological orderings) of a DAG given its adjacency matrix B.
    
        Args:
            B: p x p coefficient matrix where B[i,j] != 0 indicates an edge j -> i
    
        Returns:
            List of all valid orderings (each represented as a tuple of node indices)
        """
        children = {i: [] for i in range(self.p)}
        in_degree = [0] * self.p
    
        for i in range(self.p):
            for j in range(self.p):
                if B[i,j] != 0:
                    children[j].append(i)
                    in_degree[i] += 1
    
        orderings = []
    
        def backtrack(ordering: List[int], remaining: List[int], in_degree_copy: List[int]):
            if len(ordering) == self.p:
                orderings.append(tuple(ordering))
                return
        
            for node in remaining:
                if in_degree_copy[node] == 0:
                    new_ordering = ordering + [node]
                    new_remaining = [x for x in remaining if x != node]
                    new_degree = in_degree_copy.copy()
                
                    for child in children[node]:
                        new_degree[child] -= 1
                
                    backtrack(new_ordering, new_remaining, new_degree)
    
        backtrack([], list(range(self.p)), in_degree.copy())
        return orderings

    
