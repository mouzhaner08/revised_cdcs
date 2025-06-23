"""
Parameters
--------------
p: num of nodes
n: num of samples (p^(5/4) or p^2)
error_dist: the distribution of errors 
coef: linear coefficient for each edge

low_scale: smallest possible error std
high_scale: largest possible error std

H: list of test functions hj
L: num of residual bootstrap 
"""
# global parameters
p = 3    # [10, 15, 20, 30, 45]
dist_list = ['unif']    # ['unif', 'lognormal', 'gamma', 'weibull', 'laplace']
n_list = [1000]    #n_list = [500, 1000, 2500, 5000]
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