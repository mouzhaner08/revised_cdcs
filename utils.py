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
# rDAG.py
#p = 3   # [10, 15, 20, 30, 45]
#n_list = [500, 1000, 2500, 5000]   # [p^(5/4), p^2]
#dist_list = ['unif', 'lognormal', 'gamma', 'weibull', 'laplace']  # ['unif', 'lognormal', 'gamma', 'weibull', 'laplace']
coef = [-1, 1, -.95, .95]
low_scale = .8
high_scale = 1
# uniqueTop = "T"  # ["T", "F"]
parent_prob = 1/3


# ------------------------
# gof_test.py
# H = ['poly', 'sign', 'cubic', 'sin1', 'cos1', 'sin2', 'cos2']
bs = 400

# ------------------------
# bnb.py
alpha = 0.1