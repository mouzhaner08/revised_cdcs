import numpy as np
from scipy.stats import t, cauchy, expon, gamma, poisson, bernoulli

def generate_scenarios():

    """Returns a dict of scenario name -> generator function (n -> (X, Y))"""
    scenarios = {}
    
    # --------------------------
    # Independent Cases (30 total)
    # --------------------------
    
    # 1. Gaussian-Gaussian (10 cases)
    for i in range(1, 11):
        def make_gg(n, i=i):
            np.random.seed(i)
            X = np.random.normal(0, i, (n, 2))  # Varying variance
            Y = np.random.normal(0, 1, (n, 1))
            return X, Y
        scenarios[f"Independent GG {i}"] = make_gg
    
    # 2. Gaussian-NonGaussian (10 cases)
    for i in range(1, 11):
        def make_gn(n, i=i):
            np.random.seed(i)
            X = np.random.normal(0, 1, (n, 2))
            # Various non-Gaussian Y
            if i % 3 == 0:
                Y = np.random.uniform(-2, 2, (n, 1))
            elif i % 3 == 1:
                Y = np.random.laplace(0, 1, (n, 1))
            else:
                Y = t(df=3).rvs((n, 1))
            return X, Y
        scenarios[f"Independent GN {i}"] = make_gn
    
    # 3. NonGaussian-NonGaussian (10 cases)
    for i in range(1, 11):
        def make_nn(n, i=i):
            np.random.seed(i)
            # Various non-Gaussian X
            if i % 4 == 0:
                X = np.random.uniform(-3, 3, (n, 2))
            elif i % 4 == 1:
                X = gamma(2).rvs((n, 2))
            elif i % 4 == 2:
                X = cauchy().rvs((n, 2))
            else:
                X = expon().rvs((n, 2))
            
            # Various non-Gaussian Y
            if i % 3 == 0:
                Y = poisson(3).rvs((n, 1))
            elif i % 3 == 1:
                Y = bernoulli(0.5).rvs((n, 1))
            else:
                Y = gamma(3).rvs((n, 1))
            return X, Y
        scenarios[f"Independent NN {i}"] = make_nn
    
    # --------------------------
    # Subtle Dependent Cases (15 cases)
    # --------------------------
    
    # 1. Weak nonlinear relationships (5 cases)
    for i in range(1, 6):
        def make_weak(n, i=i):
            np.random.seed(i)
            X = np.random.normal(0, 1, (n, 2))
            noise = np.random.normal(0, 0.5, (n, 1))
            if i == 1:
                Y = 0.3 * X[:, [0]]**2 + noise
            elif i == 2:
                Y = 0.4 * np.sin(X[:, [0]]) + noise
            elif i == 3:
                Y = 0.2 * np.abs(X[:, [0]]) * X[:, [1]] + noise
            elif i == 4:
                Y = 0.1 * (X[:, [0]] > 0) * X[:, [1]] + noise
            else:
                Y = 0.25 * np.log(np.abs(X[:, [0]]) + 1) + noise
            return X, Y
        scenarios[f"Subtle Weak {i}"] = make_weak
    
    # 2. Conditional independence (5 cases)
    for i in range(1, 6):
        def make_cond(n, i=i):
            np.random.seed(i)
            Z = np.random.normal(0, 1, n)
            X = np.column_stack([
                Z + np.random.normal(0, 0.5, n),
                np.random.normal(0, 1, n)
            ])
            Y = np.column_stack([
                Z**2 + np.random.normal(0, 0.5, n)
            ])
            return X, Y
        scenarios[f"Subtle Cond {i}"] = make_cond
    
    # 3. Higher-order interactions (5 cases)
    for i in range(1, 6):
        def make_higher(n, i=i):
            np.random.seed(i)
            X = np.random.normal(0, 1, (n, 2))
            noise = np.random.normal(0, 0.3, (n, 1))
            if i == 1:
                Y = 0.2 * X[:, [0]] * X[:, [1]] + noise
            elif i == 2:
                Y = 0.15 * np.sin(X[:, [0]] + X[:, [1]]) + noise
            elif i == 3:
                Y = 0.1 * (X[:, [0]] > 0) * (X[:, [1]] > 0) + noise
            elif i == 4:
                Y = 0.25 * np.exp(-X[:, [0]]**2 - X[:, [1]]**2) + noise
            else:
                Y = 0.3 * np.sign(X[:, [0]]) * np.abs(X[:, [1]])**0.5 + noise
            return X, Y
        scenarios[f"Subtle Higher {i}"] = make_higher
    
    # --------------------------
    # Clearly Dependent Cases (10 cases)
    # --------------------------
    
    # 1. Strong nonlinear (5 cases)
    for i in range(1, 6):
        def make_strong(n, i=i):
            np.random.seed(i)
            X = np.random.uniform(-2, 2, (n, 2))
            noise = np.random.normal(0, 0.2, (n, 1))
            if i == 1:
                Y = X[:, [0]]**3 + noise
            elif i == 2:
                Y = np.sin(2 * np.pi * X[:, [0]]) + noise
            elif i == 3:
                Y = np.exp(X[:, [0]]) + noise
            elif i == 4:
                Y = 1 / (1 + np.exp(-X[:, [0]])) + noise
            else:
                Y = np.abs(X[:, [0]])**0.7 * np.sign(X[:, [0]]) + noise
            return X, Y
        scenarios[f"Dependent Strong {i}"] = make_strong
    
    # 2. Mixed distributions (5 cases)
    for i in range(1, 6):
        def make_mixed(n, i=i):
            np.random.seed(i)
            if i == 1:
                X = np.random.normal(0, 1, (n, 2))
                Y = (X[:, [0]] > 0).astype(float) + np.random.normal(0, 0.1, (n, 1))
            elif i == 2:
                X = np.random.poisson(3, (n, 2))
                Y = np.column_stack([np.log(X[:, 0] + 1)]) + np.random.normal(0, 0.2, (n, 1))
            elif i == 3:
                X = np.random.exponential(1, (n, 2))
                Y = np.column_stack([gamma(2).rvs(n) * (X[:, 0] > 1.5)])
            elif i == 4:
                X = np.random.uniform(0, 1, (n, 2))
                Y = np.column_stack([bernoulli(X[:, 0]).rvs(n)])
            else:
                X = np.random.laplace(0, 1, (n, 2))
                Y = np.column_stack([cauchy(loc=X[:, 0]/2).rvs(n)])
            return X, Y
        scenarios[f"Dependent Mixed {i}"] = make_mixed
    
    return scenarios

def get_scenario_description(name):
    """Helper function to provide descriptions for each scenario"""
    if "Independent GG" in name:
        return "Independent Gaussian-Gaussian variables"
    elif "Independent GN" in name:
        return "Independent Gaussian-NonGaussian variables"
    elif "Independent NN" in name:
        return "Independent NonGaussian-NonGaussian variables"
    elif "Subtle Weak" in name:
        return "Weak nonlinear dependence (hard to detect)"
    elif "Subtle Cond" in name:
        return "Conditional dependence through latent variable"
    elif "Subtle Higher" in name:
        return "Higher-order interaction dependence"
    elif "Dependent Strong" in name:
        return "Strong nonlinear dependence (easy to detect)"
    elif "Dependent Mixed" in name:
        return "Dependence between mixed variable types"
    return ""

# Explicitly declare exports
__all__ = ['generate_scenarios', 'get_scenario_description']