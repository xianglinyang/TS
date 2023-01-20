import numpy as np
# # Define hyperparameters
# Ns = [10,20,50]
# DATASETS = ["Uniform", "TruncatedGaussian", "Segment"]
# DISTRIBUTIONS = ["Gaussian", "Bernoulli", "Poisson", "Gamma"]
# BASELINES = ["TS", "TSGreedy","KL_UCB",  "KL_UCB_plus_plus", "MOTS", "ExpTS", "ExpTS_plus"]

# # [[1/N, 1/np.sqrt(N), 1/np.log(N)] for N in Ns]
# PROBS_fn = lambda n: [1/n, 1/np.sqrt(n), 1/np.log(n)]


# T = pow(10, 5)
# REPEAT = 3
# PERIOD = 50

# Define hyperparameters
Ns = [10,20,50]
DATASETS = ["Uniform", "TruncatedGaussian", "Segment"]
DISTRIBUTIONS = ["Gaussian", "Bernoulli", "Poisson", "Gamma"]
BASELINES = ["TS", "TSGreedy", "MOTS"]

# [[1/N, 1/np.sqrt(N), 1/np.log(N)] for N in Ns]
PROBS_fn = lambda n: [1/n, 1/np.sqrt(n), 1/np.log(n)]

T = pow(10, 5)
REPEAT = 50
PERIOD = 500
