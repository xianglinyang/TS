import numpy as np
# Define hyperparameters
Ns = [10,20,50]
DATASETS = ["Uniform", "TruncatedGaussian", "Segment"]
DISTRIBUTIONS = ["Gaussian", "Bernoulli", "Poisson", "Gamma"]
BASELINES = ["TS", "TSGreedy","KL_UCB",  "KL_UCB_plus_plus", "MOTS", "ExpTS", "ExpTS_plus"]
PROBS = [[1/N, 1/np.sqrt(N), 1/np.log(N)] for N in Ns]
T = pow(10, 6)
REPEAT = 1000
PERIOD = 500
