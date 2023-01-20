import numpy as np

# Define hyperparameters
Ns = [10, 50]
DISTRIBUTIONS = ["Gaussian", "Bernoulli", "Poisson", "Gamma"]
BASELINES = ["TS", "TSGreedy","KL_UCB",  "KL_UCB_plus_plus", "MOTS", "ExpTS", "ExpTS_plus"]
PROBS_fn = lambda n: [1/n, 1/np.sqrt(n)]
T = pow(10, 4)
PERIOD = 100

REPEAT_T = {
    "TS": 1000,
    "TSGreedy": 1000,
    "MOTS": 1000,
    "KL_UCB": 200,
    "KL_UCB_plus_plus": 200,
    "ExpTS": 50,
    "ExpTS_plus": 50
}
