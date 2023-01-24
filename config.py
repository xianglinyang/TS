import numpy as np

# Define hyperparameters
Ns = [10, 50]
DISTRIBUTIONS = ["Gamma", "Gaussian", "Bernoulli", "Poisson"]
# BASELINES = ["TS", "TSGreedy","KL_UCB",  "KL_UCB_plus_plus", "ExpTS", "ExpTS_plus", "MOTS"]
BASELINES = ["ExpTS_plus"]
PROBS_fn = lambda n: [1/n]
T = pow(10, 4)
PERIOD = 100

REPEAT_T = {
    # [, )
    "TS": [0, 4000],
    "TSGreedy": [0, 4000],
    "MOTS": [0, 4000],
    "KL_UCB": [0, 1000],
    "KL_UCB_plus_plus": [0, 1000],
    "ExpTS": [50,200],
    "ExpTS_plus": [50,1000]
}

NAME_IN_PLOT = {
    "TS": "TS",
    "TSGreedy": "(1/K)-TS",
    "MOTS": "MOTS",
    "KL_UCB": "KL-UCB",
    "KL_UCB_plus_plus": "KL-UCB++",
    "ExpTS": "ExpTS",
    "ExpTS_plus": "ExpTS+"
}
