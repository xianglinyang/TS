import numpy as np
import os
from config import *
from ts import *
from synthetic import SyntheticDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=str, choices=BASELINES)
parser.add_argument('-k', type=int, choices=Ns)
parser.add_argument('--dataset', type=str, choices=DATASETS)
parser.add_argument('--distribution', type=str, choices=DISTRIBUTIONS)
parser.add_argument("--repeat", type=int)
parser.add_argument("--period", type=int)
parser.add_argument("--prob", type=float, default=1.0)
args = parser.parse_args()

# get hyperparameters
BASELINE = args.baseline
N = args.k
DATASET = args.dataset
DISTRIBUTION = args.distribution
REPEAT = args.repeat
PERIOD = args.period
PROB = args.prob
print("Load hyperparameters...")

regret_plots = np.zeros(int(T/PERIOD))
for i in range(REPEAT):
    sd = SyntheticDataset(N, DATASET)
    mu_gt = sd.generate_dataset()
    print("Generate {} dataset {}...".format(DATASET, mu_gt))
    if BASELINE == "TS":
        ts = TS(N, mu_gt, DISTRIBUTION)
    elif BASELINE == "KL_UCB_plus_plus":
        ts = KL_UCB_plus_plus(N, mu_gt, DISTRIBUTION)
    elif BASELINE == "KL_UCB":
        ts = KL_UCB(N, mu_gt, DISTRIBUTION)
    elif BASELINE == "MOTS":
        # two distributions
        ts = MOTS(N, mu_gt, DISTRIBUTION)
    elif BASELINE == "ExpTS":
        ts = ExpTS(N, mu_gt, DISTRIBUTION)
    elif BASELINE == "TSGreedy":
        ts = TS_Greedy(N, mu_gt, DISTRIBUTION, PROB)
    elif BASELINE == "ExpTS_plus":
        ts = ExpTS_plus(N, mu_gt, DISTRIBUTION, PROB)
    else:
        raise NotImplementedError
    _,regret_line = ts.regret(T, PERIOD)
    regret_plots = regret_plots+regret_line
regret_plots = regret_plots / REPEAT

os.makedirs("./results", exist_ok=True)
np.save("./results/{}-{}-{}-{}-{}".format(BASELINE, N, DATASET, DISTRIBUTION, round(PROB, 2)), regret_plots)

