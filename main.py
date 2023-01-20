import numpy as np
import os
from config import *
from ts import *
from synthetic import SyntheticDataset
import argparse

def main():
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
    for _ in range(REPEAT):
        run(BASELINE, N, DATASET, DISTRIBUTION, PERIOD, PROB)


def run(BASELINE, N, DATASET, DISTRIBUTION, PERIOD, PROB, REPEAT_TIME):

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
    file_name = "./results/{}-{}-{}-{}-{}-{}.npy".format(BASELINE, N, DATASET, DISTRIBUTION, round(PROB, 2), REPEAT_TIME)
    np.save(file_name, regret_line)


