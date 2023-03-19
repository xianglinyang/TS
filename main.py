import numpy as np
import time
from config import BASELINES, Ns, DISTRIBUTIONS
from ts import *
from synthetic import SyntheticSimpleDataset
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, choices=BASELINES)
    parser.add_argument('-k', type=int, choices=Ns)
    parser.add_argument('--distribution', type=str, choices=DISTRIBUTIONS)
    parser.add_argument("--period", type=int)
    parser.add_argument("--prob", type=float, default=1.0)
    parser.add_argument("--t", type=int)
    args = parser.parse_args()
    # get hyperparameters
    BASELINE = args.baseline
    N = args.k
    DISTRIBUTION = args.distribution
    PERIOD = args.period
    PROB = args.prob
    T = args.t
    print("Load hyperparameters...")

    # sd = SyntheticSimpleDataset(N, DISTRIBUTION)
    # mu_gt = sd.generate_dataset()
    # file_name = "./results/{}-{}.npy".format(N, DISTRIBUTION)
    # np.save(file_name, mu_gt)

    file_name = "./results/{}-{}.npy".format(N, DISTRIBUTION)
    mu_gt = np.load(file_name)
    if BASELINE == "TS":
        ts = TS(N, mu_gt, DISTRIBUTION)
    elif BASELINE == "KL_UCB_plus_plus":
        ts = KL_UCB_plus_plus(N, mu_gt, DISTRIBUTION)
    elif BASELINE == "KL_UCB":
        ts = KL_UCB(N, mu_gt, DISTRIBUTION)
    elif BASELINE == "MOTS":
        ts = MOTS(N, mu_gt, DISTRIBUTION)
    elif BASELINE == "ExpTS":
        ts = ExpTS(N, mu_gt, DISTRIBUTION)
    elif BASELINE == "TSGreedy":
        ts = TS_Greedy(N, mu_gt, DISTRIBUTION, PROB)
    elif BASELINE == "ExpTS_plus":
        ts = ExpTS_plus(N, mu_gt, DISTRIBUTION, PROB)
    else:
        raise NotImplementedError
    t_s = time.time()
    _, regret_line = ts.regret(T, PERIOD)
    t_e = time.time()
    print("{} {} takes {} seconds...".format(N, DISTRIBUTION, round(t_e-t_s, 3)))
    return regret_line, round(t_e-t_s, 3)


def run(mu_gt, BASELINE, N, DISTRIBUTION, PERIOD, PROB, T, REPEAT_TIME, FOLDER="results"):
    np.random.seed(REPEAT_TIME)
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
    _, regret_line = ts.regret(T, PERIOD)

    file_name = "./{}/{}-{}-{}-{}-{}-{}.npy".format(FOLDER, BASELINE, N, DISTRIBUTION, T, round(PROB, 2), REPEAT_TIME)
    np.save(file_name, regret_line)


if __name__ == "__main__":
    main()
