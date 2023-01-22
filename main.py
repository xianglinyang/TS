import numpy as np
from config import *
from ts import *
from synthetic import SyntheticSimpleDataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, choices=BASELINES)
    parser.add_argument('-k', type=int, choices=Ns)
    parser.add_argument('--distribution', type=str, choices=DISTRIBUTIONS)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--period", type=int)
    parser.add_argument("--prob", type=float, default=1.0)
    args = parser.parse_args()
    # get hyperparameters
    BASELINE = args.baseline
    N = args.k
    DISTRIBUTION = args.distribution
    PERIOD = args.period
    PROB = args.prob
    print("Load hyperparameters...")

    sd = SyntheticSimpleDataset(N, DISTRIBUTION)
    mu_gt = sd.generate_dataset()
    file_name = "./results/{}-{}.npy".format(N, DISTRIBUTION)
    np.save(file_name, mu_gt)

    for r in range(REPEAT_T[BASELINE]):
        run(mu_gt, BASELINE, N, DISTRIBUTION, PERIOD, PROB, r)


def run(mu_gt, BASELINE, N, DISTRIBUTION, PERIOD, PROB, REPEAT_TIME):
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

    file_name = "./results/{}-{}-{}-{}-{}.npy".format(BASELINE, N, DISTRIBUTION, round(PROB, 2), REPEAT_TIME)
    np.save(file_name, regret_line)
    # print(BASELINE, N, DISTRIBUTION, PERIOD, PROB, round(t_e-t_s, 2))


if __name__ == "__main__":
    N = 10
    DISTRIBUTION = "Bernoulli"
    BASELINE = "ExpTS"

    sd = SyntheticSimpleDataset(N, DISTRIBUTION)
    mu_gt = sd.generate_dataset()
    file_name = "./results/{}-{}.npy".format(N, DISTRIBUTION)
    np.save(file_name, mu_gt)

    if BASELINE == "MOTS":
        if DISTRIBUTION in ["Gamma", "Poisson"]:
            pass
    if BASELINE in ["TSGreedy", "ExpTS_plus"]:
        for PROB in PROBS_fn(N):
            run(mu_gt, BASELINE, N, DISTRIBUTION, PERIOD, PROB, 0)
    else:
        run(mu_gt, BASELINE, N, DISTRIBUTION, PERIOD, 1.0, 0)
