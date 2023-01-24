import os
import argparse
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=str, choices=BASELINES)
args = parser.parse_args()
BASELINE = args.baseline

for N in Ns:
    for DISTRIBUTION in DISTRIBUTIONS:
        if BASELINE == "MOTS":
            if DISTRIBUTION in ["Gamma", "Poisson"]:
                continue
        if BASELINE in ["TSGreedy", "ExpTS_plus"]:
            for PROB in PROBS_fn(N):
                os.system("python main.py --baseline {} -k {} --distribution {} --period {} --prob {}".format(BASELINE, N, DISTRIBUTION, PERIOD, PROB))
        else:
            os.system("python main.py --baseline {} -k {} --distribution {} --period {}".format(BASELINE, N, DISTRIBUTION, PERIOD))