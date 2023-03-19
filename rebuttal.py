from multiprocessing import Pool
from main import run
from synthetic import RebuttalDataset
import time
import os

DISTRIBUTIONS = ["Gaussian", "Bernoulli"]
# DISTRIBUTIONS = ["Bernoulli"]
# BASELINES = ["KL_UCB",  "KL_UCB_plus_plus","TS", "MOTS", "ExpTS_plus", "TSGreedy"]
BASELINES = ["ExpTS_plus"]

# New Config
PROBS_fn = lambda n: [1/n]
T = 2*pow(10, 3)
PERIOD = 200
N = 500
FOLDER = "rebuttal_results"
REPEAT_T = {
    "TS": [0,936],
    "TSGreedy": [0,936],
    "MOTS": [0,936],
    "KL_UCB": [0,936],
    "KL_UCB_plus_plus": [0,936],
    "ExpTS": [0,936],
    "ExpTS_plus": [0,936]
}

NAME_IN_PLOT = {
    "TS": "TS",
    "TSGreedy": "ε-TS",
    "MOTS": "MOTS",
    "KL_UCB": "KL-UCB",
    "KL_UCB_plus_plus": "KL-UCB++",
    "ExpTS": "ExpTS",
    "ExpTS_plus": "ExpTS+"
}


def run_one_round(BASELINE, DISTRIBUTION):
    # EXP
    os.makedirs(f'./{FOLDER}', exist_ok=True)

    t_s = time.time()
    p = Pool()

    for r in range(REPEAT_T[BASELINE][0], REPEAT_T[BASELINE][1]):
        sd = RebuttalDataset(N, DISTRIBUTION)
        mu_gt = sd.generate_dataset()

        if BASELINE in ["TSGreedy", "ExpTS_plus"]:
            for PROB in PROBS_fn(N):
                p.apply_async(run, args=(mu_gt, BASELINE, N, DISTRIBUTION, PERIOD, PROB, T, r, FOLDER))
        else:
            p.apply_async(run, args=(mu_gt, BASELINE, N, DISTRIBUTION, PERIOD, 1.0, T, r, FOLDER))
    p.close()
    p.join()
    t_e = time.time()


    time_record = os.path.join(f'{FOLDER}', f'{N}_time.txt')
    with open(time_record, 'a') as file:
        file.write(f'{DISTRIBUTION}\t{BASELINE}\t{round(t_e-t_s, 2)}\n')

    print(f'{DISTRIBUTION}\t{BASELINE}\t{round(t_e-t_s, 2)}\n')

if __name__ == "__main__":
    for DISTRIBUTION in DISTRIBUTIONS:
        for BASELINE in BASELINES:
            run_one_round(BASELINE, DISTRIBUTION)
