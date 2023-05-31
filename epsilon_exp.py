from multiprocessing import Pool
from main import run
from synthetic import RebuttalDataset
import time
import os

DISTRIBUTIONS = ["Gaussian", "Bernoulli", "Poisson", "Gamma"]
BASELINES = ["TSGreedy"]
PROBS = [1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]

# New Config
T = pow(10, 3)
PERIOD = 50
N = 10
FOLDER = "epsilon_results"
REPEAT = 936

def run_one_round(BASELINE, DISTRIBUTION):
    # EXP
    os.makedirs(f'./{FOLDER}', exist_ok=True)
    t_s = time.time()
    p = Pool()

    for r in range(REPEAT):
        sd = RebuttalDataset(N, DISTRIBUTION)
        mu_gt = sd.generate_dataset()

        for PROB in PROBS:
            p.apply_async(run, args=(mu_gt, BASELINE, N, DISTRIBUTION, PERIOD, PROB, T, r, FOLDER))
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
