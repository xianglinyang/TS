from multiprocessing import Pool
from config import *
from main import run
from synthetic import SyntheticSimpleDataset
import time
import os

os.makedirs("./results", exist_ok=True)
t_s = time.time()
c = 0
p = Pool()

for N in Ns:
    for DISTRIBUTION in DISTRIBUTIONS:
        # sd = SyntheticSimpleDataset(N, DISTRIBUTION)
        # mu_gt = sd.generate_dataset()
        # file_name = "./results/{}-{}.npy".format(N, DISTRIBUTION)
        # np.save(file_name, mu_gt)
        file_name = "./results/{}-{}.npy".format(N, DISTRIBUTION)
        mu_gt = np.load(file_name)

        for BASELINE in BASELINES:
            for r in range(REPEAT_T[BASELINE][0], REPEAT_T[BASELINE][1]):

                if BASELINE == "MOTS":
                    if DISTRIBUTION in ["Gamma", "Poisson"]:
                        continue
                if BASELINE in ["TSGreedy", "ExpTS_plus"]:
                    for PROB in PROBS_fn(N):
                        p.apply_async(run, args=(mu_gt, BASELINE, N, DISTRIBUTION, PERIOD, PROB, r))
                else:
                    p.apply_async(run, args=(mu_gt, BASELINE, N, DISTRIBUTION, PERIOD, 1.0, r))
p.close()
p.join()
t_e = time.time()
print("{} seconds...".format(round(t_e-t_s, 2)))