from multiprocessing import Pool
from config import *
from main import run
import time

t_s = time.time()
c = 0
p = Pool(20)
for r in range(REPEAT):
    for BASELINE in BASELINES:
        for N in Ns:
            for DATASET in DATASETS:
                for DISTRIBUTION in DISTRIBUTIONS:
                    if BASELINE == "MOTS":
                        if DISTRIBUTION in ["Gamma", "Poisson"]:
                            continue
                    if BASELINE in ["TSGreedy", "ExpTS_plus"]:
                        for PROB in PROBS_fn(N):
                            p.apply_async(run, args=(BASELINE, N, DATASET, DISTRIBUTION, PERIOD, PROB, r))
                    else:
                        p.apply_async(run, args=(BASELINE, N, DATASET, DISTRIBUTION, PERIOD, 1.0, r))
p.close()
p.join()
t_e = time.time()
print("{} seconds to repeat {} times...".format(round(t_e-t_s, 2), REPEAT))