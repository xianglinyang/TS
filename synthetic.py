import numpy as np

class SyntheticDataset:
    def __init__(self, K, dataset):
        self.K = K
        self.dataset = dataset
        
    def generate_dataset(self):
        if self.dataset == "Uniform":
            # Uniform: μi ∼ Unif [0.2, 0.8], i.e., the mean of arms, μi, are uniformly distributed in [0.2, 0.8]
            return np.random.uniform(0.2, 0.8, self.K)
        elif self.dataset == "TruncatedGaussian":
            # Truncated Gaussian: the mean of arms are generated from N (0, 5, 0.2) and the support [0.2, 0.8].
            mu = np.random.normal(0, 5)
            mu = np.clip(mu, 0.2, 0.8)
            return mu
        elif self.dataset == "Segment":
            # Segment: μi = 0.5 for i = 2, · · · , K and θ1 = 0.6 for i = 1
            mu = np.ones(self.K)*0.5
            mu[0] = 0.6
        else:
            raise NotImplementedError
