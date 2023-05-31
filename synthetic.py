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
            mu = np.random.normal(0.5, 0.2, self.K)
            mu = np.clip(mu, 0.2, 0.8)
            return mu
        elif self.dataset == "Segment":
            # Segment: μi = 0.5 for i = 2, · · · , K and θ1 = 0.6 for i = 1
            mu = np.ones(self.K)*0.5
            mu[0] = 0.6
            return mu
        else:
            raise NotImplementedError


class SyntheticSimpleDataset:
    def __init__(self, N, distribution) -> None:
        self.N = N
        self.distribution = distribution
    
    def generate_dataset(self):
        mu = np.zeros(self.N)
        if self.distribution == "Bernoulli":
            mu[:10] = np.ones(10)*0.8
            mu[0] = 0.9
        elif self.distribution == "Gaussian" or self.distribution == "Poisson":
            mu[:10] = np.ones(10)*0.7
            mu[0] = 1.0
        elif self.distribution == "Gamma":
            mu[:10] = np.ones(10)*0.8
            mu[0] = 1.0
        else:
            raise NotImplementedError
        
        if self.N >10:
            if self.distribution == "Bernoulli":
                mu[10:] = np.random.uniform(0.5,0.7,40)
            else:
                mu[10:] = np.random.uniform(0.3,0.5,40)
        return mu


class RebuttalDataset:
    def __init__(self, N, distribution) -> None:
        self.distribution = distribution
        self.N = N
    
    def generate_dataset(self):
        if self.N == 10:
            mu = np.ones(10)
            if self.distribution == "Bernoulli" or self.distribution == "Gaussian":
                mu = mu*0.2
                mu[0] = 0.3
            else:
                mu = mu*0.2
                mu[0] = 0.3
                # raise NotImplementedError
            return mu
        elif self.N == 500:
            if self.distribution == "Gaussian":
                mu = np.zeros(self.N)
                mu[0] = 2
            elif self.distribution == "Bernoulli":
                mu = np.ones(self.N)
                mu = mu*0.25
                mu[0] = 0.75
            else:
                raise NotImplementedError
            return mu
        else:
            raise NotImplementedError



            
        