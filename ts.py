import numpy as np
import time
import matplotlib.pyplot as plt
from math import e


def find_mu(distribution, mu, upper, T):
    if distribution == "Gaussian":
        # assume sigma==1
        return np.sqrt(2*upper/T)+mu
    elif distribution == "Berboulli":
        pass
    elif distribution == "Poisson":
        pass
    elif distribution == "Exponential":
        pass
    else:
        raise NotImplementedError


class TS:
    def __init__(self, N, mu_ground_truth, distribution, init_mu=0):
        self.N = N
        self.distribution = distribution
        self.mu_ground_truth = mu_ground_truth
        self.mu = np.ones(self.N)*init_mu
        self.T = np.zeros(self.N)
    
    def _init_prior(self):
        if self.distribution == "Gaussian":
            pass
        elif self.distribution == "Berboulli":
            pass
        elif self.distribution == "Poisson":
            pass
        elif self.distribution == "Exponential":
            pass
        else:
            raise NotImplementedError

    def _pull_posterior(self, i):
        if self.distribution == "Gaussian":
            return np.random.normal(self.mu[i], np.sqrt(self.sigma[i]), 1)[0]
        elif self.distribution == "Berboulli":
            pass
        elif self.distribution == "Poisson":
            pass
        elif self.distribution == "Exponential":
            pass
        else:
            raise NotImplementedError

    def _update_posterior(self, i, reward):
        # reward mean
        self.mu[i] = (self.mu[i]*self.T[i]+reward)/(self.T[i]+1)
        self.T[i] = self.T[i]+1
    
    @property
    def _mu_theta(self):
        if self.distribution == "Gaussian":
            return self.mu
        elif self.distribution == "Berboulli":
            return self.mu
        elif self.distribution == "Poisson":
            return self.mu
        elif self.distribution == "Exponential":
            return 1 / self.mu
        else:
            raise NotImplementedError

    def _choose_arm(self):
        curr_thetas = np.zeros(self.N)
        for i in range(self.N):
            curr_thetas[i] = self._pull_posterior(i)
        return np.argmax(curr_thetas)
    
    def _pull_arm(self, i):
        if self.distribution == "Gaussian":
            return np.random.normal(self.mu_gt[i], 1, 1)[0]
        elif self.distribution == "Berboulli":
            pass
        elif self.distribution == "Poisson":
            pass
        elif self.distribution == "Exponential":
            pass
        else:
            raise NotImplementedError
    
    def regret(self, T, period):
        mu_max = np.max(self.mu_gt)
        regrets_plot = np.zeros(int(T/period))
        regret = 0
        for t in range(1, T+1, 1):
            i = self._choose_arm()
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)
            
            regret = regret + mu_max - self.mu_gt[i]
            
            if t % period == 0:
                regrets_plot[t//period] = regret
        return regret, regrets_plot

class TS_Greedy(TS):
    '''with prob probability'''
    def __init__(self, N, mu_ground_truth, distribution, prob, init_mu=0):
        super().__init__(N, mu_ground_truth, distribution, init_mu)
        self.prob = prob

    def _choose_arm(self):
        curr_thetas = np.zeros(self.N)
        for i in range(self.N):
            tmp = np.random.random_sample()
            if tmp<self.prob:
                curr_thetas[i] = self._pull_posterior(i)
            else:
                curr_thetas[i] = self._mu_theta[i]
        return np.argmax(curr_thetas)

class KL_UCB_plus_plus(TS):

    def _choose_arm(self, T):
        U_upper = np.max(np.log((np.power(np.max(np.log(T/(self.N*self.T)),0),2)+1)*T/(self.N*self.T)),0)
        curr_u = np.zeros(self.N)
        for i in range(self.N):
            # find mu kl(self.mu[i], mu)<=U_upper[i]/self.T[i]
            curr_u[i] = find_mu(self.distribution, self.mu[i], U_upper, self.T[i])
        return np.argmax(curr_u)
    

        
if __name__ == "__main__":
    N = 100
    T = 100000
    repeat = 100
    prob = 1/N
    eps = 1
    period = 1000

    record_time = np.zeros(int(T/period))
    for t in range(T):
        if t % period == 0:
            record_time[t//period] = np.log(t+1)

    mu_gt = np.ones(N) - eps
    mu_gt[0] = 1

    t0 = time.time()

    fig = plt.figure()
    plt.title("TS variants")
    plt.xlabel("T")
    plt.ylabel("Regret")
    
    # ----------TS---------
    mean_regret = np.zeros(repeat)
    regret = np.zeros(int(T/period))
    for i in range(repeat):
        ts = TS(N, mu_gt)
        mean_regret[i], regret_line = ts.regret(T, period)
        regret = regret + regret_line
    regret = regret / repeat
    print("TS mean regret is {:.3f}".format(mean_regret.mean()))
    
    line1 = plt.plot(record_time, regret, 'go--', linewidth=1, markersize=4, label="TS")

    # ----------TS Greedy---------
    mean_regret = np.zeros(repeat)
    regret = np.zeros(int(T/period))
    for i in range(repeat):
        ts = TS_Greedy(N, mu_gt)
        mean_regret[i],regret_line = ts.regret(T, prob, period)
        regret = regret_line + regret
    regret = regret / repeat
    print("TS Greedy mean regret is {:.3f}".format(mean_regret.mean()))

    plt.legend()
    plt.savefig("TS+.png")
    t1 = time.time()

    print("Time spent {:.2f}".format(t1-t0))
