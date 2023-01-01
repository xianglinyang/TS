import numpy as np
import time
import sympy as sy
import matplotlib.pyplot as plt
from math import e


def find_mu(distribution, mu, upper, T):
    x = sy.symbols("x")
    if distribution == "Gaussian":
        results = sy.solve(T/2*(mu-x)**2-upper,x)
    elif distribution == "Berboulli":
        results = sy.solve(T*(mu*sy.log(mu/x)+(1-mu)*sy.log((1-mu)/(1-x)))-upper, x)
    elif distribution == "Poisson":
        results = sy.solve(T*(mu*sy.log(mu/x)-mu+x)-upper, x)
    elif distribution == "Exponential":
        results = sy.solve(T*(sy.log(mu/x)-(mu-x)/mu)-upper, x)
    else:
        raise NotImplementedError
    return max(results)

class TS:
    def __init__(self, N, mu_ground_truth, distribution, init_mu=0):
        self.N = N
        self.distribution = distribution
        self.mu_ground_truth = mu_ground_truth
        self.mu = np.ones(self.N)*init_mu
        self.T = np.zeros(self.N)

    def _pull_posterior(self, i):
        if self.distribution == "Gaussian":
            return np.random.normal(self.mu[i], np.sqrt(1/self.T[i]), 1)[0]
        elif self.distribution == "Berboulli":
            pass
        elif self.distribution == "Poisson":
            pass
        elif self.distribution == "Exponential":
            pass
        else:
            raise NotImplementedError
    
    def _initialization(self):
        for i in range(self.N):
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)

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
            return np.random.normal(self.mu_ground_truth[i], 1, 1)[0]
        elif self.distribution == "Berboulli":
            return np.random.binomial(1, self.mu_ground_truth[i], 1)[0]
        elif self.distribution == "Poisson":
            return np.random.poisson(self.mu_ground_truth[i], 1)[0]
        elif self.distribution == "Exponential":
            return np.random.exponential(self.mu_ground_truth[i], 1)[0]
        else:
            raise NotImplementedError
    
    def regret(self, T, period):
        mu_max = np.max(self.mu_ground_truth)
        regrets_plot = np.zeros(int(T/period))
        regret = 0
        for t in range(self.N+1, T+1, 1):
            i = self._choose_arm()
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)
            
            regret = regret + mu_max - self.mu_ground_truth[i]
            
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
        U_upper = np.maximum(np.log((np.power(np.maximum(np.log(T/(self.N*self.T)),0),2)+1)*T/(self.N*self.T)),0)
        curr_u = np.zeros(self.N)
        for i in range(self.N):
            # find mu kl(self.mu[i], mu)<=U_upper[i]/self.T[i]
            curr_u[i] = find_mu(self.distribution, self.mu[i], U_upper, self.T[i])
        return np.argmax(curr_u)
    
    def regret(self, T, period):
        mu_max = np.max(self.mu_ground_truth)
        regrets_plot = np.zeros(int(T/period))
        regret = 0
        for t in range(self.N+1, T+1, 1):
            i = self._choose_arm(T)
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)
            
            regret = regret + mu_max - self.mu_ground_truth[i]
            
            if t % period == 0:
                regrets_plot[t//period] = regret
        return regret, regrets_plot

class KL_UCB(TS):

    def _choose_arm(self, t):
        U_upper = np.log(t)+3*np.log(np.log(t))
        curr_u = np.zeros(self.N)
        for i in range(self.N):
            # find mu kl(self.mu[i], mu)<=U_upper[i]/self.T[i]
            curr_u[i] = find_mu(self.distribution, self.mu[i], U_upper, self.T[i])
        return np.argmax(curr_u)
    
    def regret(self, T, period):
        mu_max = np.max(self.mu_ground_truth)
        regrets_plot = np.zeros(int(T/period))
        regret = 0
        for t in range(self.N+1, T+1, 1):
            i = self._choose_arm(t)
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)
            
            regret = regret + mu_max - self.mu_ground_truth[i]
            
            if t % period == 0:
                regrets_plot[t//period] = regret
        return regret, regrets_plot

class MOTS(TS):
    def _choose_arm(self, T):
        if self.distribution == "Gaussian":
            sigma = 1
        elif self.distribution == "Berboulli":
            sigma = 0.25
        else:
            raise NotImplementedError

        thetas = np.zeros(self.N)
        for i in range(self.N):
            thetas[i] = np.random.normal(self.mu[i], sigma/0.9/self.T[i], 1)[0]

        mus = self.mu+2/self.T*np.maximum(np.log(T/self.N*self.T),0)

        curr_thetas = np.minimum(thetas, mus)
        return np.argmax(curr_thetas)
    
    def regret(self, T, period):
        mu_max = np.max(self.mu_ground_truth)
        regrets_plot = np.zeros(int(T/period))
        regret = 0
        for t in range(self.N+1, T+1, 1):
            i = self._choose_arm(T)
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)
            
            regret = regret + mu_max - self.mu_ground_truth[i]
            
            if t % period == 0:
                regrets_plot[t//period] = regret
        return regret, regrets_plot


        
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
