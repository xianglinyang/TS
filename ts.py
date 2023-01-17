import numpy as np
import time
import sympy as sy
import matplotlib.pyplot as plt


def kl_div(distribution, mu, x):
    '''the symbolic expression of kl divergence'''
    if distribution == "Gaussian":
        kl = 1/2*(mu-x)**2
    elif distribution == "Bernoulli":
        kl = mu*sy.log(mu/x)+(1-mu)*sy.log((1-mu)/(1-x))
    elif distribution == "Poisson":
        kl = mu*sy.log(mu/x)-mu+x
    elif distribution == "Gamma":
        kl = sy.log(mu/x)-(mu-x)/mu
    else:
        raise NotImplementedError
    return kl


def find_mu(x, kl, upper, T):
    results = sy.solve(T*kl-upper, x)
    return np.max(np.array(results))


class TS:
    def __init__(self, N, mu_ground_truth, distribution, init_mu=0):
        self.N = N
        self.distribution = distribution
        self.mu_ground_truth = mu_ground_truth
        self.mu = np.ones(self.N)*init_mu
        self.T = np.zeros(self.N)
        self._initialization()

    def _pull_posterior(self, i):
        if self.distribution == "Gaussian":
            return np.random.normal(self.mu[i], np.sqrt(1/self.T[i]), 1)[0]
        elif self.distribution == "Bernoulli":
            return np.random.beta(1+self.T[i]*self.mu[i], 1+self.T[i]*(1-self.mu[i]), size=1)[0]
        elif self.distribution == "Poisson":
            return np.random.gamma(1+self.mu[i]*self.T[i],self.T[i],size=1)[0]
        elif self.distribution == "Gamma":
            return np.random.gamma(self.T[i]-1, 1/self.mu[i]*self.T[i],size=1)[0]
        else:
            raise NotImplementedError
    
    def _initialization(self):
        for i in range(self.N):
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)
        if self.distribution == "Gamma":
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
        elif self.distribution == "Bernoulli":
            return self.mu
        elif self.distribution == "Poisson":
            return self.mu
        elif self.distribution == "Gamma":
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
        elif self.distribution == "Bernoulli":
            return np.random.binomial(1, self.mu_ground_truth[i], 1)[0]
        elif self.distribution == "Poisson":
            return np.random.poisson(self.mu_ground_truth[i], 1)[0]
        elif self.distribution == "Gamma":
            return np.random.exponential(self.mu_ground_truth[i], 1)[0]
        else:
            raise NotImplementedError
    
    def regret(self, T, period):
        mu_max = np.max(self.mu_ground_truth)
        regrets_plot = np.zeros(int(T/period))
        regret = 0
        
        start = 2*self.N + 1 if self.distribution == "Gamma" else self.N + 1
        for t in range(start, T+1, 1):
            i = self._choose_arm()
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)
            
            regret = regret + mu_max - self.mu_ground_truth[i]
            
            if t % period == 0:
                regrets_plot[int(t/period)-1] = regret
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
            x = sy.symbols("x")
            kl = kl_div(self.distribution, self.mu[i], x)
            curr_u[i] = find_mu(x, kl, upper=U_upper, T=self.T[i])
        return np.argmax(curr_u)
    
    def regret(self, T, period):
        mu_max = np.max(self.mu_ground_truth)
        regrets_plot = np.zeros(int(T/period))
        regret = 0
        start = 2*self.N + 1 if self.distribution == "Gamma" else self.N + 1
        for t in range(start, T+1, 1):
            i = self._choose_arm(T)
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)
            
            regret = regret + mu_max - self.mu_ground_truth[i]
            
            if t % period == 0:
                regrets_plot[int(t/period)-1] = regret
        return regret, regrets_plot

class KL_UCB(TS):

    def _choose_arm(self, t):
        U_upper = np.log(t)+3*np.log(np.log(t))
        curr_u = np.zeros(self.N)
        for i in range(self.N):
            # find mu kl(self.mu[i], mu)<=U_upper[i]/self.T[i]
            x = sy.symbols("x")
            kl = kl_div(self.distribution, self.mu[i], x)
            curr_u[i] = find_mu(x, kl, upper=U_upper, T=self.T[i])
        return np.argmax(curr_u)
    
    def regret(self, T, period):
        mu_max = np.max(self.mu_ground_truth)
        regrets_plot = np.zeros(int(T/period))
        regret = 0
        start = 2*self.N + 1 if self.distribution == "Gamma" else self.N + 1
        for t in range(start, T+1, 1):
            i = self._choose_arm(t)
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)
            
            regret = regret + mu_max - self.mu_ground_truth[i]
            
            if t % period == 0:
                regrets_plot[int(t/period)-1] = regret
        return regret, regrets_plot

class MOTS(TS):
    def __init__(self, N, mu_ground_truth, distribution, init_mu=0, rho=0.9):
        super().__init__(N, mu_ground_truth, distribution, init_mu)
        self.rho = rho

    def _choose_arm(self, T):
        if self.distribution == "Gaussian":
            sigma = 1
        elif self.distribution == "Bernoulli":
            sigma = 0.25
        else:
            raise NotImplementedError

        thetas = np.zeros(self.N)
        for i in range(self.N):
            thetas[i] = np.random.normal(self.mu[i], sigma/self.rho/self.T[i], 1)[0]

        mus = self.mu+2/self.T*np.maximum(np.log(T/self.N*self.T),0)

        curr_thetas = np.minimum(thetas, mus)
        return np.argmax(curr_thetas)
    
    def regret(self, T, period):
        mu_max = np.max(self.mu_ground_truth)
        regrets_plot = np.zeros(int(T/period))
        regret = 0
        start = 2*self.N + 1 if self.distribution == "Gamma" else self.N + 1
        for t in range(start, T+1, 1):
            i = self._choose_arm(T)
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)
            
            regret = regret + mu_max - self.mu_ground_truth[i]
            
            if t % period == 0:
                regrets_plot[int(t/period)-1] = regret
        return regret, regrets_plot


class ExpTS(TS):
    def _choose_arm(self):
        thetas = np.zeros(self.N)
        for i in range(self.N):
            x = sy.symbols("x")
            kl = kl_div(self.distribution, self.mu[i], x)
            
            y = np.random.random_sample()
            if y>= .5:
                thetas[i] = max(sy.solve(1-0.5*sy.exp(-(self.T[i]-1)*kl)-y, x))
            else:
                thetas[i] = min(sy.solve(0.5*sy.exp(-(self.T[i]-1)*kl)-y, x))
        return np.argmax(thetas)

class ExpTS_plus(TS):
    '''with prob probability'''
    def __init__(self, N, mu_ground_truth, distribution, prob, init_mu=0):
        super().__init__(N, mu_ground_truth, distribution, init_mu)
        self.prob = prob
    
    def _choose_arm(self):
        thetas = np.zeros(self.N)
        for i in range(self.N):
            p = np.random.random_sample()
            if p<1./self.N:
                x = sy.symbols("x")
                kl = kl_div(self.distribution, self.mu[i], x)
                y = np.random.random_sample()
                if y>= .5:
                    # equivalent
                    # thetas[i] = max(sy.solve(1-0.5*sy.exp(-(self.T[i]-1)*kl)-y, x))
                    thetas[i] = max(sy.solve(sy.log(0.5/(1-y))/(self.T[i]-1)-kl, x))
                else:
                    # equivalent
                    # thetas[i] = min(sy.solve(0.5*sy.exp(-(self.T[i]-1)*kl)-y, x))
                    thetas[i] = min(sy.solve(sy.log(0.5/y)/(self.T[i]-1)-kl, x))
            else:
                thetas[i] = self.mu[i]
        return np.argmax(thetas)
