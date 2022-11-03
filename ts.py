import numpy as np
import time
import matplotlib.pyplot as plt
from math import e

class TS:
    def __init__(self, N, mu_gt, init_k=0, init_mu=0):
        self.N = N
        self.mu_gt = mu_gt
        self.init_k = init_k
        self.init_mu = init_mu
    
    def init_arm(self):
        self.mu = np.ones(self.N)*self.init_mu
        self.k = np.ones(self.N)*self.init_k
        self.sigma = np.ones(self.N)*(1/(self.init_k+1))
    
    def choose_arm(self):
        curr_thetas = np.zeros(self.N)
        for i in range(self.N):
            curr_thetas[i] = np.random.normal(self.mu[i], np.sqrt(self.sigma[i]), 1)[0]
        return np.argmax(curr_thetas)
    
    def update_arm(self, i, reward):
        self.mu[i] = (self.mu[i]*self.k[i]+reward)/(self.k[i]+2)
        self.k[i] = self.k[i]+1
        self.sigma[i] = 1/(1+self.k[i])
    
    def pull_arm(self, i):
        return np.random.normal(self.mu_gt[i], 1, 1)[0]
    
    def regret(self, T, period):
        mu_max = np.max(self.mu_gt)
        self.init_arm()
        regret = np.zeros(int(T/period))
        r = 0

        for t in range(T):
            i = self.choose_arm()
            reward = self.pull_arm(i)
            r = r + mu_max -  self.mu_gt[i]
            self.update_arm(i, reward)
            if t % period == 0:
                regret[t//period] = r
        return r, regret

class TS_Greedy(TS):

    def choose_arm(self, prob):
        curr_thetas = np.zeros(self.N)
        for i in range(self.N):
            # propability 1/N sample, (N-1)/N mu_i
            tmp = np.random.random_sample()
            # tmp = np.random.choice(self.N, 1, replace=True)[0]
            if tmp<prob:
                curr_thetas[i] = np.random.normal(self.mu[i], np.sqrt(self.sigma[i]), 1)[0]
            else:
                curr_thetas[i] = self.mu[i]
        return np.argmax(curr_thetas)
    
    def regret(self, T, prob, period):
        mu_max = np.max(self.mu_gt)
        self.init_arm()
        regret = np.zeros(int(T/period))
        r = 0
        for t in range(T):
            i = self.choose_arm(prob)
            reward = self.pull_arm(i)
            r = r + mu_max -  self.mu_gt[i]
            self.update_arm(i, reward)
            if t % period == 0:
                regret[t//period] = r
        return r, regret

class TS_plus(TS):
    def choose_arm(self):
        curr_thetas = np.zeros(self.N)
        for i in range(self.N):
            curr_thetas[i] = np.random.normal(self.mu[i], np.sqrt(self.sigma[i]), 1)[0]
            while curr_thetas[i]<self.mu[i]:
                curr_thetas[i] = np.random.normal(self.mu[i], np.sqrt(self.sigma[i]), 1)[0]
        return np.argmax(curr_thetas)

class TS_Greedy_plus(TS_Greedy):
    def choose_arm(self, prob):
        curr_thetas = np.zeros(self.N)
        for i in range(self.N):
            # propability 1/N sample, (N-1)/N mu_i
            tmp = np.random.random_sample()
            # tmp = np.random.choice(self.N, 1, replace=True)[0]
            if tmp<prob:
                curr_thetas[i] = np.random.normal(self.mu[i], np.sqrt(self.sigma[i]), 1)[0]
                while curr_thetas[i]<self.mu[i]:
                    curr_thetas[i] = np.random.normal(self.mu[i], np.sqrt(self.sigma[i]), 1)[0]
            else:
                curr_thetas[i] = self.mu[i]
        return np.argmax(curr_thetas)


class TS_plusplus(TS):
    def update_arm(self, i, t, reward):
        self.mu[i] = (self.mu[i]*self.k[i]+reward)/(self.k[i]+2)
        self.k[i] = self.k[i]+1
        self.sigma[i] = self.sigma[i]*(1-1/(np.log(e*e*t/self.k[i])))
    
    def regret(self, T, period):
        mu_max = np.max(self.mu_gt)
        self.init_arm()
        regret = np.zeros(int(T/period))
        r = 0

        for t in range(T):
            i = self.choose_arm()
            reward = self.pull_arm(i)
            r = r + mu_max -  self.mu_gt[i]
            self.update_arm(i, t+1, reward)
            if t % period == 0:
                regret[t//period] = r
        return r, regret



class TS_Greedy_plusplus(TS_Greedy):

    def update_arm(self, i, t, reward):
        self.mu[i] = (self.mu[i]*self.k[i]+reward)/(self.k[i]+2)
        self.k[i] = self.k[i]+1
        self.sigma[i] = self.sigma[i]*(1-1/(np.log(e*e*t/self.k[i])))

    def regret(self, T, prob, period):
        mu_max = np.max(self.mu_gt)
        self.init_arm()
        regret = np.zeros(int(T/period))
        r = 0
        for t in range(T):
            i = self.choose_arm(prob)
            reward = self.pull_arm(i)
            r = r + mu_max -  self.mu_gt[i]
            self.update_arm(i, t+1, reward)
            if t % period == 0:
                regret[t//period] = r
        return r, regret



        



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

    line2 = plt.plot(record_time, regret, 'ro--', linewidth=1, markersize=4, label="TS Greedy")

    # ----------TS plus---------
    mean_regret = np.zeros(repeat)
    regret = np.zeros(int(T/period))
    for i in range(repeat):
        ts = TS_plus(N, mu_gt)
        mean_regret[i], regret_line = ts.regret(T, period)
        regret = regret + regret_line
    regret = regret / repeat
    print("TS+ mean regret is {:.3f}".format(mean_regret.mean()))
    
    line3 = plt.plot(record_time, regret, 'bo--', linewidth=1, markersize=4, label="TS+")

    # ----------TS Greedy plus---------
    mean_regret = np.zeros(repeat)
    regret = np.zeros(int(T/period))
    for i in range(repeat):
        ts = TS_Greedy_plus(N, mu_gt)
        mean_regret[i],regret_line = ts.regret(T, prob, period)
        regret = regret_line + regret
    regret = regret / repeat
    print("TS Greedy+ mean regret is {:.3f}".format(mean_regret.mean()))

    line4 = plt.plot(record_time, regret, 'mo--', linewidth=1, markersize=4, label="TS Greedy+")

    # ----------TS plusplus---------
    mean_regret = np.zeros(repeat)
    regret = np.zeros(int(T/period))
    for i in range(repeat):
        ts = TS_plusplus(N, mu_gt)
        mean_regret[i], regret_line = ts.regret(T, period)
        regret = regret + regret_line
    regret = regret / repeat
    print("TS++ mean regret is {:.3f}".format(mean_regret.mean()))
    
    line5 = plt.plot(record_time, regret, 'yo--', linewidth=1, markersize=4, label="TS++")

    # ----------TS Greedy plusplus---------
    mean_regret = np.zeros(repeat)
    regret = np.zeros(int(T/period))
    for i in range(repeat):
        ts = TS_Greedy_plusplus(N, mu_gt)
        mean_regret[i],regret_line = ts.regret(T, prob, period)
        regret = regret_line + regret
    regret = regret / repeat
    print("TS Greedy++ mean regret is {:.3f}".format(mean_regret.mean()))

    line6 = plt.plot(record_time, regret, 'co--', linewidth=1, markersize=4, label="TS Greedy++")

    plt.legend()
    plt.savefig("TS+.png")
    t1 = time.time()

    print("Time spent {:.2f}".format(t1-t0))
