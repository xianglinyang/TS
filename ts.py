import numpy as np
import sympy as sy
from tqdm import tqdm
from scipy.optimize import least_squares, newton


def kl_div(distribution, mu, x):
    '''the symbolic expression of kl divergence'''
    if distribution == "Gaussian":
        kl = 1/2*(mu-x)**2
    elif distribution == "Bernoulli":
        kl = mu*sy.log(mu/x)+(1-mu)*sy.log((1-mu)/(1-x))
    elif distribution == "Poisson":
        kl = mu*sy.log(mu/x)-mu+x
    elif distribution == "Gamma":
        kl = sy.log(x/mu)-(x-mu)/x
    else:
        raise NotImplementedError
    return kl


def find_mu(x, x_min, kl, upper, T, distribution):
    eqn = sy.Eq(T*kl, upper)
    if eqn == False:
        return np.inf
    func_np = sy.lambdify(x, T*kl-upper, modules=['numpy'])
    if distribution == "Bernoulli":
        solution = least_squares(func_np, (x_min+0.01), bounds = ((x_min), (1))).x
        # solution = newton(func_np, x)
    elif distribution == "Gamma" or distribution == "Gaussian" or distribution == "Poisson":
        solution = least_squares(func_np, (x_min+0.01), bounds = ((x_min), (np.inf))).x
        # solution = newton(func_np, x)
    else:
        raise NotImplementedError
    return solution[0]
        

class TS:
    def __init__(self, N, mu_ground_truth, distribution, init_mu=0, verbose=0):
        self.N = N
        self.distribution = distribution
        self.mu_ground_truth = mu_ground_truth
        self.mu = np.ones(self.N)*init_mu
        self.T = np.zeros(self.N)
        self.verbose = verbose
        self._initialization()

    def _pull_posterior(self, i):
        if self.distribution == "Gaussian":
            return np.random.normal(self.mu[i], np.sqrt(1/self.T[i]), 1)[0]
        elif self.distribution == "Bernoulli":
            return np.random.beta(1+self.T[i]*self.mu[i], 1+self.T[i]*(1-self.mu[i]), size=1)[0]
        elif self.distribution == "Poisson":
            return np.random.gamma(1+self.mu[i]*self.T[i],1/self.T[i],size=1)[0]
        elif self.distribution == "Gamma":
            return 1/np.random.gamma(self.T[i]-1, 1/self.T[i]/self.mu[i],size=1)[0]
        else:
            raise NotImplementedError
    
    def _initialization(self):
        if self.distribution == "Poisson":
            for i in range(self.N):
                # in case 0
                reward = self._pull_arm(i)+1
                self._update_posterior(i, reward)
        else:
            for i in range(self.N):
                reward = self._pull_arm(i)
                self._update_posterior(i, reward)

        if self.distribution == "Gamma":
            for i in range(self.N):
                reward = self._pull_arm(i)
                self._update_posterior(i, reward)
        if self.verbose>0:
            print("Initialize arms...")

    def _update_posterior(self, i, reward):
        # reward mean
        self.mu[i] = (self.mu[i]*self.T[i]+reward)/(self.T[i]+1)
        self.T[i] = self.T[i]+1

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
        # for t in tqdm(range(start, T+1, 1)):
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
                curr_thetas[i] = self.mu[i]
        return np.argmax(curr_thetas)


class KL_UCB_plus_plus(TS):

    def __init__(self, N, mu_ground_truth, distribution, init_mu=0, verbose=0):
        super().__init__(N, mu_ground_truth, distribution, init_mu, verbose)
        # for faster computation, as memory of curr_u
        self.curr_u = None
    
    def _update_curr_u(self, i, T):
        U_upper = np.max(np.log((np.power(np.max(np.log(T/(self.N*self.T[i]))),2)+1)*T/(self.N*self.T[i])))
        # find mu kl(self.mu[i], mu)<=U_upper[i]/self.T[i]
        x = sy.symbols("x", real=True)
        kl = kl_div(self.distribution, self.mu[i], x)
        self.curr_u[i] = find_mu(x, self.mu[i], kl, upper=U_upper, T=self.T[i], distribution=self.distribution)

    def _choose_arm(self, T):
        U_upper = np.maximum(np.log((np.power(np.maximum(np.log(T/(self.N*self.T)),0),2)+1)*T/(self.N*self.T)),0)
        if self.curr_u is None:
            self.curr_u = np.zeros(self.N)
            for i in range(self.N):
                # find mu kl(self.mu[i], mu)<=U_upper[i]/self.T[i]
                x = sy.symbols("x", real=True)
                kl = kl_div(self.distribution, self.mu[i], x)
                self.curr_u[i] = find_mu(x, self.mu[i], kl, upper=U_upper[i], T=self.T[i], distribution=self.distribution)
        return np.argmax(self.curr_u)
        
    def regret(self, T, period):
        mu_max = np.max(self.mu_ground_truth)
        regrets_plot = np.zeros(int(T/period))
        regret = 0
        start = 2*self.N + 1 if self.distribution == "Gamma" else self.N + 1
        # for t in tqdm(range(start, T+1, 1)):
        for t in range(start, T+1, 1):
            i = self._choose_arm(T)
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)
            # update memory
            self._update_curr_u(i, T)
    
            regret = regret + mu_max - self.mu_ground_truth[i]
            
            if t % period == 0:
                regrets_plot[int(t/period)-1] = regret
        return regret, regrets_plot

class KL_UCB(TS):

    def __init__(self, N, mu_ground_truth, distribution, init_mu=0, verbose=0):
        super().__init__(N, mu_ground_truth, distribution, init_mu, verbose)
        self.name = "KL_UCB"
        self.curr_u = None
        if self.distribution == "Gamma":
            self._update_curr_u(2*self.N)
        else:
            self._update_curr_u(self.N)

    def _update_curr_u(self, t):
        U_upper = np.log(t)+ 3 * np.log(np.log(t))
        self.curr_u = np.zeros(self.N)
        for i in range(self.N):
            # find mu kl(self.mu[i], mu)<=U_upper[i]/self.T[i]
            x = sy.symbols("x", real=True)
            kl = kl_div(self.distribution, self.mu[i], x)
            if self.verbose:
                print("Arm {} kl divergence:\n{}".format(i, kl))
            self.curr_u[i] = find_mu(x, self.mu[i], kl, upper=U_upper, T=self.T[i], distribution=self.distribution)

    def _update_curr_u_i(self, t, i):
        U_upper = np.log(t)+ 3 * np.log(np.log(t))
        x = sy.symbols("x", real=True)
        kl = kl_div(self.distribution, self.mu[i], x)
        if self.verbose:
            print("Arm {} kl divergence:\n{}".format(i, kl))
        self.curr_u[i] = find_mu(x, self.mu[i], kl, upper=U_upper, T=self.T[i], distribution=self.distribution)

    def _choose_arm(self):
        return np.argmax(self.curr_u)

    def regret(self, T, period):
        mu_max = np.max(self.mu_ground_truth)
        regrets_plot = np.zeros(int(T/period))
        regret = 0
        start = 2*self.N + 1 if self.distribution == "Gamma" else self.N + 1
        # for t in tqdm(range(start, T+1, 1)):
        for t in range(start, T+1, 1):
            i = self._choose_arm()
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)
            if t<100 or np.log10(t) % 1 ==0:
                self._update_curr_u(t)
            else:
                self._update_curr_u_i(t, i)
            
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
            thetas[i] = np.random.normal(self.mu[i], np.sqrt(sigma/self.rho/self.T[i]), 1)[0]

        mus = self.mu+4/self.T*np.maximum(np.log(T/self.N*self.T),0)

        curr_thetas = np.minimum(thetas, mus)
        return np.argmax(curr_thetas)
    
    def regret(self, T, period):
        mu_max = np.max(self.mu_ground_truth)
        regrets_plot = np.zeros(int(T/period))
        regret = 0
        start = 2*self.N + 1 if self.distribution == "Gamma" else self.N + 1
        # for t in tqdm(range(start, T+1, 1)):
        for t in range(start, T+1, 1):
            i = self._choose_arm(T)
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)
            
            regret = regret + mu_max - self.mu_ground_truth[i]
            
            if t % period == 0:
                regrets_plot[int(t/period)-1] = regret
        return regret, regrets_plot


class ExpTS(TS):

    def _initialization(self):
        if self.distribution == "Poisson":
            for i in range(self.N):
                # in case 0
                reward = self._pull_arm(i)+1
                self._update_posterior(i, reward)
                # reward = self._pull_arm(i)+1
                # self._update_posterior(i, reward)
        else:
            for i in range(self.N):
                reward = self._pull_arm(i)
                self._update_posterior(i, reward)
                # reward = self._pull_arm(i)
                # self._update_posterior(i, reward)
        print("Initialize arms...")
    
    def _solve_theta(self, i):
        x = sy.symbols("x", real=True)
        kl = kl_div(self.distribution, self.mu[i], x)
        if kl == sy.nan:
            return np.inf
        y = np.random.random_sample()
        if y>= .5:
            # equivalent
            # thetas[i] = max(sy.solve(1-0.5*sy.exp(-(self.T[i]-1)*kl)-y, x))
            func_np = sy.lambdify(x, sy.log(0.5/(1-y))/(self.T[i]-0.99)-kl, modules=['numpy'])
            if self.distribution == "Bernoulli":
                solution = least_squares(func_np, ((self.mu[i]+1)/2.0), bounds = ((self.mu[i]), (1))).x
            elif self.distribution == "Poisson" or self.distribution== "Gamma":
                solution = least_squares(func_np, (self.mu[i]+0.1), bounds = ((self.mu[i]), (np.inf))).x
            elif self.distribution == "Gaussian":
                solution = least_squares(func_np, (self.mu[i]+0.1), bounds = ((self.mu[i]), (np.inf))).x
            else:
                raise NotImplementedError
        else:
            # equivalent
            # thetas[i] = min(sy.solve(0.5*sy.exp(-(self.T[i]-1)*kl)-y, x))
            func_np = sy.lambdify(x, sy.log(0.5/y)/(self.T[i]-0.99)-kl, modules=['numpy'])
            if self.distribution == "Bernoulli":
                solution =least_squares(func_np, (self.mu[i]*0.5), bounds = ((0), (self.mu[i]))).x
            elif self.distribution == "Poisson" or self.distribution== "Gamma":
                solution =least_squares(func_np, (self.mu[i]*0.5), bounds = ((0), (self.mu[i]))).x
            elif self.distribution == "Gaussian":
                solution = least_squares(func_np, (self.mu[i]-0.1), bounds = ((-np.inf), (self.mu[i]))).x
            else:
                raise NotImplementedError
        return solution[0]

    def _choose_arm(self):
        thetas = np.zeros(self.N)
        for i in range(self.N):
            thetas[i] = self._solve_theta(i)
        return np.argmax(thetas)

    def regret(self, T, period):
        mu_max = np.max(self.mu_ground_truth)
        regrets_plot = np.zeros(int(T/period))
        regret = 0

        start = self.N + 1
        # for t in tqdm(range(start, T+1, 1)):
        for t in range(start, T+1, 1):
            i = self._choose_arm()
            reward = self._pull_arm(i)
            self._update_posterior(i, reward)
            
            regret = regret + mu_max - self.mu_ground_truth[i]
            
            if t % period == 0:
                regrets_plot[int(t/period)-1] = regret
        return regret, regrets_plot

class ExpTS_plus(ExpTS):
    '''with prob probability'''
    def __init__(self, N, mu_ground_truth, distribution, prob, init_mu=0):
        super().__init__(N, mu_ground_truth, distribution, init_mu)
        self.prob = prob
    
    def _choose_arm(self):
        thetas = np.zeros(self.N)
        for i in range(self.N):
            p = np.random.random_sample()
            if p<1./self.N:
                thetas[i] = self._solve_theta(i)
            else:
                thetas[i] = self.mu[i]
        return np.argmax(thetas)

    