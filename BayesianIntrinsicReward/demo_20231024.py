# %% 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize

class Agent:
    pass

class Agent_BayesianDecision(Agent):
    def __init__(self, alpha, beta, p_self=0.5, p_others=0.5, gamma_self=1, gamma_others=0):
        self.alpha = alpha
        self.beta = beta
        self.p_self = p_self
        self.p_others = p_others
        self.gamma_self = gamma_self
        self.gamma_others = gamma_others
        
    def likelihood_self(self):
        # likelihood of the performance given self as the cause
        return stats.beta(self.alpha, self.beta)

    def likelihood_others(self):
        # likelihood of the performance given others as the cause
        return stats.uniform(0, 1)
        
    def posterior(self, x):
        # likelihood of the performance given self as the cause
        p_x_given_self = self.likelihood_self().logpdf(x)

        # likelihood of the performance given others as the cause
        p_x_given_others = self.likelihood_others().logpdf(x)

        # Compute log posterior belief of the cause of the performance
        p_self_given_x_ = (
            np.log(self.p_self) + 
            p_x_given_self - 
            np.logaddexp(np.log(self.p_self) + p_x_given_self,
                         np.log(self.p_others) + p_x_given_others)
            )
        p_self_given_x = np.exp(p_self_given_x_)
        p_others_given_x = 1 - p_self_given_x
        return p_self_given_x, p_others_given_x
    
    def Utility(self, x):
        # Utility function
        p_self_given_x, p_others_given_x = self.posterior(x)
        U_self = self.gamma_self * p_self_given_x * x
        U_others = self.gamma_others * p_others_given_x * x
        U = U_self + U_others
        return U
    
    def self_efficacy(self):
        # self efficacy: mean of the likelihood of the performance given self as the cause
        return self.alpha / (self.alpha + self.beta)
    
    def find_optimal_performance(self):
        # find the optimal performance
        res = minimize(lambda x: -self.Utility(x), 0.5, method='nelder-mead')
        return res.x[0]
    
    def find_optimal_assistance(self):
        # find the optimal assistance
        a_opt = self.find_optimal_performance() - self.self_efficacy()
        return a_opt
    
    def plot_utility(self, fig=None, ax=None):
        x = np.linspace(0, 1, 100)
        
        # likelihood of the performance given self as the cause
        p = self.likelihood_self().pdf(x)
        p = p / np.max(p)
        se = self.self_efficacy()

        # utility function
        U = self.Utility(x)
        U = U / np.max(U)
        
        # optimal performance
        x_opt = self.find_optimal_performance()
        a_opt = x_opt - self.self_efficacy()

        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, p, 'b-', label='Likelihood function')
        ax.plot(x, U, 'y-', label='Utility function')
        ax.vlines(se, 0, 1, 
                  linestyles='dotted', color='b',
                  label=f'Self efficacy (={se:.2f})')
        ax.vlines(x_opt, 0, 1, 
                  linestyles='dashed', color='y',
                  label=f'Optimal performance (={x_opt:.2f})')
        # plot the optimal assistance
        ax.hlines(0, se, x_opt, 
              linestyles='dashed', color='r',
              label=f'Optimal assistance (={a_opt:.2f})')
        ax.set_xlabel('Performance')
        ax.set_ylabel('Utility')
        ax.set_xlim([0, 1])
        ax.legend()
        fig.show()
        return fig, ax
    

Agent_BayesianDecision(20, 4, p_self=0.9, p_others=0.1, gamma_self=1, gamma_others=0).plot_utility()