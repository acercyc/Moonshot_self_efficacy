# %% 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize

from model import model
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import jax.random as random
import jax.numpy as jnp
import arviz as az



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

class Visualizer:
    def __init__(self, figsize=(12, 12), windowTitle=None):
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        self.h_idata = []  # history of idata
        self.fig = fig
        
    def plot_model(self, model):
        fig = self.fig
        ax_model = fig.add_subplot(2, 2, 1)
        model.plot_utility(fig, ax_model)
        
    def plot_data(self, a, se):
        fig = self.fig
        ax_data = fig.add_subplot(2, 2, 2)
        ax_data.plot(se, a, 'o', fillstyle='none')
        ax_data.set_xlabel('Self efficacy')
        ax_data.set_ylabel('Assistance level')
        ax_data.set_xlim([0, 1])
        ax_data.set_ylim([0, 1])
        ax_data.set_title('Self efficacy vs Assistance level')
    
    def plot_joint_posterior(self, mcmc):
        # extact median alpha and beta
        alpha = np.median(mcmc.get_samples()['alpha'])
        beta = np.median(mcmc.get_samples()['beta'])
        ax_joint = self.fig.add_subplot(2, 2, 3)
        # plot joint posterior of alpha and beta
        az.plot_pair(mcmc, var_names=['alpha', 'beta'], kind='kde', ax=ax_joint)
        ax_joint.set_title("Joint distribution of alpha and beta", fontsize=10)
        
    def plot_inference(self, mcmc):
        # extact median alpha and beta
        alpha = np.median(mcmc.get_samples()['alpha'])
        beta = np.median(mcmc.get_samples()['beta'])
        ax_inference = self.fig.add_subplot(2, 2, 4)
        # plot joint posterior of alpha and beta
        Agent_BayesianDecision(alpha, beta).plot_utility(self.fig, ax_inference)



if __name__ == '__main__':
    # generate dummy data        
    agent = Agent_BayesianDecision(10, 10, p_self=0.5, p_others=0.5)
    a = np.random.normal(agent.find_optimal_assistance(), 0.03, 10)
    se = np.random.normal(agent.self_efficacy(), 0.01, 10)
    x = np.linspace(0, 1, 100)
    
    # Inference
    a_ = jnp.array(a)
    se_ = jnp.array(se)
    kernel = numpyro.infer.NUTS(model, init_strategy=numpyro.infer.init_to_median)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=1000)
    mcmc.run(random.PRNGKey(0), a_, se_)
    mcmc.print_summary()
    
    # %% 
    # Visualize
    v = Visualizer()
    v.plot_model(agent)
    v.plot_data(a, se)
    v.plot_joint_posterior(mcmc)
    v.plot_inference(mcmc)
    v.fig.canvas.manager.window.activateWindow()

# %%
