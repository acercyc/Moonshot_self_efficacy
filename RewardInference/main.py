# %% 
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.random as random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

class BayesianVis:
    def __init__(self, samples):
        self.samples = samples

    def plot_posteriors(self, params=None):
        """
        Plot the posterior distributions of the given parameters.
        
        :param params: List of parameter names to plot. If None, plots all.
        """
        if params is None:
            params = self.samples.keys()
        
        for param in params:
            sns.histplot(self.samples[param], kde=True, label=param)
        
        plt.xlabel('Parameter Value')
        plt.ylabel('Density')
        plt.legend()
        plt.title('Posterior Distributions')
        plt.show()
    
    def plot_pair(self):
        """
        Plot pair plot for the samples to visualize the joint distribution.
        """
        sns.pairplot(pd.DataFrame(self.samples))
        plt.suptitle('Pair Plot of Parameters')
        plt.show()
        
import tkinter as tk
from tkinter import ttk
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.random as random

# Data lists to store trial inputs
x_values = []
A_values = []

# Bayesian model as defined previously
def model(x, A):
    alpha = numpyro.sample('alpha', dist.Normal(0, 10))
    beta = numpyro.sample('beta', dist.Normal(0, 10))
    B1 = 0.5
    B2 = 0.5
    y_pred = (alpha * (1 - x) + B1) * (beta * (1 - x) + x) + B2
    numpyro.sample('obs', dist.Normal(y_pred, 0.1), obs=A)

def run_inference():
    x = np.array(x_values, dtype=float)
    A = np.array(A_values, dtype=float)
    
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1500)
    mcmc.run(random.PRNGKey(0), x, A)
    
    samples = mcmc.get_samples()
    visualizer = BayesianVis(samples)
    visualizer.plot_posteriors(['alpha', 'beta'])

def add_trial():
    try:
        x = float(x_entry.get())
        A = float(A_entry.get())
        x_values.append(x)
        A_values.append(A)
        trials_listbox.insert(tk.END, f"x: {x}, A: {A}")
    except ValueError:
        # Handle invalid float input
        error_label.config(text="Please enter valid float values for x and A.")
        
np.random.seed(0)
num_samples = 100
x = np.random.rand(num_samples)
alpha_true = 2.0
beta_true = 3.0
noise = 0.1
B1 = 0.5  # some arbitrary constant
B2 = 0.5  # some arbitrary constant
y = (alpha_true * (1 - x) + B1) * (beta_true * (1 - x) + x) + B2 + noise * np.random.randn(num_samples)
A = y  # Total reward

# 2. numpyro model
def model(x, A):
    alpha = numpyro.sample('alpha', dist.Normal(0, 10))
    beta = numpyro.sample('beta', dist.Normal(0, 10))
    B1 = 0.5  # constant
    B2 = 0.5  # constant
    y_pred = (alpha * (1 - x) + B1) * (beta * (1 - x) + x) + B2
    numpyro.sample('obs', dist.Normal(y_pred, noise), obs=A)

# 3. run inference
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1500)
mcmc.run(random.PRNGKey(0), x, A)
mcmc.print_summary()

# 4. plot the result
samples = mcmc.get_samples()
sns.histplot(samples['alpha'], kde=True, label='alpha')
sns.histplot(samples['beta'], kde=True, color='orange', label='beta')
plt.xlabel('Parameter Value')
plt.ylabel('Density')
plt.legend()
plt.title('Posterior Distributions of alpha and beta')
plt.show()


# %%
samples = mcmc.get_samples()
visualizer = BayesianVis(samples)

# To plot the posteriors of alpha and beta
visualizer.plot_posteriors(['alpha', 'beta'])

# To plot the pair plot
visualizer.plot_pair()
