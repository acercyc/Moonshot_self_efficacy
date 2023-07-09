# %%

# import pymc as pm
# import numpy as np

# # Define the function to estimate
# def my_function(x, alpha, beta):
#     return alpha * x + beta

# # Generate some sample data
# x = np.linspace(0, 1, 100)
# alpha_true = 3
# beta_true = 3
# y = my_function(x, alpha_true, beta_true) + np.random.normal(0, 0.1, 100)

# # Define the PyMC3 model
# with pm.Model() as model:
#     # Define the priors for the parameters
#     alpha = pm.Normal('alpha', mu=3, sigma =1)
#     beta = pm.Normal('beta', mu=3, sigma =1)

#     # Define the likelihood function
#     likelihood = pm.Normal('y', mu=my_function(x, alpha, beta), sigma =0.1, observed=y)

#     # Run the MCMC sampler to estimate the posterior distribution
#     # trace = pm.sample(10, tune=10, chains=2)
#     trace = pm.sample()
# # Print the summary statistics of the posterior distribution
# pm.summary(trace)
# %%
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # %config InlineBackend.figure_format = 'retina'
    # Initialize random number generator
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)
    az.style.use("arviz-darkgrid")
    # True parameter values
    alpha, sigma = 1, 1
    beta = [1, 2.5]

    # Size of dataset
    size = 100

    # Predictor variable
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2

    # Simulate outcome variable
    Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma

    import pymc as pm

    print(f"Running on PyMC v{pm.__version__}")

    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Expected value of outcome
        mu = alpha + beta[0] * X1 + beta[1] * X2

        # Likelihood (sampling distribution) of observationsH
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

    with basic_model:
        # draw 1000 posterior samples
        idata = pm.sample(tune=200, chains=2, cores=2)


# %%
# New data
new_X1 = np.random.randn(size)
new_X2 = np.random.randn(size) * 0.2
new_Y = alpha + beta[0] * new_X1 + beta[1] * new_X2 + rng.normal(size=size) * sigma

# Update the observed values
with basic_model:
    pm.set_data({"Y_obs": new_Y, "X1": new_X1, "X2": new_X2})
    idata = pm.sample()
