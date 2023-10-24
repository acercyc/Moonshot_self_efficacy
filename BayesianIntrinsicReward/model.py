# %%
import numpy as np
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import jax.random as random
import jax.numpy as jnp
from jax.scipy.optimize import minimize


def model(a, se):
    
    def posterior(x, alpha, beta, p_self, p_others):
        # define the likelihood
        p_x_given_self = numpyro.deterministic("p_x_given_self", dist.Beta(alpha, beta).log_prob(x))
        p_x_given_others = numpyro.deterministic("p_x_given_others", dist.Uniform(0, 1).log_prob(x))   

        # Compute log posterior belief of the cause of the performance
        p_self_given_x_ = jnp.log(p_self) + p_x_given_self
        p_others_given_x_ = jnp.log(p_others) + p_x_given_others
        p_self_given_x = numpyro.deterministic("p_self_given_x", p_self_given_x_ - jnp.logaddexp(p_self_given_x_, p_others_given_x_))
        p_others_given_x = numpyro.deterministic("p_others_given_x", p_others_given_x_ - jnp.logaddexp(p_self_given_x_, p_others_given_x_))
        return jnp.exp(p_self_given_x), jnp.exp(p_others_given_x)
    
    def Utility(x, alpha, beta, p_self, p_others, gamma_self, gamma_others):
        p_self_given_x, p_others_given_x = posterior(x, alpha, beta, p_self, p_others)

        # utility function
        U_self = gamma_self * p_self_given_x * x
        U_others = gamma_others * p_others_given_x * x
        U = U_self + U_others
        return U

    # define the hyper prior
    alpha = numpyro.sample("alpha", dist.Uniform(0, 50))
    beta = numpyro.sample("beta", dist.Uniform(0, 50))  
    
    # utility cooefficients
    gamma_self = numpyro.deterministic("gamma_self", 1)
    gamma_others = numpyro.deterministic("gamma_others", 0)      

    # Prior probability of self as the cause of the performance
    p_self = numpyro.deterministic("p_self", 0.5)
    p_others = numpyro.deterministic("p_others", 1-p_self)
    
    # find maximum utility
    xs = jnp.linspace(0, 1, 1000)
    Us = Utility(xs, alpha, beta, p_self, p_others, gamma_self, gamma_others)
    x_opt = xs[jnp.argmax(Us)]
    
    with numpyro.plate("data", len(a)):
        self_efficacy = numpyro.sample("self_efficacy", dist.Normal(alpha / (alpha + beta), 0.1), obs=se)
        numpyro.sample("assistance", dist.Normal(x_opt - self_efficacy, 0.1), obs=a)





