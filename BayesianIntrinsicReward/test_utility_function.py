# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

x = 0.7

# hyperparameters of the beta distribution
alpha = 30
beta = 30

# Prior probability of self as the cause of the performance
p_self = 0.5
p_others = 1 - p_self

# likelihood of the performance given self as the cause
## beta distribution
# create a beta distribution object
beta_dist = stats.beta(alpha, beta)
p_x_given_self = beta_dist.pdf(x)

## uniform distribution
p_x_given_others = stats.uniform(0, 1).pdf(x)


# Compute log posterior belief of the cause of the performance
p_self_given_x_ = np.log(p_self) + np.log(p_x_given_self) - np.logaddexp(np.log(p_self) + np.log(p_x_given_self), np.log(p_others) + np.log(p_x_given_others))
p_self_given_x = np.exp(p_self_given_x_)
p_others_given_x = 1 - p_self_given_x

# utility cooefficients
gamma_self = 1
gamma_others = 0

# utility function
U_self = gamma_self * p_self_given_x * x
U_others = gamma_others * p_others_given_x * x
U = U_self + U_others


# %% 
# Posterior belief of the cause of the performance
def posterior(x):
    # hyperparameters of the beta distribution
    alpha = 30
    beta = 30

    # Prior probability of self as the cause of the performance
    p_self = 0.5
    p_others = 1 - p_self

    # likelihood of the performance given self as the cause
    ## beta distribution
    # create a beta distribution object
    beta_dist = stats.beta(alpha, beta)
    p_x_given_self = beta_dist.pdf(x)

    ## uniform distribution
    p_x_given_others = stats.uniform(0, 1).pdf(x)


    # Compute log posterior belief of the cause of the performance
    p_self_given_x_ = np.log(p_self) + np.log(p_x_given_self) - np.logaddexp(np.log(p_self) + np.log(p_x_given_self), np.log(p_others) + np.log(p_x_given_others))
    p_self_given_x = np.exp(p_self_given_x_)
    return p_self_given_x


# Utility function
def Utility(x):
    # posterior belief of the cause of the performance
    p_self_given_x = posterior(x)
    p_others_given_x = 1 - p_self_given_x


    # utility cooefficients
    gamma_self = 1
    gamma_others = 0

    # utility function
    U_self = gamma_self * p_self_given_x * x
    U_others = gamma_others * p_others_given_x * x
    U = U_self + U_others
    return -U

# argmax of the utility function
from scipy.optimize import minimize
x0 = 0.5
res = minimize(Utility, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
print(res.x)
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

alpha = 63.07
beta = 56.08

# Utility function
def Utility(x, alpha, beta, p_self=0.5, gamma_self=1, gamma_others=0):
# Prior probability of self as the cause of the performance
    p_others = 1 - p_self

    # likelihood of the performance given self as the cause
    ## beta distribution
    # create a beta distribution object
    beta_dist = stats.beta(alpha, beta)
    p_x_given_self = beta_dist.pdf(x)

    ## uniform distribution
    p_x_given_others = stats.uniform(0, 1).pdf(x)


    # Compute log posterior belief of the cause of the performance
    p_self_given_x_ = np.log(p_self) + np.log(p_x_given_self) - np.logaddexp(np.log(p_self) + np.log(p_x_given_self), np.log(p_others) + np.log(p_x_given_others))
    p_self_given_x = np.exp(p_self_given_x_)
    p_others_given_x = 1 - p_self_given_x

    # utility function
    U_self = gamma_self * p_self_given_x * x
    U_others = gamma_others * p_others_given_x * x
    U = U_self + U_others
    return -U

# self efficacy
def self_efficacy(alpha, beta):
    # mean performance
    return alpha / (alpha + beta)

# argmax of the utility function
from scipy.optimize import minimize
x0 = 0.5

# involve alpha and beta
res = minimize(Utility, x0, method='nelder-mead', args=(alpha, beta), options={'xatol': 1e-8, 'disp': True})

# compute assistance level
a = res.x[0] - self_efficacy(alpha, beta)
a

# %%
x = np.linspace(0, 1, 100)
U = Utility(x, alpha, beta)
plt.plot(x, -U)
plt.xlabel("x")
plt.ylabel("Utility")
plt.title("Utility function")