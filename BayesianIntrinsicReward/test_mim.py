# %%

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.optimize import minimize
# Define the function to minimize
def quadratic(x):
    return jnp.sum((x - 2) ** 2)

# Set the initial guess for the minimum
x0 = jnp.array([0.])

# Minimize the function
res = minimize(quadratic, x0, method='BFGS')

# Print the result
print(res.x[0])