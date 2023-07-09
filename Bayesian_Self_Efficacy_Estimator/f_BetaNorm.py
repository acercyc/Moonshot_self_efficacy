# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def f_BetaNorm(alpha, beta_param, x):
    # with scipy.stats.beta
    from scipy.stats import beta
    y = beta.pdf(x, alpha, beta_param)
    x_max = (alpha - 1) / (alpha + beta_param - 2)
    y_max = beta.pdf(x_max, alpha, beta_param)
    y_normalized = y / y_max
    return y_normalized


def f_BetaNorm_flatted(alpha, beta_param, x):
    # without scipy.stats.beta
    x_max = (alpha - 1) / (alpha + beta_param - 2)
    y = x**(alpha-1) * (1 - x)**(beta_param-1)
    y_max = x_max**(alpha-1) * (1 - x_max)**(beta_param-1)
    y_normalized = y / y_max
    return y_normalized


# Define the parameters
alpha = 10
beta_param = 9

# Define the x range
x = np.linspace(0, 1, 1000)
y_normalized = f_BetaNorm_flatted(alpha, beta_param, x)

# Plot the normalized Beta distribution
plt.figure(figsize=(8, 6))
plt.plot(x, y_normalized, label=f'Beta({alpha}, {beta_param})')
plt.title('Modified Beta Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
# %%
