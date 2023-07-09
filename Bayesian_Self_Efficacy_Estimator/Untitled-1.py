# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.special import gamma

# Define the parameters
alpha = 2
beta = 5

# Define the x range
x = np.linspace(0, 1, 1000)

# Compute the Beta distribution
y = beta.pdf(x, alpha, beta)

# Compute the maximum of the Beta distribution using alpha and beta
x_max = (alpha - 1) / (alpha + beta - 2)
y_max = (x_max**(alpha-1) * (1 - x_max)**(beta-1)) / ((gamma(alpha) * gamma(beta)) / gamma(alpha + beta))

# Normalize the Beta distribution
y_normalized = y / y_max

# Plot the normalized Beta distribution
plt.figure(figsize=(8, 6))
plt.plot(x, y_normalized, label=f'Beta({alpha}, {beta})')
plt.title('Modified Beta Distribution')
plt.xlabel('x')
plt.ylabel('f_mod(x)')
plt.legend()
plt.grid(True)
plt.show()
