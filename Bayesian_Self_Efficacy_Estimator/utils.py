# %%
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm


def f_BetaNorm(alpha, beta, x):
    # without scipy.stats.beta
    x_max = (alpha - 1) / (alpha + beta - 2)
    y = x ** (alpha - 1) * (1 - x) ** (beta - 1)
    y_max = x_max ** (alpha - 1) * (1 - x_max) ** (beta - 1)
    y_normalized = y / y_max
    return y_normalized


# Estimator
class Estimator:
    pass


# visualizer
class Visualizer:
    pass


class Participant:
    def __init__(self):
        pass

    def response(self, x):
        pass


class Participant_BetaNorm(Participant):
    def __init__(self, alpha=3, beta=6, sigma=0.1):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    def response(self, x):
        y = f_BetaNorm(self.alpha, self.beta, x) + np.random.normal(0, sigma, len(x))
        y = np.clip(y, 0, 1)
        return y

    def plot(self):
        x = np.linspace(0, 1, 1000)
        y = f_BetaNorm(self.alpha, self.beta, x) 
        plt.figure(figsize=(8, 6))
        plt.plot(x, y)
        plt.title("Internal model of participant")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # create participant
    alpha_true = 3
    beta_true = 6
    sigma_true = 0.1

    # create participant
    participant = Participant_BetaNorm(alpha_true, beta_true, sigma_true)

    # plot participant
    participant.plot()


# %%
