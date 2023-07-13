# %%
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import os


def f_BetaNorm(alpha, beta, x):
    # without scipy.stats.beta
    x_max = (alpha - 1) / (alpha + beta - 2)
    y = x ** (alpha - 1) * (1 - x) ** (beta - 1)
    y_max = x_max ** (alpha - 1) * (1 - x_max) ** (beta - 1)
    y_normalized = y / y_max
    return y_normalized


# Task
class Task:
    pass


class Task_dummy:
    # AssistRate1, AssistRate2, Botton Length
    def __init__(self):
        # load csv
        data = pd.read_csv("20230705_Dummydata.csv")
        self.data = data

    def step(self, i):
        # return 1 to i-th row
        data_ = self.data.iloc[:i, :]
        ass1 = np.array(data_["AssistRate1"] / 100)
        ass2 = np.array(data_["AssistRate2"] / 100)
        bot = np.array(data_["Botton_Length"])
        # normalise bot to 0 to 1
        bot = (bot - bot.min()) / (bot.max() - bot.min())
        return ass1, ass2, bot

    def plot_step(self, i):
        # plot 2 subplots for AssistRate against Botton Length at step i
        # Set x and y limit to 0 to 1
        ass1, ass2, bot = self.step(i)
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].scatter(ass1, bot)
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(0, 1)
        ax[0].set_title("AssistRate1 vs Botton Length")
        ax[0].set_xlabel("AssistRate1")
        ax[0].set_ylabel("Botton Length")
        ax[0].grid(True)
        ax[1].scatter(ass2, bot)
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(0, 1)
        ax[1].set_title("AssistRate2 vs Botton Length")
        ax[1].set_xlabel("AssistRate2")
        ax[1].set_ylabel("Botton Length")
        ax[1].grid(True)
        plt.show()
        return fig, ax


# Estimator
class Estimator:
    pass


class Estimator_BetaNorm(Estimator):
    def __init__(self):
        model = pm.Model()
        with model:
            # prior
            alpha = pm.Uniform("alpha", lower=2, upper=8)
            beta = pm.Uniform("beta", lower=2, upper=8)
            sigma = pm.Gamma("sigma", alpha=1, beta=1)

        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    def fit(self, x=None, y=None):
        if x is None:
            with self.model:
                # inference
                idata = pm.sample(tune=200, chains=1, cores=1)
        else:
            with self.model:
                # likelihood
                y_obs = pm.Normal("y_obs", mu=f_BetaNorm(self.alpha, self.beta, x), sigma=self.sigma, observed=y)
                # inference
                idata = pm.sample(tune=200, chains=1, cores=1)
            self.y_obs = y_obs
        self.idata = idata

        return idata


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
        y = f_BetaNorm(self.alpha, self.beta, x) + np.random.normal(0, self.sigma, len(x))
        y = np.clip(y, 0, 1)
        return y

    def plot(self):
        x = np.linspace(0, 1, 1000)
        y = f_BetaNorm(self.alpha, self.beta, x)
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(x, y)
        plt.title("Internal model of participant")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()

        return fig, ax


class Participant_dummy(Participant):
    def __init__(self):
        data = pd.read_csv("20230705_Dummydata.csv")
        self.data = data

    def getData(self, iTrial):
        data_ = self.data.iloc[:iTrial, :]
        ass1 = np.array(data_["AssistRate1"] / 100)
        ass2 = np.array(data_["AssistRate2"] / 100)
        self_efficacy = np.array(data_["Botton_Length"])
        # normalise bot to 0 to 1
        self_efficacy = (self_efficacy - self_efficacy.min()) / (self_efficacy.max() - self_efficacy.min())
        return ass1, ass2, self_efficacy
    
    
if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    #                              create participant                              #
    # ---------------------------------------------------------------------------- #
    alpha_true = 3
    beta_true = 6
    sigma_true = 0.1
    participant = Participant_BetaNorm(alpha_true, beta_true, sigma_true)

    # plot participant
    fig, ax = participant.plot()

    # response
    n = 100
    x_sampled = np.random.uniform(0, 1, n)
    y_sampled = participant.response(x_sampled)

    # plot response
    plt.figure(figsize=(8, 6))
    plt.scatter(x_sampled, y_sampled)
    plt.title("Response of participant")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    # ---------------------------------------------------------------------------- #
    #                                  Create task                                 #
    # ---------------------------------------------------------------------------- #
    # ass1 and ass2 are the assist rate of the two agents
    task = Task_dummy()
    for step in range(0, 11):
        estimator_1 = Estimator_BetaNorm()
        estimator_2 = Estimator_BetaNorm()
        
        if step == 0:
            idata_1 = estimator_1.fit(None)
            idata_2 = estimator_2.fit(None)
            
        else:
            ass1, ass2, bot = task.step(step)
            idata_1 = estimator_1.fit(ass1, bot)
            idata_2 = estimator_2.fit(ass2, bot)

        # summary of idata
        alpha_1, beta_1, sigma_1 = pm.summary(idata_1, var_names=["alpha", "beta", "sigma"])["mean"]
        alpha_2, beta_2, sigma_2 = pm.summary(idata_2, var_names=["alpha", "beta", "sigma"])["mean"]
        
        # print(f"alpha: {alpha}, beta: {beta}, sigma: {sigma}")

        # ---------------------------------------------------------------------------- #
        #                                 Visualization                                #
        # ---------------------------------------------------------------------------- #
        x = np.linspace(0, 1, 1000)
        y_1 = f_BetaNorm(alpha_1, beta_1, x)
        y_2 = f_BetaNorm(alpha_2, beta_2, x)
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].plot(x, y_1)
        if step != 0:
            ax[0].scatter(ass1, bot, color="red")
        ax[0].set_title("Estimated Internal model 1 of participant")
        ax[0].set_xlabel("Normalised Assistance Rate 1")
        ax[0].set_ylabel("Normalised Self-efficiency")
        ax[0].grid(True)
        
        ax[1].plot(x, y_2)
        if step != 0:
            ax[1].scatter(ass2, bot, color="red")
        ax[1].set_title("Estimated Internal model 2 of participant")
        ax[1].set_xlabel("Normalised Assistance Rate 2")
        ax[1].set_ylabel("Normalised Self-efficiency")
        ax[1].grid(True)
        
        plt.show()
        
        # save figure
        # create result folder
        if not os.path.exists("./result"):
            os.mkdir("./result")
        fig.savefig(f"./result/{step}.png")
        


# %%
