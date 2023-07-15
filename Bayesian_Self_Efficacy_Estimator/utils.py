# %matplotlib qt
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


def get_var_names(idata):
    variables = idata.posterior.data_vars
    var_names = list(variables.keys())
    return var_names


def get_posterior_mean(idata):
    summary = pm.summary(idata)
    posterior_mean = summary["mean"]
    posterior_mean
    return posterior_mean


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
            alpha = pm.Uniform("alpha", lower=0, upper=10)
            beta = pm.Uniform("beta", lower=0, upper=10)
            sigma = pm.Gamma("sigma", alpha=1, beta=1)

        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    def sample_prior(self):
        with self.model:
            idata = pm.sample(tune=200, chains=2, cores=1)
        self.idata = idata
        return idata

    def fit(self, x, y):
        with self.model:
            # likelihood
            y_obs = pm.Normal("y_obs", mu=f_BetaNorm(self.alpha, self.beta, x), sigma=self.sigma, observed=y)

            # inference
            idata = pm.sample(tune=200, chains=2, cores=1)
            self.y_obs = y_obs
        self.idata = idata
        return idata


# visualizer
class Visualizer:
    pass


class Visualizer_BetaNorm(Visualizer):
    def __init__(self, figsize=(12, 12), windowTitle=None):
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        self.h_idata = []  # history of idata
        self.fig = fig

    def cal_model(self, idata):
        posterior_mean = get_posterior_mean(idata)
        alpha_estimated = posterior_mean["alpha"]
        beta_estimated = posterior_mean["beta"]
        x = np.linspace(0, 1, 1000)
        y_estimated = f_BetaNorm(alpha_estimated, beta_estimated, x)
        return x, y_estimated

    def add_idata(self, idata):
        self.h_idata.append(idata)

    def plot(self, idata=None):
        if idata is None:
            idata = self.h_idata[-1]
        else:
            self.add_idata(idata)
        # clean ax
        plt.clf()
        self.plot_model(idata)
        self.plot_joint_posterior(idata)
        self.plot_posterior_alpha(idata)
        self.plot_posterior_beta(idata)
        plt.show(block=False)

    def plot_history(self, idata=None):
        if idata is None:
            idata = self.h_idata[-1]
        else:
            self.add_idata(idata)
        # clean ax
        plt.clf()
        self.plot_model_history()
        self.plot_joint_posterior(idata)
        self.plot_posterior_alpha(idata)
        self.plot_posterior_beta(idata)
        plt.show(block=False)

    def plot_model(self, idata):
        fig = self.fig
        ax_model = fig.add_subplot(2, 2, 1)
        x, y_estimated = self.cal_model(idata)
        ax_model.plot(x, y_estimated, label=f"Updated model")
        ax_model.set_xlabel("Assistance")
        ax_model.set_ylabel("Normalised self-efficacy")
        ax_model.set_title(
            "Assistance vs Normalised self-efficacy",
            fontsize=10,
        )
        ax_model.set_xlim(0, 1.1)
        ax_model.set_ylim(0, 1.1)
        ax_model.grid(True)
        self.ax_model = ax_model

    def plot_model_history(self):
        fig = self.fig
        ax_model = fig.add_subplot(2, 2, 1)
        h_idata = self.h_idata
        n_history = len(h_idata)
        if n_history > 10:
            n_history = 10
        cmap = plt.get_cmap("Wistia")
        cmap = cmap.reversed()

        for i in range(n_history):
            color = cmap(i / n_history)
            idata = h_idata[-i - 1]
            x, y_estimated = self.cal_model(idata)
            transparency = 1 - (i / n_history) * 0.3
            ax_model.plot(x, y_estimated, color=color, alpha=transparency, label=f"trial {n_history-i}")
            
        ax_model.set_xlabel("Assistance")
        ax_model.set_ylabel("Normalised self-efficacy")
        ax_model.set_title(
            "Assistance vs Normalised self-efficacy",
            fontsize=10,
        )
        ax_model.set_xlim(0, 1.1)
        ax_model.set_ylim(0, 1.1)
        ax_model.grid(True)
        ax_model.legend()
        self.ax_model = ax_model

    def plot_joint_posterior(self, idata):
        fig = self.fig
        ax_joint = fig.add_subplot(2, 2, 2)
        az.plot_pair(
            idata,
            var_names=["alpha", "beta"],
            kind="kde",
            figsize=[8, 8],
            divergences=True,
            textsize=10,
            colorbar=True,
            point_estimate="mean",
            ax=ax_joint,  # specify the ax parameter
        )
        ax_joint.set_xlim(0, 10)
        ax_joint.set_ylim(0, 10)
        ax_joint.set_title("Joint distribution of alpha and beta", fontsize=10)
        self.ax_joint = ax_joint

    def plot_posterior_alpha(self, idata):
        fig = self.fig
        ax_alpha = fig.add_subplot(2, 2, 3)
        az.plot_posterior(idata, var_names="alpha", ax=ax_alpha, textsize=10, point_estimate="mean", round_to=2)
        self.ax_alpha = ax_alpha

    def plot_posterior_beta(self, idata):
        fig = self.fig
        ax_beta = fig.add_subplot(2, 2, 4)
        az.plot_posterior(idata, var_names="beta", ax=ax_beta, textsize=10, point_estimate="mean", round_to=2)
        self.ax_beta = ax_beta


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

class Participant_BetaNorm_2(Participant):
    def __init__(self, alpha=3, beta=6, sigma=0.1, n_trial=100):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        x = np.random.uniform(0, 1, n_trial)
        y = f_BetaNorm(self.alpha, self.beta, x) + np.random.normal(0, self.sigma, len(x))
        x_true = np.linspace(0, 1, 1000)
        y_true = f_BetaNorm(self.alpha, self.beta, x_true)
        self.TrueModel = (x_true, y_true)
        self.data = (x, y)
        
    def getData(self, itrial):
        x, y = self.data
        return x[:itrial], y[:itrial]
    
    # def getTrueModel(self):
    #     x = np.linspace(0, 1, 1000)
    #     y = f_BetaNorm(self.alpha, self.beta, x)
    #     return x, y

    def response(self, x):
        y = f_BetaNorm(self.alpha, self.beta, x) + np.random.normal(0, self.sigma, len(x))
        y = np.clip(y, 0, 1)
        return y


class Participant_dummy(Participant):
    def __init__(self):
        data = pd.read_csv("20230705_Dummydata.csv")
        self.data = data

    def getData(self, itrial):
        data_ = self.data.iloc[:itrial, :]
        ass1 = np.array(data_["AssistRate1"] / 100)
        ass2 = np.array(data_["AssistRate2"] / 100)
        self_efficacy = np.array(data_["Botton_Length"])
        # normalise bot to 0 to 1
        self_efficacy = (self_efficacy - self_efficacy.min()) / (self_efficacy.max() - self_efficacy.min())
        return ass1, ass2, self_efficacy


if __name__ == "__main__":
    pass