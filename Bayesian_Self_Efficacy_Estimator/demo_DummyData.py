import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import os
from utils import *


if __name__ == '__main__':
    # ---------------------------------------------------------------------------- #
    #                              create participant                              #
    # ---------------------------------------------------------------------------- #
    participant = Participant_dummy()

    # ---------------------------------------------------------------------------- #
    #                               create estimator                               #
    # ---------------------------------------------------------------------------- #
    estimator = Estimator_BetaNorm()
    idata = estimator.sample_prior()
    var_name = get_var_names(idata)

    # ---------------------------------------------------------------------------- #
    #                               create visualizer                              #
    # ---------------------------------------------------------------------------- #
    plt.close("all") 
    visualizer = Visualizer_BetaNorm()
    plt.gcf().canvas.manager.window.activateWindow()
    plt.pause(0.1)

    # ---------------------------------------------------------------------------- #
    # %%

    for i in range(1, 11):
        ass1, ass2, self_efficacy = participant.getData(i)
        estimator = Estimator_BetaNorm()
        idata = estimator.fit(ass1, self_efficacy)
        visualizer.plot_history(idata)
        visualizer.ax_model.scatter(ass1, self_efficacy, color="red", label=f"trial {i}")
        # refresh and show the figure
        plt.suptitle(f"trial {i}")
        plt.show(block=False)
        plt.pause(0.1)
    plt.show(block=True)