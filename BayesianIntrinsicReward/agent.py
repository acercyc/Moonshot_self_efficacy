# %%
# Given a certain task, the agent has a belief distribution over the task parameters
# The performance is normmalizied to be between 0 and 1
# The distribution can be represented by a beta distribution
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# %%
# belief of performance is moddeled by a beta distribution