# %% Preprocessing of EEG data with MNE
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne

# %% Importing the dataset
# Importing the dataset
data = pd.read_csv('C:/Users/Chaitanya/Desktop/EEG_Data/EEG_Data.csv')

# band pass filter from 1 to 30 hz
data = mne.filter.filter_data(data.T, sfreq=1000, l_freq=1, h_freq=30, verbose=True).T


# Perform ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(data)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
ica.plot_properties(data, picks=ica.exclude)


### Pack avobe code into a function ###
def preprocess(data):
    data = mne.filter.filter_data(data.T, sfreq=1000, l_freq=1, h_freq=30, verbose=True).T
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(data)
    ica.exclude = [1, 2]  # details on how we picked these are omitted here
    ica.plot_properties(data, picks=ica.exclude)
    return data



