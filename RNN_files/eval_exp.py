import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import layers, models, Input
import xarray as xr
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import sys, os
import pandas as pd


def return_data(filename):
    if filename == 'LaB6_brac':
        files_wo_path = [
                'LaB6_brac1_xrd_calib_20250720-183936_bdf715_primary-1_mean_tth.chi',
                ]
        path = '/home/shasko/Downloads/standard_patterns_for_Sonia'
        files = [f'{path}/{filename}' for filename in files_wo_path]
        intens = [pd.read_csv(os.path.join(file), delimiter='\s+', header=None, skiprows=1)[1].values for file in files]
        tth = [pd.read_csv(os.path.join(file), delimiter='\s+', header=None, skiprows=1)[0].values for file in files]

    elif filename == 'CeO2':
        files_wo_path = [
                 'xrd_CeO2_std_brac1_20250720-194639_811d45_primary-1_mean_tth.chi',
                 'xrd_CeO2_std_brac1_20250721-001045_b26be7_primary-1_mean_tth.chi',
                 'xrd_CeO2_std_brac1_20250721-043453_3a683f_primary-1_mean_tth.chi'
                ]
        path = '/home/shasko/Downloads/standard_patterns_for_Sonia/xrd_jogged'
        files = [f'{path}/{filename}' for filename in files_wo_path]
        intens = [pd.read_csv(os.path.join(file), delimiter='\s+', header=None, skiprows=1)[1].values for file in files]
        tth = [pd.read_csv(os.path.join(file), delimiter='\s+', header=None, skiprows=1)[0].values for file in files]

    elif filename == 'empty_kapton':
        files_wo_path = [
                 'xrd_empty_kapton_brac1_20250720-194025_5e1923_primary-1_mean_tth.chi',
                 'xrd_empty_kapton_brac1_20250721-000431_bff40a_primary-1_mean_tth.chi',
                 'xrd_empty_kapton_brac1_20250721-042839_b15c4c_primary-1_mean_tth.chi'
                ]
        path = '/home/shasko/Downloads/standard_patterns_for_Sonia/xrd_jogged'
        files = [f'{path}/{filename}' for filename in files_wo_path]
        intens = [pd.read_csv(os.path.join(file), delimiter='\s+', header=None, skiprows=1)[1].values for file in files]
        tth = [pd.read_csv(os.path.join(file), delimiter='\s+', header=None, skiprows=1)[0].values for file in files]

    elif filename == 'LaB6_660c':
        files_wo_path = [
                 'xrd_LaB6_660c_std_brac1_20250720-194230_f86ca9_primary-1_mean_tth.chi',
                 'xrd_LaB6_660c_std_brac1_20250721-000636_f3c480_primary-1_mean_tth.chi',
                 'xrd_LaB6_660c_std_brac1_20250721-043043_030b08_primary-1_mean_tth.chi', 
                 'xrd_LaB6_660c_std_brac2_20250720-205807_6c9c36_primary-1_mean_tth.chi',
                 'xrd_LaB6_660c_std_brac2_20250721-012210_6fd6a8_primary-1_mean_tth.chi', 
                 'xrd_LaB6_660c_std_brac2_20250721-054620_e845a0_primary-1_mean_tth.chi'
                ]
        path = '/home/shasko/Downloads/standard_patterns_for_Sonia/xrd_jogged'
        files = [f'{path}/{filename}' for filename in files_wo_path]
        intens = [pd.read_csv(os.path.join(file), delimiter='\s+', header=None, skiprows=1)[1].values for file in files]
        tth = [pd.read_csv(os.path.join(file), delimiter='\s+', header=None, skiprows=1)[0].values for file in files]
    
    elif filename == 'Ni':
        files = ['/home/shasko/Downloads/standard_patterns_for_Sonia/xrd_jogged/xrd_Ni_std_brac1_20250720-194434_7d7ebd_primary-1_mean_tth.chi',
                 '/home/shasko/Downloads/standard_patterns_for_Sonia/xrd_jogged/xrd_Ni_std_brac1_20250721-000840_b6458b_primary-1_mean_tth.chi',
                 '/home/shasko/Downloads/standard_patterns_for_Sonia/xrd_jogged/xrd_Ni_std_brac1_20250721-043248_1fec02_primary-1_mean_tth.chi']

        intens = [pd.read_csv(os.path.join(file), delimiter='\s+', header=None, skiprows=1)[1].values for file in files]
        tth = [pd.read_csv(os.path.join(file), delimiter='\s+', header=None, skiprows=1)[0].values for file in files]
    
    return intens, tth

filename_str = 'CeO2'
num_weights = 5 # training weights file to load in 
intens, tth = return_data(filename_str)

tth_exp_unpadded = np.mean(tth, axis=0)
inten_exp_unpadded = np.mean(intens, axis=0)

tth_exp_unpadded = np.array(tth_exp_unpadded)
inten_exp_unpadded = np.array(inten_exp_unpadded)

inten_exp = np.zeros((11837, ))

tth_exp = np.linspace(1,10,11837)

for i in range(inten_exp_unpadded.shape[0]):
    inten_exp[i] = inten_exp_unpadded[i]
inten_exp = inten_exp.reshape(1, inten_exp.shape[0])

window_size = tth_exp.shape[0]

# Scale the data
inten_exp_sc = np.zeros_like(inten_exp)

for j in range(inten_exp.shape[0]):
    max_inten = np.max(inten_exp[j])
    min_inten = np.min(inten_exp[j])
    inten_exp_sc[j] = (inten_exp[j] - min_inten) / (max_inten - min_inten)

inten_exp_reshaped = inten_exp_sc.reshape(inten_exp_sc.shape[0], inten_exp_sc.shape[1], 1)

# binary_new = np.array(binary_new)
# binary_new_reshaped = binary_new.reshape(binary_new.shape[0], binary_new.shape[1], 1)

n_batch, n_timesteps, n_input_dim = 64, window_size, 1

def build_model():
    model = models.Sequential()
    model.add(Input(shape=(n_timesteps, n_input_dim)))
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Create model instance
model = build_model()

# Load saved weights
model.load_weights(f'/home/shasko/Desktop/internship_2025/training_only_analytical_{num_weights}/weights.weights.h5')

# Predict
predictions = model.predict(inten_exp_reshaped)
print(f'predictions are {predictions}')

def vis_subplots():

    fig, axs = plt.subplots(ncols=2,nrows=2, figsize=(24,14), sharex=True)

    for num, ax in enumerate(axs.flat):

        # linear vs sqrt scale
        if num == 0:
            ax.plot(tth_exp, inten_exp_reshaped[0], color='orange', label=f'Pattern for {filename_str}')
        elif num == 1:
            ax.plot(tth_exp, np.sqrt(inten_exp_reshaped[0]), color='orange', label=f'Sqrt Pattern')
        elif num in [2,3]:
            ax.plot(tth_exp, predictions[0], color='purple', label='Prediction Probabilities')

        if num in [0,2]:
            ax.set_ylim(0,1)
        elif num in [3]:
            ax.set_ylim(0,0.1)

        ax.set_xlim(0,5)
        ax.legend(loc='upper left')

    # plt.savefig(f'saved_figures/avg_{filename_str}_exp_trainingweights_{num_weights}.png')
    plt.show()

vis_subplots()