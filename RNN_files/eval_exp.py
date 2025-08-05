import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import layers, models, Input
import xarray as xr
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import csv

filename_str = sys.argv[1]
num_weights = sys.argv[2]

def return_data(filename):
    path = '/home/shasko/Downloads/patterns_for_sonia'

    file_map = {
        'LaB6_brac': ['LaB6_brac1_xrd_calib_20250720-183936_bdf715_primary-1_mean_tth.chi'],
        'CeO2_std_avg': ['xrd_CeO2_std_brac1_20250720-194639_811d45_primary-1_mean_tth.chi',
                         'xrd_CeO2_std_brac1_20250721-001045_b26be7_primary-1_mean_tth.chi',
                         'xrd_CeO2_std_brac1_20250721-043453_3a683f_primary-1_mean_tth.chi'],
        'empty_kapton_avg': ['xrd_empty_kapton_brac1_20250720-194025_5e1923_primary-1_mean_tth.chi',
                        'xrd_empty_kapton_brac1_20250721-000431_bff40a_primary-1_mean_tth.chi',
                        'xrd_empty_kapton_brac1_20250721-042839_b15c4c_primary-1_mean_tth.chi'],
        'LaB6_660c_avg': ['xrd_LaB6_660c_std_brac1_20250720-194230_f86ca9_primary-1_mean_tth.chi',
                 'xrd_LaB6_660c_std_brac1_20250721-000636_f3c480_primary-1_mean_tth.chi',
                 'xrd_LaB6_660c_std_brac1_20250721-043043_030b08_primary-1_mean_tth.chi', 
                 'xrd_LaB6_660c_std_brac2_20250720-205807_6c9c36_primary-1_mean_tth.chi',
                 'xrd_LaB6_660c_std_brac2_20250721-012210_6fd6a8_primary-1_mean_tth.chi', 
                 'xrd_LaB6_660c_std_brac2_20250721-054620_e845a0_primary-1_mean_tth.chi'],
        'Ni_avg': ['xrd_Ni_std_brac1_20250720-194434_7d7ebd_primary-1_mean_tth.chi',
                 'xrd_Ni_std_brac1_20250721-000840_b6458b_primary-1_mean_tth.chi',
                 'xrd_Ni_std_brac1_20250721-043248_1fec02_primary-1_mean_tth.chi'],
        'LaB6_argonne': ['LaB6_from_Dan_rebinned.csv'],
        'MgWO10': ['xrd_10_LMT_MgW_O_10_20250725-041024_5788da_primary-1_mean_tth.chi'],
        'TiWO15': ['xrd_14_LMNb_TiW_O_15_20250725-041833_dbc9cd_primary-1_mean_tth.chi'],
        'GaNbO': ['xrd_25_LM_GaNb_O_20250725-044721_015154_primary-1_mean_tth.chi'],
        'CaAlCl': ['xrd_CaAlCl_MMO_CO2_20240627-160840_3a269a_primary-1_mean_tth.chi'],
        'CeO2_std': ['xrd_CeO2_std_20250724-184324_d9d30a_primary-1_mean_tth.chi'],
        'Cu_NPM_600': ['ExSituXRDdata/Cu_NPM_600Cannealed_XRD.txt'],
        'Cu_NPM_original': ['ExSituXRDdata/Cu_NPM_as-synthesized_XRD.txt'],
        'CuLiCl_NC_as_synth': ['ExSituXRDdata/CuLiCl_NC_as-synthesized_XRD.txt'],
        'CuLiCl_NC_300': ['ExSituXRDdata/CuLiCl_NC_300Cannealed_XRD.txt'],
        'CuLiCl_NC_600': ['ExSituXRDdata/CuLiCl_NC_600Cannealed_XRD.txt'],
        'Fe_NPM_as_synth': ['ExSituXRDdata/Fe_NPM_as-synthesized_XRD.xye.txt'],
        'Fe_NPM_300': ['ExSituXRDdata/Fe_NPM_300Cannealed_XRD.txt'],
        'Fe_NPM_600': ['ExSituXRDdata/Fe_NPM_600Cannealed_XRD.txt'],
        'FeLiCl_NC_as_synth': ['ExSituXRDdata/FeLiCl_NC_as-synthesized_XRD.txt'],
        'FeLiCl_NC_300': ['ExSituXRDdata/FeLiCl_NC_300Cannealed_XRD.txt'],
        'FeLiCl_NC_600': ['ExSituXRDdata/FeLiCl_NC_600Cannealed_XRD.txt']
    }

    if filename not in file_map:
        raise ValueError(f'{filename} not found in file map')
    
    files = [os.path.join(path, f) for f in file_map[filename]]
    
    if filename == 'LaB6_argonne':  # Load in csvs
        df = pd.read_csv(files[0])
        intens = [df['y'].values]
        tth = [df['x_rescaled'].values]
 
    else:  # generic loader for chi files
        intens = [pd.read_csv(f, delim_whitespace=True, header=None, skiprows=1)[1].values for f in files]
        tth = [pd.read_csv(f, delim_whitespace=True, header=None, skiprows=1)[0].values for f in files]

    return intens, tth

intens, tth = return_data(filename_str)

tth_exp_unpadded = np.mean(tth, axis=0)
inten_exp_unpadded = np.mean(intens, axis=0)

tth_exp_unpadded = np.array(tth_exp_unpadded)
inten_exp_unpadded = np.array(inten_exp_unpadded)

inten_exp = np.zeros((11837, ))

tth_exp = np.linspace(1,10,11837)
print(inten_exp_unpadded.shape)

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

# save the predictions to a npy file or a csv


def vis_subplots():

    fig, axs = plt.subplots(ncols=1,nrows=2, figsize=(24,14), sharex=True)

    for num, ax in enumerate(axs.flat):

        # linear vs sqrt scale
        if num == 0:
            ax.plot(tth_exp, inten_exp_reshaped[0], color='#25B574', label=f'Pattern for {filename_str}')
        # elif num == 1:
        #     ax.plot(tth_exp, np.sqrt(inten_exp_reshaped[0]), color='#25B574', label=f'Sqrt Pattern')
        elif num == 1:
            ax.plot(tth_exp, predictions[0], color='#00ADDC', label='Prediction Probabilities')

        if num in [0]:
            ax.set_ylim(-0.08,1.04)
        elif num in [1]:
            ax.set_ylim(0, 1.04)
        ax.legend()
        
        if filename_str == 'LaB6_argonne':
            ax.set_xlim(1,10)
        else:
            ax.set_xlim(1,5)

    # plt.savefig(f'saved_figures/avg_{filename_str}_exp_trainingweights_{num_weights}.png')
    plt.show()

vis_subplots()

# make function that takes out the padding for the predictions and for vis - strip out everything that is irrelevant after making predictions