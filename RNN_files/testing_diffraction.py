# Importing necessary libraries
import sys
import os
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from scipy.signal import resample_poly
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras import layers, models, Input
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import csv
import pandas as pd

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(".."))

path = '/home/shasko/Desktop/internship_2025/'
filenames = [
             'saved_data/compare_pv_small_2000.nc',
             'saved_data/compare_pv_medium_2000.nc',
             'saved_data/compare_pv_large_2000.nc',
             'saved_data/compare_pv_very_large_2000.nc',
             'saved_data/compare_L_small_2000.nc',
             'saved_data/compare_L_medium_2000.nc',
             'saved_data/compare_L_large_2000.nc',
             'saved_data/compare_L_very_large_2000.nc',
             'saved_data/compare_G_small_2000.nc',
             'saved_data/compare_G_medium_2000.nc',
             'saved_data/compare_G_large_2000.nc',
             'saved_data/compare_G_very_large_2000.nc'      
             ]
trial = f'test'
             
# List comprehension to get all path names
full_paths = [f'{path}{i}' for i in filenames]
all_datasets = [xr.open_dataset(p, engine='netcdf4') for p in full_paths] # list of all the Datasets

combined = xr.concat(all_datasets, dim="pattern")
window_size = combined["x"].shape[0]
gaussians = combined["Intensities"]
binary = combined["BinaryArr"]
x = combined["x"].values

# Train-val, test split

tv_gaussians, test_gaussians, tv_binary, test_binary = train_test_split(gaussians, binary, test_size=0.2, shuffle=True, random_state=42)
# Train, val split
train_gaussians, val_gaussians, train_binary, val_binary = train_test_split(tv_gaussians, tv_binary, test_size=0.25, shuffle=True, random_state=42)

# Scale the data
train_gaussians_sc = np.zeros_like(train_gaussians)
val_gaussians_sc = np.zeros_like(val_gaussians)
test_gaussians_sc = np.zeros_like(test_gaussians)

for j in range(train_gaussians.shape[0]):
    max_inten = np.max(train_gaussians[j])
    min_inten = np.min(train_gaussians[j])
    train_gaussians_sc[j] = (train_gaussians[j] - min_inten) / (max_inten - min_inten)

for j in range(val_gaussians.shape[0]):
    max_inten = np.max(val_gaussians[j])
    min_inten = np.min(val_gaussians[j])
    val_gaussians_sc[j] = (val_gaussians[j] - min_inten) / (max_inten - min_inten)

for j in range(test_gaussians.shape[0]):
    max_inten = np.max(test_gaussians[j])
    min_inten = np.min(test_gaussians[j])
    test_gaussians_sc[j] = (test_gaussians[j] - min_inten) / (max_inten - min_inten)

n_timesteps, batch_size, n_input_dim = window_size, 64, 1

model = models.Sequential()
model.add(Input(shape=(n_timesteps, n_input_dim)))
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Create callback to save model weights
checkpoint_path = f'training_only_analytical_{trial}/weights.weights.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create callback to save model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=2) 

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=5,
                                               min_delta=1e-4,
                                               verbose=2,
                                               restore_best_weights=True)

# Load in and read LaB6 data
def eval_LaB6():    
    files = ['/home/shasko/Downloads/patterns_for_sonia/LaB6_brac1_xrd_calib_20250720-183936_bdf715_primary-1_mean_tth.chi']
    intens = [pd.read_csv(f, delim_whitespace=True, header=None, skiprows=1)[1].values for f in files]
    tth = [pd.read_csv(f, delim_whitespace=True, header=None, skiprows=1)[0].values for f in files]

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

    return tth_exp, inten_exp_reshaped

LaB6_tth_exp, LaB6_inten_exp_reshaped = eval_LaB6()

peak_x_values = [
    [1.353, 1.357], [1.545, 1.550], [1.693, 1.685], [1.818, 1.823],
    [1.9285, 1.933], [2.0275, 2.0340], [2.204, 2.209], [2.285, 2.289],
    [2.360, 2.365], [2.432, 2.437], [2.501, 2.506], [2.567, 2.572],
    [2.6305, 2.6355], [2.751, 2.756], [2.809, 2.8135], [2.865, 2.869],
    [2.919, 2.924], [2.972, 2.977], [3.024, 3.028], [3.074, 3.079],
    [3.171, 3.177], [3.219, 3.224], [3.265, 3.270], [3.311, 3.316],
    [3.4, 3.404], [3.442, 3.448], [3.526, 3.532], [3.567, 3.574],
    [3.608, 3.614], [3.648, 3.654], [3.687, 3.694], [3.726, 3.7325],
    [3.764, 3.771], [3.839, 3.847], [3.876, 3.883], [3.912, 3.920],
    [3.949, 3.955], [3.984, 3.991], [4.020, 4.026], [4.054, 4.061]
]

# Ensure each range is sorted (to correct any flipped ranges)
peak_ranges = [sorted(rng) for rng in peak_x_values]

# Initialize labels
LaB6_true_labels = np.zeros_like(LaB6_tth_exp, dtype=int)

# Label all x values that fall within any of the peak ranges
for low, high in peak_ranges:
    mask = (LaB6_tth_exp >= low) & (LaB6_tth_exp <= high)
    LaB6_true_labels[mask] = 1

# Create callback to plot LaB6 predictions after every epoch
# lab6_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: (
    # plt.figure(),
    # plt.plot(LaB6_tth_exp, LaB6_inten_exp_reshaped[0] + 1.04, label='Pattern'),
    # plt.plot(LaB6_tth_exp, model.predict(LaB6_inten_exp_reshaped, verbose=0)[0], label='Pred'),
    # plt.legend(),
    # plt.show()
# )

def on_epoch_end(epoch, logs):
    pred = model.predict(LaB6_inten_exp_reshaped, verbose=0)[0]
    residual = np.abs(pred.flatten() - LaB6_true_labels) # take absolute values of the difference between each entry. residual should be + so sum of residuals will be +. big-small and small-big will be penalized based on magnitude, not sign
    print(residual.shape)
    print(f"\nEpoch {epoch+1}")
    print("Residual:", residual)
    print("Sum of residuals:", np.sum(residual))

lab6_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

# Reshape intensity data
train_gaussians_reshaped = train_gaussians_sc.reshape(train_gaussians_sc.shape[0], train_gaussians_sc.shape[1], 1)
val_gaussians_reshaped = val_gaussians_sc.reshape(val_gaussians_sc.shape[0], val_gaussians_sc.shape[1], 1)
test_gaussians_reshaped = test_gaussians_sc.reshape(test_gaussians_sc.shape[0], test_gaussians_sc.shape[1], 1)

# Reshape labels
train_binary = np.array(train_binary)
val_binary = np.array(val_binary)
test_binary = np.array(test_binary)

train_binary_reshaped = train_binary.reshape(train_binary.shape[0], train_binary.shape[1], 1)
val_binary_reshaped = val_binary.reshape(val_binary.shape[0], val_binary.shape[1], 1)
test_binary_reshaped = test_binary.reshape(test_binary.shape[0], test_binary.shape[1], 1)

# Train model
model.fit(x=train_gaussians_reshaped,
          y=train_binary_reshaped,
          batch_size=batch_size, # DECREASE the batch size
          epochs=75, 
          validation_data=(val_gaussians_reshaped, val_binary_reshaped),
          callbacks=[cp_callback, es_callback, lab6_callback])

# Make predictions on held-out test set
binary_pred = model.predict(test_gaussians_reshaped,
                            verbose=2)

# Find f1 score after setting a threshold, using held-out test set
threshold = 0.5
binary_pred_adjusted_sklearn = (binary_pred >= threshold).astype(int)
test_binary_reshaped = test_binary_reshaped.astype(int)
f1 = f1_score(test_binary_reshaped.squeeze(), binary_pred_adjusted_sklearn.squeeze(), average='micro')
print(f1)

print(f'shape of test_binary_reshaped is {test_binary_reshaped.shape}')
print(f'shape of binary_pred is {binary_pred.shape}')
print(f'shape of test_gaussians_sc is {test_gaussians_sc.shape}')
# Save data for further analysis
ds_with_results = xr.Dataset(
    {
        "true_y": (("sample", "x", "channel"), test_binary_reshaped),
        "predicted_y": (("sample", "x", "channel"), binary_pred),
        "test_intensities": (("sample", "x"), test_gaussians_sc)
    },
    coords={
        "x": x,
        "sample": np.arange(test_binary_reshaped.shape[0]),
        "channel": np.arange(1)
    },
    attrs={
        "checkpoint_filepath": checkpoint_path, 
        "filenames_training": filenames,
        "filenames_testing": filenames,
        "test_split": 0.20,
        "params": 'loss=binary_crossentropy, optimizer=adam',
        "f1 w thresh": f'f1={f1}, threshold={threshold}'
    }
)

ds_with_results.to_netcdf(f"/home/shasko/Desktop/internship_2025/saved_results_only_analytical_{trial}.nc")

# for each epoch plot how the model performs on LaB6