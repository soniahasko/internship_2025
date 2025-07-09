# Add the parent directory to sys.path
sys.path.append(os.path.abspath(".."))

# Importing necessary libraries
import sys
import os
from matplotlib import pyplot as plt
from RNN_files import Laitala_data_original_file
import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from scipy.signal import resample_poly
import xarray as xr
import matplotlib.pyplot as plt
from RNN_files import Laitala_data_original_file
import tensorflow
from tensorflow.keras import layers, models, Input
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import csv

path = '/nsls2/users/shasko/Repos/internship_2025/saved_data/'
filenames = ['lorentzian_functions_smalldataset_noisy_11763',
             'gaussian_functions_smalldataset_varying_amps_noisy_11763',
             'psuedovoigt_functions_smalldataset_noisy_11763',
             'ds_combined_100_patterns_NaCl_cubic_width_peakslabeled_noisy'
             ]

# List comprehension to get all path names
full_paths = [f'{path}{i}.nc' for i in filenames]
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

n_batch, n_timesteps, n_input_dim = 64, window_size, 1

model = models.Sequential()
model.add(Input(shape=(n_timesteps, n_input_dim)))
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Create callback to save model weights
checkpoint_path = 'training_5/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create callback to save model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=2) 

es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                              patience=3,
                                              min_delta=1e-5,
                                              verbose=2)

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
          batch_size=64,
          epochs=20, 
          validation_data=(val_gaussians_reshaped, val_binary_reshaped),
          callbacks=[cp_callback, es_callback])

# Make predictions on held-out test set
binary_pred = model.predict(test_gaussians_reshaped,
                            verbose=2)

# Find f1 score after setting a threshold, using held-out test set
threshold = 0.5
binary_pred_adjusted_sklearn = (binary_pred >= threshold).astype(int)
test_binary_reshaped = test_binary_reshaped.astype(int)
f1 = f1_score(test_binary_reshaped.squeeze(), binary_pred_adjusted_sklearn.squeeze(), average='micro')
print(f1)

# Save data for further analysis
ds_with_results = xr.Dataset(
    {
        "true_y": (("x"), test_binary_reshaped),
        "predicted_y": (("x"), binary_pred),
        "test_intensities": (("x"), test_gaussians_sc)
    },
    coords={
        "x": x
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

ds_with_results.to_netcdf("/nsls2/users/shasko/Repos/internship_2025/saved_data/saved_results.nc")