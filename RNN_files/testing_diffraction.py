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

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(".."))

path = '/home/shasko/Desktop/internship_2025/'
filenames = [
             'saved_data/compare_G_small_2000.nc',
             'saved_data/compare_L_small_2000.nc',
             'saved_data/compare_pv_small_2000.nc',
             'saved_data/compare_G_medium_2000.nc',
             'saved_data/compare_L_medium_2000.nc',
             'saved_data/compare_pv_medium_2000.nc',
             'saved_data/compare_G_large_2000.nc',
             'saved_data/compare_L_large_2000.nc',
             'saved_data/compare_pv_large_2000.nc',
             'saved_data/compare_G_very_large_2000.nc',
             'saved_data/compare_L_very_large_2000.nc',
             'saved_data/compare_pv_very_large_2000.nc'
             ]
trial = 9
             
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

n_timesteps, n_batch, n_input_dim = window_size, 64, 1

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
          batch_size=n_batch,
          epochs=25, 
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