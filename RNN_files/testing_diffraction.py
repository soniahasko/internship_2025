# Add the parent directory to sys.path
sys.path.append(os.path.abspath(".."))

# Importing necessary libraries
import sys
import os
from matplotlib import pyplot as plt
from RNN_files import Laitala_data_original_file
import wfdb
from wfdb.io import get_record_list
from wfdb import rdsamp
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

diffraction_sim = False
gauss_sim = True
visualize_example = False

if diffraction_sim:
    path = '/nsls2/users/shasko/Repos/internship_2025/datasets/ds_combined_500_patterns_NaCl.nc'
    ds = xr.open_dataset(path, engine="netcdf4")
    gaussians = ds["Intensities"].values
    binary = ds["binary_arr"].values
    x = ds["tth"].values

    window_size = 11763

if gauss_sim:
    path = '/nsls2/users/shasko/Repos/internship_2025/saved_data/gaussian_functions_smalldataset_varying_amps.nc' 
    ds = xr.open_dataset(path, engine="netcdf4")

    gaussians = ds["Gaussians"].values
    binary = ds["BinaryArr"].values
    print(type(binary))

    x = ds["x"].values
    window_size = 1000


# Pad peaks with two 1s on either side
for j in range(binary.shape[0]):
    idx = np.where(binary[j] == 1)[0][0] # because there's only 1 "1" we can use [0][0]
    binary[j][idx - 2] = 1
    binary[j][idx - 1] = 1
    binary[j][idx + 1] = 1
    binary[j][idx + 2] = 1

# Train-val, test split
tv_gaussians, test_gaussians, tv_binary, test_binary = train_test_split(gaussians, binary, test_size=0.2, shuffle=False)

# Train, val split
train_gaussians, val_gaussians, train_binary, val_binary = train_test_split(tv_gaussians, tv_binary, test_size=0.25, shuffle=False)

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

model = models.Sequential() # initialize model
model.add(Input(shape=(n_timesteps, n_input_dim)))
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# Create callback to save model weights
checkpoint_path = 'training/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=2) 

# Create callback to implement earliy stopping
es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                              patience=3,
                                              verbose=2)

# Reshape intensity data
train_gaussians_reshaped = train_gaussians_sc.reshape(train_gaussians_sc.shape[0], train_gaussians_sc.shape[1], 1)
val_gaussians_reshaped = val_gaussians_sc.reshape(val_gaussians_sc.shape[0], val_gaussians_sc.shape[1], 1)
test_gaussians_reshaped = test_gaussians_sc.reshape(test_gaussians_sc.shape[0], test_gaussians_sc.shape[1], 1)

# Reshape labels
train_binary_reshaped = train_binary.reshape(train_binary.shape[0], train_binary.shape[1], 1)
val_binary_reshaped = val_binary.reshape(val_binary.shape[0], val_binary.shape[1], 1)
test_binary_reshaped = test_binary.reshape(test_binary.shape[0], test_binary.shape[1], 1)

# Train model
model.fit(x=train_gaussians_reshaped,
          y=train_binary_reshaped,
          batch_size=64,
          epochs=40, 
          validation_data=(val_gaussians_reshaped, val_binary_reshaped),
          callbacks=[cp_callback, es_callback])


# Make predictions on held-out test set
binary_pred = model.predict(test_gaussians_reshaped,
                            verbose=2)

def f1_w_threshold(threshold, test, pred):
    pred_adjusted = (pred >= threshold).astype(int)
    test = test.astype(int)
    f1 = f1_score(test.squeeze(), pred_adjusted.squeeze(), average='micro')
    return f1

f1_gauss = f1_w_threshold(threshold=0.3, test=test_binary_reshaped, pred=binary_pred)
print(f'f1 for gaussian test set is {f1_gauss}')

# Evaulate on non-Gauss data. In this case, Lorentzian signals.
path = '/nsls2/users/shasko/Repos/internship_2025/saved_data/lorentzian_functions_smalldataset.nc' 
ds = xr.open_dataset(path, engine="netcdf4")

intensities_lor = ds["Intensities"].values
binary = ds["BinaryArr"].values
x = ds["x"].values
window_size = 1000

# Scale signals
intensities_lor_sc = np.zeros_like(intensities_lor)

for j in range(intensities_lor.shape[0]):
    max_inten = np.max(intensities_lor[j])
    min_inten = np.min(intensities_lor[j])
    intensities_lor_sc[j] = (intensities_lor[j] - min_inten) / (max_inten - min_inten)

# Reshape intensities
intensities_lor_sc_reshaped = intensities_lor_sc.reshape(intensities_lor_sc.shape[0],
                                                         intensities_lor_sc.shape[1],
                                                         1)

# Reshape labels
binary_lor_reshaped = binary.reshape(binary.shape[0],
                                 binary.shape[1],
                                 1)

# Make prediction
pred_lor = model.predict(intensities_lor_sc_reshaped,
                         verbose=2)

f1_lor = f1_w_threshold(threshold=0.3, test=binary_lor_reshaped, pred=pred_lor)
print(f'f1 for lorentzian test set is {f1_lor}')


# Evaulate on non-Gauss data. In this case, cubic NaCl diffraction signals.
path = '/nsls2/users/shasko/Repos/internship_2025/datasets/ds_combined_500_patterns_NaCl.nc'
ds = xr.open_dataset(path, engine="netcdf4")
intensities_diff = ds["Intensities"].values
binary_diff = ds["binary_arr"].values
x = ds["tth"].values

window_size = 11763

# Scale signals
intensities_diff_sc = np.zeros_like(intensities_diff)

for j in range(intensities_diff.shape[0]):
    max_inten = np.max(intensities_diff[j])
    min_inten = np.min(intensities_diff[j])
    intensities_diff_sc[j] = (intensities_diff[j] - min_inten) / (max_inten - min_inten)

# Reshape signals and labels
intensities_diff_sc_reshaped = intensities_diff_sc.reshape(intensities_diff_sc.shape[0],
                                                           intensities_diff_sc.shape[1],
                                                           1)
binary_diff_reshaped = binary_diff.reshape(binary_diff.shape[0],
                                           binary_diff.shape[1],
                                           1)

# Make predictions
pred_diff = model.predict(intensities_diff_sc_reshaped,
                         verbose=2)

f1_diff = f1_w_threshold(threshold=0.3, test=binary_diff_reshaped, pred=pred_diff)
print(f'f1 for lorentzian test set is {f1_diff}')
