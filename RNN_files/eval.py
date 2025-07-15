import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import layers, models, Input
import xarray as xr
from sklearn.metrics import f1_score

file = '/home/shasko/Desktop/internship_2025/evaluation_set/ds_combined_20_patterns_CsCl_cubic_width_peakslabeled_noisy.nc'

# List comprehension to get all path names
ds = xr.open_dataset(file, engine='netcdf4')

window_size = ds["x"].shape[0]
gaussians_new = ds["Intensities"]
binary_new = ds["BinaryArr"]
x_new = ds["x"].values

# Scale the data
gaussians_new_sc = np.zeros_like(gaussians_new)

for j in range(gaussians_new.shape[0]):
    max_inten = np.max(gaussians_new[j])
    min_inten = np.min(gaussians_new[j])
    gaussians_new_sc[j] = (gaussians_new[j] - min_inten) / (max_inten - min_inten)

gaussians_new_reshaped = gaussians_new_sc.reshape(gaussians_new_sc.shape[0], gaussians_new_sc.shape[1], 1)

binary_new = np.array(binary_new)
binary_new_reshaped = binary_new.reshape(binary_new.shape[0], binary_new.shape[1], 1)




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
model.load_weights('training_5/weights.weights.h5')






# Predict
predictions_new = model.predict(gaussians_new_reshaped)
print(f'predictions_new are {predictions_new}')


# Find f1 score after setting a threshold, using held-out test set
threshold = 0.5
binary_pred_adjusted_sklearn = (predictions_new >= threshold).astype(int)
test_binary_reshaped = binary_new_reshaped.astype(int)
f1 = f1_score(test_binary_reshaped.squeeze(), binary_pred_adjusted_sklearn.squeeze(), average='micro')
print(f1)