import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import layers, models, Input
import xarray as xr
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

file = '/home/shasko/Desktop/internship_2025/evaluation_set/test_1_patterns_PSn_tetragonal_COD_peakslabeled_noisy.nc'

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




n_batch, n_timesteps, n_input_dim = 128, window_size, 1

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
model.load_weights('/home/shasko/Desktop/internship_2025/training_only_analytical_5/weights.weights.h5')

# Predict
predictions_new = model.predict(gaussians_new_reshaped)
print(f'predictions_new are {predictions_new}')


# Find f1 score after setting a threshold, using held-out test set
threshold = 0.5
binary_pred_adjusted_sklearn = (predictions_new >= threshold).astype(int)
test_binary_reshaped = binary_new_reshaped.astype(int)
f1 = f1_score(test_binary_reshaped.squeeze(), binary_pred_adjusted_sklearn.squeeze(), average='micro')
print(f1)

print(f'x_new.shape = {x_new.shape} and test_binary_reshaped.shape = {test_binary_reshaped.shape}')

binary_sequence = test_binary_reshaped[0, :, 0]  # shape (11837,)

# Step 2: Find indices where the value is 1
idx_ones = np.where(binary_sequence == 1)[0] 
x_new_ones = x_new[idx_ones]                     
binary_ones = binary_sequence[idx_ones]        

idx = 0

def vis():
    plt.figure(figsize=(10,8))
    plt.plot(x_new, predictions_new[idx], color='#00ADDC', label='Prediction Probabilities')

    plt.vlines(x_new_ones, 1.025, 1.050, label="True Peaks", linewidth=1.0, color='#B72467')

    # plt.plot(x_new, test_binary_reshaped[idx] * 0.05 + 1, color='green', label='True Peaks')
    plt.plot(x_new, gaussians_new_reshaped[idx] + 1.075, color='#B2D33B', label='Pattern')
    plt.legend(bbox_to_anchor=(1.01, 1.02), loc='upper left', fontsize=15)
    plt.tight_layout()
    # plt.savefig('/nsls2/users/shasko/Repos/internship_2025/saved_figures/gaussian_jul2_idx12')
    plt.show()

vis()