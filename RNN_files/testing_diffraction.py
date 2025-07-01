import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(".."))

# Importing needed libraries
from matplotlib import pyplot as plt
from RNN_files import Laitala_data_original_file
import wfdb
from wfdb.io import get_record_list
from wfdb import rdsamp
import tensorflow as tf
from tensorflow.keras import layers
from scipy.signal import resample_poly
from RNN_files import Laitala_data_original_file
import tensorflow
from tensorflow.keras import layers, models, Input
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from wfdb.processing import (
    resample_singlechan,
    find_local_peaks,
    correct_peaks,
    normalize_bound
)

diffraction_sim = False
gauss_sim = True
visualize_example = False

if diffraction_sim:
    path = '/nsls2/users/shasko/Repos/internship_2025/datasets/ds_combined_500_patterns_NaCl.nc'
    ds = xr.open_dataset(path, engine="netcdf4")
    gaussians = ds["Intensities"]
    binary = ds["binary_arr"]
    x = ds["tth"].values
    window_size = 11753

if gauss_sim:
    path = '/nsls2/users/shasko/Repos/internship_2025/datasets/math_functions_smalldataset.nc' 
    ds = xr.open_dataset(path, engine="netcdf4")

    gaussians = ds["Gaussians"].values
    binary = ds["BinaryArr"].values
    print(type(binary))

    x = ds["x"].values
    window_size = 990

# Train-val, test split
tv_gaussians, test_gaussians, tv_binary, test_binary = train_test_split(gaussians, binary, test_size=0.2, shuffle=False)

# Train, val split
train_gaussians, val_gaussians, train_binary, val_binary = train_test_split(tv_gaussians, tv_binary, test_size=0.25, shuffle=False)

# Make lists from train, val, test data
gauss_signals_train = [signal for signal in train_gaussians]
gauss_signals_val = [signal for signal in val_gaussians]
gauss_signals_test = [signal for signal in test_gaussians]

binary_labels_train = [label for label in train_binary]
binary_labels_val = [label for label in val_binary]
binary_labels_test = [label for label in test_binary]

# Data generator adapted from Laitala et al.
def data_generator(signals, labels, win_size, batch_size):
   
    while True:
        X, y = [], []

        while len(X) < batch_size:
            i = np.random.randint(0, len(signals))
            sig = signals[i]
            lbl = labels[i]

            if len(sig) <= win_size + 4:
                continue  # skip short signals

            start = np.random.randint(2, len(sig) - win_size - 2)
            end = start + win_size

            data_win = sig[start:end]
            label_win = lbl[start:end]

            # Pad 1s ±2 samples around every 1 in label_win
            padded_label = label_win.copy()
            ones = np.where(label_win == 1)[0]
            for p in ones:
                for offset in [-2, -1, 1, 2]:
                    if 0 <= p + offset < win_size:
                        padded_label[p + offset] = 1

            # Normalize signal window to (0, 1)
            # data_win = normalize_bound(data_win, lb=0, ub=1)

            # Normalized locally
            data_win = (data_win - np.min(data_win))/(np.max(data_win) - np.min(data_win))

            # Normalized globally
            # low = np.min(signals)
            # high = np.max(signals)
            # data_win = (data_win - low)/(high - low)

            X.append(data_win)
            y.append(padded_label)

        X = np.array(X).reshape(batch_size, win_size, 1)
        y = np.array(y).reshape(batch_size, win_size, 1).astype(int)

        yield X, y

if visualize_example:
    gen = data_generator(gauss_signals_train, binary_labels_train, win_size=window_size, batch_size=64)
    X_batch, y_batch = next(gen)
    n = next(gen)

    # Plot 4 training examples with labels
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(10), fig.set_figwidth(18)
    fig.suptitle('Some examples of training data with labels', size=20)

    # first index refers to whether it's an input or label, so noisy signal is input and peaks are labels
    axs[0, 0].plot(n[0][0], color='orange')
    axs[0, 0].plot(n[1][0]+1, color='green')

    axs[0, 1].plot(n[0][1], color='purple')
    axs[0, 1].plot(n[1][1]+1, color='blue')

    axs[1, 0].plot(n[0][2], color='yellow')
    axs[1, 0].plot(n[1][2]+1, color='red')

    axs[1, 1].plot(n[0][3], color='black')
    axs[1, 1].plot(n[1][3]+1, color='grey')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()


n_batch, n_timesteps, n_input_dim = 64, window_size, 1

model = models.Sequential() # initialize model
model.add(Input(shape=(n_timesteps, n_input_dim)))
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
model.add(layers.Dense(1, activation='sigmoid'))

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

# Generate training, validation, and testing data
training_gen = data_generator(gauss_signals_train, binary_labels_train, win_size=n_timesteps, batch_size=n_batch)
validation_gen = data_generator(gauss_signals_val, binary_labels_val, win_size=window_size, batch_size=n_batch)
testing_gen = data_generator(gauss_signals_test, binary_labels_test, win_size=n_timesteps, batch_size=n_batch)

# Train model
model.fit(training_gen,
          steps_per_epoch=40,
          epochs=20, 
          validation_data=validation_gen,
          validation_steps=10,
          callbacks=[cp_callback, es_callback])

gauss_test, label_test = next(testing_gen)

binary_pred = model.predict(gauss_test) # Make predictions

plt.figure()

for j in range(1):
    plt.plot(gauss_test[j], color='green', label='Signal')
    plt.plot(binary_pred[j] + 1, color='orange', label='Prediction Probabilities')
    plt.plot(label_test[j] + 2, color='purple', label='True Labels')

plt.tick_params(top=True, bottom=True, left=True, right=True, direction='in')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
# plt.savefig('test_results_multigauss1.png')
plt.show()
