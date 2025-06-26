import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os 
from zipfile import ZipFile, ZIP_DEFLATED

np.random.seed(42) # for reproducibility
num_repetitions = 10000
x = np.linspace(-5, 5, 500)
all_gauss = np.zeros((num_repetitions, len(x)))

def make_gauss(x, amp, mean, std_dev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

def add_noise_by_percentage(signal, noise_percentage):
        """
        Add random noise to a NumPy array based on a percentage of the maximum signal value.

        Parameters:
            signal (numpy.ndarray): The original signal array.
            noise_percentage (float): Percentage of the maximum signal value to use as noise.

        Returns:
            noisy_signal (numpy.ndarray): The signal array with added noise.
        """
        # Find the maximum value in the signal
        max_value = np.max(np.abs(signal))

        # Calculate the noise standard deviation as a percentage of the max signal value
        noise_std = (noise_percentage / 100) * max_value

        # Generate random Gaussian noise with zero mean and calculated standard deviation
        noise = np.random.normal(0.005, noise_std, signal.shape)

        # Add the noise to the original signal
        noisy_signal = signal + noise

        return noisy_signal

for i in range(num_repetitions):
    amp = np.random.uniform(0.5, 1.5)
    mean = np.random.uniform(-3, 3)
    std_dev = np.random.uniform(0.005,0.01)
    signal = make_gauss(x, amp, mean, std_dev)

    all_gauss[i] = add_noise_by_percentage(signal, 0.2)

binary_peaks = np.zeros_like(all_gauss, dtype=int)

peak_idx = np.argmax(all_gauss, axis=1)

rows = np.arange(all_gauss.shape[0])

binary_peaks[rows, peak_idx] = 1

ds = xr.Dataset(
    {
        "Gaussians": (["pattern", "x"], all_gauss),
        "BinaryArr": (["pattern", "x"], binary_peaks)
    },
    coords = {
        "pattern": np.arange(num_repetitions),
        "x": x
    }
)

path = 'saved_data/'
file = 'math_functions_single_narrow_noisy.nc'


ds.to_netcdf(os.path.join(path, file))
with ZipFile(os.path.join(path,file.replace('.nc','.zip')), 'w', ZIP_DEFLATED) as zObject:
    zObject.write(os.path.join(path,file), arcname=file)

# directory_path = f'{path}{file}'

# # Load .nc file
# ds = xr.open_dataset(directory_path)

# gaussians = ds["Gaussians"].values
# binary = ds["BinaryArr"].values

# plt.figure()
# for i in range(5):
#     plt.plot(x, all_gauss[i])

# for i in range(5):
#     plt.scatter(x[peak_idx[i]], all_gauss[i][peak_idx[i]])

# plt.show()