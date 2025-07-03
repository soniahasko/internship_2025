import numpy as np
import xarray as xr
import os, sys
from zipfile import ZipFile, ZIP_DEFLATED

name = 'lorentzian_functions_smalldataset'
file=f'/nsls2/users/shasko/Repos/internship_2025/saved_data/{name}.nc'
ds = xr.open_dataset(file)

gaussians = ds["Intensities"].values
binary = ds["BinaryArr"].values
x = ds["x"].values
pattern = ds["pattern"].values

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

noisy_signal = add_noise_by_percentage(gaussians, 1)

ds_noisy = xr.Dataset(
    {
        "Gaussians": (["pattern", "x"], noisy_signal),
        "BinaryArr": (["pattern", "x"], binary)
    },
    coords = {
        "pattern": pattern,
        "x": x
    }
)

path = '/nsls2/users/shasko/Repos/internship_2025/saved_data/'
file = f'{name}.nc'


ds.to_netcdf(os.path.join(path, file))
with ZipFile(os.path.join(path,file.replace('.nc','.zip')), 'w', ZIP_DEFLATED) as zObject:
    zObject.write(os.path.join(path,file), arcname=file)