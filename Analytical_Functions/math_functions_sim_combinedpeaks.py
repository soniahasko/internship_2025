import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os 
from zipfile import ZipFile, ZIP_DEFLATED

np.random.seed(42) # for reproducibility
num_repetitions = 10000
num_xs = 500
x = np.linspace(-5, 5, num_xs)

signals = np.zeros((num_repetitions, num_xs))
gauss_parameters = []












binary_peaks = np.zeros((num_repetitions, len(x)))

peak_idx = np.argmax(signals, axis=1)

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
file = 'math_functions.nc'

ds.to_netcdf(os.path.join(path, file))
with ZipFile(os.path.join(path,file.replace('.nc','.zip')), 'w', ZIP_DEFLATED) as zObject:
    zObject.write(os.path.join(path,file), arcname=file)

directory_path = f'saved_data/math_functions.nc'

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