import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os 
from zipfile import ZipFile, ZIP_DEFLATED

np.random.seed(42) # for reproducibility
num_repetitions = 20000
x = np.linspace(-10, 10, 50)
all_gauss = np.zeros((num_repetitions, len(x)))

def make_gauss(x, amp, mean, std_dev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

for i in range(num_repetitions):
    amp = np.random.uniform(0.5, 1.5)
    mean = np.random.uniform(-6, 6)
    std_dev = np.random.uniform(0.2,2.0)
    all_gauss[i] = make_gauss(x, amp, mean, std_dev)

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