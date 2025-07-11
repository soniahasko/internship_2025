import Dans_Diffraction as dif
import Dans_Diffraction.functions_scattering as fs
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import csv
from tqdm import tqdm
import xarray as xr
from zipfile import ZipFile, ZIP_DEFLATED

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

cif_list = ['NaCl_cubic', 'BaSO4_orthorhombic']
cif_files_lst = [f'cif_files/{f}.cif' for f in cif_list]

crystals = [dif.Crystal(cif_file) for cif_file in cif_files_lst]
multi_xtl = dif.MultiCrystal(crystals)

# Set up scattering parameters
energy_kev = 12.398 / 0.1665
min_twotheta = 1
max_twotheta = 10
scattering_type = 'xray'
num_patterns = 1

# Set up scatter
multi_xtl.setup_scatter(
    scattering_type=scattering_type, 
    powder_units='twotheta', 
    energy_kev=energy_kev,
    min_twotheta=min_twotheta,
    max_twotheta=max_twotheta,
    output=False,
    powder_lorentz=1
)

# Vary lattice parameters
all_lps_nacl = np.linspace(crystals[0].Cell.lp(), crystals[0].Cell.lp()*np.array([2,2,2,1,1,1]), num_patterns)
all_lps_baso4 = np.linspace(crystals[1].Cell.lp(), crystals[1].Cell.lp()*np.array([2,2,2,1,1,1]), num_patterns)

all_lps = np.array((all_lps_nacl, all_lps_baso4))

# Make lists to hold data
tths = []
intensities = []
all_reflections = []

# Iter thru patterns for all crystals

for i in tqdm(range(num_patterns), desc="Simulating multiphase patterns"):
    # Set lattice for each crystal
    for num, xtl in enumerate(multi_xtl.crystal_list):
        xtl.Cell.latt(all_lps[num][i])

    # Initialize arrays for summing
    combined_intensity = None
    combined_tth = None
    combined_refl = []

    for xtl in multi_xtl.crystal_list:
        tth, inten, refl = xtl.Scatter.powder(return_hkl=True)

        if combined_intensity is None:
            combined_tth = tth
            combined_intensity = inten
        else:
            combined_intensity += inten  # sum intensities across phases

        combined_refl.extend(refl)

    tths.append(combined_tth)
    intensities.append(combined_intensity)
    all_reflections.append(combined_refl)

# Turn lists to arrays
tths = np.array(tths)
intensities = np.array(intensities)

# Prepare binary arrays for indexing peaks
binary_peaks = []
tol = 0.1
for i in range(num_patterns):
    peaks = np.zeros_like(tths[i], dtype=int)
    for ref in all_reflections[i]:
        tth_val = ref[3]
        idx = np.argmin(np.abs(tths[i] - tth_val))
        if np.abs(tths[i][idx] - tth_val) < tol:
            peaks[idx-5:idx+8] = 1
    binary_peaks.append(peaks)

binary_peaks = np.array(binary_peaks)

noisy_intensities = add_noise_by_percentage(intensities, 0.05)

ds = xr.Dataset(
    {
        "Intensities": (["pattern", "x"], noisy_intensities),
        "BinaryArr": (["pattern", "x"], binary_peaks)
    },
    coords={
        "pattern": np.arange(num_patterns),
        "x": np.linspace(min_twotheta, max_twotheta, noisy_intensities.shape[1])
    },
    attrs={
        "CIFs": cif_list,
        "description": "Multiphase powder diffraction simulated with MultiCrystal",
        "energy_kev": energy_kev
    }
)

def vis_first_three():
    plt.figure()
    intens = ds["Intensities"].values
    bins = ds["BinaryArr"].values
    x = ds["x"].values

    colors = ['purple', 'green', 'orange']

    for i in range(3):
        plt.plot(x, intens[i], color=colors[i], label=f'Pattern {i}')

    plt.legend()
    plt.show()

def vis_start_mid_end():
    plt.figure()

    intens = ds["Intensities"].values
    bins = ds["BinaryArr"].values
    x = ds["x"].values

    plt.plot(x, intens[0], color='purple', label='Pattern 0')
    plt.plot(x, intens[9], color='green', label='Pattern 9')
    plt.plot(x, intens[19], color='orange', label='Pattern 19')

    plt.legend()
    plt.show()


from zipfile import ZipFile, ZIP_DEFLATED
import os

path = 'saved_data/'
file = f'ds_multiphase_dans_nacl_baso4_1.nc'

ds.to_netcdf(os.path.join(path, file))
with ZipFile(os.path.join(path,file.replace('.nc','.zip')), 'w', ZIP_DEFLATED) as zObject:
    zObject.write(os.path.join(path,file), arcname=file)