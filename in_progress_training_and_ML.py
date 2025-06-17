# Import relevant libraries

import Dans_Diffraction as dif
import Dans_Diffraction.functions_scattering as fs
import numpy as np
import matplotlib.pyplot as plt
import sys
import xarray as xr
from zipfile import ZipFile, ZIP_DEFLATED
import os

cif_file = 'cif_files/NaCl_cubic.cif'
xtl = dif.Crystal(cif_file)

orig_lps = xtl.Cell.lp() # starting lattice parameters
lp_multiplier = (2,2,2,1,1,1) # separate multiplier for cell prms
max_lps = np.array(orig_lps) * np.array(lp_multiplier) # max lp_a, lp_b, lp_c, alpha, beta, gamma
num_patterns = 100 # number of variations in lattice prms

all_lps = np.linspace(orig_lps, max_lps, num_patterns) # all variations, including original

# wavelength = 1.5406 # Cu K-alpha
# energy_kev = 8.0478 # Cu K-alpha energy in kEV
wavelength = 0.1665
energy_kev = 74
min_twotheta = 0
max_twotheta = 10
scattering_type = 'xray'

new_structure_factor = xtl.Scatter.structure_factor # find the structure factor

# This function has been adapted from the Dans-Diffraction library
def intensity(hkl=None, scattering_type=None, int_hkl=None, **options):
    """
    Returns the structure factor squared
            I = |sum( f_i * occ_i * dw_i * exp( -i * 2 * pi * hkl.uvw ) |^2
    Where f_i is the elemental scattering factor, occ_i is the site occupancy, dw_i
    is the Debye-Waller thermal factor, hkl is the reflection and uvw is the site position.

    The following options for scattering_type are  supported:
      'xray'  - uses x-ray form factors
      'neutron' - uses neutron scattering lengths
      'xray magnetic' - calculates the magnetic (non-resonant) component of the x-ray scattering
      'neutron magnetic' - calculates the magnetic component of neutron scattering
      'xray resonant' - calculates magnetic resonant scattering
      'xray dispersion' - uses x-ray form factors including f'-if'' components

    :param hkl: array[n,3] : reflection indexes (h, k, l)
    :param scattering_type: str : one of ['xray','neutron', 'electron', 'xray magnetic','neutron magnetic','xray resonant']
    :param int_hkl: Bool : when True, hkl values are converted to integer.
    :param kwargs: additional options to pass to scattering function
    :return: float array[n] : array of |SF|^2
    """
    return fs.intensity(new_structure_factor(hkl, scattering_type, int_hkl, **options))
new_intensity = intensity

# This function has been adapted from the Dans-Diffraction library

def list_all_reflections(energy_kev=None, print_symmetric=False,
                              min_intensity=0.01, max_intensity=None, units=None):

    '''
    Returns an np array containing lists of hkl indices, two theta values, and intensities for reflections
    All of these reflections are valid and at the right two theta values; however, the intensity values correspond
    to the integrated areas rather than the peak heights.
    '''
        
    if energy_kev is None:
        energy_kev = ENERGY_KEV
    
    if min_intensity is None: min_intensity = -1
    if max_intensity is None: max_intensity = np.inf
    
    hkl = xtl.Cell.all_hkl(energy_kev, max_twotheta)
    if not print_symmetric:
        hkl = xtl.Symmetry.remove_symmetric_reflections(hkl)
        
    hkl = xtl.Cell.sort_hkl(hkl)

    tth = xtl.Cell.tth(hkl, energy_kev)
    inrange = np.all([tth < max_twotheta, tth > min_twotheta], axis=0)
    hkl = hkl[inrange, :]
    tth = tth[inrange]
    #inten = np.sqrt(self.intensity(hkl)) # structure factor
    inten = intensity(hkl)

    all_info = []

    count = 0
    for n in range(1, len(tth)):
        if inten[n] < min_intensity: continue
        if inten[n] > max_intensity: continue
        count += 1
        all_info.append([hkl[n,0], hkl[n,1], hkl[n,2],tth[n],inten[n]])
    
    return np.array(all_info)

valid_refs = [] # contains valid reflections with integrated areas intensities
tths = [] # each lst entry contains an array with all the two theta vals for that set of lattice prms
intensities = [] # same as above, but for intensities
powder_ref1s = [] # contains powder reflections

xtl.Scatter.setup_scatter(
    scattering_type=scattering_type, 
    powder_units='twotheta', 
    energy_kev=energy_kev,
    min_twotheta=min_twotheta,
    max_twotheta=max_twotheta,
    output=False,
    powder_lorentz=1
)

import csv
from tqdm import tqdm

for i in tqdm(range(num_patterns), desc="Setting lattice parameters"):
    xtl.Cell.latt(all_lps[i]) # set lattice prms

    tth1, intensity1, ref1 = xtl.Scatter.powder() # record two theta, intensities, and all reflections including aphysical ones
    tths.append(tth1)
    intensities.append(intensity1)
    powder_ref1s.append(ref1)
    
    real_reflections = list_all_reflections(energy_kev=energy_kev)
    valid_refs.append(real_reflections)

tths = np.array(tths) # array containing all simulated tths values, not just peaks
intensities = np.array(intensities) # array containing all simulated intensity values, not just peaks

max_refs = max(len(sublist) for sublist in valid_refs)

refs_arr_tths = np.zeros((num_patterns, max_refs)) # array that will contain reflection tths values
refs_arr_ints = np.zeros((num_patterns, max_refs)) # array that will contain reflection intensities

for i in range(num_patterns):
    for j in range(len(valid_refs[i])):
        refs_arr_tths[i][j] = valid_refs[i][j][3]
        refs_arr_ints[i][j] = valid_refs[i][j][4]

# Separate ouput of Dans-Diffraction powder method into hkls, tths, intensities, for all variations
powder_ref1_hkls = [] 
powder_ref1_tths = []
powder_ref1_intensities = []

for j in range(num_patterns):
    ref1_tths_pattern = []
    ref1_intensities_pattern = []
    ref1_hkls_pattern = []
    
    for i in range(len(powder_ref1s[j])):
        ref1_hkls_pattern.append((powder_ref1s[j][i][0], powder_ref1s[j][i][1], powder_ref1s[j][i][2]))
        ref1_tths_pattern.append(powder_ref1s[j][i][3])
        ref1_intensities_pattern.append(powder_ref1s[j][i][4])
    
    powder_ref1_hkls.append(ref1_hkls_pattern)
    powder_ref1_tths.append(ref1_tths_pattern)
    powder_ref1_intensities.append(ref1_intensities_pattern)

binary_peaks = [] # empty list that will contain the np arrays

peak_dtype = np.dtype([
    ('hkl', '3i4'),           # tuple of 3 integers
    ('2theta_calc', 'f8'),   # calculated 2θ
    ('2theta_nearest', 'f8'),# nearest 2θ
    ('intensity', 'f8')      # intensity
]) # this is an array that will hold all of the data we want to keep

all_variations = [] # Create a list to hold each variation's reflections

tol = 0.1
max_refs = max(len(sublist) for sublist in powder_ref1_tths)

for j in range(num_patterns):
    binary_peaks_pattern = np.zeros(tths[j].shape[0], dtype=int)
    non_zero_count = np.count_nonzero(refs_arr_tths[j])
    variation_data = np.zeros((non_zero_count), dtype=peak_dtype)

    count = 0

    for i in range(non_zero_count):
        if refs_arr_tths[j][i] == 0:
            continue  # skip zero entries

        diffs = np.abs(powder_ref1_tths[j] - refs_arr_tths[j][i])
        min_idx_powder = np.argmin(diffs)
        
        if diffs[min_idx_powder] < tol:
            idx_powder = min_idx_powder
            variation_data[i] = (
                powder_ref1_hkls[j][idx_powder],
                refs_arr_tths[j][i],
                powder_ref1_tths[j][idx_powder],
                powder_ref1_intensities[j][idx_powder]
            )    
        
        # For binary pattern
        diffs_alldata = np.abs(tths[j] - refs_arr_tths[j][i])
        min_idx_alldata = np.argmin(diffs_alldata)
        if diffs_alldata[min_idx_alldata] < tol:
            binary_peaks_pattern[min_idx_alldata] = 1
            count+=1
            # print(f"Yes: match in tths[{j}] for ref peak {refs_arr_tths[j][i]} (i={i} at hkl {powder_ref1_hkls[j][i]}")
            # print(tths[j][min_idx_alldata])

        else:
            print(f"No match in tths[{j}] for ref peak {refs_arr_tths[j][i]} (i={i})")
    # print(f'count: {count}')

    # print(f"Pattern {j} - Peaks marked: {np.count_nonzero(binary_peaks_pattern)}")
    all_variations.append(variation_data)
    binary_peaks.append(binary_peaks_pattern)
 
 # Find the maximum number of peaks across all variations
max_reflections = max(len(arr) for arr in all_variations)

# Initialize arrays with NaNs (to pad if necessary)
hkl_arr = np.full((num_patterns, max_reflections, 3), np.nan)
theta_calc_arr = np.full((num_patterns, max_reflections), np.nan)
theta_nearest_arr = np.full((num_patterns, max_reflections), np.nan)
intensity_arr = np.full((num_patterns, max_reflections), np.nan)

# Fill in values
for i, arr in enumerate(all_variations):
    num_reflections = len(arr)
    hkl_arr[i, :num_reflections] = np.stack([row[0] for row in arr])
    theta_calc_arr[i, :num_reflections] = [row[1] for row in arr]
    theta_nearest_arr[i, :num_reflections] = [row[2] for row in arr]
    intensity_arr[i, :num_reflections] = [row[3] for row in arr]

# Store all simulated data and reflections in an xarray dataset
ds_combined = xr.Dataset(
    {
        # These are your xr dataarrays - they can be multidimensional and are indexed to coords (which are also xr dataarrays)
        "Intensities": (["pattern", "tth"], intensities),
        "hkl": (("variation", "peak", "hkl_index"), hkl_arr),
        "2theta_calc": (("variation", "peak"), theta_calc_arr),
        "2theta_nearest": (("variation", "peak"), theta_nearest_arr),
        "intensity": (("variation", "peak"), intensity_arr)
    },
    coords={ # coordinates for indexing your dataarrays 
        "pattern": np.arange(num_patterns),
        "tth": np.linspace(min_twotheta, max_twotheta, tths.shape[1]), # np.linspace(min_tth,max_tth, 11763) or whatever your tth values are
        "variation": np.arange(num_patterns),
        "peak": np.arange(max_reflections),
        "hkl_index": ["h", "k", "l"]
    },
    attrs={ # metadata
        "CIF": cif_file, # you can throw a whole json in here if you like
        "tth_range": (min_twotheta, max_twotheta)
    }
)

# Save file to path
path = 'saved_data/'
file = 'ds_combined_1000_patterns_NaCl_1.nc'

ds_combined.to_netcdf(os.path.join(path, file))
with ZipFile(os.path.join(path,file.replace('.nc','.zip')), 'w', ZIP_DEFLATED) as zObject:
    zObject.write(os.path.join(path,file), arcname=file)