import Dans_Diffraction as dif
import Dans_Diffraction.functions_scattering as fs
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

class SimulateDiff:
    def __init__(self, wavelength, energy_kev, min_twotheta, max_twotheta, scattering_type, xtl=None):
        self.wavelength = wavelength
        self.energy_kev = energy_kev
        self.min_twotheta = min_twotheta
        self.max_twotheta = max_twotheta
        self.scattering_type = scattering_type
        self.xtl = xtl

    def load(self, cif_file):
        # load in CIF file
        self.xtl = dif.Crystal(cif_file)

        orig_lps = xtl.Cell.lp() # starting lattice parameters
        lp_multi-plier = (2,2,2,1,1,1) # separate multiplier for cell prms
        max_lps = np.array(orig_lps) * np.array(lp_multiplier) # max lp_a, lp_b, lp_c, alpha, beta, gamma
        num_patterns = 3 # number of variations in lattice prms

        all_lps = np.linspace(orig_lps, max_lps, num_patterns) # all variations, including original

        return all_lps
    
    def _sf(self):
        if self.xtl is None:
            raise ValueError("xtl has not been loaded.")
        return self.xtl.Scatter.structure_factor

    def intensity(self, hkl=None, scattering_type=None, int_hkl=None, **options):
        new_sf = self._sf()
        return fs.intensity(new_sf(hkl, scattering_type, int_hkl, **options))

    new_intensity = intensity

def list_all_reflections(energy_kev=None, print_symmetric=False, min_intensity=0.01, max_intensity=None, units=None):
    '''
    Lists all valid reflections, with their integrated areas as intensities
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

def sim_scattering():
    valid_refs = [] # contains valid reflections with integrated areas intensities
    tths = [] # each lst entry contains an array with all the two theta vals for that set of lattice prms
    intensities = [] # same as above, but for intensities
    powder_ref1s = [] # contains powder reflections
    
    # Set up the scattering prms for a 2theta x-ray scan
    xtl.Scatter.setup_scatter(
        scattering_type=scattering_type, 
        powder_units='twotheta', 
        energy_kev=energy_kev,
        min_twotheta=min_twotheta,
        max_twotheta=max_twotheta,
        output=False,
        powder_lorentz=1 # lorentz fraction of 1 was arbitrarily chosens
    )
    
    for i in range(num_patterns):
        xtl.Cell.latt(all_lps[i]) # set lattice prms
    
        tth1, intensity1, ref1 = xtl.Scatter.powder() # record two theta, intensities, and all reflections including aphysical ones
        tths.append(tth1)
        intensities.append(intensity1)
        powder_ref1s.append(ref1)
        
        real_reflections = list_all_reflections(energy_kev=energy_kev)
        valid_refs.append(real_reflections)
    
    tths = np.array(tths)
    intensities = np.array(intensities)

    return valid_refs, tths, intensities, powder_ref1s

valid_refs, tths, intensities, powder_ref1s = sim_scattering()

def find_valid_ref_int():
    max_refs = max(len(sublist) for sublist in valid_refs)
    
    refs_arr_tths = np.zeros((num_patterns, max_refs))
    refs_arr_ints = np.zeros((num_patterns, max_refs))
    
    # Populate separate arrays with the valid reflection tth values and valid reflection integrated area intensities
    for i in range(num_patterns):
        for j in range(len(valid_refs[i])):
            refs_arr_tths[i][j] = valid_refs[i][j][3]
            refs_arr_ints[i][j] = valid_refs[i][j][4]
    
    # Separate ouput of Dans-Diffraction powder method into hkls, tths, intensities
    powder_ref1_hkls = [] 
    powder_ref1_tths=[]
    powder_ref1_intensities=[]
    
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

    return refs_arr_tths, refs_arr_ints, powder_ref1_hkls, powder_ref1_tths, powder_ref1_intensities

refs_arr_tths, refs_arr_ints, ref1_hkls, ref1_tths, ref1_intensities = find_valid_ref_int()

def create_binary_output():
    # Define the data type
    peak_dtype = np.dtype([
        ('hkl', '3i4'),           # tuple of 3 integers
        ('2theta_calc', 'f8'),   # calculated 2θ
        ('2theta_nearest', 'f8'),# nearest 2θ
        ('intensity', 'f8')      # intensity
    ]) # this is an array that will hold all of the data we want to keep
    
    # Create a list to hold each variation's peaks
    all_variations = []
    
    tol = 1e-6 # tolerance value
    max_refs = max(len(sublist) for sublist in ref1_tths)
    binary_peaks = [] # list to hold arrays of binary values indicating whether or not a peak is present
    
    for j in range(num_patterns):
        binary_peaks_pattern = np.zeros((tths[0].shape[0],))
        
        non_zero_count = np.count_nonzero(refs_arr_tths[j]) # how many valid peak positions there are that we should check
            
        variation_data = np.zeros((non_zero_count), dtype=peak_dtype) # create np array to hold this pattern's data
    
        for i in range(non_zero_count):
            idx = np.where(np.abs(ref1_tths[j] - refs_arr_tths[j][i]) < tol)[0][0] # find idx where valid peak matches powder method peak
            variation_data[i] = (ref1_hkls[j][idx], refs_arr_tths[j][i], ref1_tths[j][idx], ref1_intensities[j][idx])
            binary_peaks_pattern[idx] = int(1) # fill binary array with 1s if there is reflection at that idx
            
        all_variations.append(variation_data)
        binary_peaks.append(binary_peaks_pattern)

    return binary_peaks
        
if name == "__main__":
    cif_file = 'cif_files/NaCl_cubic.cif'

    mySim = SimulateDiff(wavelength=0.1665, energy_kev=74, min_twotheta=0, max_twotheta=10, scattering_type='xray')

    all_lps = mySim.load(cif_file)

    mySim.sf()






    