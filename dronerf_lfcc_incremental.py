import os
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
import pickle

from loading_functions import *
from file_paths import *
from feat_gen_functions import *

from scipy.fft import fft
from scipy.signal import get_window
import librosa

t_seg = 250
fs = 40e6 #40 MHz
features_folder = dronerf_feat_path
n_per_seg = 4096 # length of each segment (powers of 2)
high_low = 'LH' #'L', 'H' # high or low range of frequency
arr_lfcc_folder = "ARR_LFCC_"+high_low+'_'+str(t_seg)+"/"
perturbation_filepath = "/home/zebra/shriniwas/RFClassification/two_class_perturbation_001.npy"
main_folder = dronerf_raw_path

def compute_improved_lfcc(signal, fs, num_filters=24, num_coeffs=12, fmin=0, fmax=None):
    """
    Compute improved LFCC features from an RF signal as a single vector.
    
    Parameters:
    - signal: Input RF signal (numpy array, e.g., 0.25s long)
    - fs: Sampling frequency in Hz
    - num_filters: Number of filter bank channels (adjusted to 24)
    - num_coeffs: Number of cepstral coefficients (adjusted to 12)
    - fmin: Minimum frequency (e.g., 2380 MHz for high-band)
    - fmax: Maximum frequency (e.g., 2480 MHz for high-band)
    
    Returns:
    - lfcc_vector: LFCC feature vector (1D array)
    """
    N = len(signal)  # Use full signal length (e.g., 0.25s)
    if fmax is None:
        fmax = fs / 2

    # Step 1: Windowing (single frame)
    hamming_window = get_window("hamming", N)
    windowed_signal = signal * hamming_window

    # Step 2: Compute DFT and absolute-squared DFT
    dft = fft(windowed_signal, n=N)
    abs_squared_dft = np.abs(dft) ** 2

    # Step 3: Construct linear filter bank
    filter_bank = librosa.filters.constant_q(
        sr=fs, fmin=fmin, fmax=fmax, n_bins=num_filters, bins_per_octave=num_filters,
        scale=False, filter_scale=1, norm=1, pad_fft=False
    )
    filter_bank = filter_bank[:, :N//2 + 1]

    # Step 4: Apply filter bank to power spectrum
    power_spectrum = abs_squared_dft[:N//2 + 1]
    filter_bank_output = np.dot(power_spectrum, filter_bank.T)
    filter_bank_output = np.where(filter_bank_output == 0, np.finfo(float).eps, filter_bank_output)

    # Step 5: Compute logarithm
    log_filter_bank_output = np.log(filter_bank_output)

    # Step 6: Compute DCT for cepstral coefficients
    lfcc = librosa.util.dct(log_filter_bank_output, n_mfcc=num_coeffs + 1)[1:]  # Exclude 0th initially

    # Step 7: Add zero-order coefficient and log energy
    zero_order = librosa.util.dct(log_filter_bank_output, n_mfcc=1)[0]
    log_energy = np.log(np.sum(power_spectrum) + np.finfo(float).eps)
    lfcc_vector = np.concatenate([zero_order, lfcc, log_energy])  # 1D vector

    return lfcc_vector

def process_and_save_incrementally(checkpoint_dir='/home/zebra/shriniwas/checkpoints_lfcc'):
    """
    Processes drone RF data incrementally, calculates PSD, and saves results incrementally.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load checkpoint if exists
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pkl')
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            last_processed_file = checkpoint['last_processed_file']
            start_idx = checkpoint['last_processed_idx'] + 1
    else:
        last_processed_file = ""
        start_idx = 0

    # Collect all files
    high_freq_files = []
    low_freq_files = []

    for dirpath, _, filenames in os.walk(main_folder):
        for filename in filenames:
            full_filepath = os.path.join(dirpath, filename)
            if 'H' in filename:
                high_freq_files.append([filename, full_filepath])
            elif 'L' in filename:
                low_freq_files.append([filename, full_filepath])

    high_freq_files.sort()
    low_freq_files.sort()

    print("TOTAL HIGH/LOW FREQ FILES", len(high_freq_files))

    # Process files incrementally
    for i in range(start_idx, len(high_freq_files)):
        high_freq_file = high_freq_files[i]
        low_freq_file = None

        # Find corresponding low-frequency file
        for lff in low_freq_files:
            if lff[0][:5] + lff[0][6:] == high_freq_file[0][:5] + high_freq_file[0][6:]:
                low_freq_file = lff
                break

        if not low_freq_file:
            print(f"No matching low-frequency file for {high_freq_file[0]}")
            continue

        # Load high-frequency data
        try:
            rf_data_h = pd.read_csv(high_freq_file[1], header=None).values.flatten()
        except Exception as e:
            print(f"EXCEPTION loading {high_freq_file[0]}: {e}")
            continue

        # Load low-frequency data
        try:
            rf_data_l = pd.read_csv(low_freq_file[1], header=None).values.flatten()
        except Exception as e:
            print(f"EXCEPTION loading {low_freq_file[0]}: {e}")
            continue

        if len(rf_data_h) != len(rf_data_l):
            print(f"Length mismatch: {high_freq_file[0]} and {low_freq_file[0]}")
            continue

        ########## ADDING PERTURBATION
        p_array = np.load(perturbation_filepath)
    
        # Calculate the tiling factor needed
        tiling_factor = len(rf_data_l) // len(p_array)
        
        # Check if the division is clean
        if len(rf_data_l) % len(p_array) != 0:
            print(f"Warning: RF array length ({len(rf_data_l)}) is not a multiple of perturbation length ({len(p_array)})")
            # Round up to ensure the tiled array is at least as long as the target
            tiling_factor += 1
        
        # Tile the array
        tiled_array = np.tile(p_array, tiling_factor)
        
        # Trim if necessary to match the target array length
        if len(tiled_array) > len(rf_data_l):
            tiled_array = tiled_array[:len(rf_data_l)]
        
        # Add to the target array
        # print("IS TILED ARRAY 0? ", np.isclose(tiled_array, 0.0, atol=1e-5))
        # print("ARE ORIG N PERTURBED CLOSE?", np.allclose(rf_data_l, rf_data_l + tiled_array, atol=1e-5))
        # print("NORM RATIO", np.linalg.norm(tiled_array)/np.linalg.norm(rf_data_l))
        current_ratio = np.linalg.norm(tiled_array) / np.linalg.norm(rf_data_l)
        desired_ratio = 0.4
        scaling_factor = desired_ratio / current_ratio
        tiled_array = tiled_array * scaling_factor
        # rf_data_l = rf_data_l + tiled_array
        ############ PERTURBED!

        print("rf_data_h, rf_data_l shapes ", rf_data_h.shape, rf_data_l.shape)
        # Stack high and low frequency data
        rf_sig = np.vstack((rf_data_h, rf_data_l))
        print("rf_sig shape ", rf_sig.shape)

        # Segment the data
        len_seg = int(t_seg / 1e3 * fs)
        n_segs = len(rf_data_h) // len_seg
        n_keep = n_segs * len_seg
        print("len_seg, n_segs, n_keep : ", len_seg, n_segs, n_keep)

        try:
            rf_sig_segments = np.split(rf_sig[:, :n_keep], n_segs, axis=1)
        except Exception as e:
            print(f"Error splitting {high_freq_file[0]}: {e}")
            continue

        print("rf_sig_segments shape ", len(rf_sig_segments))

        # Process each segment
        F_LFCC = []
        BILABEL = []
        DRONELABEL = []
        MODELALBEL = []

        for seg in rf_sig_segments:
            lfcc_low = compute_improved_lfcc(seg[1], fs, fmin=0, fmax=100e6)
            lfcc_high = compute_improved_lfcc(seg[0], fs, fmin=2380e6, fmax=2480e6)
            lfcc_combined = np.concatenate([lfcc_low, lfcc_high])
            print("LFCC shape ", lfcc_combined.shape)
            F_LFCC.append(lfcc_combined)

            # Labels
            BILABEL.append(int(low_freq_file[0][0]))  # 2-class label
            DRONELABEL.append(int(low_freq_file[0][:3]))  # 4-class label
            MODELALBEL.append(int(low_freq_file[0][:5]))  # 10-class label
        
        print("len F_PSD and component ", len(F_LFCC), F_LFCC[0].shape)

        # Save results for this file
        save_array_rf(features_folder+arr_lfcc_folder, F_LFCC, BILABEL, DRONELABEL, MODELALBEL, 'LFCC', n_per_seg, i)

        # Update checkpoint
        checkpoint = {
            'last_processed_file': high_freq_file[0],
            'last_processed_idx': i
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Processed and saved {high_freq_file[0]}")

process_and_save_incrementally()