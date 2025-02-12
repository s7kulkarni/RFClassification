import os
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
import pickle

from loading_functions import *
from file_paths import *
from feat_gen_functions import *

t_seg = 20
fs = 40e6 #40 MHz
features_folder = dronerf_feat_path
n_per_seg = 1024 # length of each segment (powers of 2)
n_overlap_spec = 120
win_type = 'hamming' # make ends of each segment match
high_low = 'L' #'L', 'H' # high or low range of frequency
arr_psd_folder = "ARR_PSD_"+high_low+'_'+str(n_per_seg)+"_"+str(t_seg)+"/"
main_folder = dronerf_raw_path


def process_and_save_incrementally(checkpoint_dir='/home/zebra/shriniwas/checkpoints'):
    """
    Processes drone RF data incrementally, calculates PSD, and saves results incrementally.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(features_folder):
        os.makedirs(features_folder)

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

        # Stack high and low frequency data
        rf_sig = np.vstack((rf_data_h, rf_data_l))

        # Segment the data
        len_seg = int(t_seg / 1e3 * fs)
        n_segs = len(rf_data_h) // len_seg
        n_keep = n_segs * len_seg

        try:
            rf_sig_segments = np.split(rf_sig[:, :n_keep], n_segs, axis=1)
        except Exception as e:
            print(f"Error splitting {high_freq_file[0]}: {e}")
            continue

        # Process each segment
        F_PSD = []
        BILABEL = []
        DRONELABEL = []
        MODELALBEL = []

        for seg in rf_sig_segments:
            # Calculate PSD for high-frequency data (assuming index 0)
            h_l = 0 if high_low == 'H' else 1
            fpsd, Pxx_den = signal.welch(seg[h_l], fs, window=win_type, nperseg=n_per_seg)
            F_PSD.append(Pxx_den)

            # Labels
            BILABEL.append(int(low_freq_file[0][0]))  # 2-class label
            DRONELABEL.append(int(low_freq_file[0][:3]))  # 4-class label
            MODELALBEL.append(int(low_freq_file[0][:5]))  # 10-class label

        # Save results for this file
        save_array_rf(features_folder+arr_psd_folder, F_PSD, BILABEL, DRONELABEL, MODELALBEL, 'PSD', n_per_seg, i)

        # Update checkpoint
        checkpoint = {
            'last_processed_file': high_freq_file[0],
            'last_processed_idx': i
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.load(checkpoint, f)

        print(f"Processed and saved {high_freq_file[0]}")

process_and_save_incrementally()