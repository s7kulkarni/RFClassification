import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft
from tqdm import tqdm
import pickle

from loading_functions import *
from file_paths import *
from feat_gen_functions import *

t_seg = 250
fs = 40e6 #40 MHz
features_folder = dronerf_feat_path
n_per_seg = 4096 # length of each segment (powers of 2)
n_overlap_spec = 120
win_type = 'hamming' # make ends of each segment match
high_low = 'L' #'L', 'H' # high or low range of frequency
arr_psd_folder = "ARR_PSD_"+high_low+'_'+str(n_per_seg)+"_"+str(t_seg)
main_folder = dronerf_raw_path

def compute_dft_average_streaming(main_folder, t_seg, chunk_size=1000, 
                                checkpoint_file="/project/shriniwas/dft_attack/dft_avg_checkpoint.npz"):
    """
    Compute running average DFT per class in a streaming fashion.
    Returns:
        - binary_avg_dfts: Dict {label: avg_dft}
        - fourclass_avg_dfts: Dict {label: avg_dft}
        - tenclass_avg_dfts: Dict {label: avg_dft}
    """
    # Checkpoint loading
    if os.path.exists(checkpoint_file):
        checkpoint = np.load(checkpoint_file, allow_pickle=True)
        binary_avg_dfts = checkpoint['binary_avg_dfts'].item()
        fourclass_avg_dfts = checkpoint['fourclass_avg_dfts'].item()
        tenclass_avg_dfts = checkpoint['tenclass_avg_dfts'].item()
        start_index = int(checkpoint['start_index'])
        print(f"Resuming from chunk {start_index}")
    else:
        binary_avg_dfts = {}  # Format: {label: {'sum': sum_dft, 'count': N}}
        fourclass_avg_dfts = {}
        tenclass_avg_dfts = {}
        start_index = 0

    # # Initialize data generator
    # data_gen = load_dronerf_raw_stream(main_folder, t_seg, chunk_size=chunk_size, stream=True)
    
    # # Process first chunk to verify dimensions
    # first_chunk, _, _, _ = next(data_gen)
    # test_dft = fft(first_chunk[0, 0, :])  # DFT of first sample, first channel
    # assert test_dft.shape == (first_chunk.shape[2],), \
    #        f"DFT axis mismatch! Expected length {first_chunk.shape[2]}, got {test_dft.shape}"
    # print(f"DFT verification passed. Processing {first_chunk.shape[2]} frequency bins.")

    # Reset generator
    data_gen = load_dronerf_raw_stream(main_folder, t_seg, chunk_size=chunk_size, stream=True)

    for chunk_idx, (X_chunk, y_binary, y4, y10) in enumerate(data_gen):
        if start_index > 218:
            print("Already processed all chunks, proceeding")
            break
        if chunk_idx < start_index:
            print("skipping chunk ", chunk_idx)
            continue
        print(f"Processing chunk {chunk_idx}")

        # Compute DFT along time axis (axis=2)
        dfts = np.abs(fft(X_chunk, axis=2))  # Shape: [chunk_size, 2, freq_bins]

        # Average across channels (axis=1)
        dfts_avg = np.mean(dfts, axis=1)  # Shape: [chunk_size, freq_bins]

        # Update running averages
        def update_avg_dict(avg_dict, dfts_chunk, labels_chunk):
            for label in np.unique(labels_chunk):
                mask = (labels_chunk == label)
                label_dfts = dfts_chunk  # Use entire chunk since all have same label
                if label not in avg_dict:
                    avg_dict[label] = {'sum': np.zeros_like(label_dfts), 'count': 0}
                avg_dict[label]['sum'] += label_dfts * mask.sum()  # Scale by count
                avg_dict[label]['count'] += mask.sum()
            return avg_dict

        binary_avg_dfts = update_avg_dict(binary_avg_dfts, dfts_avg, y_binary)
        fourclass_avg_dfts = update_avg_dict(fourclass_avg_dfts, dfts_avg, y4)
        tenclass_avg_dfts = update_avg_dict(tenclass_avg_dfts, dfts_avg, y10)

        # Save checkpoint
        np.savez(checkpoint_file,
                binary_avg_dfts=binary_avg_dfts,
                fourclass_avg_dfts=fourclass_avg_dfts,
                tenclass_avg_dfts=tenclass_avg_dfts,
                start_index=chunk_idx + 1)
    print("DFT avg calculation done")

    # Finalize averages
    def finalize_avg(avg_dict):
        return {label: avg_dict[label]['sum'] / avg_dict[label]['count'] 
                for label in avg_dict}

    binary_avg = finalize_avg(binary_avg_dfts)
    fourclass_avg = finalize_avg(fourclass_avg_dfts)
    tenclass_avg = finalize_avg(tenclass_avg_dfts)

    return binary_avg, fourclass_avg, tenclass_avg


def process_and_save_incrementally(avg_dft_dict, checkpoint_dir='/project/shriniwas/checkpoints_attack'):
    """
    Processes drone RF data with optimized perturbation strategy.
    """
    # Checkpoint setup (unchanged)
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

    # Class representative cache (NEW)
    class_representatives = {}  # {label: (filepath, signal)}

    def get_representative_signal(target_label):
        """Get or load a representative signal for the target class"""
        if target_label not in class_representatives:
            # Find first file with this label
            for lf in low_freq_files:
                if int(lf[0][:3]) == target_label:
                    try:
                        signal = pd.read_csv(lf[1], header=None).values.flatten()
                        class_representatives[target_label] = (lf[1], signal)
                        break
                    except Exception as e:
                        print(f"Error loading representative for {target_label}: {e}")
                        continue
        return class_representatives.get(target_label, (None, None))[1]

    # File collection (unchanged)
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

    # Main processing loop
    for i in range(start_idx, len(high_freq_files)):
        print(i, 'of', len(high_freq_files))
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

        # Load data (unchanged)
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

        # ===== IMPROVED PERTURBATION LOGIC =====
        current_label = int(low_freq_file[0][:3])
        false_labels = [l for l in avg_dft_dict.keys() if l != current_label]

        print("KEYS", avg_dft_dict.keys())
        
        if 1:
            # Adversarial: Choose the farthest class representative
            target_label = max(false_labels, 
                             key=lambda x: np.linalg.norm(avg_dft_dict[x] - avg_dft_dict[current_label]))
        else:
            # Random selection
            target_label = np.random.choice(false_labels)
        
        # Get representative signal
        target_signal = get_representative_signal(target_label)
        if target_signal is None:
            print(f"No representative found for {target_label}")
            continue

        # Compute perturbation
        current_dft = np.fft.fft(rf_data_l)
        target_dft = np.fft.fft(target_signal[:len(rf_data_l)])  # Truncate if needed
        perturbation = np.real(np.fft.ifft(target_dft - current_dft))
        
        # Scale perturbation
        ratio = 0.4
        perturbation *= ratio / (np.linalg.norm(perturbation)/np.linalg.norm(rf_data_l))
        print('PEERTURBATION NORM', np.linalg.norm(perturbation))
        print('PEERTURBATION NORM RATIO', np.linalg.norm(perturbation)/np.linalg.norm(rf_data_l))
        
        # Apply perturbations
        rf_data_l_adv = rf_data_l + perturbation
        # ===== END PERTURBATION LOGIC =====

        print("rf_data_h, rf_data_l shapes ", rf_data_h.shape, rf_data_l.shape)
        # Stack high and low frequency data
        rf_sig = np.vstack((rf_data_h, rf_data_l_adv))
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
        
        print("len F_PSD and component ", len(F_PSD), F_PSD[0].shape)

        # Save results for this file
        save_path = features_folder+arr_psd_folder+'_'+'dft_attack'+'_'+str(int(ratio*100))+'/'
        save_array_rf(save_path, F_PSD, BILABEL, DRONELABEL, MODELALBEL, 'PSD', n_per_seg, i)

        # Update checkpoint
        checkpoint = {
            'last_processed_file': high_freq_file[0],
            'last_processed_idx': i
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Processed and saved {high_freq_file[0]}")

        # # Process all versions
        # for data_l, suffix in [(rf_data_l_adv, 'dft_attack'), 
        #                       (rf_data_l_rand, 'random')]:
        #     # Original processing pipeline (UNCHANGED)
        #     rf_sig = np.vstack((rf_data_h, data_l))
        #     len_seg = int(t_seg / 1e3 * fs)
        #     n_segs = len(rf_data_h) // len_seg
        #     rf_sig_segments = np.split(rf_sig[:, :n_segs*len_seg], n_segs, axis=1)

        #     F_PSD = []
        #     for seg in rf_sig_segments:
        #         fpsd, Pxx_den = signal.welch(seg[1], fs, window=win_type, nperseg=n_per_seg)
        #         F_PSD.append(Pxx_den)

        #     # Save to version-specific subfolder (only path modified)
        #     save_path = features_folder+arr_psd_folder+'_'+suffix+'_'+str(int(ratio*100))+'/'
        #     save_array_rf(save_path, F_PSD, 
        #                  [int(low_freq_file[0][0])]*len(F_PSD),
        #                  [int(low_freq_file[0][:3])]*len(F_PSD),
        #                  [int(low_freq_file[0][:5])]*len(F_PSD),
        #                  'PSD', n_per_seg, i)

        # # Checkpoint update (unchanged)
        # checkpoint = {
        #     'last_processed_file': high_freq_file[0],
        #     'last_processed_idx': i
        # }
        # with open(checkpoint_file, 'wb') as f:
        #     pickle.dump(checkpoint, f)

        # print(f"Processed {high_freq_file[0]} with all perturbations")

binary_avg, fourclass_avg, tenclass_avg = compute_dft_average_streaming(
    main_folder=main_folder,
    t_seg=t_seg,
    chunk_size=1000
)

process_and_save_incrementally(
    avg_dft_dict=fourclass_avg,  # Use 4-class DFT averages
)