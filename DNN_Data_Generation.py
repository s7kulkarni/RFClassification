import numpy as np
import os
from scipy.io import loadmat, savemat

# Paths
load_filename = '/project/shriniwas/DroneRF_extracted/'  # Path of raw RF data
save_filename = '/project/shriniwas/DNN/'  # Path to save the final CSV

# Parameters
BUI = {
    0: ['00000'],  # BUI of RF background activities
    1: ['10000', '10001', '10010', '10011'],  # BUI of the Bebop drone RF activities
    2: ['10100', '10101', '10110', '10111'],  # BUI of the AR drone RF activities
    3: ['11000']  # BUI of the Phantom drone RF activities
}
M = 2048  # Total number of frequency bins
L = int(1e5)  # Total number of samples in a segment
Q = 10  # Number of returning points for spectral continuity

# Initialize variables for concatenation and labeling
DATA = []
LN = []
SEGMENT_INFO = []  # To store segment information (n and i)

# Function to find and load the file
def find_and_load_file(root_dir, b, n, suffix):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == f"{b}{suffix}_{n}.csv":
                return np.loadtxt(os.path.join(dirpath, filename), delimiter=',')
    return None

# Main processing loop
for opt in BUI:
    for b in BUI[opt]:
        print(f"Processing {b}")
        if b == '00000':
            N = 40  # Number of segments for RF background activities
        elif b == '10111':
            N = 17
        else:
            N = 20  # Number of segments for drones RF activities

        data = []
        cnt = 0
        for n in range(N + 1):
            # Loading raw CSV files
            x = find_and_load_file(load_filename, b, n, 'L')
            y = find_and_load_file(load_filename, b, n, 'H')

            if x is None or y is None:
                continue

            # Re-segmenting and signal transformation
            for i in range(len(x) // L):
                st = i * L
                fi = (i + 1) * L
                x_segment = x[st:fi]
                y_segment = y[st:fi]

                xf = np.abs(np.fft.fftshift(np.fft.fft(x_segment - np.mean(x_segment), M)))
                xf = xf[M//2:]

                yf = np.abs(np.fft.fftshift(np.fft.fft(y_segment - np.mean(y_segment), M)))
                yf = yf[M//2:]

                scaling_factor = np.mean(xf[-Q:]) / np.mean(yf[:Q])
                data.append(np.concatenate([xf, yf * scaling_factor]))
                SEGMENT_INFO.append([n, i])  # Store segment info (n and i)
                cnt += 1

            print(f"{100 * n / N:.2f}%")

        # Normalize and concatenate data
        Data = np.array(data).T ** 2
        Data = Data / np.max(Data)  # Normalize
        DATA.append(Data)
        LN.append(Data.shape[1])  # Store the number of columns

    print(f"{100 * (opt + 1) / len(BUI):.2f}%")

# Concatenate all data
DATA = np.hstack(DATA)

# Convert segment info to a numpy array
SEGMENT_INFO = np.array(SEGMENT_INFO).T  # Shape: (2, num_samples)

# Labeling
T = len(BUI)
Label = np.zeros((3, DATA.shape[1]))

# Label(1,:) = [0*ones(1,LN(1)) 1*ones(1,sum(LN(2:end)))]
Label[0, :LN[0]] = 0
Label[0, LN[0]:] = 1

# Label(2,:) = [0*ones(1,LN(1)) 1*ones(1,sum(LN(2:5))) 2*ones(1,sum(LN(6:9))) 3*ones(1,LN(10)))]
Label[1, :LN[0]] = 0
Label[1, LN[0]:LN[0] + sum(LN[1:5])] = 1
Label[1, LN[0] + sum(LN[1:5]):LN[0] + sum(LN[1:9])] = 2
Label[1, LN[0] + sum(LN[1:9]):] = 3

# Label(3,:) = temp
temp = []
for i in range(len(LN)):
    temp.extend([i] * LN[i])
Label[2, :] = temp

# Combine DATA, Label, and Segment Info
final_data = np.vstack([DATA, Label, SEGMENT_INFO])

# Save to CSV
np.savetxt(f"{save_filename}RF_Data.csv", final_data.T, delimiter=',', fmt='%.6f')

print("Processing complete. Data saved to RF_Data.csv.")