import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from helper_functions import *
from latency_helpers import *
from file_paths import *
from models import *
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchmetrics
from tqdm import tqdm
import torchhd
from torchhd.models import Centroid
import copy
import argparse
import random
import random
import importlib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# remove to run locally
import loading_functions
importlib.reload(loading_functions)
from loading_functions import *
#

def get_arrays_efficient(dataset, batch_size=64):
    """    
    Args:
        dataset (DroneRFTorch): The dataset object.
        batch_size (int): Batch size for loading data. Adjust based on memory constraints.
    Returns:
        X (np.ndarray): Concatenated feature arrays.
        y (np.ndarray): Concatenated labels.
    """
    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize lists to collect batches
    X_batches = []
    y_batches = []

    # Iterate through the DataLoader
    for batch_idx, (data, labels) in enumerate(data_loader):
        X_batches.append(np.array(data))  # Convert tensors to numpy arrays
        y_batches.append(np.array(labels))
        print(f"Processed batch {batch_idx + 1}")

    # Concatenate all batches
    X = np.concatenate(X_batches, axis=0)
    y = np.concatenate(y_batches, axis=0)

    return X, y

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print("Using {} device".format(device))

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility if using GPU


# Random Projection Encoder
class RandomProjectionEncoder(nn.Module):
    def __init__(self, out_features, in_features):
        super(RandomProjectionEncoder, self).__init__()
        self.projection_matrix = torch.randn(in_features, out_features).to(device)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = x.squeeze(0)
        sample_hv = torch.matmul(x, self.projection_matrix)
        sample_hv = torch.sign(sample_hv)
        sample_hv = sample_hv.unsqueeze(0)
        return sample_hv


def train_full_precision(encode, model):
    # Training loop
    with torch.no_grad():
        for samples, labels in tqdm(train_ld, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)
            # print(labels.shape, )
            # Encode the samples using the random projection matrix
            samples_hv = encode(samples)
            # Add the encoded hypervectors to the model
            model.add(samples_hv, labels)

def test_model(encode, model):
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=len(label_encoder.classes_)).to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for samples, labels in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)
            labels = labels.to(device)

            # Encode the test samples using the random projection matrix
            samples_hv = encode(samples)

            # Get the predictions from the Centroid model by passing encoded hypervectors
            preds = model(samples_hv)

            # Compute accuracy
            correct += accuracy(preds, labels)
            total += labels.size(0)

    accuracy_value = correct / total
    return accuracy_value.item()

feat_name = 'PSD'
t_seg = 250 #ms
n_per_seg = 4096
interferences = ['WIFI', 'BLUE', 'BOTH', 'CLEAN']
output_name = 'bi'
feat_format = 'ARR'
which_dataset = 'dronerf'
output_tensor = False
in_features = 2049
DIMENSIONS = 10000
seed = 3

print('Loading DroneRF Dataset')
highlow = 'L'
dataset = DroneRFTorch(dronerf_feat_path, feat_name, t_seg, n_per_seg,
                    feat_format, output_name, output_tensor, highlow)

print("dataset loaded")
# X_use, y_use = dataset.get_arrays()
X, Y = get_arrays_efficient(dataset, batch_size=64)

# ################# DUMMY DATA
# def generate_dummy_data(num_samples=10, num_features=2049):
#     """
#     Generate dummy data where:
#     - If >85% of elements in a sample are 1, label is 1.
#     - If <15% of elements are 1, label is 0.
#     - No sample has between 15% and 85% ones.
    
#     Args:
#     - num_samples: Number of samples to generate.
#     - num_features: Number of features in each sample.
    
#     Returns:
#     - X: Input data (num_samples, num_features).
#     - y: Labels (num_samples, 1).
#     """
#     X = np.zeros((num_samples, num_features), dtype=int)
#     y = np.zeros((num_samples, 1), dtype=int)

#     for i in range(num_samples):
#         if np.random.rand() < 0.5:  # Randomly decide label 0 or 1
#             ones_count = np.random.randint(0, int(0.4 * num_features) + 1)  # ≤15% ones
#             y[i] = 0
#         else:
#             ones_count = np.random.randint(int(0.6 * num_features), num_features + 1)  # ≥85% ones
#             y[i] = 1

#         # Randomly pick the indices to set to 1
#         ones_indices = np.random.choice(num_features, ones_count, replace=False)
#         X[i, ones_indices] = 1  # Set the selected indices to 1

#     return X, y

# X, Y = generate_dummy_data()
# print("DUMMY DATA SHAPE",X.shape, Y.shape)

###############################

label_encoder = LabelEncoder()
Y_int = label_encoder.fit_transform(Y)

X_tensor = torch.tensor(X, dtype=torch.float32)  # (219, 4097)
Y_tensor = torch.tensor(Y_int, dtype=torch.long)  # (219,)

# K-Fold Cross-Validation
k_folds = 5  # You can change this
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Store accuracy for each fold
fold_accuracies = []



for fold, (train_idx, test_idx) in enumerate(skf.split(X_tensor, Y_tensor)):
    print(f"Fold {fold + 1}/{k_folds}")

    # Split data into train and test sets for this fold
    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    Y_train, Y_test = Y_tensor[train_idx], Y_tensor[test_idx]

    # Create DataLoader for training and testing
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    train_ld = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_ld = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    set_seed(89)


    encode = RandomProjectionEncoder(DIMENSIONS, in_features).to(device)
    model = Centroid(DIMENSIONS, len(label_encoder.classes_)).to(device)

    
    # Train the model
    train_full_precision(encode, model)

    # Test the model
    accuracy_value = test_model(encode, model)
    fold_accuracies.append(accuracy_value)
    print(f"Fold {fold + 1} Accuracy: {accuracy_value:.4f}")

# Print average accuracy across all folds
print(f"Average Accuracy: {np.mean(fold_accuracies):.4f}")
