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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

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
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
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
    train_residuals = []
    with torch.no_grad():
        for samples, labels in tqdm(train_ld, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)
            # print(labels.shape, )
            # Encode the samples using the random projection matrix
            samples_hv = encode(samples)
            # Add the encoded hypervectors to the model
            model.add_online(samples_hv, labels)

            # Compute and store training residuals
            similarities = model(samples_hv)  # Shape: [batch_size, num_classes]
            batch_residuals = 1 - similarities.max(dim=1).values
            train_residuals.extend(batch_residuals.cpu().numpy())
    return np.array(train_residuals)

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
output_name = 'drones'
norm_ratio = '05' # 0.xxx mapped to xxx
feat_format = 'ARR'
which_dataset = 'dronerf'
output_tensor = False
in_features = 2049
DIMENSIONS = 10000
seed = 11

print('Loading DroneRF Dataset')
highlow = 'L'
dataset = DroneRFTorch(dronerf_feat_path, feat_name, t_seg, n_per_seg,
                    feat_format, output_name, output_tensor, highlow)
perturbed_dataset = DroneRFTorchPerturbed(dronerf_feat_path, feat_name, t_seg, n_per_seg,
                    feat_format, output_name, output_tensor, highlow, norm_ratio, output_name)
print("dataset loaded")
# X_use, y_use = dataset.get_arrays()
X, Y = get_arrays_efficient(dataset, batch_size=64)
X_perturbed, Y_perturbed = get_arrays_efficient(perturbed_dataset, batch_size=64)
print("SHAPES ", X.shape, Y.shape, X_perturbed.shape, Y_perturbed.shape)

# --- Anomaly Detection Adaptation ---
label_encoder = LabelEncoder()
Y_int = label_encoder.fit_transform(Y)  # Convert labels to 0,1,2,3
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y_int, dtype=torch.long)

# Metrics storage (averaged over all unknown class choices)
final_metrics = {
    'precision': [],
    'recall': [],
    'f1': [],
    'auroc': [],
    'specificity': []
}

k_folds = 5
set_seed(seed)
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

for unknown_class in range(4):  # Test each class as unknown
    print(f"\n=== Evaluating Class {unknown_class} as Unknown ===")
    
    fold_metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'auroc': [],
        'specificity': []
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_tensor, Y_tensor)):
        # Split data (exclude unknown_class from training)
        X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
        Y_train, Y_test = Y_tensor[train_idx], Y_tensor[test_idx]
        train_mask = (Y_train != unknown_class)
        X_train_known = X_train[train_mask]
        Y_train_known = Y_train[train_mask]

        # Create DataLoaders (using your existing setup)
        global train_ld, test_ld  # Required for your train/test methods
        train_dataset = torch.utils.data.TensorDataset(X_train_known, Y_train_known)
        test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
        train_ld = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, random_state = seed)
        test_ld = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Train HDC model (3 classes)
        encode = RandomProjectionEncoder(DIMENSIONS, in_features).to(device)
        model = Centroid(DIMENSIONS, len(label_encoder.classes_)).to(device)
        train_residuals = train_full_precision(encode, model)

        # Compute residuals for anomaly detection
        residuals = []
        y_true = []
        for x, y in zip(X_test, Y_test):
            x = x.unsqueeze(0).to(device)
            x_hv = encode(x)
            
            # Get similarities to all centroids
            similarities = model(x_hv)
            
            # Residual = 1 - max similarity (higher = more anomalous)
            residual = 1 - similarities.max().item()
            residuals.append(residual)
            y_true.append(1 if y == unknown_class else 0)  # 1=anomaly, 0=normal

        residuals = np.array(residuals)
        y_true = np.array(y_true)

        # AUROC (no threshold needed)
        fold_metrics['auroc'].append(roc_auc_score(y_true, residuals))

        # Thresholding (95th percentile of normal class residuals)
        threshold = np.percentile(train_residuals, 90)
        y_pred = (residuals > threshold).astype(int)

        tn = ((y_pred == 0) & (y_true == 0)).sum()  # True negatives
        fp = ((y_pred == 1) & (y_true == 0)).sum()  # False positives
        specificity = tn / (tn + fp)                # (out of 100 true labels how many were detected true)
        fold_metrics['specificity'].append(specificity)

        # Compute metrics
        fold_metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
        fold_metrics['recall'].append(recall_score(y_true, y_pred))
        fold_metrics['f1'].append(f1_score(y_true, y_pred))

    # Store average metrics for this unknown class
    for metric in fold_metrics:
        final_metrics[metric].append(np.mean(fold_metrics[metric]))

# Print final averaged metrics
print("\n=== Final Metrics (Averaged Over All Unknown Classes) ===")
for metric in final_metrics:
    print(f"Mean {metric}: {np.mean(final_metrics[metric]):.4f}")