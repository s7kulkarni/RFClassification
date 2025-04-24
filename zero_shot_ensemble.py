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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as pl

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

class RFFEncoder(torch.nn.Module):
    def __init__(self, fdim: int, dim: int = 1024, bw: float = 100, seed: int = 86, device=None):
        super().__init__()
        self.dim = dim
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Covariance matrix for the random projection
        cov = torch.eye(fdim, device=device) * (fdim / (bw**2))
        
        # Create mean vector (zeros)
        mean = torch.zeros(fdim, device=device)
        
        # Sample from multivariate normal for each row
        mvn = torch.distributions.MultivariateNormal(mean, cov)
        projection = mvn.sample((dim,))
        
        # Register as parameters
        self.projection = nn.Parameter(projection, requires_grad=False)
        self.bias = nn.Parameter(torch.rand(dim, device=device) * 2 * torch.pi, requires_grad=False)
        
        self.flatten = torch.nn.Flatten()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on the same device as parameters
        if x.device != self.projection.device:
            x = x.to(self.projection.device)
            
        x = self.flatten(x)
        x = x.squeeze(0)
        
        # Matrix multiplication (equivalent to einsum in JAX)
        proj = torch.matmul(x, self.projection.transpose(0, 1))
        proj = proj + self.bias
        
        # Apply cosine and scaling
        proj = torch.cos(proj) * torch.sqrt(torch.tensor(2.0, device=proj.device) / self.dim)
        
        # Apply sign function and add back batch dimension
        proj = torch.sign(proj)
        proj = proj.unsqueeze(0)
        
        # Check for zero vectors (debugging)
        if (proj.abs().sum(dim=1) == 0).any():
            print("Warning: Zero vector detected in encoder output")
        
        return proj

class SinusoidEncoder(nn.Module):
    def __init__(self, out_features, in_features):
        super(SinusoidEncoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.nonlinear_projection = torchhd.embeddings.Sinusoid(in_features, out_features)

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = self.nonlinear_projection(x)
        return torchhd.hard_quantize(sample_hv)

class LevelEncoder(nn.Module):
    def __init__(self, out_features, in_features, levels):
        super(LevelEncoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = torchhd.embeddings.Random(in_features, out_features)
        self.value = torchhd.embeddings.Level(levels, out_features)

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)

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

def test_model(rp_encode, rff_encode, sin_encode, rp_model, rff_model, sin_model):
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=len(label_encoder.classes_)).to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for samples, labels in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)
            labels = labels.to(device)

            # Encode the test samples
            rp_samples_hv = rp_encode(samples)
            rff_samples_hv = rff_encode(samples)
            sin_samples_hv = sin_encode(samples)

            # Get the predictions from the Centroid model by passing encoded hypervectors
            rp_preds = rp_model(rp_samples_hv)
            rff_preds = rff_model(rff_samples_hv)
            sin_preds = sin_model(sin_samples_hv)

            total_preds = rp_preds + rff_preds + sin_preds

            # Compute accuracy
            correct += accuracy(total_preds, labels)
            total += labels.size(0)

    accuracy_value = correct / total
    return accuracy_value.item()

def test_model_0(encode, model):
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

# ===== INITIALIZATION =====
best_seed_data = {
    'seed': None,
    'y_true': [],
    'residuals': [],
    'auc': 0
}

k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
seeds = [5*i for i in range(14,15)]

for seed in seeds:
    set_seed(seed)
    seed_y_true = []
    seed_residuals = []
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
            train_ld = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(seed))
            test_ld = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

            # Train HDC model (3 classes)
            rp_encode = RandomProjectionEncoder(DIMENSIONS, in_features).to(device)
            rp_model = Centroid(DIMENSIONS, len(label_encoder.classes_)).to(device)

            rff_encode = RFFEncoder(in_features, DIMENSIONS, 21, seed).to(device)
            rff_model = Centroid(DIMENSIONS, len(label_encoder.classes_)).to(device)

            sin_encode = SinusoidEncoder(DIMENSIONS, in_features)
            sin_model = Centroid(DIMENSIONS, len(label_encoder.classes_)).to(device)

            # Train the model
            rp_residual = train_full_precision(rp_encode, rp_model)
            rff_residual = train_full_precision(rff_encode, rff_model)
            sin_residual = train_full_precision(sin_encode, sin_model)

            train_residuals = rp_residual + rff_residual + sin_residual

            # Compute residuals for anomaly detection
            residuals = []
            y_true = []
            for x, y in zip(X_test, Y_test):
                x = x.unsqueeze(0).to(device)

                rp_samples_hv = rp_encode(x)
                rff_samples_hv = rff_encode(x)
                sin_samples_hv = sin_encode(x)

                # Get the predictions from the Centroid model by passing encoded hypervectors
                rp_preds = rp_model(rp_samples_hv)
                rff_preds = rff_model(rff_samples_hv)
                sin_preds = sin_model(sin_samples_hv)

                similarities = rp_preds + rff_preds + sin_preds
                
                # Residual = 1 - max similarity (higher = more anomalous)
                residual = 1 - similarities.max().item()
                residuals.append(residual)
                y_true.append(1 if y == unknown_class else 0)

            # === STORE FOR SEED-LEVEL AUC ===
            seed_y_true.extend(y_true)
            seed_residuals.extend(residuals)
            # AUROC (no threshold needed)
            fold_metrics['auroc'].append(roc_auc_score(y_true, residuals))

            # Thresholding (95th percentile of normal class residuals)
            threshold = np.percentile(train_residuals, 85)
            y_pred = (residuals > threshold).astype(int)

            tn = ((y_pred == 0) & (y_true == 0)).sum()  # True negatives
            fp = ((y_pred == 1) & (y_true == 0)).sum()  # False positives
            specificity = tn / (tn + fp)                # (out of 100 true labels how many were detected true)
            fold_metrics['specificity'].append(specificity)

            # Compute metrics
            fold_metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            fold_metrics['recall'].append(recall_score(y_true, y_pred)) # (out of 100 anomalies, how many were detected anomalies)
            fold_metrics['f1'].append(f1_score(y_true, y_pred))

        # Store average metrics for this unknown class
        for metric in fold_metrics:
            final_metrics[metric].append(np.mean(fold_metrics[metric]))
    seed_auc = roc_auc_score(seed_y_true, seed_residuals)
    if seed_auc > best_seed_data['auc']:
        best_seed_data = {
            'seed': seed,
            'y_true': seed_y_true,
            'residuals': seed_residuals,
            'auc': seed_auc
        }

# ===== PLOT BEST ROC (ONLY ADDITION) =====
fpr, tpr, _ = roc_curve(best_seed_data['y_true'], best_seed_data['residuals'])
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', 
         label=f'Best Seed {best_seed_data["seed"]} (AUC = {best_seed_data["auc"]:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Best Seed ROC Curve')
plt.legend(loc="lower right")
plt.savefig('best_seed_roc.png', dpi=300, bbox_inches='tight')
print(f"\nSaved ROC for best seed {best_seed_data['seed']} (AUC={best_seed_data['auc']:.3f})")

# === ORIGINAL METRIC REPORTING ===
print("\n=== Final Metrics (Averaged Over All Unknown Classes) ===")
for metric in final_metrics:
    print(f"Mean {metric}: {np.mean(final_metrics[metric]):.4f}")
