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
import torchhd.embeddings
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
    
class LevelEncoder(nn.Module):
    def __init__(self, out_features, in_features, levels=1000):
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

        ## NEW DYNAMIC THRESHOLDING
        # Tunable parameter k
        k = 0
        mu = sample_hv.mean()
        sigma = sample_hv.std(unbiased=False)  # use unbiased=False for population std
        threshold = mu + k * sigma
        sample_hv = torch.sign(sample_hv - 0*threshold)
        ## NEW DYNAMIC THRESHOLDING END

        # sample_hv = torch.sign(sample_hv)
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
            model.add_online(samples_hv, labels)

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
seed = 86
perturbation_type = 'uap'

print('Loading DroneRF Dataset')
highlow = 'L'
dataset = DroneRFTorch(dronerf_feat_path, feat_name, t_seg, n_per_seg,
                    feat_format, output_name, output_tensor, highlow)
perturbed_dataset = DroneRFTorchPerturbed(dronerf_feat_path, feat_name, t_seg, n_per_seg,
                    feat_format, output_name, output_tensor, highlow, norm_ratio, perturbation_type, output_name)
print("dataset loaded")
# X_use, y_use = dataset.get_arrays()
X, Y = get_arrays_efficient(dataset, batch_size=64)
X_perturbed, Y_perturbed = get_arrays_efficient(perturbed_dataset, batch_size=64)

label_encoder = LabelEncoder()
Y_int = label_encoder.fit_transform(Y)

X_tensor = torch.tensor(X, dtype=torch.float32)  # (219, 4097)
Y_tensor = torch.tensor(Y_int, dtype=torch.long)  # (219,)

# Y_perturbed = label_encoder.fit_transform(Y_perturbed)
# X_perturbed = torch.tensor(X_perturbed, dtype=torch.float32)  # (219, 4097)
# Y_perturbed = torch.tensor(Y_perturbed, dtype=torch.long)  # (219,)

# K-Fold Cross-Validation
k_folds = 5  # You can change this
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Store accuracy for each fold
fold_accuracies = []

# Few-shot learning: Number of samples per class for training
n_samples_per_class = 5
bws = [0.25*i for i in range(1, 101)]
bws = [21]
seeds = [50]
# seeds.append(11)
# seeds.append(86)
optimal_params = {'accuracy':0,
                  'bw':0,
                  'seed':0}
for bw in bws:
    for seed in seeds:
        fold_accuracies = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_tensor, Y_tensor)):
            set_seed(seed)
            print(f"Fold {fold + 1}/{k_folds}")

            # Split data into train and test sets for this fold
            X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
            Y_train, Y_test = Y_tensor[train_idx], Y_tensor[test_idx]

            # Few-shot learning: Select `n_samples_per_class` for each class
            few_shot_train_indices = []
            for class_label in torch.unique(Y_train):
                class_indices = torch.where(Y_train == class_label)[0]
                selected_indices = np.random.choice(class_indices, size=n_samples_per_class, replace=False)
                few_shot_train_indices.extend(selected_indices)

            # Use only the selected few-shot samples for training
            # X_train = X_train[few_shot_train_indices]
            # Y_train = Y_train[few_shot_train_indices]

            # Create DataLoader for training and testing
            train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
            test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

            train_ld = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
            test_ld = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


            # encode = RandomProjectionEncoder(DIMENSIONS, in_features).to(device)
            # encode = RFFEncoder(in_features, DIMENSIONS, bw, seed).to(device)
            encode = LevelEncoder(DIMENSIONS, in_features, levels=1000)
            model = Centroid(DIMENSIONS, len(label_encoder.classes_)).to(device)

            
            # Train the model
            train_full_precision(encode, model)
            # model.normalize()

            # Test the model
            accuracy_value = test_model(encode, model)
            fold_accuracies.append(accuracy_value)
            print(f"Fold {fold + 1} Accuracy: {accuracy_value:.4f}")

        # Print average accuracy across all folds
        print(f"Average Accuracy: {np.mean(fold_accuracies):.4f}")
        if np.mean(fold_accuracies) > optimal_params['accuracy']:
            optimal_params['accuracy'] = np.mean(fold_accuracies)
            optimal_params['bw'] = bw
            optimal_params['seed'] = seed
print(optimal_params)