import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader

from helper_functions import *
from latency_helpers import *
from loading_functions import *
from file_paths import *
from models import *
import importlib


import random

import importlib
import loading_functions
importlib.reload(loading_functions)
from loading_functions import *

"""### Load Features"""


def get_arrays_efficient(dataset, batch_size=64):
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


feat_name = 'LFCC'
t_seg = 250 #ms
n_per_seg = 4096
interferences = ['WIFI', 'BLUE', 'BOTH', 'CLEAN']
output_name = 'modes'
norm_ratio = '04' # 0.xxx mapped to xxx
feat_format = 'ARR'
which_dataset = 'dronerf'
output_tensor = False

if which_dataset == 'dronerf':
    print('Loading DroneRF Dataset')
    highlow = 'LH'
    dataset = DroneRFTorch(dronerf_feat_path, feat_name, t_seg, n_per_seg,
                       feat_format, output_name, output_tensor, highlow)
elif which_dataset == 'dronedetect':
    print('Loading DroneDetect Dataset')
    dataset = DroneDetectTorch(dronedetect_feat_path, feat_name, t_seg, n_per_seg, feat_format,
                                    output_name, output_tensor, interferences)
print("dataset loaded")
X_use, y_use = get_arrays_efficient(dataset, batch_size=64)


print(X_use.shape)
print("loading finished")

print(y_use[:50])
print(np.sum(y_use=='None'))

"""### Run Model"""

model = PsdSVM(t_seg, n_per_seg)

# search through parameters
Cs=list(map(lambda x:pow(2,x),range(-3,10,2)))
gammas=list(map(lambda x:pow(2,x),range(-3,10,2)))
parameters = {'C':Cs, 'gamma':gammas}

k_fold=5

accs, f1s, runts, best_params = model.run_gridsearch(X_use, y_use, parameters, k_fold)
#################################### MY OWN TESTING ########################################
# Store accuracy for each fold
fold_accuracies = []
# K-Fold Cross-Validation
k_folds = 5  # You can change this
n_samples_per_class = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=11)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_use, y_use)):
    print(f"Fold {fold + 1}/{k_folds}")
    # Split data into train and test sets for this fold
    X_train, X_test = X_use[train_idx], X_use[test_idx]
    Y_train, Y_test = y_use[train_idx], y_use[test_idx]
    svc = svm.SVC(kernel='rbf', C=512, gamma = 0.5, class_weight='balanced') # 8, 512

    # Few-shot learning: Select `n_samples_per_class` for each class
    few_shot_train_indices = []
    for class_label in np.unique(Y_train):
        class_indices = np.where(Y_train == class_label)[0]
        selected_indices = np.random.choice(class_indices, size=n_samples_per_class, replace=False)
        few_shot_train_indices.extend(selected_indices)

    # Use only the selected few-shot samples for training
    # X_train = X_train[few_shot_train_indices]
    # Y_train = Y_train[few_shot_train_indices]


    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    accuracy = accuracy_score(Y_pred, Y_test)
    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
    fold_accuracies.append(accuracy)

print(f"Average Accuracy: {np.mean(fold_accuracies):.4f}")
#################################### MY OWN TESTING DONE ########################################