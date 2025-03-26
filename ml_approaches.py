# -*- coding: utf-8 -*-
"""ML Approaches.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sT4Wv61YmI35K3EceCkT3z7rCYnhQcYc

## PSD+SVM Method
"""

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


feat_name = 'PSD'
t_seg = 250 #ms
n_per_seg = 4096
interferences = ['WIFI', 'BLUE', 'BOTH', 'CLEAN']
output_name = 'drones'
norm_ratio = '02' # 0.xxx mapped to xxx
feat_format = 'ARR'
which_dataset = 'dronerf'
output_tensor = False

if which_dataset == 'dronerf':
    print('Loading DroneRF Dataset')
    highlow = 'L'
    dataset = DroneRFTorch(dronerf_feat_path, feat_name, t_seg, n_per_seg,
                       feat_format, output_name, output_tensor, highlow)
    perturbed_dataset = DroneRFTorchPerturbed(dronerf_feat_path, feat_name, t_seg, n_per_seg,
                        feat_format, output_name, output_tensor, highlow, norm_ratio, output_name)
elif which_dataset == 'dronedetect':
    print('Loading DroneDetect Dataset')
    dataset = DroneDetectTorch(dronedetect_feat_path, feat_name, t_seg, n_per_seg, feat_format,
                                    output_name, output_tensor, interferences)
print("dataset loaded")
# X_use, y_use = dataset.get_arrays()
X_use, y_use = get_arrays_efficient(dataset, batch_size=64)
X_perturbed, y_perturbed = get_arrays_efficient(perturbed_dataset, batch_size=64)

print("ARE WE EVEN PERTURBING: ", not np.allclose(X_use, X_perturbed, atol=1e-5))

## RAND X_PERT ##
# max_value = np.max(X_use)
# X_perturbed = np.random.uniform(0, max_value, size=X_use.shape)
# y_perturbed = y_use

# X_tmp, y_tmp = dataset.get_arrays()

# print("all shapes", X_use.shape, y_use.shape, X_tmp.shape, y_tmp.shape)

# print("Xs equal", np.array_equal(X_use, X_tmp))
# print("Ys equal", np.array_equal(y_use, y_tmp))

# Set fixed number of samples
# n_samps = 15500
# i_test= random.sample(range(len(dataset)), n_samps)
# # i_test= list(range(0,2712,10))
# X_use, y_use = dataset.__getitem__(i_test)
print(X_use.shape)
print("loading finished")

print(y_use[:50])
print(np.sum(y_use=='None'))

"""### Run Model"""

model = PsdSVM(t_seg, n_per_seg)

# accs, f1s, runts = model.run_cv(X_use, y_use, k_fold=5)
accs, f1s, runts = model.run_cv_perturbed(X_use, y_use, X_perturbed, y_perturbed, k_fold=5)

for icv in range(5):
    print(model.cv_models[icv].support_vectors_.shape)

np.mean([12345,12395,12450,12469,12383])

# search through parameters
Cs=list(map(lambda x:pow(2,x),range(-3,10,2)))
gammas=list(map(lambda x:pow(2,x),range(-3,10,2)))
parameters = {'C':Cs, 'gamma':gammas}

k_fold=5

# accs, f1s, runts, best_params = model.run_gridsearch(X_use, y_use, parameters, k_fold)
accs, f1s, runts, best_params = model.run_gridsearch_perturbed(X_use, y_use, X_perturbed, y_perturbed, parameters, k_fold)

#################################### MY OWN TESTING ########################################
# Store accuracy for each fold
fold_accuracies = []
# K-Fold Cross-Validation
k_folds = 5  # You can change this
n_samples_per_class = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_use, y_use)):
    print(f"Fold {fold + 1}/{k_folds}")
    # Split data into train and test sets for this fold
    X_train, X_test = X_use[train_idx], X_use[test_idx]
    Y_train, Y_test = y_use[train_idx], y_use[test_idx]
    svc = svm.SVC(kernel='rbf', C=8, gamma = 512, class_weight='balanced')

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
    Y_pred = svc.predict(X_perturbed[test_idx])
    accuracy = accuracy_score(Y_pred, y_perturbed[test_idx])
    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
    fold_accuracies.append(accuracy)

print(f"Average Accuracy: {np.mean(fold_accuracies):.4f}")
#################################### MY OWN TESTING DONE ########################################

"""### Visualize Results"""

# Set-up train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_use,
                                                    y_use,
                                                    test_size=0.33,
                                                    random_state=None)

# Train & Test
model.train(X_train, y_train)
preds = model.predict(X_test)
show_confusion_matrix(y_test, preds)

"""### Save Model"""

to_train_all = True # whether to retrain using all the data
model_name = which_dataset+'_'+'SVM'+'_'+str(feat_name)+'_'+str(n_per_seg)+'_'+str(t_seg)+'_'+str(int(to_train_all))
model_path = '../temp_figs/'

if to_train_all:
    model.train(X_use, y_use)

model.save(model_path, model_name)

# pickle.dump(svc, open(model_path+model_name, 'wb'))

"""-----------------"""

# import loading_functions
# importlib.reload(loading_functions)
# from loading_functions import *

"""## Pilot Test- Try Model on GamutRF data"""

data_path = '/home/kzhou/Data/S3/leesburg_worker1/Features/'
Xgamut = load_gamut_features(data_path, 'psd')

# normalize data
## Apply normalization
X_gamut_norm = Xgamut
for n in range(len(Xgamut)):
    X_gamut_norm[n] = Xgamut[n]/max(Xgamut[n])

X_gamut_norm.max()

# Feed data through trained SVM model
y_gamut_pred = model.predict(X_gamut_norm)

y_gamut_pred