# -*- coding: utf-8 -*-
"""RFUAV-Net.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EL9io24J0M6_d_w_a4SAvvEzOHRVhK32

## Implementation of RFUAV-net
efficient CNN method - 1D convolution
"""

import os
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from helper_functions import *
from loading_functions import *

import time

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import torchvision.models as models
# from torchmetrics import F1Score

# reload functions & modules
import importlib
import loading_functions
importlib.reload(loading_functions)
from loading_functions import *

from loading_functions import load_dronerf_raw_stream

def compute_min_max_streaming(main_folder, t_seg, chunk_size=1000):
    """
    Compute min/max values in a streaming fashion to avoid memory overload.
    Also determine the full dataset shape.
    """
    min_vals = None
    max_vals = None
    total_samples = 0
    first_chunk_shape = None
    
    data_gen = load_dronerf_raw_stream(main_folder, t_seg, chunk_size=chunk_size, stream=True)  # Streaming loader

    for X_chunk, _, _, _ in data_gen:
        if min_vals is None:
            min_vals = np.min(X_chunk, axis=(0, 2), keepdims=True)
            max_vals = np.max(X_chunk, axis=(0, 2), keepdims=True)
            first_chunk_shape = X_chunk.shape[1:]  # Store feature dimensions (excluding batch size)
        else:
            min_vals = np.minimum(min_vals, np.min(X_chunk, axis=(0, 2), keepdims=True))
            max_vals = np.maximum(max_vals, np.max(X_chunk, axis=(0, 2), keepdims=True))

        total_samples += X_chunk.shape[0]  # Track total number of samples

    dataset_shape = (total_samples,) + first_chunk_shape  # Construct final dataset shape
    return min_vals, max_vals, dataset_shape

# def normalize_data_memmap(main_folder, t_seg, min_vals, max_vals, output_path, dataset_shape, chunk_size=1000):
#     """
#     Normalize data in a memory-mapped fashion, avoiding full memory load.
#     Also returns the corresponding labels.
#     """
#     data_gen = load_dronerf_raw_stream(main_folder, t_seg, chunk_size=chunk_size, stream=True)
    
#     Xs_norm = np.memmap(output_path, dtype=np.float32, mode='w+', shape=dataset_shape)

#     ys_list = []
#     y4s_list = []
#     y10s_list = []

#     idx = 0
#     for X_chunk, ys_chunk, y4s_chunk, y10s_chunk in data_gen:
#         norm_chunk = (X_chunk - min_vals) / (max_vals - min_vals)
#         Xs_norm[idx:idx + X_chunk.shape[0]] = norm_chunk
#         idx += X_chunk.shape[0]

#         ys_list.append(ys_chunk)
#         y4s_list.append(y4s_chunk)
#         y10s_list.append(y10s_chunk)

#     # Concatenate labels into arrays
#     ys_arr = np.concatenate(ys_list, axis=0)
#     y4s_arr = np.concatenate(y4s_list, axis=0)
#     y10s_arr = np.concatenate(y10s_list, axis=0)

#     return Xs_norm, ys_arr, y4s_arr, y10s_arr

def normalize_data_memmap(main_folder, t_seg, min_vals, max_vals, output_path, labels_output_path, dataset_shape, chunk_size=1000, checkpoint_file='checkpoint.txt'):
    """
    Normalize data in a memory-mapped fashion, avoiding full memory load.
    Includes checkpointing to resume if interrupted.
    Stores processed data and labels incrementally.
    """
    # Check if there's a checkpoint file to resume from
    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            start_index = int(f.read().strip())  # Read last processed chunk index
    
    data_gen = load_dronerf_raw_stream(main_folder, t_seg, chunk_size=chunk_size, stream=True)
    
    # Get the full shape for the memmap files
    Xs_norm = np.memmap(output_path, dtype=np.float32, mode='w+', shape=dataset_shape)  # For normalized features
    
    # Assuming ys, y4s, y10s have the same shape as Xs for labeling
    ys_shape = (dataset_shape[0],)
    y4s_shape = (dataset_shape[0],)
    y10s_shape = (dataset_shape[0],)
    
    ys_memmap = np.memmap(labels_output_path[0], dtype=np.int32, mode='w+', shape=ys_shape)
    y4s_memmap = np.memmap(labels_output_path[1], dtype=np.int32, mode='w+', shape=y4s_shape)
    y10s_memmap = np.memmap(labels_output_path[2], dtype=np.int32, mode='w+', shape=y10s_shape)
    
    idx = 0
    for chunk_idx, (X_chunk, ys_chunk, y4s_chunk, y10s_chunk) in enumerate(data_gen):
        # Skip processed chunks if resuming
        if chunk_idx < start_index:
            continue
        
        # Normalize the chunk and store in memory-mapped file
        norm_chunk = (X_chunk - min_vals) / (max_vals - min_vals)
        Xs_norm[idx:idx + X_chunk.shape[0]] = norm_chunk
        ys_memmap[idx:idx + ys_chunk.shape[0]] = ys_chunk
        y4s_memmap[idx:idx + y4s_chunk.shape[0]] = y4s_chunk
        y10s_memmap[idx:idx + y10s_chunk.shape[0]] = y10s_chunk
        
        idx += X_chunk.shape[0]
        
        # Save progress by updating the checkpoint file
        with open(checkpoint_file, 'w') as f:
            f.write(str(chunk_idx + 1))  # Save the index of the next chunk to process
    
    return Xs_norm, ys_memmap, y4s_memmap, y10s_memmap

# Main execution
main_folder = '/home/zebra/shriniwas/DroneRF_extracted/'
t_seg = 0.25  # ms

min_vals, max_vals, dataset_shape = compute_min_max_streaming(main_folder, t_seg)
output_path = '/home/zebra/shriniwas/RFUAV/normalized_x.dat'
labels_output_path = ['/home/zebra/shriniwas/RFUAV/ys.dat',
                      '/home/zebra/shriniwas/RFUAV/y4s.dat',
                      '/home/zebra/shriniwas/RFUAV/y10s.dat']
Xs_norm, ys_arr, y4s_arr, y10s_arr = normalize_data_memmap(main_folder, t_seg, min_vals, max_vals, output_path, dataset_shape)

dataset = DroneData(Xs_norm, y10s_arr)  # Assume y10s_arr is loaded correctly

print("INFO: Loaded the dataset, length is -", len(dataset))
"""## Model"""

class RFUAVNet(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, num_classes):
        super(RFUAVNet, self).__init__()
        self.num_classes = num_classes

        self.dense = nn.Linear(320, num_classes)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.smax = nn.Softmax(dim=1)

        # for r unit
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=5, stride=5)
        self.norm1 = nn.BatchNorm1d(num_features=64)
        self.elu1 = nn.ELU(alpha=1.0, inplace=False)

        # setup for components of the gunit
        self.groupconvlist = []
        self.norm2list = []
        self.elu2list = []
        for i in range(4):
            self.groupconvlist.append( nn.Conv1d(
                  in_channels=64,
                  out_channels=64,
                  kernel_size=3,
                  stride = 2,
                  groups=8,
    #               bias=False,
                  dtype=torch.float32
                ))
            self.norm2list.append(nn.BatchNorm1d(num_features=64))
            self.elu2list.append(nn.ELU(alpha=1.0, inplace=False))
        self.groupconv = nn.ModuleList(self.groupconvlist)
        self.norm2 = nn.ModuleList(self.norm2list)
        self.elu2 = nn.ModuleList(self.elu2list)

        # multi-gap implementation
        self.avgpool1000 = nn.AvgPool1d(kernel_size=1000)
        self.avgpool500 = nn.AvgPool1d(kernel_size=500)
        self.avgpool250 = nn.AvgPool1d(kernel_size=250)
        self.avgpool125 = nn.AvgPool1d(kernel_size=125)

    # Progresses data across layers
    def forward(self, x):
        # runit first
        x1 = self.runit(x)
        xg1 = self.gunit(F.pad(x1, (1,0)), 0)
        x2 = self.pool(x1)
        x3 = xg1+x2

        # series of gunits
        xg2 = self.gunit(F.pad(x3, (1,0)), 1)
        x4 = self.pool(x3)
        x5 = xg2+x4

        xg3 = self.gunit(F.pad(x5, (1,0)), 2)
        x6 = self.pool(x5)
        x7 = x6+xg3

        xg4 = self.gunit(F.pad(x7, (1,0)), 3)
        x8 = self.pool(x7)
        x_togap = x8+xg4


        # gap and multi-gap
        f_gap_1 = self.avgpool1000(xg1)
        f_gap_2 = self.avgpool500(xg2)
        f_gap_3 = self.avgpool250(xg3)
        f_gap_4 = self.avgpool125(xg4)

        f_multigap = torch.cat((f_gap_1,f_gap_2, f_gap_3, f_gap_4), 1)

        f_gap_add = self.avgpool125(x_togap)

        f_final = torch.cat((f_multigap, f_gap_add),1)
        f_flat = f_final.flatten(start_dim=1)

        out = self.dense(f_flat)
#         out = self.smax(f_fc)
        # fc_layer

        return out

    def runit(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.elu1(x)
        return x

    def gunit(self, x, n):
        # group convolution layer 8 by 8
        # norm
        # elu
        # n indicates which gunit
        x = self.groupconv[n](x)
        x = self.norm2[n](x)
        x = self.elu2[n](x)
        return x

    def reset_weights(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()

net = RFUAVNet(3)

for layer in net.children():
    if isinstance(layer, nn.ModuleList):
        for item in layer.children():
            print(item)



## Test network
input1 = dataset.__getitem__(40)[0]
# input1 = input1.float()
# input1= input1.type(torch.float)
print(input1.shape)
input1 = torch.unsqueeze(input1, 0)
# input = input.reshape(1, 2, 10000)
# input1 = torch.rand(128, 2, 10000)

# print(input1.shape)

# input_1d = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype = torch.float)

net = RFUAVNet(3)
out = net(input1)

# print(out.shape)
# print(out)

"""## Training & Testing"""

from nn_functions import runkfoldcv

# Network Hyperparameters
num_classes = 10
batch_size = 128 # 128
learning_rate = 0.01
num_epochs = 5 # 0
momentum = 0.95
l2reg = 1e-4

## Set up Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = RFUAVNet(num_classes)
# model = model.to(device)

k_folds = 5
avg_acc, mean_f1s, mean_runtime = runkfoldcv(
    model, dataset, device, k_folds, batch_size, learning_rate, num_epochs, momentum, l2reg)

"""### Single fold train & test development code"""

# Set up data and parameters
batch_size = 128
num_classes = len(set(ys_arr))
learning_rate = 0.01
num_epochs = 5 # 0
momentum = 0.95
l2reg = 1e-4

## Set up Data
train_split_percentage = 0.9
split_lengths = [int(train_split_percentage*len(dataset)), len(dataset)-int(train_split_percentage*len(dataset))]
train_set, test_set = torch.utils.data.random_split(dataset, split_lengths)

train_loader = torch.utils.data.DataLoader(dataset = train_set,
                                           batch_size = batch_size,
                                           shuffle = True)


test_loader = torch.utils.data.DataLoader(dataset = test_set,
                                           batch_size = batch_size,
                                           shuffle = True)


## Set up Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = RFUAVNet(num_classes)
model = model.to(device)

# Set Loss function with criterion
criterion = nn.CrossEntropyLoss()

# Set optimizer with optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2reg, momentum = momentum)

total_step = len(train_loader)


# Training
# We use the pre-defined number of epochs to determine how many iterations to train the network on
for epoch in range(num_epochs):
    #Load in the data in batches using the train_loader object
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.float()
        inputs = torch.squeeze(inputs, 1)
#         labels = labels.type(torch.float)

        # Move tensors to the configured device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%50 == 49:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

## Check accuracy
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = torch.squeeze(inputs, 1)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
#         print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} train images: {} %'.format(total, 100 * correct / total))

# # KFoldCV original implementation
# k_folds = 10

# num_classes =10

# if num_classes == 2:
#     f1type = 'binary'
# else:
#     f1type = 'weighted' # is this the best choice


# # For fold results
# results = {}
# runtimes = np.zeros(k_folds)
# f1s = np.zeros(k_folds)

# # Define the K-fold Cross Validator
# kfold = KFold(n_splits=k_folds, shuffle=True)

# # Start print
# print('--------------------------------')

# # K-fold Cross Validation model evaluation
# for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
#     # Print
#     print(f'FOLD {fold}')
#     print('--------------------------------')

#     # Sample elements randomly from a given list of ids, no replacement.
#     train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
#     test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

#     # Define data loaders for training and testing data in this fold
#     trainloader = torch.utils.data.DataLoader(
#                       dataset,
#                       batch_size=batch_size, sampler=train_subsampler)
#     testloader = torch.utils.data.DataLoader(
#                       dataset,
#                       batch_size=batch_size, sampler=test_subsampler)

#     # Init the neural network
#     network = RFUAVNet(num_classes)
#     network = network.to(device)
# #     network.apply(reset_weights)

#     criterion = nn.CrossEntropyLoss()

#     # Initialize optimizer
# #     optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
#     optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, weight_decay=l2reg, momentum = momentum)

#     # Run the training loop for defined number of epochs
#     for epoch in range(0, num_epochs):
#         # Print epoch
#         print(f'Starting epoch {epoch+1}')

#         # Set current loss value
#         current_loss = 0.0

#         # Iterate over the DataLoader for training data
#         for i, data in enumerate(trainloader):
#             # Get inputs
#             inputs, targets = data
#             targets= targets.type(torch.long)

#             # Move tensors to configured device
#             inputs = inputs.to(device)
#             targets = targets.to(device)

#             # Perform forward pass
#             outputs = network(inputs)

#             # Compute loss
#             loss = criterion(outputs, targets)

#             # Zero the gradients
#             optimizer.zero_grad()

#             # Perform backward pass
#             loss.backward()

#             # Perform optimization
#             optimizer.step()

#             # Print statistics
#             current_loss += loss.item()
#             if i % 50 == 49:
#                 print('    Loss after mini-batch %5d: %.5f' %
#                       (i + 1, current_loss / 50))
#                 current_loss = 0.0
# #         print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

#     # Process is complete.
# #     print('Training process has finished. Saving trained model.')

#     # Print about testing
#     print('Starting testing')
#     print('----------------')

#     # Saving the model
# #     save_path = f'./model-fold-{fold}.pth'
# #     torch.save(network.state_dict(), save_path)

#     # Evaluation for this fold
#     correct, total = 0, 0
#     network.eval()
#     with torch.no_grad():
#         starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#         runtimes_thisfold = []
#         f1s_thisfold = []
#         # Iterate over the test data and generate predictions
#         for i, data in enumerate(testloader, 0):
#             # Get inputs
#             inputs, targets = data
#             inputs = inputs.to(device)
#             targets = targets.to(device)

#             # Generate outputs
#             n_instances = len(inputs)
#             ys = torch.empty(n_instances)
#             ys = ys.to(device)

#             for i in range(n_instances):
#                 instance = inputs[i]
#                 instance = instance.float()
#                 start = time.time()
#                 starter.record()
#                 yi = network(instance)
#                 _,pred = torch.max(yi,1)
#                 ender.record()

#                 torch.cuda.synchronize()
#                 curr_time = starter.elapsed_time(ender) #miliseconds

#                 runtimes_thisfold.append(curr_time*1e-3)
#                 ys[i] = pred


#             # Set total and correct
#             total += targets.size(0)
#             correct += (ys == targets).sum().item()
#             f1i = f1_score(ys.cpu().numpy(), targets.cpu().numpy(), average=f1type)
#             f1s_thisfold.append(f1i)

#         mean_runtime = np.mean(np.array(runtimes_thisfold))
#         mean_f1 = np.mean(np.array(f1s_thisfold))

#     # Summarize and print results
#     results[fold] = 100.0 * (correct / total)
#     runtimes[fold] = mean_runtime
#     f1s[fold] = mean_f1
#     print('Accuracy for fold %d: %.2f %%' % (fold, 100.0 * correct / total))
#     print('F1 for fold %d: %.2f ' % (fold, mean_f1))
#     print('Runtime for fold %d: %.4f s' % (fold, mean_runtime))
#     print('--------------------------------')

# # Print fold results
# print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
# print('--------------------------------')
# sum = 0.0
# for key, value in results.items():
#     print(f'Fold {key}: {value} %')
#     sum += value
# print(f'Average Accuracy: {sum/len(results.items())} %')
# print(f'Average F1: {np.mean(f1s)}')
# print(f'Average Runtime: {np.mean(runtimes)} s')