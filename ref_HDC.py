############################## Libraries ################################
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torchhd.models import Centroid
from torchmetrics import Accuracy
from tqdm import tqdm

############################## Functions ###############################
def decode(datum):
    y = np.zeros((datum.shape[0], 1))
    for i in range(datum.shape[0]):
        y[i] = np.argmax(datum[i])
    return y

def encode(datum):
    return np.eye(int(np.max(datum) + 1))[datum.astype(int)].squeeze()

############################# Parameters ###############################
np.random.seed(1)
K = 10  # Number of folds for cross-validation
dimension = 10000  # Hypervector dimension (typically 10,000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################ Dummy Data for testing HDC model
def generate_dummy_data(num_samples=1000, num_features=2047):
    """
    Generate dummy data where:
    - If >85% of elements in a sample are 1, label is 1.
    - If <15% of elements are 1, label is 0.
    - No sample has between 15% and 85% ones.
    
    Args:
    - num_samples: Number of samples to generate.
    - num_features: Number of features in each sample.
    
    Returns:
    - X: Input data (num_samples, num_features).
    - y: Labels (num_samples, 1).
    """
    X = np.zeros((num_samples, num_features), dtype=int)
    y = np.zeros((num_samples, 1), dtype=int)

    for i in range(num_samples):
        if np.random.rand() < 0.5:  # Randomly decide label 0 or 1
            ones_count = np.random.randint(0, int(0.1 * num_features) + 1)  # ≤15% ones
            y[i] = 0
        else:
            ones_count = np.random.randint(int(0.9 * num_features), num_features + 1)  # ≥85% ones
            y[i] = 1

        # Randomly pick the indices to set to 1
        ones_indices = np.random.choice(num_features, ones_count, replace=False)
        X[i, ones_indices] = 1  # Set the selected indices to 1

    return X, y

x, y = generate_dummy_data(num_samples=219, num_features=2047)
y = encode(y)


################################ Main ####################################
# Random Projection Encoder
class RandomProjectionEncoder(nn.Module):
    def __init__(self, out_features, in_features):
        super(RandomProjectionEncoder, self).__init__()
        self.projection_matrix = torch.randn(out_features, in_features).to(device)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        sample_hv = torch.matmul(self.projection_matrix, x.T).T  # Project input into hypervector space
        sample_hv = torch.sign(sample_hv)  # Binarize hypervectors
        return sample_hv

# Training function
def train_full_precision(encode, model, x_train, y_train):
    """
    Train the Centroid model using encoded hypervectors.
    """
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    with torch.no_grad():
        for i in tqdm(range(len(x_train_tensor)), desc="Training"):
            sample = x_train_tensor[i].unsqueeze(0)  # Add batch dimension
            label = y_train_tensor[i]

            # Encode the sample using the random projection matrix
            sample_hv = encode(sample)

            # Add the encoded hypervector to the model
            model.add(sample_hv, label)

# Testing function
def test_model(encode, model, x_test, y_test):
    """
    Test the Centroid model using encoded hypervectors.
    """
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    accuracy = Accuracy(task="multiclass", num_classes=int(np.max(y_test) + 1)).to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(x_test_tensor)), desc="Testing"):
            sample = x_test_tensor[i].unsqueeze(0)  # Add batch dimension
            label = y_test_tensor[i]

            # Encode the test sample using the random projection matrix
            sample_hv = encode(sample)

            # Get the prediction from the Centroid model
            pred = model(sample_hv)

            # print("pred and label ", pred, label)

            # Compute accuracy
            correct += (torch.argmax(pred, dim=1) == label).sum().item()  # pred_probs: [1, num_classes], label: [1]
            # print("-----", correct, torch.argmax(pred, dim=1) == label, torch.argmax(pred, dim=1), label)
            total += 1

    accuracy_value = correct / total
    return accuracy_value

# Cross-validation
cvscores = []
cnt = 0
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)

for train, test in kfold.split(x, decode(y)):
    cnt += 1
    print(f"Fold {cnt}")

    # Initialize Random Projection Encoder
    encoder = RandomProjectionEncoder(out_features=dimension, in_features=x.shape[1]).to(device)

    # Initialize Centroid Model
    centroid_model = Centroid(dimension, 2).to(device)

    # Train the Centroid Model
    train_full_precision(encoder, centroid_model, x[train], decode(y[train]))

    # Evaluate the Centroid Model on the test set
    accuracy = test_model(encoder, centroid_model, x[test], decode(y[test]))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    cvscores.append(accuracy * 100)

    # Save results (optional)
    # np.savetxt(f"Results_3_{cnt}.csv", np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s')

# Print average accuracy across all folds
print(f"Average Accuracy: {np.mean(cvscores):.2f}%")