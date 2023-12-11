import os
import glob
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Configuration
config = {
    "lr": 0.001,
    "epochs": 500,
    "batch_size": 45
}

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        abs_percentage_error = torch.abs((y_true - y_pred) / (y_true))
        mape = torch.mean(abs_percentage_error)

        return mape
# Define path for audio files
path = '/content/drive/MyDrive/splitaudio/'

# Function to extract MFCC and save dataset
def extract_mfcc_and_save_dataset(audio_path, csv_path, output_file='rain_dataset.npz'):
    mfccs = []
    for filename in glob.glob(os.path.join(audio_path, '*.wav')):
        data, samplerate = librosa.load(filename, res_type='kaiser_best')
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=22050, n_mfcc=21)[1:].T, axis=0)
        mfccs.append(mfcc)

    csv_file = pd.read_csv(csv_path)
    labels = csv_file["rain intensity(mm/5mins)"]
    np.savez_compressed(output_file, mfcc=np.array(mfccs), label=np.array(labels))

# CNN model definition
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=36, kernel_size=9, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=36, out_channels=72, kernel_size=5, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=72, out_channels=128, kernel_size=3, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(768, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, input_data):
        input_data = input_data.reshape((input_data.shape[0], 20, 1)).permute(0, 2, 1)
        return self.model(input_data)

# Large CNN model definition
class LargeCNNModel(nn.Module):
    def __init__(self):
        super(LargeCNNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.activation1 = nn.GELU()  # Use GELU activation
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.activation2 = nn.GELU()  # Use GELU activation
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.activation3 = nn.GELU()  # Use GELU activation
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.activation4 = nn.GELU()  # Use GELU activation
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.activation5 = nn.GELU()  # Use GELU activation
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers with Dropout
        self.fc1 = nn.Linear(9216, 1024)
        self.dropout1 = nn.Dropout(0.1)  # Add Dropout with a probability of 0.5
        self.activation6 = nn.GELU()

        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.1)
        self.activation7 = nn.GELU()

        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.1)
        self.activation8 = nn.GELU()

        self.fc4 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(0.1)
        self.activation9 = nn.GELU()

        self.fc5 = nn.Linear(128, 64)
        self.dropout5 = nn.Dropout(0.1)
        self.activation10 = nn.GELU()

        self.fc6 = nn.Linear(64, 32)
        self.dropout6 = nn.Dropout(0.1)
        self.activation11 = nn.GELU()

        self.fc7 = nn.Linear(32, 16)
        self.dropout7 = nn.Dropout(0.1)
        self.activation12 = nn.GELU()

        self.fc8 = nn.Linear(16, 1)

    def forward(self, x):
        # Input shape: (batch_size, channels, height, width)
        x = x.unsqueeze(1)  # Add a channel dimension for grayscale image

        # Convolutional layers
        x = self.pool1(self.activation1(self.conv1(x)))
        x = self.pool2(self.activation2(self.conv2(x)))
        x = self.pool3(self.activation3(self.conv3(x)))
        x = self.pool4(self.activation4(self.conv4(x)))
        x = self.pool5(self.activation5(self.conv5(x)))

        # Flatten before fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with Dropout
        x = self.activation6(self.dropout1(self.fc1(x)))
        x = self.activation7(self.dropout2(self.fc2(x)))
        x = self.activation8(self.dropout3(self.fc3(x)))
        x = self.activation9(self.dropout4(self.fc4(x)))
        x = self.activation10(self.dropout5(self.fc5(x)))
        x = self.activation11(self.dropout6(self.fc6(x)))
        x = self.activation12(self.dropout7(self.fc7(x)))

        x = self.fc8(x)

        return x

# Rainfall dataset class
class RainfallDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.Tensor(self.features[idx])
        y = torch.Tensor([self.labels[idx]])

        return x, y

# Functions for training and validation
def train(model, train_loader, criterion1, criterion2, optimizer, device):
    model.train()
    total_loss1 = 0.0
    total_loss2 = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU

        optimizer.zero_grad()
        outputs = model(inputs)

        # Calculate the losses using both criteria
        loss1 = criterion1(outputs, targets)
        loss2 = criterion2(outputs, targets)

        loss = loss1 + loss2
        loss.backward()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()

        optimizer.step()

    avg_loss1 = total_loss1 / len(train_loader)
    avg_loss2 = total_loss2 / len(train_loader)

    return avg_loss1, avg_loss2

def validate(model, val_loader, criterion1, criterion2, device):
    model.eval()
    val_loss1 = 0.0
    val_loss2 = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)  # Move data to GPU

            val_outputs = model(val_inputs)

            # Calculate the validation losses using both criteria
            val_loss_1 = criterion1(val_outputs, val_targets)
            val_loss_2 = criterion2(val_outputs, val_targets)

            val_loss1 += val_loss_1.item()
            val_loss2 += val_loss_2.item()

    avg_val_loss1 = val_loss1 / len(val_loader)
    avg_val_loss2 = val_loss2 / len(val_loader)

    return avg_val_loss1, avg_val_loss2

# Example usage of extract_mfcc_and_save_dataset function
audio_folder_path = '/content/drive/MyDrive/splitaudio/'
csv_file_path = '/content/drive/MyDrive/splitaudio/AI4AfricaSWS.csv'
extract_mfcc_and_save_dataset(audio_folder_path, csv_file_path)

# Load dataset, preprocess, and split
rain_dataset = np.load('/content/rain_dataset.npz')
X = rain_dataset["mfcc"]
Y = rain_dataset["label"]

# Standardize MFCC data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Fold Cross-Validation setup
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=0)

# Main training loop with K-Fold Cross-Validation

# Lists to store the training and validation loss for each epoch for both criteria across all folds
all_train_losses_criterion1 = []
all_train_losses_criterion2 = []
all_val_losses_criterion1 = []
all_val_losses_criterion2 = []

for fold, (train_index, val_index) in enumerate(kf.split(X, Y)):
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]

    train_dataset = RainfallDataset(X_train, Y_train)
    val_dataset = RainfallDataset(X_val, Y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config["batch_size"], shuffle=False)

    model = CNNModel()  # Initialize your model here
    criterion1 = MAPELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    best_val_loss_criterion1 = float('inf')
    best_val_loss_criterion2 = float('inf')

    train_losses_criterion1 = []
    train_losses_criterion2 = []
    val_losses_criterion1 = []
    val_losses_criterion2 = []

    for epoch in range(config["epochs"]):

        train_loss_criterion1, train_loss_criterion2 = train(model, train_loader, criterion1, criterion2, optimizer)
        val_loss_criterion1, val_loss_criterion2 = validate(model, val_loader, criterion1, criterion2)

        # Append the training and validation loss for both criteria to the lists
        train_losses_criterion1.append(train_loss_criterion1)
        train_losses_criterion2.append(train_loss_criterion2)
        val_losses_criterion1.append(val_loss_criterion1)
        val_losses_criterion2.append(val_loss_criterion2)

        # Print or log the training and validation loss for both criteria
        print(f'Fold {fold + 1}, Epoch [{epoch + 1}/{config["epochs"]}], Train Loss (MAPE): {train_loss_criterion1:.4f}, Train Loss (MSE): {train_loss_criterion2:.4f}, Val Loss (MAPE): {val_loss_criterion1:.4f}, Val Loss (MSE): {val_loss_criterion2:.4f}')

        # Check if the current validation loss for MAPE is the best so far
        if val_loss_criterion1 < best_val_loss_criterion1 and val_loss_criterion2 < best_val_loss_criterion2:
            # Save the model state for MAPE
            torch.save(model.state_dict(), f'best_model_criterion1_fold{fold + 1}.pth')
            best_val_loss_criterion1 = val_loss_criterion1
            best_val_loss_criterion2 = val_loss_criterion2

    # After the epoch loop for each fold
    # Append the final training and validation losses for this fold
    all_train_losses_criterion1.append(train_losses_criterion1)
    all_train_losses_criterion2.append(train_losses_criterion2)
    all_val_losses_criterion1.append(val_losses_criterion1)
    all_val_losses_criterion2.append(val_losses_criterion2)

# Calculate the average training and validation losses over all folds
avg_train_loss_criterion1 = np.mean(all_train_losses_criterion1, axis=0)
avg_train_loss_criterion2 = np.mean(all_train_losses_criterion2, axis=0)
avg_val_loss_criterion1 = np.mean(all_val_losses_criterion1, axis=0)
avg_val_loss_criterion2 = np.mean(all_val_losses_criterion2, axis=0)

# Print or log the average performance
print(f'Average Train Loss (MAPE): {avg_train_loss_criterion1[-1]:.4f}')
print(f'Average Train Loss (MSE): {avg_train_loss_criterion2[-1]:.4f}')
print(f'Average Val Loss (MAPE): {avg_val_loss_criterion1[-1]:.4f}')
print(f'Average Val Loss (MSE): {avg_val_loss_criterion2[-1]:.4f}')

# Plotting the average performance
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(avg_train_loss_criterion1) + 1), avg_train_loss_criterion1, label='Average Train Loss (MAPE)')
plt.plot(range(1, len(avg_val_loss_criterion1) + 1), avg_val_loss_criterion1, label='Average Val Loss (MAPE)')

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Training and Validation Loss over Epochs for MAPE')
plt.legend()
plt.show()

# Instantiate the model, loss function, and optimizer
model = CNNModel()
criterion1 = MAPELoss()
criterion2 = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

# Load the rain dataset
rain_dataset = np.load('rain_dataset.npz')

# Extract the MFCC and label data from the dataset
X = rain_dataset["mfcc"]
Y = rain_dataset["label"]

# Define the number of folds (e.g., k = 5)
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=0)

# Lists to store the training and validation loss for each epoch for both criteria across all folds
all_train_losses_criterion1 = []
all_train_losses_criterion2 = []
all_val_losses_criterion1 = []
all_val_losses_criterion2 = []

for fold, (train_index, val_index) in enumerate(kf.split(X, Y)):
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]

    train_dataset = RainfallDataset(X_train, Y_train)
    val_dataset = RainfallDataset(X_val, Y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config["batch_size"], shuffle=False)

    model = CNNModel()  # Initialize your model here
    criterion1 = MAPELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    best_val_loss_criterion1 = float('inf')
    best_val_loss_criterion2 = float('inf')

    train_losses_criterion1 = []
    train_losses_criterion2 = []
    val_losses_criterion1 = []
    val_losses_criterion2 = []

    for epoch in range(config["epochs"]):

        train_loss_criterion1, train_loss_criterion2 = train(model, train_loader, criterion1, criterion2, optimizer)
        val_loss_criterion1, val_loss_criterion2 = validate(model, val_loader, criterion1, criterion2)

        # Append the training and validation loss for both criteria to the lists
        train_losses_criterion1.append(train_loss_criterion1)
        train_losses_criterion2.append(train_loss_criterion2)
        val_losses_criterion1.append(val_loss_criterion1)
        val_losses_criterion2.append(val_loss_criterion2)

        # Print or log the training and validation loss for both criteria
        print(f'Fold {fold + 1}, Epoch [{epoch + 1}/{config["epochs"]}], Train Loss (MAPE): {train_loss_criterion1:.4f}, Train Loss (MSE): {train_loss_criterion2:.4f}, Val Loss (MAPE): {val_loss_criterion1:.4f}, Val Loss (MSE): {val_loss_criterion2:.4f}')

        # Check if the current validation loss for MAPE is the best so far
        if val_loss_criterion1 < best_val_loss_criterion1 and val_loss_criterion2 < best_val_loss_criterion2:
            # Save the model state for MAPE
            torch.save(model.state_dict(), f'best_model_criterion1_fold{fold + 1}.pth')
            best_val_loss_criterion1 = val_loss_criterion1
            best_val_loss_criterion2 = val_loss_criterion2

    # After the epoch loop for each fold
    # Append the final training and validation losses for this fold
    all_train_losses_criterion1.append(train_losses_criterion1)
    all_train_losses_criterion2.append(train_losses_criterion2)
    all_val_losses_criterion1.append(val_losses_criterion1)
    all_val_losses_criterion2.append(val_losses_criterion2)

# Calculate the average training and validation losses over all folds
avg_train_loss_criterion1 = np.mean(all_train_losses_criterion1, axis=0)
avg_train_loss_criterion2 = np.mean(all_train_losses_criterion2, axis=0)
avg_val_loss_criterion1 = np.mean(all_val_losses_criterion1, axis=0)
avg_val_loss_criterion2 = np.mean(all_val_losses_criterion2, axis=0)

# Print or log the average performance
print(f'Average Train Loss (MAPE): {avg_train_loss_criterion1[-1]:.4f}')
print(f'Average Train Loss (MSE): {avg_train_loss_criterion2[-1]:.4f}')
print(f'Average Val Loss (MAPE): {avg_val_loss_criterion1[-1]:.4f}')
print(f'Average Val Loss (MSE): {avg_val_loss_criterion2[-1]:.4f}')

# Plotting the average performance
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(avg_train_loss_criterion1) + 1), avg_train_loss_criterion1, label='Average Train Loss (MAPE)')
plt.plot(range(1, len(avg_val_loss_criterion1) + 1), avg_val_loss_criterion1, label='Average Val Loss (MAPE)')

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Training and Validation Loss over Epochs for MAPE')
plt.legend()
plt.show()
