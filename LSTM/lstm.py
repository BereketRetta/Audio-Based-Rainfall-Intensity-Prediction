import os
import glob
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

def extract_raw_and_save_dataset(audio_path, csv_path, output_file='rnn_raw_rain_dataset.npz'):
    mfccs, labels_to_save = [], []
    check_shape = 0
    csv_file = pd.read_csv(csv_path)
    labels = csv_file["rain intensity(mm/5mins)"]
    idx = 0

    for filename in glob.glob(os.path.join(audio_path, '*.wav')):
        resampled_data, samplerate = librosa.load(filename, res_type='kaiser_best', sr=8000)
        if len(resampled_data) == 40000:
            mfccs.append(resampled_data)
            labels_to_save.append(labels[idx])
        else:
            print(f"Skipping file {filename} as it is less than 5 seconds.")
            continue
        idx += 1

    np.savez_compressed(output_file, mfcc=np.array(mfccs), label=np.array(labels_to_save))

audio_folder_path = 'splitaudio/'
csv_file_path = 'splitaudio/AI4AfricaSWS.csv'
extract_raw_and_save_dataset(audio_folder_path, csv_file_path)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", DEVICE)

class LSTMModel(nn.Module):
    def __init__(self, input_size, num_layers, batch_size):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.input_size,
            num_layers=self.num_layers,
            dropout=0.2,
            bidirectional=False,
        )
        self.fc = nn.Linear(self.input_size, 1)

    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        res = self.fc(output)

        return res, state

    def init_state(self):
        return (
            torch.zeros(self.num_layers, self.batch_size, self.input_size).to(DEVICE),
            torch.zeros(self.num_layers, self.batch_size, self.input_size).to(DEVICE),
        )

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        abs_percentage_error = torch.abs((y_true - y_pred) / (y_true))
        mape = torch.mean(abs_percentage_error)

        return mape

class RainfallDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.Tensor(self.features[idx])
        y = torch.Tensor([self.labels[idx]])

        return x, y

# Load and prepare dataset
rain_dataset = np.load('rnn_raw_rain_dataset.npz', allow_pickle=True)
X = rain_dataset["mfcc"]
Y = rain_dataset["label"]

config = {
    "lr": 0.002,
    "epochs": 300,
    "batch_size": 32,
    "sequence_length": 40,
    "input_size": int(len(X[0]) / 40),  # Updated dynamically
    "num_layers": 3
}

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
train_dataset = RainfallDataset(X_train, Y_train)
val_dataset = RainfallDataset(X_val, Y_val)

train_loader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=True)

# Model, loss, and optimizer
model = LSTMModel(input_size=config["input_size"], num_layers=config["num_layers"], batch_size=config["batch_size"])
criterion_1 = nn.MSELoss()
criterion_2 = MAPELoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, criterion, optimizer):
    state_h, state_c = model.init_state()
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        # print("##########################################")
        # print("Input shape:", inputs.shape)
        inputs = inputs.reshape(config["sequence_length"], config["batch_size"], config["input_size"])
        inputs = inputs.to(torch.float32).to(DEVICE)
        targets = targets.to(DEVICE)

        # Calculate the losses using both criteria


        y_pred, (state_h, state_c)  = model(inputs,  (state_h, state_c))
        y_pred = y_pred[-1:, :, :].squeeze(0)
        loss = criterion(y_pred, targets)
        loss.backward()
        total_loss += loss.item()
        # print("Output length:", len(y_pred))
        # print("1 output shape:", y_pred.shape)
        # print("Output", y_pred.shape)

        state_h = state_h.detach()
        state_c = state_c.detach()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)

    return avg_loss

def validate(model, val_loader, criterion1, criterion2):
    state_h, state_c = model.init_state()
    model.eval()
    total_val_loss_1 = 0.0
    total_val_loss_2 = 0.0

    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs = val_inputs.reshape(config["sequence_length"], config["batch_size"], config["input_size"])
            val_inputs = val_inputs.to(torch.float32).to(DEVICE)
            val_targets = val_targets.to(DEVICE)

            val_outputs, (state_h, state_c) = model(val_inputs, (state_h, state_c))

            # Select the last time step output
            val_outputs = val_outputs[-1:, :, :].squeeze(0)


            # Calculate the validation losses using both criteria
            val_loss_1 = criterion1(val_outputs, val_targets)
            val_loss_2 = criterion2(val_outputs, val_targets)

            total_val_loss_1 += val_loss_1.item()
            total_val_loss_2 += val_loss_2.item()

            state_h = state_h.detach()
            state_c = state_c.detach()

    avg_val_loss_1 = total_val_loss_1 / len(val_loader)
    avg_val_loss_2 = total_val_loss_2 / len(val_loader)

    return avg_val_loss_1, avg_val_loss_2

# Lists to store the training and validation loss for each epoch for both criteria
train_losses = []
val_losses_1, val_losses_2 = [], []

# Variables to keep track of the best validation losses for both criteria
best_val_loss = float('inf')

for epoch in range(config["epochs"]):

    train_loss = train(model, train_loader, criterion_1, optimizer)
    val_loss_1, val_loss_2 = validate(model, val_loader, criterion_1, criterion_2)

    # Append the training and validation loss for both criteria to the lists
    train_losses.append(train_loss)
    val_losses_1.append(val_loss_1)
    val_losses_2.append(val_loss_2)

    # Print or log the training and validation loss for both criteria
    print(f'Epoch [{epoch + 1}/{config["epochs"]}], Train Loss : {train_loss:.4f}, Val Loss MAPE : {val_loss_1:.4f}, Val Loss MSE : {val_loss_2:.4f}')

    # Check if the current validation loss for criterion 1 is the best so far
    if val_loss_1 < best_val_loss:
        # Save the model state for criterion 1
        torch.save(model.state_dict(), 'best_lstm_model.pth')
        best_val_loss = val_loss_1

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(range(1, config["epochs"] + 1), train_losses, label='Train Loss (MAPE)')
plt.plot(range(1, config["epochs"] + 1), val_losses_1, label='Val Loss (MAPE)')

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs for MAPE')
plt.legend()
plt.show()

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