import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import KFold
from PIL import Image
from torchvision import transforms
import librosa
import librosa.display
import matplotlib.pyplot as plt
import zipfile

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
config = {
    "lr": 0.002,
    "epochs": 10,
    "batch_size": 32
}

# Extract audio files from zip
path = './splitaudio-20231202T200607Z-001.zip'
with zipfile.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall('./')

# Define the path to the directory containing audio files
audio_path = './splitaudio'

# Function to extract MFCCs and save dataset
def extract_mfcc_and_save_dataset(audio_path, csv_path, output_file='rain_dataset.npz'):
    # Initialize list to store mfccs
    mfccs = []

    # Loop through audio files
    for filename in glob.glob(os.path.join(audio_path, '*.wav')):
        # Load audio data
        data, samplerate = librosa.load(filename, res_type='kaiser_best')

        # Extract mean of MFCCs (excluding the first coefficient)
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=22050, n_mfcc=21)[1:].T, axis=0)
        mfccs.append(mfcc)

    # Read labels from CSV file
    csv_file = pd.read_csv(csv_path)
    labels = csv_file["rain intensity(mm/5mins)"]

    # Save mfccs and labels to npz file
    np.savez_compressed(output_file, mfcc=np.array(mfccs), label=np.array(labels))

# Example usage of the function
csv_file_path = './splitaudio/AI4AfricaSWS.csv'
extract_mfcc_and_save_dataset(audio_path, csv_file_path)

# Load the rain dataset
rain_dataset = np.load('./rain_dataset.npz')
X, Y = rain_dataset["mfcc"], rain_dataset["label"]

# Generate and save Mel-spectrogram for each audio file
audio_files = [os.path.join(audio_path, file) for file in os.listdir(audio_path) if file.endswith('.wav')]
for file in audio_files:
    y, sr = librosa.load(file)

    # Generate a Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot and save the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-frequency spectrogram of {os.path.basename(file)}')
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'./spectrograph/{os.path.basename(file).replace(".wav", ".png")}')
    plt.close()

# Dataset for spectrogram images
class SpectrogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor()           # Convert images to PyTorch tensors
])

# Loading dataset
spectrogram_dir = './spectrograph'
spectrogram_paths = [os.path.join(spectrogram_dir, fname) for fname in os.listdir(spectrogram_dir) if fname.endswith('.png')]
dataset = SpectrogramDataset(spectrogram_paths, Y, transform=transform)

# Split dataset into training and validation sets
total_size = len(dataset)
train_size = int(total_size * 0.8)  # 80% for training
validation_size = total_size - train_size  # 20% for validation

labels = rain_dataset["label"]

unique_labels = np.unique(labels)
unique_labels = set(labels)

num_unique_labels = len(unique_labels)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=num_unique_labels):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 32, num_blocks[3], stride=2)
        self.linear = nn.Linear(32 * 8 * 8, num_unique_labels)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        # print(out.size())
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

# Create a ResNet18 model
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        abs_percentage_error = torch.abs((y_true - y_pred) / (y_true))
        mape = torch.mean(abs_percentage_error)

        return mape

# Training and validation functions
def train(model, train_loader, criterion1, criterion2, optimizer):
    model.train()
    total_loss1 = 0.0
    total_loss2 = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        outputs = outputs.float()
        labels = labels.float()

        outputs = outputs.view(-1)
        labels = labels.view(-1, 1)

        # Calculate the losses using both criteria
        loss1 = criterion1(outputs, labels)
        loss2 = criterion2(outputs, labels)

        loss = loss1 + loss2
        loss.backward()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()

        optimizer.step()

    avg_loss1 = total_loss1 / len(train_loader)
    avg_loss2 = total_loss2 / len(train_loader)

    return avg_loss1, avg_loss2

def validate(model, val_loader, criterion1, criterion2):
    model.eval()
    val_loss1 = 0.0
    val_loss2 = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.long()
            val_outputs = model(inputs)

            val_outputs = val_outputs.view(-1)
            labels = labels.view(-1, 1)

            # Calculate the validation losses using both criteria
            val_loss_1 = criterion1(val_outputs, labels)
            val_loss_2 = criterion2(val_outputs, labels)

            val_loss1 += val_loss_1.item()
            val_loss2 += val_loss_2.item()

    avg_val_loss1 = val_loss1 / len(val_loader)
    avg_val_loss2 = val_loss2 / len(val_loader)

    return avg_val_loss1, avg_val_loss2

# K-Fold Cross-Validation setup
k = 10
kf = KFold(n_splits=k, shuffle=True)

# Lists to store the training and validation loss for each epoch for both criteria across all folds
all_train_losses_criterion1 = []
all_train_losses_criterion2 = []
all_val_losses_criterion1 = []
all_val_losses_criterion2 = []

for fold, (train_index, val_index) in enumerate(kf.split(X, Y)):
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    # Create DataLoaders for both sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    model = ResNet18()  # Initialize your model here
    model.to(device)
    criterion1 = MAPELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    best_val_loss_criterion1 = float('inf')
    best_val_loss_criterion2 = float('inf')

    train_losses_criterion1 = []
    train_losses_criterion2 = []
    val_losses_criterion1 = []
    val_losses_criterion2 = []

    for epoch in range(config["epochs"]):

        train_loss_criterion1, train_loss_criterion2 = train(model, train_loader, criterion1, criterion2, optimizer)
        val_loss_criterion1, val_loss_criterion2 = validate(model, validation_loader, criterion1, criterion2)

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
