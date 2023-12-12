# Rainfall Prediction Using 1D and 2D CNN

This repository contains code for a deep learning project aimed at predicting rainfall intensity from audio data. The project leverages various Python libraries and deep learning frameworks, including Pandas, Librosa, TensorFlow, PyTorch, and Sklearn, to preprocess audio data, extract features, and train convolutional neural network (CNN) models for prediction.

## Features

- **Audio Data Processing**: Utilizes Librosa for loading and processing audio files.
- **Feature Extraction**: Extracts Mel-frequency cepstral coefficients (MFCCs) from audio data.
- **Model Training**: Implements CNN models using PyTorch for rainfall intensity prediction.
- **Model Evaluation**: Uses K-fold cross-validation for robust model evaluation.
- **Dataset Management**: Efficient handling of large datasets using Numpy and PyTorch's DataLoader.
- **Loss Function Customization**: Incorporates Mean Absolute Percentage Error (MAPE) and Mean Squared Error (MSE) as loss functions.

## Prerequisites

- Python 3.x
- Pandas
- Librosa
- TensorFlow
- PyTorch
- Sklearn
- Matplotlib
- Numpy

## Installation

1. Clone the repository:
    `git clone [repository-url]`
2. Install required Python packages:
    `pip install -r requirements.txt`

## Usage

1. Prepare your audio data and label CSV in the specified format.
2. Run the script to extract MFCCs and save them:

`extract_mfcc_and_save_dataset(audio_folder_path, csv_file_path)`

Load the dataset, preprocess, and split it for training and validation.
Train the model using CNN with customized architecture.
Evaluate the model performance using the provided evaluation functions.

### Folder Structure
splitaudio/: Directory containing audio files.
CNNModel: File defining the CNN model architecture for training.
RainfallDataset: Custom PyTorch Dataset class for handling rainfall data.
Training scripts and utility functions.


### Contributing
Contributions, issues, and feature requests are welcome. Feel free to check issues page if you want to contribute.




