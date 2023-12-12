# Rainfall Intensity using LSTM

This repository contains a deep learning-based approach for predicting rainfall intensity from audio data. The project utilizes Python, along with several libraries for data processing, feature extraction, and model training.

## Features

- **Audio Data Processing**: Uses Librosa for audio file manipulation.
- **Feature Extraction**: Extracts audio features suitable for time-series prediction.
- **Deep Learning Model**: Implements an LSTM model using PyTorch for sequence-based predictions.
- **Custom Loss Function**: Includes Mean Absolute Percentage Error (MAPE) for model optimization.

## Prerequisites

Ensure you have Python installed on your system. The project uses several libraries which can be installed via the provided `requirements.txt`.

## Installation

1. Clone the repository:
    `pip install `
2. Navigate to the project directory:
    `cd Directory`
3. Install the required libraries:
    `pip install -r requirements.txt`

## Usage

1. Prepare your audio data and label CSV in the specified format.
2. Extract features and save them using the provided script.
3. Load the dataset, preprocess it, and split it for training and validation.
4. Train the LSTM model using the predefined configurations.
5. Evaluate the model's performance using the included evaluation scripts.

## Folder Structure

- `/splitaudio/`: Directory containing audio files.
- Model definition and training scripts.
- Dataset processing and utility functions.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.

