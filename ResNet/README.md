# Audio-Based Rainfall Intensity Prediction Using ResNet

This project involves using a ResNet model to predict rainfall intensity from audio data. It includes steps for preprocessing audio files, extracting Mel-frequency cepstral coefficients (MFCCs), generating Mel-spectrograms, and training a ResNet model.

## Features

- **Audio Preprocessing**: Utilizes Librosa for audio file processing.
- **MFCC Extraction**: Extracts MFCCs from audio data.
- **Mel-Spectrogram Generation**: Converts audio files to Mel-spectrograms.
- **ResNet Model**: Implements a ResNet model for classification tasks.
- **K-Fold Cross-Validation**: Ensures model robustness via cross-validation.

## Prerequisites

- Python environment
- Libraries as listed in `requirements.txt`

## Installation

1. Clone the repository:
    `git clone `
2. Install required dependencies:
    `pip install -r requirements.txt`

## Usage

1. Prepare your audio dataset.
2. Run the preprocessing scripts to extract MFCCs and generate Mel-spectrograms.
3. Train the ResNet model using the provided training scripts.
4. Evaluate the model's performance.

## Folder Structure

- `./splitaudio`: Directory containing audio files.
- `./spectrograph`: Directory where Mel-spectrograms are stored.
- Main scripts for data preprocessing, model definition, training, and evaluation.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.

