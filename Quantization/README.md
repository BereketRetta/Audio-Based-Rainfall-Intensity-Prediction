# Quantization

This repository contains scripts for quantizing a deep learning model, specifically focused on a model used for predicting rainfall intensity from audio data. The project demonstrates the use of different quantization techniques to optimize the model for performance.

## Features

- **Model Conversion**: Converts PyTorch models to ONNX format.
- **Half-Precision (FP16) Quantization**: Reduces model size and potentially increases performance on compatible hardware.
- **Dynamic Quantization**: Applies dynamic quantization on linear layers.
- **Performance Measurement**: Includes scripts to measure and compare the inference time of different model versions.
- **ONNX Inference**: Demonstrates how to run inference using ONNX Runtime.

## Prerequisites

- Python environment
- Libraries as listed in `requirements.txt`

## Installation

1. Clone the repository:
    `git clone `
2. Install required dependencies:
    `pip install -r requirements.txt`

## Usage

1. Place your trained PyTorch model in the root directory.
2. Run the provided scripts to convert and quantize the model.
3. Use the inference scripts to compare the performance of different model versions.

## Contributing

Contributions to improve the quantization scripts or extend their functionality are welcome. Feel free to submit issues and pull requests.

