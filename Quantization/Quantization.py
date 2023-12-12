import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import librosa
import matplotlib.pyplot as plt
import onnxruntime as ort
from sklearn.model_selection import KFold
import os

# Configuration
config = {
    "beam_width": 2,
    "lr": 0.002,
    "epochs": 10,
    "batch_size": 32
}

# Load your trained model
model = torch.load('best_model.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare dummy input for model conversion
dummy_input = torch.randn(32, 1, 20, 1)
torch.onnx.export(model, dummy_input, "model_converted.onnx", export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'])

# Convert to Half Precision (FP16)
torch.save(model.half(), 'best_model_fp16.pth')

# Dynamic Quantization on Linear layers
model_fp32_prepared = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
torch.save(model_fp32_prepared.state_dict(), 'model_quantized.pth')

# Function to measure inference time
def measure_inference_time(model, input_data, iterations=10):
    with torch.no_grad():
        # Warm-up iterations
        for _ in range(2):
            _ = model(input_data.to(device))

        # Measure inference time
        start_time = time.time()
        for _ in range(iterations):
            _ = model(input_data.to(device))
        end_time = time.time()

        # Calculate average time per inference
        avg_inference_time = (end_time - start_time) / iterations
        return avg_inference_time

# Measure inference time for original, quantized, and fp16 models
original_time = measure_inference_time(model, dummy_input)
quantized_model = torch.load('model_quantized.pth')
quantized_time = measure_inference_time(quantized_model, dummy_input)
model_fp16 = torch.load('best_model_fp16.pth')
fp16_time = measure_inference_time(model_fp16, dummy_input)

print(f"Original Model Time: {original_time} seconds")
print(f"Quantized Model Time: {quantized_time} seconds")
print(f"FP16 Model Time: {fp16_time} seconds")

# ONNX Inference
def run_onnx_inference(onnx_model_path, input_data_np, use_cuda=False):
    # Check if CUDA is requested and available
    if use_cuda and 'AzureExecutionProvider' in ort.get_available_providers():
        session = ort.InferenceSession(onnx_model_path, providers=['AzureExecutionProvider'])
    else:
        session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    # Get the name of the input node
    input_name = session.get_inputs()[0].name

    # Prepare the input data as a dictionary
    input_dict = {input_name: input_data_np}


    start = time.time()
    # Perform inference
    output = session.run(None, input_dict)
    end = time.time()

    return output[0], end - start

# Run ONNX inference on CPU and CUDA
onnx_model_path = 'model_converted.onnx'
output_cpu, time_cpu = run_onnx_inference(onnx_model_path, dummy_input.numpy(), use_cuda=False)
output_cuda, time_cuda = run_onnx_inference(onnx_model_path, dummy_input.numpy(), use_cuda=True)

print("CPU Inference Time:", time_cpu)
print("CUDA Inference Time:", time_cuda)
