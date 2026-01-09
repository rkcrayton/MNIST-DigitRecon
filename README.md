# MNIST Handwritten Digit Recognition

A convolutional neural network implementation in PyTorch for classifying handwritten digits from the MNIST dataset.

## Overview

This project implements a CNN-based image classifier that achieves 98% accuracy on the MNIST test set. The model uses convolutional layers for feature extraction and fully connected layers for classification, trained on 60,000 grayscale images of handwritten digits (0-9).

## Model Architecture

The network consists of:
- Two convolutional layers (1->10 and 10->20 feature maps)
- Max pooling layers for dimensionality reduction
- Dropout regularization to prevent overfitting
- Two fully connected layers (320->50->10)
- ReLU activation functions
- Softmax output for classification

## Technical Implementation

**Framework:** PyTorch
**Optimizer:** Adam (learning rate: 0.001)
**Loss Function:** Cross-Entropy Loss
**Training:** 10 epochs with batch size of 100
**Hardware:** CPU/CUDA compatible

## Dataset

MNIST Dataset:
- Training set: 60,000 images
- Test set: 10,000 images
- Image size: 28x28 grayscale
- Classes: 10 (digits 0-9)

## Requirements
```
torch
torchvision
matplotlib
numpy
```

## Installation
```bash
git clone https://github.com/yourusername/mnist-digit-recognition.git
cd mnist-digit-recognition
pip install -r requirements.txt
```

## Usage

Run the training script:
```bash
python main.py
```

The script will:
1. Download the MNIST dataset automatically
2. Train the model for 10 epochs
3. Display training progress and test accuracy
4. Save a visualization of a test prediction

## Results

**Performance Metrics:**
- Test Accuracy: 98%
- Training Accuracy: 99%
- Training Time: Approximately 5 minutes on CPU

## Key Features

- Custom CNN architecture built from scratch
- Efficient data loading with PyTorch DataLoader
- Dropout regularization for improved generalization
- GPU acceleration support
- Training/test accuracy monitoring
- Visualization of model predictions

## Future Improvements

- Add confusion matrix analysis
- Implement early stopping
- Extend to Fashion-MNIST dataset
- Add model checkpointing
- Create web interface for real-time predictions
- Visualize learned convolutional filters

## Learning Outcomes

This project demonstrates understanding of:
- Convolutional neural network architecture
- Backpropagation and gradient descent
- PyTorch framework fundamentals
- Model training and evaluation workflows
- Overfitting prevention techniques
- Computer vision fundamentals

## Author

Raheem Crayton - University of Alabama
```
