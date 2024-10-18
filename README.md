# Digits-Classifier

This repository contains the code and resources for building a Handwritten Digits Classifier using PyTorch. This project demonstrates image classification using deep learning and convolutional neural networks (CNNs).

## Project Overview

The goal of this project is to develop a model that accurately classifies handwritten digits from 0 to 9. Such models are essential for applications like optical character recognition (OCR) systems and digit recognition tasks.

## Prerequisites

Before you begin, ensure you have the following prerequisites installed:

- Python 3.6+
- PyTorch (latest version)
- torchvision (latest version)
- NumPy
- Matplotlib (for visualization)
- Jupyter Notebook (optional, for running notebooks)

You can install the required packages using pip:

```bash
pip install torch torchvision numpy matplotlib
```

## Dataset

We use the MNIST dataset, which includes 28x28 grayscale images of handwritten digits. It consists of 60,000 training images and 10,000 test images. PyTorch provides convenient access to this dataset through the torchvision library.

## Project Structure

The project is organized as follows:

- **data/**: Contains data loading and preprocessing code.
- **models/**: Includes the PyTorch model architecture.
- **train.ipynb**: Jupyter Notebook for training the model.
- **evaluate.ipynb**: Jupyter Notebook for evaluating the model's performance.
- **utils.py**: Utility functions for data loading and visualization.

## Training the Model

To train the model:
- Open and run `train.ipynb`. This notebook loads the MNIST dataset, defines and trains the CNN model, and saves the trained model's weights.

## Evaluating the Model

To evaluate the model's performance:
- Open and run `evaluate.ipynb`. This notebook loads the trained model and tests it on the test dataset, providing accuracy and other relevant metrics.

## Results

Our model achieves an accuracy of over 98% on the MNIST test dataset, demonstrating its effectiveness in handwritten digit classification.
