# MNIST Handwritten Digit Classification using PyTorch
[![Python](https://img.shields.io/badge/python-3.x-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)](https://pytorch.org/)
[![MNIST](https://img.shields.io/badge/Dataset-MNIST-green)](http://yann.lecun.com/exdb/mnist/)

## Introduction

Welcome to the repository for MNIST Handwritten Digit Classification using PyTorch. This project aims to provide a comprehensive guide to building an end-to-end machine learning pipeline for classifying handwritten digits. Leveraging the power of PyTorch, the project covers essential steps from data preparation and preprocessing to model training and evaluation. Special attention is given to data visualization and performance metrics, making this a robust solution for anyone looking to delve into image classification with neural networks.

## File Structure

The project has a simple file structure for ease of navigation and usage. Below is the directory layout along with a brief description of each file:


- `MNIST/`: Contains the MNIST dataset used for training and testing the model.
- `Digit_Classifier_Model.ipynb`: The Jupyter Notebook containing Python code for the entire pipeline, from data preparation to model evaluation.
- `.gitignore`: A file specifying untracked files that Git should ignore.
- `README.md`: The file you are currently reading, explaining the project setup, structure, and usage.

## Features

The project is designed as a comprehensive machine learning pipeline, and it includes the following key features:

### Data Preparation

- **MNIST Dataset**: Utilizes the MNIST dataset for training and testing.
- **Data Loaders**: Implements data loaders with batch processing to efficiently handle large datasets.

### Data Visualization

- **Initial Inspection**: Displays the first 30 images from the MNIST dataset for initial data inspection.

### Model Architecture

The neural network model used in this project is designed for the specific task of handwritten digit recognition. Below are the details of the architecture, including the rationale behind the choices made.

### Layers

1. **Input Layer**: 
    - **Neurons**: 784 
    - **Activation**: None
    - **Comment**: The input layer consists of 784 neurons corresponding to the 28x28 pixels in each grayscale image of a handwritten digit.

2. **Hidden Layer 1**: 
    - **Neurons**: 64 
    - **Activation**: ReLU (Rectified Linear Unit)
    - **Comment**: The first hidden layer is designed to begin the pattern recognition process typical in neural networks.

3. **Hidden Layer 2**: 
    - **Neurons**: 32 
    - **Activation**: ReLU
    - **Comment**: The second hidden layer continues the pattern recognition and begins consolidating features for classification.

4. **Output Layer**: 
    - **Neurons**: 10 
    - **Activation**: Softmax (Not explicitly implemented, but implied in the loss function)
    - **Comment**: The output layer consists of 10 neurons, each representing one of the 10 possible digits (0-9).

### Architecture Diagram

Here's a simple diagram to illustrate the architecture:
Input Layer (784) ---> Hidden Layer 1 (64) ---> Hidden Layer 2 (32) ---> Output Layer (10)


### Rationale

The architecture was chosen based on empirical testing and aims to balance model complexity against computational efficiency. Specifically, the model is complex enough to capture the intricate patterns in handwritten digits but is also streamlined to train quickly and efficiently. The use of ReLU activation functions in the hidden layers also contributes to faster training.

### Model Training

The model is trained using the MNIST dataset, which is a large dataset of handwritten digits. The dataset is automatically downloaded and preprocessed for training.

### Hyperparameters

Here are the key hyperparameters used for training the model:

- **Batch Size**: 64
- **Learning Rate**: 0.1
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Epochs**: 100

### Training Loop

The training process involves iterating over the MNIST training dataset multiple times, defined by the number of epochs. In each epoch, the model's parameters are updated to minimize the loss function.

### Monitoring

During training, the loss is monitored to ensure that the model is learning effectively. The loss values are printed at the end of each epoch to provide insight into the training process.

## Evaluation

After the model is trained, it is important to evaluate its performance to ensure that it is learning effectively and is capable of making accurate predictions.

### Test Dataset

The model is evaluated using a separate test dataset provided by the MNIST dataset. This dataset is not seen by the model during training and serves as a good indicator of how the model will perform on unseen data.

### Metrics

The primary metric used for evaluation is accuracy, which is the percentage of correctly classified images out of the total images in the test dataset. The accuracy is printed at the end of the evaluation process in the notebook.

### Results

The results of the evaluation are available in the Jupyter notebook. The model achieves competitive performance, successfully recognizing a high percentage of handwritten digits.

## Technologies

This project is implemented using a range of technologies and libraries to ensure robustness and ease of use. Below are the key technologies used:

- **Python**: The backbone of the project, used for all computational tasks.
- **PyTorch**: A leading deep learning framework, used for building and training the neural network.
- **NumPy**: Used for numerical operations, especially for handling arrays.
- **Matplotlib**: Employed for data visualization tasks, particularly for displaying MNIST images.

