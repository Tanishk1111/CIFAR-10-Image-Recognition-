# CIFAR-10 Image Recognition Model

This repository contains a Convolutional Neural Network (CNN) model built using TensorFlow and Keras to classify images from the CIFAR-10 dataset into one of 10 classes.

## Introduction
This project aims to develop a CNN model that can accurately classify images from the CIFAR-10 dataset. The CIFAR-10 dataset is widely used for machine learning and computer vision research.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Model Architecture
The CNN model consists of the following layers:
- Convolutional Layer: 32 filters, kernel size (3, 3), ReLU activation
- Max Pooling Layer: pool size (2, 2)
- Convolutional Layer: 64 filters, kernel size (3, 3), ReLU activation
- Max Pooling Layer: pool size (2, 2)
- Convolutional Layer: 64 filters, kernel size (3, 3), ReLU activation
- Flatten Layer
- Dense Layer: 64 units, ReLU activation
- Dense Layer: 10 units (output layer)

## Training
The model is compiled with the Adam optimizer and Sparse Categorical Crossentropy loss function. It is trained for 10 epochs with the training data and validated using the test data.

## Evaluation
The model is evaluated on the test data to determine its accuracy. Additionally, training and validation accuracy and loss are plotted to visualize the model's performance over epochs.
