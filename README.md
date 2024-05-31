# Separate-two-type-of-pistachio-with-KNN-algorithm

# Pistachio Classification using K-Nearest Neighbors Algorithm

This Python code demonstrates the use of the K-Nearest Neighbors (K-NN) algorithm to classify two types of pistachios based on certain features.

## Overview

The code uses the K-Nearest Neighbors algorithm to classify two types of pistachios. It performs the following steps:
1. Imports necessary libraries such as numpy, pandas, MinMaxScaler, KNeighborsClassifier, and train_test_split.
2. Reads training data from a CSV file located at '../data/train.csv' and scales the features using MinMaxScaler.
3. Splits the dataset into training and testing sets.
4. Trains a K-Nearest Neighbors classifier on the training data with 5 neighbors, Euclidean distance metric, and uniform weights.
5. Predicts the target variable on the test data using the trained model.
6. Evaluates the model using the F1 score metric, which considers both precision and recall.

## Prerequisites

Before running the code, ensure that you have the following libraries installed in your Python environment:
- numpy
- pandas
- scikit-learn

You can install these libraries using pip install numpy pandas scikit-learn.

## Usage

1. Clone the repository to your local machine.
2. Replace the path to the training data file with the correct one in your local environment.
3. Run the Python script to see the classification results.

## Evaluation Metric

The F1 score is used as the evaluation metric in this code. The F1 score is a measure of a model's accuracy that considers both precision and recall. It is calculated as the weighted average of precision and recall, with values ranging from 0 to 1. A higher F1 score indicates better model performance.

The predicted F1 score for this model can be found in the variable prediction.
