# 🌸 Iris Dataset Analysis and KNN Classification

This repository contains two Python scripts for analyzing the Iris dataset. One script focuses on K-Nearest Neighbors (KNN) classification, while the other is dedicated to data visualization and basic data exploration.

## 📊 Project Overview

### `knn_classification.py` (Assumed filename)
This script performs K-Nearest Neighbors (KNN) classification on the Iris dataset. It includes the following steps:
- Loading and preparing the Iris dataset.
- Splitting the data into training and testing sets.
- Scaling and normalizing the features using `StandardScaler`.
- Determining the optimal 'K' value for the KNN algorithm by evaluating the error rate for different 'K' values.
- Training the KNN model with the best 'K' value.
- Evaluating the model's performance using a confusion matrix and classification report.
- Visualizing the error rate for different 'K' values.

### `iris_visualization.py` (Assumed filename)
This script focuses on visualizing and exploring the Iris dataset. It includes:
- Loading the Iris dataset.
- Extracting sepal and petal measurements.
- Generating scatter plots to visualize the distribution of sepal length vs. sepal width and petal length vs. petal width for different Iris species.
- Displaying a DataFrame representation of the Iris dataset.
- Providing descriptive statistics for the dataset.

## 📦 Requirements

- Python 3.x
- Libraries:
  - `scikit-learn`
  - `pandas`
  - `matplotlib`
  - `numpy`
