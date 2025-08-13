# 🌸 Iris Dataset Analysis and KNN Classification

This repository contains two Python scripts for analyzing the Iris dataset. One script focuses on K-Nearest Neighbors (KNN) classification, while the other is dedicated to data visualization and basic data exploration.

## 📊 Project Overview

### `KNN.py` 
This script performs K-Nearest Neighbors (KNN) classification on the Iris dataset. It includes the following steps:
- Loading and preparing the Iris dataset.
- Splitting the data into training and testing sets.
- Scaling and normalizing the features using `StandardScaler`.
- Determining the optimal 'K' value for the KNN algorithm by evaluating the error rate for different 'K' values.
- Training the KNN model with the best 'K' value.
- Evaluating the model's performance using a confusion matrix and classification report.
- Visualizing the error rate for different 'K' values.

### `Graphics.py` 
This script focuses on visualizing and exploring the Iris dataset. It includes:
- Loading the Iris dataset.
- Extracting sepal and petal measurements.
- Generating scatter plots to visualize the distribution of sepal length vs. sepal width and petal length vs. petal width for different Iris species.
- Displaying a DataFrame representation of the Iris dataset.
- Providing descriptive statistics for the dataset.

## 📊 Visualizations

### Data Visualization
![Graphics Image](Classificação%20elementos%20images/Graphics_image.png)

### KNN Evaluation
![KNN Image](Classificação%20elementos%20images/KNN_image.png)

## 📦 Requirements

- Python 3.10 or higher
- Libraries:
  - `scikit-learn`
  - `pandas`
  - `matplotlib`
  - `numpy`


## 💻 How to Run

1.  **Install Dependencies**

    ```bash
    # NOTE: You can run these commands either in your system terminal
    # (Windows, Mac, Linux) or in the Python console (e.g., PyCharm).

    # ----- Windows -----
    pip install scikit-learn pandas matplotlib numpy
    # or, to ensure the correct pip is used:
    python -m pip install scikit-learn pandas matplotlib numpy

    # ----- Mac -----
    # If using Homebrew to manage Python:
    # brew install python   # only if Python 3 is not installed
    python3 -m pip install scikit-learn pandas matplotlib numpy

    # ----- Linux (Ubuntu/Debian) -----
    sudo apt update
    sudo apt install python3-pip  # only if pip is missing
    python3 -m pip install scikit-learn pandas matplotlib numpy

    # ----- Conda (optional) -----
    # Works on any system with Anaconda/Miniconda installed:
    conda install -c anaconda scikit-learn pandas matplotlib numpy

    # ----- Via .whl file (optional, rare) -----
    # 1. Download the .whl file matching your Python version and OS
    # 2. Install with:
    # pip install path/to/file.whl
    ```

2.  **Download the Scripts**

    If you have Git installed, you can clone this repository:
    ```bash
    git clone https://github.com/YOUR-USERNAME/ML-KNN_DataSet-Iris.git
    cd ML-KNN_DataSet-Iris
    ```
    Alternatively, you can download the Python files directly.

3.  **Run the Scripts**

    To run the KNN classification script:
    ```bash
    python KNN.py
    ```

    To run the Iris visualization script:
    ```bash
    python Graphics.py
    ```
