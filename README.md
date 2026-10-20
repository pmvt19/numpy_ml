# Numpy ML

<p align="center">
<img src="./assets/numpy_ml.png" alt="NumpyML Logo" width="75%">
</p>


- Neural Network
-- Linear Layers
-- Loss Functions
--- BCE Loss
--- Cross Entropy Loss
--- MSE Loss
-- Activation Functions
--- ReLu
--- LeakyReLu

Completed:
- K-Means

- KNN Classification

- MinMaxScaler

# K-Means Example
# KNN Example
# Linear Regression Example? (maybe not)

# neural network example

# Training Curves
# Toy Example
# MNIST Example

# Numpy ML
This repository holds numpy implementations of common machine learning algorithms

## Datasets
### GridDataSet
(Image of grid dataset)

## K-Means

```python
from data_generator import GridDataSet
from numpy_ml.k_means import KMeans

# Initialize Dataset
dataset = GridDataSet()

# Create KMeans
kmeans = KMeans(k=9)

# Fit data to get k centroids
centroids = kmeans.fit(dataset.train_data)

# Assign cluster labels to test dataset
labels = kmeans.predict(dataset.test_data)
```

Results:
Centroids

## K Nearest Neighbors (KNN)

```python
from numpy_ml.knn import KNN
from data_generator import GridDataSet

# Initialize Dataset
dataset = GridDataSet()

# Create K-Nearest Neighbors
knn = KNN()

# Give KNN the datasets
knn.fit(dataset.train_data, dataset.train_labels)

# Predict Class Labels for Test Data with k neighbors
predicted_labels = knn.predict(dataset.test_data, k=3)
```

## Neural Network

```python 
from numpy_ml.neuarl_network import LinearLayer, ReLu
```
