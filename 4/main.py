from __future__ import annotations
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import sklearn
from sklearn.metrics import classification_report
from tqdm import tqdm
from time import time
from copy import deepcopy

with open("./assets/data.pkl", "rb") as f:
    data = pickle.load(f)

with open("./assets/label.pkl", "rb") as f:
    labels = pickle.load(f)

data_resized = [cv2.resize(img, (20, 20)) for img in data]

num_samples = len(data_resized)
num_train = int(0.8 * num_samples)
indices = list(range(num_samples))
random.shuffle(indices)
train_indices = indices[:num_train]
test_indices = indices[num_train:]
train_data = [data_resized[i] for i in train_indices]
test_data = [data_resized[i] for i in test_indices]
train_labels = [labels[i] for i in train_indices]
test_labels = [labels[i] for i in test_indices]

num_samples = 10
samples = random.sample(train_data, num_samples)
for i, sample in enumerate(samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample, cmap="gray")
    plt.axis("off")
    
plt.suptitle("Training Data")
plt.show()

num_samples = 10
samples = random.sample(test_data, num_samples)
for i, sample in enumerate(samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample, cmap="gray")
    plt.axis("off")
    
plt.suptitle("Testing Data")
plt.show()
CLASSES = range(10)
for i in CLASSES:
    plt.subplot(2, 5, i + 1)
    plt.imshow(train_data[train_labels.index(i)], cmap="gray")
    plt.axis("off")
    plt.title(f"{i}")

plt.suptitle("samples for each class")
plt.show()
class_counts = np.zeros(10)
for label in train_labels:
    class_counts[label] += 1

# Plot a bar graph of the class counts
plt.bar(range(10), class_counts)
plt.xticks(range(10))
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Counts in Training Data")
plt.show()


class_counts = np.zeros(10)
for label in test_labels:
    class_counts[label] += 1

# Plot a bar graph of the class counts
plt.bar(range(10), class_counts)
plt.xticks(range(10))
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Counts in Test Data")
plt.show()
train_data = [[i / 255 for i in row] for row in train_data]
test_data = [[i / 255 for i in row] for row in test_data]
# A better way to do this is to use the
# sklearn.preprocessing.MinMaxScaler() 
# but we are not allowed to do that
train_data = np.array(train_data)
train_data = train_data.reshape(train_data.shape[0], -1)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_data = test_data.reshape(test_data.shape[0], -1)
test_labels = np.array(test_labels)
NUMBER_OF_PIXELS = 400
NUMBER_OF_CLASSES = 10

mean = np.zeros([NUMBER_OF_CLASSES, NUMBER_OF_PIXELS])
variance = np.zeros([NUMBER_OF_CLASSES, NUMBER_OF_PIXELS])
class_probability = [0 for _ in range(NUMBER_OF_CLASSES)]

for i in CLASSES:
    class_indices = train_data[train_labels == i]
    class_probability[i] = len(class_indices) / len(train_data)
    mean[i] = class_indices.mean(axis=0)
    variance[i] = class_indices.var(axis=0)