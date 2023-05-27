from __future__ import annotations

# data manipulation
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from abc import ABC, abstractmethod
import math

# data visualization
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from wordcloud import WordCloud

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import make_moons
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split

# text processing
import re
import nltk as nlt
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# types
from dataclasses import dataclass
from enum import Enum
from typing import Union, Iterable, Callable, Optional
from collections import namedtuple, Counter

# tensorflow
import tensorflow as tf
from tensorflow import keras

## torch
import torch
from torch import nn
from torch.nn import ReLU, Sigmoid, Linear
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import grad

# utils
import os
from tqdm import tqdm

stopwords = set(stopwords.words('english'))

X, Y = make_moons(n_samples=2000, noise=0.1)

x_min, x_max = -1.5, 2.5
y_min, y_max = -1, 1.5
fig, ax = plt.subplots(1, 1, facecolor='#4B6EA9')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

def plot_data(ax: Axes, X: NDArray, Y: NDArray):
    plt.axis('off')
    ax.scatter(X[:, 0], X[:, 1], s=1, c=Y, cmap='bone')

plot_data(ax, X, Y)
plt.show()

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
to_forward = np.array(list(zip(xx.ravel(), yy.ravel())))

def plot_decision_boundary(ax: Axes, X, Y, classifier):
    Z = classifier.forward(to_forward)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z > 0.5, cmap="Blues")
    plot_data(ax, X, Y)
    
class MyReLU(object):
    def __init__(self):
        self.input_x = None

    def forward(self, x) -> NDArray:
        self.input_x = x
        return np.maximum(0, x)

    def backward(self, grad_output) -> NDArray:
        if self.input_x is None:
            raise ValueError("Forward pass must be called before backward pass.")
        grad_input = grad_output.copy()
        grad_input[self.input_x <= 0] = 0
        return grad_input

    def step(self, learning_rate):
        pass

class MySigmoid(object):
    def forward(self, x) -> NDArray:
        x = np.clip(x, -100, 100)
        return 1. / (1. + np.exp(-x))

    
    def backward(self, grad_output) -> NDArray:
        sigmoid = self.forward(grad_output)
        return grad_output * sigmoid * (1 - sigmoid)
    
    def step(self, learning_rate):
        pass

class MyLinear(object):
    def __init__(self, n_input, n_output):
        self.W = np.random.randn(n_input, n_output)
        self.b = np.random.randn(n_output)

    def forward(self, x: NDArray):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output):
        if self.x.ndim == 1:
            self.dW = np.outer(self.x, grad_output)
        else:
            self.dW = np.dot(self.x.T, grad_output)
        
        self.db = np.sum(grad_output, axis=0)
        dx = np.dot(grad_output, self.W.T)
        
        return dx
        
    def step(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        
class Sequential(object):
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def compute_loss(self, out, label):
        eps = 1e-10
        loss = -(label * np.log(out + eps) + (1 - label) * np.log(1 - out + eps))
        self.grad_loss = -label / (out + eps) + (1 - label) / (1 - out + eps)
        return loss

    def backward(self):
        grad = self.grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def step(self, learning_rate):
        for layer in self.layers:
            layer.step(learning_rate)

my_model = Sequential()
my_model.add_layer(MyLinear(n_input=2, n_output=5))
my_model.add_layer(MyReLU())
my_model.add_layer(MyLinear(n_input=5, n_output=5))
my_model.add_layer(MyReLU())
my_model.add_layer(MyLinear(n_input=5, n_output=5))
my_model.add_layer(MyReLU())
my_model.add_layer(MyLinear(n_input=5, n_output=1))
my_model.add_layer(MySigmoid())

losses = []
learning_rate = 1e-3
epochs = 10
for epoch in range(epochs):
    epoch_loss = np.array([])
    for _ in range(len(X)):
        idx = np.random.randint(0, len(X))
        x = X[idx]
        y = Y[idx]
        out = my_model.forward(x)
        loss = my_model.compute_loss(out, y)
        epoch_loss = np.append(epoch_loss, loss)
        my_model.backward()
        my_model.step(learning_rate)
    losses.append(np.mean(epoch_loss))
plt.plot(losses)
plt.show()