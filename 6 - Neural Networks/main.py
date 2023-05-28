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
import plotly.express as px
import plygdata as pg

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import make_moons, make_circles, make_classification
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

# nlt.download('stopwords')
# nlt.download('wordnet')
# nlt.download('punkt')

stopwords = set(stopwords.words("english"))

X, Y = None, None
x_min, x_max = -1.5, 2.5
y_min, y_max = -1, 1.5


def create_X_Y(mode: str = "moon"):
    global X, Y
    global x_min, x_max
    global y_min, y_max

    if mode == "moon":
        X, Y = make_moons(n_samples=2000, noise=0.1)

    elif mode == "circle":
        X, Y = make_circles(n_samples=2000, noise=0.1)
        x_min, x_max = -1.5, 1.5
        y_min, y_max = -1.5, 1.5

    elif mode == "classification":
        X, Y = make_classification(
            n_samples=2000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0
        )
        x_min, x_max = -7, 7
        y_min, y_max = -4, 4

    elif mode == "spiral":
        data_array = np.array(pg.generate_data(pg.DatasetType.ClassifySpiralData, 0.0))
        X = data_array[:, :-1]
        Y = data_array[:, -1]
        Y = (Y + 1) / 2
        x_min, x_max = X.min() - 2, X.max() + 2
        y_min, y_max = Y.min() - 5, Y.max() + 4


create_X_Y()

fig, ax = plt.subplots(1, 1, facecolor="#4B6EA9")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)


def plot_data(ax: Axes, X: NDArray, Y: NDArray):
    plt.axis("off")
    ax.scatter(X[:, 0], X[:, 1], s=1, c=Y, cmap="bone")


plot_data(ax, X, Y)
plt.show()

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
to_forward = np.array(list(zip(xx.ravel(), yy.ravel())))


def plot_decision_boundary(
    ax: Axes,
    X: NDArray,
    Y: NDArray,
    classifier: Union["Sequential", "MyLinear", "MySigmoid", "MyReLU"],
):
    Z = classifier.forward(to_forward.T)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z > 0.5, cmap="Blues")

    plot_data(ax, X, Y)


class MyReLU(object):
    def forward(self, x) -> NDArray:
        self.relu = np.maximum(0, x)
        return self.relu

    def backward(self, grad_output) -> NDArray:
        return np.where(self.relu > 0, grad_output, 0)

    def step(self, learning_rate):
        pass


class MySigmoid(object):
    def forward(self, x) -> NDArray:
        x = np.clip(x, -1000, 1000)
        self.sigmoid = np.reciprocal(1 + np.exp(-x))
        return self.sigmoid

    def backward(self, grad_output) -> NDArray:
        return grad_output * self.sigmoid * (1 - self.sigmoid)

    def step(self, learning_rate):
        pass

class MyLinear(object):
    def __init__(self, n_input, n_output):
        self.W = np.random.randn(n_output, n_input)
        self.b = np.random.randn(n_output, 1)

    def forward(self, x: NDArray):
        self.x = x
        return self.W @ x + self.b

    def backward(self, grad_output) -> NDArray:
        self.dW = grad_output @ self.x.T
        self.db = np.sum(grad_output, axis=1, keepdims=True)
        grad_input = self.W.T @ grad_output
        return grad_input

    def step(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        
class Sequential(object):
    def __init__(self):
        self.layers: list[Union[MyLinear, MyReLU, MySigmoid]]
        self.reset()

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        current = x
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def compute_loss(self, out, label):
        eps = 1e-10
        if out == 0:
            out += eps
        elif out == 1:
            out -= eps
        loss = -(label * np.log(out) + (1 - label) * np.log(1 - out))
        self.grad_output = (out - label) / (out * (1 - out))
        return loss

    def backward(self):
        current = self.grad_output
        for layer in reversed(self.layers):
            current = layer.backward(current)

    def step(self, learning_rate):
        for layer in self.layers:
            layer.step(learning_rate)
            
    def reset(self):
        self.layers = []
        
my_model = Sequential()
my_model.add_layer(MyLinear(n_input=2, n_output=10))
my_model.add_layer(MyReLU())
my_model.add_layer(MyLinear(n_input=10, n_output=1))
my_model.add_layer(MySigmoid())

losses = []
learning_rate = 1e-2
epochs = 10
for epoch in range(epochs):
    loss_sum = 0 
    for it in range(len(X)):
        idx = np.random.randint(0, len(X))
        x = X[idx]
        x = x.reshape(2, 1)
        label = Y[idx]
        out = my_model.forward(x)
        loss_sum += my_model.compute_loss(out, label)
        my_model.backward()
        my_model.step(learning_rate)
    losses.append(loss_sum / len(X))

losses = np.array(losses)
losses = losses.reshape((-1, 1))
plt.plot(losses)
plt.show()

fig, ax = plt.subplots(1, 1, facecolor='#4B6EA9')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
plot_decision_boundary(ax, X, Y, my_model)
fig.canvas.draw()

# ------------------------------------------------------------------------------
