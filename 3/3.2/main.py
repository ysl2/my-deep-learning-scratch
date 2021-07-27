# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
import random
import torch
import tsensor as ts
from d2l import torch as d2l
import numpy as np
from matplotlib import pyplot as plt


# +
def synthetic_data(w, b, num_examples):
    with ts.explain():
        X = torch.normal(0, 1, (num_examples, len(w)))
        y = torch.matmul(X, w) + b
        y += torch.normal(0, 0.01, y.shape)
        y1 = y.reshape((-1, 1))
    return X, y1

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
with ts.explain():
    features
    labels
# -


# +
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
plt.show()
# -


# +
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
# -


# +
batch_size = 10
with ts.explain():
    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break
# -

with ts.explain():
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for params in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()



