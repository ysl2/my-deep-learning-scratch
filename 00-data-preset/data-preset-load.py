#!/bin/python3
import pandas as pd
import os
import torch

data_file = os.path.join('.', 'data', 'house_tiny.csv')
data = pd.read_csv(data_file)
# print(data)
# print('---')
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# print(inputs)
# print('---')
# print(outputs)
# print('---')
inputs = inputs.fillna(inputs.mean())
# print(inputs)
# print(inputs, outputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
# print(inputs)
# print(outputs)
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(y)
