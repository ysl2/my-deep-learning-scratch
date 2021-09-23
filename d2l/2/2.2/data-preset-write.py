#!/bin/python3
# import torch

# # python3.9
# # x = torch.arange(0, 12, out=torch.LongTensor())

# x = torch.arange(12)
# x = x.reshape(3, 4)
# x = torch.zeros((2, 3, 4))
# x = torch.ones((2, 3, 4))
# x = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# # print(x)
# # print(x.shape)
# # print(x.numel())

# x = torch.tensor([1.0, 2, 4, 8])
# y = torch.tensor([2, 2, 2, 2])

# # print(x + y)
# # print(x - y)
# # print(x * y)
# # print(x / y)
# # print(x ** y)

import os

os.makedirs(os.path.join('.', 'data'), exist_ok=True)
data_file = os.path.join('.', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,1410000\n')

