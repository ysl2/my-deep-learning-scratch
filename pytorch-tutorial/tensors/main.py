# ===============
# === Tensors ===
# ===============

import torch
import numpy as np


# ===
# === Initializing a Tensor
# ===

# Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# From a NumPy array
np_array = np.array(data)
x_np = torch.tensor(data)

# From another tensor
x_ones = torch.ones_like(x_data)
print(f'Ones Tensor: \n {x_ones} \n')

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f'Random Tensor: \n {x_rand} \n')

# With random or constant values
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f'Random Tensor: \n {rand_tensor} \n')
print(f'Ones Tensor: \n {ones_tensor} \n')
print(f'Zeros Tensor: \n {zeros_tensor}')


# ===
# === Attributes of a Tensor
# ===

tensor = torch.rand(3, 4)
print(f'Shape of tensor: {tensor.shape}')
print(f'Datatype of tensor: {tensor.dtype}')
print(f'Device tensor is stored on: {tensor.device}')


# ===
# === Operations on Tensors
# ===

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# Standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
# Ref:https://blog.csdn.net/z13653662052/article/details/78010654
# 三个点是什么鬼，Matlab里面这不是换行的操作么，但这里不是，它是省略所有的冒号来用省略号代替。
# 大家看这个a[:, :, None]和a[…, None]的输出是一样的，就是因为…代替了前面两个冒号。这下应该清楚了。
print('Last column: ', tensor[..., -1])
tensor[:, 1] = 0
print(tensor)

# Join tensors
# torch.stack也能拼接tensor，但是stack与cat有一点点不同
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Arithmetic operations
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Single-element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(tensor, '\n')
# add_ 是in-place操作，而add会再产生一块新的内存
tensor.add_(5)
print(tensor)


# ===
# === Bridge with NumPy
# ===

t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}')

t.add_(1)
print(f't: {t}')
print(f'n: {n}')

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f't: {t}')
print(f'n: {n}')


# ===
# === NumPy array to Tensor
# ===

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f't: {t}')
print(f'n: {n}')

