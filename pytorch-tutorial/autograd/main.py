# =======================================================
# === AUTOMATIC DIFFERENTIATION WITH `TORCH.AUTOGRAD` ===
# =======================================================

import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(loss)


# ===
# === Tensors, Functions and Computational graph
# ===

# You can set the value of `requires_grad` when creating a tensor, or later by using `x.requires_grad_(True)` method.

print('Gredient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)


# ===
# === Computing Gradients
# ===

loss.backward()
print(w.grad)
print(b.grad)


# ===
# === Disabling Gradient Tracking
# ===

z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)


# ===
# === More on Computational Graphs
# ===


# ===
# === Optional Reading: Tensor Gradients and Jacobian Products
# ===

inp = torch.eye(5, requires_grad=True)
out = (inp + 1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print('First call\n', inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print('First call\n', inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print('\nCall after zeroing gradients\n', inp.grad)
