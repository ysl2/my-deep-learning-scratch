#!/bin/python3
import torch

def myprint():
    a = torch.arange(12)
    # print(a)
    b = a.reshape((3, 4))
    # print(b)
    # ! 注意：尽量不要改东西。比如这里对b做了改动，实际上把a也给改了
    b[:] = 2
    # print(b)
    print(a)

