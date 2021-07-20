#!/bin/python3
import torch
from torch.distributions import multinomial
from d2l import torch as d2l
from matplotlib import pyplot as plt
import os

fair_probs = torch.ones(6)
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()

pic_dir = './image'
pic_name = 'groups-of-experiments.png'
pic_path = os.path.join(pic_dir, pic_name)
os.makedirs(pic_dir, exist_ok=True)
plt.savefig(pic_path)

# plt.show()

