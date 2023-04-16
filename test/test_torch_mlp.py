import sys
sys.path.append('.')

from contextlib import contextmanager

import time
import torch
import graphnet as GNN

num_iters = 100000

NN = 128*1024

D = 128
DT = torch.float16

dev = torch.device('cuda:0')
is_cuda = dev.type == 'cuda'

#net = GNN.Mlp(3 * D, [D, D, D], layernorm=False).to(DT).to(dev)
net = torch.nn.Linear(128, 128).to(DT).to(dev)
x = torch.randn(NN, D, device=dev, dtype=DT)

t0 = time.perf_counter()
for _ in range(num_iters):
    net(x)

t1 = time.perf_counter()

tt = t1 - t0

print(NN * (128*128) * 2 * num_iters / tt / 1e9)

