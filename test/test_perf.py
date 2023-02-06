
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from contextlib import contextmanager
import dataset
import model
import time

ni = 0

dev = torch.device('cuda:0')
net = model.CfdModel().to(dev)
opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

t0 = time.perf_counter()
for i in range(100):
    ds = dataset.CylinderFlowData(f'./data/cylinder_flow_np/train/t{i}.npz')
    batch = ds[0]

    for _ in range(10):
        net.loss(
            batch['node_type'].to(dev),
            batch['velocity'].to(dev),
            batch['mesh_pos'].to(dev),
            batch['srcs'].to(dev),
            batch['dests'].to(dev),
            batch['target_velocity'].to(dev)
        ).backward()

        opt.step()
        opt.zero_grad()
        ni += 1

t1 = time.perf_counter()
print(f'{ni / (t1 - t0)}')
