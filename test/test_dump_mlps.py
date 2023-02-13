import sys
sys.path.append('.')

import time
import pypin
import torch
import deforming_plate as M

orig_fwd = torch.nn.Linear.forward

def new_fwd(self, x):
    M, K = x.shape
    N, _ = self.weight.shape
    print(f'Linear: {M}\t{N}\t{K}')
    return orig_fwd(self, x)

# torch.nn.Linear.forward = new_fwd

BATCH_SIZE = 1

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

dev = torch.device('cpu')
net = M.DeformingPlateModel().to(dev)

ds = M.DeformingPlateData('./data/deforming_plate_np/train/')
dl = torch.utils.data.DataLoader(
    ds,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=1,
    pin_memory=dev.type == 'cuda',
    pin_memory_device=str(dev),
    collate_fn=M.collate_fn)

batch = next(iter(dl))


print(f'# Nodes = {batch["node_type"].shape[0]}')
print(f'# Edges = {batch["srcs"].shape[0]}')
print(f'# WEdges = {batch["wsrcs"].shape[0]}')

with pypin.pinhooks.perf_roi(0, 'mgn', ''):
    loss = net.loss(
        node_type=batch['node_type'].to(dev),
        mesh_pos=batch['mesh_pos'].to(dev),
        world_pos=batch['world_pos'].to(dev),
        target_world_pos=batch['target_world_pos'].to(dev),
        srcs=batch['srcs'].to(dev),
        dsts=batch['dsts'].to(dev),
        wsrcs=batch['wsrcs'].to(dev),
        wdsts=batch['wdsts'].to(dev),
    )
