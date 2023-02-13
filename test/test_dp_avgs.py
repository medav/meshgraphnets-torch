import sys
sys.path.append('.')

import os
import torch
import numpy as np

import time
import deforming_plate as M


batch_size = 1

ds = M.DeformingPlateData('./data/deforming_plate_np/valid/')

dl = torch.utils.data.DataLoader(
    ds,
    shuffle=False,
    batch_size=batch_size,
    num_workers=16,
    collate_fn=M.collate_fn)

total_samples = 0
total_nodes = []
total_edges = []
total_wedges = []

for i, batch in enumerate(dl):
    if i % 100 == 0:
        print(f'Processed {i} / {len(ds)} samples')
    total_samples += batch_size
    total_nodes.append(batch['node_type'].shape[0])
    total_edges.append(batch['srcs'].shape[0])
    total_wedges.append(batch['wsrcs'].shape[0])

print(f'Total samples: {total_samples}')

print(f'Average nodes per sample: {sum(total_nodes) / total_samples:.2f} (std: {np.std(total_nodes):.2f})')
print(f'Average edges per sample: {sum(total_edges) / total_samples:.2f} (std: {np.std(total_edges):.2f})')
print(f'Average wedges per sample: {sum(total_wedges) / total_samples:.2f} (std: {np.std(total_wedges):.2f})')

