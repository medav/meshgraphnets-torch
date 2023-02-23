import os
import torch
import sys
import numpy as np

import time
import deforming_plate as M

BATCH_SIZE = int(sys.argv[1])
ds = M.DeformingPlateData('./data/deforming_plate_np/train/')

print('Setting up data loader...')
dl = torch.utils.data.DataLoader(
    ds,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=16,
    collate_fn=M.collate_fn)

batch = next(iter(dl))

batch_np = {
    'node_offs': batch['node_offs'].numpy(),
    'node_type': batch['node_type'].numpy(),
    'mesh_pos': batch['mesh_pos'].numpy(),
    'world_pos': batch['world_pos'].numpy(),
    'target_world_pos': batch['target_world_pos'].numpy(),
    'srcs': batch['srcs'].numpy(),
    'dsts': batch['dsts'].numpy()
}

np.savez_compressed(f'data/b{BATCH_SIZE}.npz', **batch_np)
