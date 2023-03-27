import os
import torch
import sys
import numpy as np

import time
import cfd as M

BATCH_SIZE = int(sys.argv[1])
ds = M.CylinderFlowData('./data/cylinder_flow_np/train/')

print('Setting up data loader...')
dl = torch.utils.data.DataLoader(
    ds,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=4, # changed to 4 from 16
    collate_fn=M.collate_fn)

batch = next(iter(dl))

batch_np = {
    'node_type': batch['node_type'].numpy(),
    'velocity': batch['velocity'].numpy(),
    'mesh_pos': batch['mesh_pos'].numpy(),
    'srcs': batch['srcs'].numpy(),
    'dsts': batch['dsts'].numpy(),
    'target_velocity': batch['target_velocity'].numpy()
}

np.savez_compressed(f'data/b{BATCH_SIZE}.npz', **batch_np)
