import sys
sys.path.append('.')


import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

import time
import deforming_plate as M
import unsorted_segsum


filename = sys.argv[1]
np_data = np.load(filename)

NUM_ITERS = 20
BATCH_SIZE = len(np_data['node_offs'])

checkpoint_file = './deforming_plate_ckpt.pth'
torch.backends.cuda.matmul.allow_tf32 = True

dev = torch.device('cuda:0')
net = M.DeformingPlateModel().to(dev)

batch = {
    'node_offs': torch.LongTensor(np_data['node_offs']),
    'node_type': torch.LongTensor(np_data['node_type']),
    'mesh_pos': torch.Tensor(np_data['mesh_pos']),
    'world_pos': torch.Tensor(np_data['world_pos']),
    'target_world_pos': torch.Tensor(np_data['target_world_pos']),
    'srcs': torch.LongTensor(np_data['srcs']),
    'dsts': torch.LongTensor(np_data['dsts'])
}

dsts = batch['dsts'].to(dev)
num_nodes = batch['mesh_pos'].shape[0]
num_edges = batch['dsts'].shape[0]
edge_features = torch.randn(num_edges, 128, dtype=torch.float16).to(dev)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    with torch.no_grad():
        print('running...')
        t0 = time.perf_counter()
        for _ in range(NUM_ITERS):
            y = unsorted_segsum.unsorted_segment_sum(
                edge_features, dsts, num_nodes)

            prof.step()
        t1 = time.perf_counter()
        print('done')


print(f'Batch Size: {BATCH_SIZE}')
print(f'Num Iters: {NUM_ITERS}')
print(f'Elapsed time: {t1 - t0:.2f} seconds')
print(f'Throughput: {NUM_ITERS * BATCH_SIZE / (t1 - t0):.2f} samp/sec')

print(prof \
    .key_averages(group_by_input_shape=False) \
    .table(sort_by="self_cuda_time_total", row_limit=100, top_level_events_only=False))
