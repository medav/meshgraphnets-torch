import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import sys
import numpy as np

import time
import deforming_plate as M


filename = sys.argv[1]
np_data = np.load(filename)

NUM_ITERS = 10
BATCH_SIZE = int(filename.replace('b', '').replace('.npz', ''))

checkpoint_file = './deforming_plate_ckpt.pth'
torch.backends.cuda.matmul.allow_tf32 = True

dev = torch.device('cuda:0')
net = M.DeformingPlateModel()
batch_size = BATCH_SIZE

warmup = True
if os.path.exists(checkpoint_file):
    print('Loading checkpoint...')
    warmup = False
    net.load_state_dict(torch.load(checkpoint_file))

net.eval().half().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-4)

print('Setting up data loader...')
batch = {
    'node_type': torch.LongTensor(np_data['node_type']),
    'mesh_pos': torch.Tensor(np_data['mesh_pos']),
    'world_pos': torch.Tensor(np_data['world_pos']),
    'target_world_pos': torch.Tensor(np_data['target_world_pos']),
    'srcs': torch.LongTensor(np_data['srcs']),
    'dsts': torch.LongTensor(np_data['dsts']),
    'wsrcs': torch.LongTensor(np_data['wsrcs']),
    'wdsts': torch.LongTensor(np_data['wdsts']),
}

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    profile_memory=True,
) as prof:
    with torch.no_grad():
        print('running...')
        t0 = time.perf_counter()
        for _ in range(NUM_ITERS):
            loss = net.loss(
                node_type=batch['node_type'].to(dev),
                mesh_pos=batch['mesh_pos'].half().to(dev),
                world_pos=batch['world_pos'].half().to(dev),
                target_world_pos=batch['target_world_pos'].half().to(dev),
                srcs=batch['srcs'].to(dev),
                dsts=batch['dsts'].to(dev),
                wsrcs=batch['wsrcs'].to(dev),
                wdsts=batch['wdsts'].to(dev),
            )

            print('Mem Allocated:', torch.cuda.memory_allocated(dev) / 2**30, 'GB')

            del loss
        t1 = time.perf_counter()

        prof.step()

print(f'Batch Size: {BATCH_SIZE}')
print(f'Num Iters: {NUM_ITERS}')
print(f'Elapsed time: {t1 - t0:.2f} seconds')
print(f'Throughput: {NUM_ITERS * BATCH_SIZE / (t1 - t0):.2f} samp/sec')

print(prof \
    .key_averages(group_by_input_shape=False, group_by_stack_n=4) \
    .table(sort_by="self_cuda_time_total", row_limit=-1, top_level_events_only=False))
