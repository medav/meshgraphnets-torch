import sys
sys.path.append('.')

import time
import torch
import cfd

NUM_ITERS = 5
BATCH_SIZE = 8
NUM_THREADS = 32

torch.set_num_threads(NUM_THREADS)

dev = torch.device('cpu')
net = cfd.CfdModel().to(dev)
opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

ds = cfd.CylinderFlowData('./mesh0.npz')
dl = torch.utils.data.DataLoader(
    ds,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=1,
    pin_memory=dev.type == 'cuda',
    pin_memory_device=str(dev),
    collate_fn=cfd.collate_cfd)

batch = next(iter(dl))


t0 = time.perf_counter()
for _ in range(NUM_ITERS):
    opt.zero_grad()

    loss = net.loss(
        batch['node_type'].to(dev),
        batch['velocity'].to(dev),
        batch['mesh_pos'].to(dev),
        batch['srcs'].to(dev),
        batch['dsts'].to(dev),
        batch['target_velocity'].to(dev)
    ).backward()

    opt.step()

t1 = time.perf_counter()
print(f'Batch Size: {BATCH_SIZE}')
print(f'Num Iters: {NUM_ITERS}')
print(f'Num Threads: {NUM_THREADS}')
print(f'Elapsed time: {t1 - t0:.2f} seconds')
print(f'Throughput: {NUM_ITERS * BATCH_SIZE / (t1 - t0):.2f} samp/sec')
