import os
import torch

import time
import deforming_plate as M


NUM_ITERS = 100
BATCH_SIZE = 256

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
ds = M.DeformingPlateData('./data/deforming_plate_np/train/')

dl = torch.utils.data.DataLoader(
    ds,
    shuffle=False,
    batch_size=batch_size,
    num_workers=1,
    pin_memory=dev.type == 'cuda',
    pin_memory_device=str(dev),
    collate_fn=M.collate_fn)


with torch.no_grad():
    batch = next(iter(dl))

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
    t1 = time.perf_counter()

print(f'Batch Size: {BATCH_SIZE}')
print(f'Num Iters: {NUM_ITERS}')
print(f'Elapsed time: {t1 - t0:.2f} seconds')
print(f'Throughput: {NUM_ITERS * BATCH_SIZE / (t1 - t0):.2f} samp/sec')

