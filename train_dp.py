import os
import torch

import time
import deforming_plate as M

checkpoint_file = './deforming_plate_ckpt.pth'
torch.backends.cuda.matmul.allow_tf32 = True

dev = torch.device('cuda:0')
net = M.DeformingPlateModel()
batch_size = 48
warmup_samples = 1000

warmup = True
if os.path.exists(checkpoint_file):
    print('Loading checkpoint...')
    warmup = False
    net.load_state_dict(torch.load(checkpoint_file))

net.to(dev)

opt = torch.optim.Adam(net.parameters(), lr=1e-4)
ds = M.DeformingPlateData('./data/deforming_plate_np/train/')

dl = torch.utils.data.DataLoader(
    ds,
    shuffle=True,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=dev.type == 'cuda',
    pin_memory_device=str(dev),
    collate_fn=M.collate_fn)

if warmup:
    for i, batch in enumerate(dl):
        print(f'Warming normalizers ({i * batch_size}/{warmup_samples})...')
        net.loss(
            node_type=batch['node_type'].to(dev),
            mesh_pos=batch['mesh_pos'].to(dev),
            world_pos=batch['world_pos'].to(dev),
            target_world_pos=batch['target_world_pos'].to(dev),
            srcs=batch['srcs'].to(dev),
            dsts=batch['dsts'].to(dev),
            wsrcs=batch['wsrcs'].to(dev),
            wdsts=batch['wdsts'].to(dev),
        )

        if i * batch_size > warmup_samples: break


total_iters = 0
should_break = False
t0 = time.perf_counter()

try:
    while True:
        for i, batch in enumerate(dl):
            with torch.amp.autocast('cuda'):
                opt.zero_grad()
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
                loss.backward()
                opt.step()

                print(i, loss.item())
                total_iters += 1

finally:
    t1 = time.perf_counter()
    print(f'Training Speed: {total_iters / (t1 - t0):.2f} step/s')
    print(f'Throughput: {batch_size * total_iters / (t1 - t0):.2f} samp/s')
    print('Saving checkpoint...')
    torch.save(net.state_dict(), checkpoint_file)

