import os
import torch

import time
import deforming_plate as M

checkpoint_file = './deforming_plate_ckpt.pth'
torch.backends.cuda.matmul.allow_tf32 = True

dev = torch.device('cuda:0')
net = M.DeformingPlateModel()


warmup = True
if os.path.exists(checkpoint_file):
    print('Loading checkpoint...')
    warmup = False
    net.load_state_dict(torch.load(checkpoint_file))

net.to(dev)

opt = torch.optim.Adam(net.parameters(), lr=1e-4)


if warmup:
    ds = M.DeformingPlateData('./data/cylinder_flow_np/train/t0.npz')
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=True,
        batch_size=8,
        num_workers=8,
        pin_memory=dev.type == 'cuda',
        pin_memory_device=str(dev),
        collate_fn=M.collate_fn)

    for i, batch in enumerate(dl):
        print(f'Warming normalizers ({i}/{len(ds)})...')
        net.loss(
            batch['node_type'].to(dev),
            batch['velocity'].to(dev),
            batch['mesh_pos'].to(dev),
            batch['srcs'].to(dev),
            batch['dsts'].to(dev),
            batch['target_velocity'].to(dev)
        )


def train_on_data(ds):
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=True,
        batch_size=8,
        num_workers=8,
        pin_memory=dev.type == 'cuda',
        pin_memory_device=str(dev),
        collate_fn=M.collate_fn)

    ni = 0

    try:
        for i, batch in enumerate(dl):
            loss = net.loss(
                batch['node_type'].to(dev),
                batch['velocity'].to(dev),
                batch['mesh_pos'].to(dev),
                batch['srcs'].to(dev),
                batch['dests'].to(dev),
                batch['target_velocity'].to(dev)
            )

            loss.backward()
            opt.step()
            opt.zero_grad()

            print(i, loss.item())

            ni += 1

    except KeyboardInterrupt:
        return ni, True

    return ni, False

t0 = time.perf_counter()
total_iters = 0
should_break = False

while True:
    for ti in range(1000):
        filename = f'./data/cylinder_flow_np/train/t{ti}.npz'
        print(f'Training on {filename}...')
        ds = M.DeformingPlateData(filename)
        num_iters, should_break = train_on_data(ds)
        total_iters += num_iters

        if should_break: break
    if should_break: break

t1 = time.perf_counter()
print(f'Throughput: {total_iters / (t1 - t0):.2f} step/s')
print('Saving checkpoint...')
torch.save(net.state_dict(), checkpoint_file)
