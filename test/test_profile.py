import sys
sys.path.append('.')

import torch
from torch.profiler import profile, record_function, ProfilerActivity

import cfd

torch.backends.cuda.matmul.allow_tf32 = True

dev = torch.device('cuda:0')
net = cfd.CfdModel().to(dev)
opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

ds = cfd.CylinderFlowData('./data/cylinder_flow_np/train/t0.npz')
dl = torch.utils.data.DataLoader(
    ds,
    shuffle=True,
    batch_size=1,
    num_workers=1,
    pin_memory=dev.type == 'cuda',
    pin_memory_device=str(dev),
    collate_fn=cfd.collate_cfd)

batch = next(iter(dl))

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=5, active=10),
    with_stack=True
) as prof:
    for _ in range(1 + 5 + 10):
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

        prof.step()

# print(prof)
print(prof \
    .key_averages(group_by_input_shape=False, group_by_stack_n=0) \
    .table(sort_by="self_cuda_time_total", row_limit=-1, top_level_events_only=False))
