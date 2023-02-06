
import torch
from torch.profiler import profile, record_function, ProfilerActivity

import dataset
import model
# import perftools

torch.backends.cuda.matmul.allow_tf32 = True

dev = torch.device('cuda:0')
net = model.CfdModel().to(dev)
opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# print(net)

ds = dataset.CylinderFlowData('./data/cylinder_flow_np/train/t0.npz')
dl = torch.utils.data.DataLoader(
    ds,
    shuffle=True,
    batch_size=8,
    num_workers=8,
    pin_memory=dev.type == 'cuda',
    pin_memory_device=str(dev),
    collate_fn=dataset.collate_cfd)

batch = next(iter(dl))

for _ in range(10):
    opt.zero_grad()

    loss = net.loss(
        batch['node_type'].to(dev),
        batch['velocity'].to(dev),
        batch['mesh_pos'].to(dev),
        batch['srcs'].to(dev),
        batch['dests'].to(dev),
        batch['target_velocity'].to(dev)
    ).backward()

    opt.step()

# with perftools.pinperf.perf_roi(0, 'mgn', 'train'):
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:

    opt.zero_grad()

    loss = net.loss(
        batch['node_type'].to(dev),
        batch['velocity'].to(dev),
        batch['mesh_pos'].to(dev),
        batch['srcs'].to(dev),
        batch['dests'].to(dev),
        batch['target_velocity'].to(dev)
    ).backward()

    opt.step()

print(prof \
    .key_averages(group_by_input_shape=False, group_by_stack_n=0) \
    .table(sort_by="cuda_time_total", row_limit=-1, top_level_events_only=False))
