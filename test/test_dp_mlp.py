import sys
sys.path.append('.')

import time
import pypin
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import graphnet as GNN

NUM_ITERS = 10
BATCH_SIZE = 32

D = 128


class PsuedoGraphNetBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_mlp  = GNN.Mlp(3 * D, [D, D, D])
        self.edge_mlp  = GNN.Mlp(3 * D, [D, D, D])
        self.world_mlp = GNN.Mlp(3 * D, [D, D, D])

    def forward(self, n, e, w):
        n = self.node_mlp(torch.cat([n, n, n], dim=1))
        e = self.edge_mlp(torch.cat([e, e, e], dim=1))
        w = self.world_mlp(torch.cat([w, w, w], dim=1))
        return n, e, w

class PsuedoGraphNet(torch.nn.Module):
    def __init__(self, num_mp_steps=15):
        super().__init__()
        self.num_mp_steps = num_mp_steps
        self.blocks = torch.nn.ModuleList([
            PsuedoGraphNetBlock() for _ in range(num_mp_steps)])

    def forward(self, n, e, w):
        for block in self.blocks:
            n, e, w = block(n, e, w)
        return n, e, w


dev = torch.device('cuda:0')
net = PsuedoGraphNet().half().to(dev)
n = torch.randn(BATCH_SIZE * 1226,  D, device=dev, dtype=torch.float16)
e = torch.randn(BATCH_SIZE * 12281, D, device=dev, dtype=torch.float16)
w = torch.randn(BATCH_SIZE * 1660,  D, device=dev, dtype=torch.float16)

# with pypin.pinhooks.perf_roi(0, 'mlp', ''):

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    profile_memory=True,
) as prof:
    t0 = time.perf_counter()
    for i in range(NUM_ITERS):
        net(n, e, w)
        prof.step()
    t1 = time.perf_counter()

print(f'Batch Size: {BATCH_SIZE}')
print(f'Num Iters: {NUM_ITERS}')
print(f'Elapsed time: {t1 - t0:.2f} seconds')
print(f'Throughput: {NUM_ITERS * BATCH_SIZE / (t1 - t0):.2f} samp/sec')

# print(prof \
#     .key_averages(group_by_input_shape=False, group_by_stack_n=4) \
#     .table(sort_by="self_cuda_time_total", row_limit=-1, top_level_events_only=False))
