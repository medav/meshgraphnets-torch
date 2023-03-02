import sys
sys.path.append('.')

import time
import torch
import graphnet as GNN

BATCH_SIZE = int(sys.argv[1])
NUM_ITERS = 2048 * 64 // BATCH_SIZE

NN = 1226
NE = 12281
NW = 1660

D = int(sys.argv[2])

class PsuedoGraphNetBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_mlp  = GNN.Mlp(3 * D, [D, D, D], layernorm=False)
        self.edge_mlp  = GNN.Mlp(3 * D, [D, D, D], layernorm=False)
        self.world_mlp = GNN.Mlp(3 * D, [D, D, D], layernorm=False)

    def forward(self, n, e, w):
        n = self.node_mlp(n)
        e = self.edge_mlp(e)
        w = self.world_mlp(w)
        return n, e, w


dev = torch.device('cuda:0')
is_cuda = dev.type == 'cuda'
dtype = torch.float16
net = PsuedoGraphNetBlock().to(dtype).to(dev)

flops = 10 * D * D * (NN + NE + NW)

n1 = torch.randn(BATCH_SIZE * NN, 3 * D, device=dev, dtype=dtype)
e1 = torch.randn(BATCH_SIZE * NE, 3 * D, device=dev, dtype=dtype)
w1 = torch.randn(BATCH_SIZE * NW, 3 * D, device=dev, dtype=dtype)


t0 = time.perf_counter()
for i in range(NUM_ITERS):
    net(n1, e1, w1)

if is_cuda: torch.cuda.synchronize()
t1 = time.perf_counter()

print(f'Batch Size: {BATCH_SIZE}')
print(f'Num Iters: {NUM_ITERS}')
print(f'Elapsed time: {t1 - t0:.2f} seconds')
print(f'Throughput: {NUM_ITERS * BATCH_SIZE / (t1 - t0):.2f} samp/sec')
print(f'Compute: {flops * NUM_ITERS * BATCH_SIZE / (t1 - t0) / 1e9:.2f} GFLOPS')

