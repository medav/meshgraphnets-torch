import sys
sys.path.append('.')

from contextlib import contextmanager

import time
import torch
import graphnet as GNN


NN = 1226
NE = 12281
NW = 1660

D = 128
DT = torch.float32

dev = torch.device('cuda:0')
is_cuda = dev.type == 'cuda'

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

class BenchmarkResults:
    def __init__(self):
        self.header = []
        self.lines = []
        self.cur_line = 0

    def new_benchmark(self, title):
        self.header.append(title)
        self.cur_line = 0

    def add(self, str):
        if self.cur_line >= len(self.lines):
            self.lines.append([])
        self.lines[self.cur_line].append(str)
        self.cur_line += 1

    def print_csv(self):
        print(','.join(self.header))
        for line in self.lines:
            print(','.join(line))

results = BenchmarkResults()

@contextmanager
def benchmark(title):
    global results
    header = '='*20 + ' ' + title + ' ' + '='*20
    print(header)
    results.new_benchmark(title)
    yield results
    print('=' * len(header))
    print()

def run_benchmark(b, bs, ni, net, x, flops):
    t0 = time.perf_counter()
    for i in range(ni):
        net(*x)

    if is_cuda: torch.cuda.synchronize()
    t1 = time.perf_counter()


    # print(f'Batch Size: {bs}')
    # print(f'Num Iters: {ni}')
    # print(f'Elapsed time: {t1 - t0:.2f} seconds')
    # print(f'Throughput: {ni * bs / (t1 - t0):.2f} samp/sec')
    print(f'{flops * ni * bs / (t1 - t0) / 1e12:.2f}')
    b.add(f'{flops * ni * bs / (t1 - t0) / 1e12:.2f}')

def benchmark_linear(b, M, I, O):
    dtype = DT
    net = torch.nn.Linear(I, O).to(dtype).to(dev)

    flops = M * I * O * 2

    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        x = torch.randn(bs * M, I, device=dev, dtype=dtype)
        run_benchmark(b, bs, 4096 // bs, net, (x,), flops)

def benchmark_mlp(b, M):
    dtype = DT
    net = GNN.Mlp(3 * D, [D, D, D], layernorm=False).to(dtype).to(dev)

    flops = M * (3 * D * D + D * D + D * D) * 2

    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        x = torch.randn(bs * M, 3 * D, device=dev, dtype=dtype)
        run_benchmark(b, bs, 4096 // bs, net, (x,), flops)

with benchmark('Nodes 384x128') as b: benchmark_linear(b, NN, 384, 128)
with benchmark('Nodes 128x128') as b: benchmark_linear(b, NN, 128, 128)
with benchmark('Nodes MLP') as b: benchmark_mlp(b, NN)

with benchmark('Edges 384x128') as b: benchmark_linear(b, NE, 384, 128)
with benchmark('Edges 128x128') as b: benchmark_linear(b, NE, 128, 128)
with benchmark('Edges MLP') as b: benchmark_mlp(b, NE)

with benchmark('Wedges 384x128') as b: benchmark_linear(b, NW, 384, 128)
with benchmark('Wedges 128x128') as b: benchmark_linear(b, NW, 128, 128)
with benchmark('Wedges MLP') as b: benchmark_mlp(b, NW)



with benchmark('All MLPs') as b:
    dtype = DT
    net = PsuedoGraphNetBlock().to(dtype).to(dev)

    flops = 10 * D * D * (NN + NE + NW)

    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        n1 = torch.randn(bs * NN, 3 * D, device=dev, dtype=dtype)
        e1 = torch.randn(bs * NE, 3 * D, device=dev, dtype=dtype)
        w1 = torch.randn(bs * NW, 3 * D, device=dev, dtype=dtype)

        run_benchmark(b, bs, 4096 // bs, net, (n1, e1, w1), flops)

with benchmark('1024x384x128') as b: benchmark_linear(b, 1024, 384, 128)
with benchmark('1024x128x128') as b: benchmark_linear(b, 1024, 128, 128)
with benchmark('1024x MLP') as b: benchmark_mlp(b, 1024)

results.print_csv()
