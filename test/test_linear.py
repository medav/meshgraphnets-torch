import sys
sys.path.append('.')

import time
# import pypin
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import graphnet as GNN
import cudaprofile

NUM_ITERS = 5000

BS = int(sys.argv[1])
M = int(sys.argv[2])
N = int(sys.argv[3])
K = int(sys.argv[4])


flops = 2 * BS * M * N * K

NS = 1

streams = [torch.cuda.Stream() for _ in range(NS)]

dev = torch.device('cuda:0')
net = torch.nn.Linear(N, K).half().to(dev)

# xs = [
#     torch.randn(BS * M, K, device=dev, dtype=torch.float16)
#     for _ in range(NS)
# ]

x = torch.randn(BS * M, K, device=dev, dtype=torch.float16)

cudaprofile.start()
t0 = time.perf_counter()
for i in range(NUM_ITERS):
    with torch.cuda.stream(streams[i % NS]):
        net(x)

for si in range(NS):
    torch.cuda.current_stream().wait_stream(streams[si])
torch.cuda.synchronize()
t1 = time.perf_counter()
cudaprofile.stop()

print(f'Batch Size: {BS}')
print(f'Num Iters: {NUM_ITERS}')
print(f'Elapsed time: {t1 - t0:.2f} seconds')
print(f'Throughput: {NUM_ITERS * BS / (t1 - t0):.2f} samp/sec')
compute_tflops = NUM_ITERS * flops / (t1 - t0) / 1e12
print(f'Compute: {compute_tflops:.2f} TFLOP/s')
print(f'A100: {compute_tflops / 312 * 100:.2f}%, V100: {compute_tflops / 115 * 100:.2f}%')

