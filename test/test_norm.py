import torch
import logging
import graphnet as GNN

import torch._dynamo as torchdynamo

torchdynamo.config.log_level = logging.DEBUG
torchdynamo.config.verbose = True
torchdynamo.config.output_code = True


norm = GNN.InvertableNorm((128,))

norm = torch.compile(norm)

for _ in range(100):
    norm(torch.randn(1024, 128))

print(norm)

for n, b in norm.named_buffers():
    print(n, b)
