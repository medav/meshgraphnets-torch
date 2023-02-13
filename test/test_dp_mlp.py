import sys
sys.path.append('.')

import time
import pypin
import torch
import graphnet as GNN

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class PsuedoGraphNetBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_mlp = GNN.Mlp(384, [128, 128, 128])
        self.edge_mlp = GNN.Mlp(384, [128, 128, 128])
        self.world_mlp = GNN.Mlp(384, [128, 128, 128])

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


net = PsuedoGraphNet()
n = torch.randn(840, 128)
e = torch.randn(8210, 128)
w = torch.randn(1914, 128)

with pypin.pinhooks.perf_roi(0, 'mlp', ''):
    n, e, w = net(n, e, w)
