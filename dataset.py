from dataclasses import dataclass
import enum
from typing import Optional
import torch
import functools
import numpy as np

@functools.lru_cache(maxsize=8)
def load_npz_cached(fname):
    # print('Loading', fname)
    return np.load(fname)

def cells_to_edges(cells : torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    cells: int32[M, 3]
    :ret: int32[E], int32[E]
    """

    raw_edges = torch.cat([
        cells[:, 0:2],
        cells[:, 1:3],
        torch.stack([cells[:, 2], cells[:, 0]], dim=1)
    ], dim=0)

    srcs = raw_edges.max(dim=1).values
    dsts = raw_edges.min(dim=1).values

    edges = torch.stack([srcs, dsts], dim=1)
    unique_edges = edges.unique(dim=0, sorted=False)
    srcs, dsts = unique_edges[:, 0], unique_edges[:, 1]

    return torch.cat([srcs, dsts], dim=0), torch.cat([dsts, srcs], dim=0)

class CylinderFlowData(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.filename = filename

        data = load_npz_cached(self.filename)
        self.num_samples = len(data['cells']) - 1

        self.cells = torch.LongTensor(data['cells'][0, ...])
        self.node_type = torch.LongTensor(data['node_type'][0, ...])
        self.srcs, self.dests = cells_to_edges(self.cells)
        self.mesh_pos = data['mesh_pos'].copy()
        self.pressure = data['pressure'].copy()
        self.velocity = data['velocity'].copy()

    def __len__(self): return self.num_samples

    def __getitem__(self, idx : int) -> dict:
        assert idx < self.num_samples

        with torch.no_grad():
            return dict(
                cells=self.cells,
                mesh_pos=torch.Tensor(self.mesh_pos[idx, ...]),
                node_type=self.node_type,
                pressure=torch.Tensor(self.pressure[idx, ...]),
                velocity=torch.Tensor(self.velocity[idx, ...]),
                target_velocity=torch.Tensor(self.velocity[idx + 1, ...]),
                srcs=self.srcs,
                dests=self.dests
            )

def collate_cfd(batch):
    return {
        'cells': batch[0]['cells'],
        'mesh_pos': torch.stack([x['mesh_pos'] for x in batch], dim=0),
        'node_type': batch[0]['node_type'],
        'pressure': torch.stack([x['pressure'] for x in batch], dim=0),
        'velocity': torch.stack([x['velocity'] for x in batch], dim=0),
        'target_velocity': torch.stack([x['target_velocity'] for x in batch], dim=0),
        'srcs': batch[0]['srcs'],
        'dests': batch[0]['dests']
    }

if __name__ == '__main__':
    ds = CylinderFlowData('./data/cylinder_flow_np/train/t0.npz')

    dl = torch.utils.data.DataLoader(
        ds, shuffle=True, num_workers=4, collate_fn=lambda x: x)

    for i, sample in enumerate(dl):
        print(i, sample[0]['cells'].shape)
