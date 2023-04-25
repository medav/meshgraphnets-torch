import enum
import torch
import json
import os
import numpy as np
import graphnet as GNN
from dataclasses import dataclass


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9

@dataclass
class ClothSample(GNN.GraphNetSample):
    world_pos : torch.Tensor
    prev_world_pos : torch.Tensor
    target_world_pos : torch.Tensor

@dataclass
class ClothSampleBatch(GNN.GraphNetSampleBatch):
    world_pos : torch.Tensor
    prev_world_pos : torch.Tensor
    target_world_pos : torch.Tensor

class ClothModel(torch.nn.Module):
    def __init__(
        self,
        input_dim : int = 3 + NodeType.SIZE,
        output_dim : int = 3,
        latent_size : int = 128,
        num_edge_sets : int = 1,
        num_layers : int = 2,
        num_mp_steps : int = 15
    ):
        super().__init__()
        self.graph_net = GNN.GraphNetModel(
            input_dim,
            [7],
            output_dim,
            latent_size,
            num_edge_sets,
            num_layers,
            num_mp_steps)

        self.out_norm = GNN.InvertableNorm((output_dim,))
        self.node_norm = GNN.InvertableNorm((input_dim,))
        self.edge_norm = GNN.InvertableNorm((7,)) # 2D coord + 3D coord + 2*length = 7

    def forward(self, x : ClothSampleBatch, unnorm : bool = True) -> torch.Tensor:
        """Predicts Delta V"""

        node_type_oh = \
            torch.nn.functional.one_hot(x.node_type, num_classes=NodeType.SIZE) \
                .squeeze()

        velocity = x.world_pos - x.prev_world_pos

        node_features = torch.cat([velocity, node_type_oh], dim=-1)

        srcs, dsts = GNN.cells_to_edges(x.cells)
        rel_mesh_pos = x.mesh_pos[srcs, :] - x.mesh_pos[dsts, :]
        rel_world_pos = x.world_pos[srcs, :] - x.world_pos[dsts, :]

        edge_features = torch.cat([
            rel_mesh_pos,
            torch.norm(rel_mesh_pos, dim=-1, keepdim=True),
            rel_world_pos,
            torch.norm(rel_world_pos, dim=-1, keepdim=True)
        ], dim=-1)

        graph = GNN.MultiGraph(
            node_features=self.node_norm(node_features),
            edge_sets=[ GNN.EdgeSet(self.edge_norm(edge_features), srcs, dsts) ]
        )

        net_out = self.graph_net(graph)

        if unnorm: return self.out_norm.inverse(net_out)
        else: return net_out

    def loss(self, x : ClothSampleBatch) -> torch.Tensor:
        pred = self.forward(x, unnorm=False)

        with torch.no_grad():
            target_accel = x.target_world_pos - 2 * x.world_pos + x.prev_world_pos
            target_accel_norm = self.out_norm(target_accel)

        residuals = (target_accel_norm - pred).sum(dim=-1)
        mask = (x.node_type == NodeType.NORMAL).squeeze()
        return residuals[mask].pow(2).mean()


class ClothData(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.meta = json.loads(open(os.path.join(path, 'meta.json')).read())
        self.files = self.meta['files']
        self.num_samples = sum(self.files[f] - 1 for f in self.files)

    @property
    def avg_nodes_per_sample(self):
        total_nodes = 0
        total_samples = 0
        for fname, num_steps in self.files.items():
            data = np.load(os.path.join(self.path, fname))
            total_nodes += data['mesh_pos'].shape[1] * (num_steps - 1)
            total_samples += num_steps - 1

        return total_nodes / total_samples


    def idx_to_file(self, sample_id):
        for fname, num_steps in self.files.items():
            if sample_id < (num_steps - 1): return fname, sample_id
            else: sample_id -= (num_steps - 1)
        raise IndexError()

    def __len__(self): return self.num_samples

    def __getitem__(self, idx : int) -> dict:
        fname, sid = self.idx_to_file(idx)
        data = np.load(os.path.join(self.path, fname))

        return ClothSample(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]).squeeze(),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            prev_world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 2, ...])
        )

def collate_fn(batch): return GNN.collate_common(batch, ClothSampleBatch)
def make_model(): return ClothModel()
def make_dataset(path): return ClothData(path)

def load_batch_npz(path : str, dtype : torch.dtype, dev : torch.device):
    return GNN.load_batch_npz_common(path, dtype, dev, ClothSampleBatch)
