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
class CfdSample(GNN.GraphNetSample):
    velocity : torch.Tensor
    target_velocity : torch.Tensor
    pressure : torch.Tensor

@dataclass
class CfdSampleBatch(GNN.GraphNetSampleBatch):
    velocity : torch.Tensor
    target_velocity : torch.Tensor
    pressure : torch.Tensor

class CfdModel(torch.nn.Module):
    def __init__(
        self,
        input_dim : int = 2 + NodeType.SIZE, # vx, vy, one_hot(type)
        output_dim : int = 2, # vx, vy
        latent_size : int = 128,
        num_edge_sets : int = 1,
        num_layers : int = 2,
        num_mp_steps : int = 15
    ):
        super().__init__()
        self.graph_net = GNN.GraphNetModel(
            input_dim,
            [3], # 2D rel pos. + length
            output_dim,
            latent_size,
            num_edge_sets,
            num_layers,
            num_mp_steps)

        self.out_norm = GNN.InvertableNorm((output_dim,))
        self.node_norm = GNN.InvertableNorm((input_dim,))
        self.edge_norm = GNN.InvertableNorm((2 + 1,)) # 2D coord + length

    def forward(self, x : CfdSampleBatch, unnorm : bool = True) -> torch.Tensor:
        """Predicts Delta V"""

        node_type_oh = \
            torch.nn.functional.one_hot(x.node_type, num_classes=NodeType.SIZE) \
                .squeeze()

        node_features = torch.cat([x.velocity, node_type_oh], dim=-1)
        rel_mesh_pos = x.mesh_pos[x.srcs, :] - x.mesh_pos[x.dsts, :]

        edge_features = torch.cat([
            rel_mesh_pos,
            torch.norm(rel_mesh_pos, dim=-1, keepdim=True)
        ], dim=-1)

        graph = GNN.MultiGraph(
            node_features=self.node_norm(node_features),
            edge_sets=[ GNN.EdgeSet(self.edge_norm(edge_features), x.srcs, x.dsts) ]
        )

        net_out = self.graph_net(graph)

        if unnorm: return self.out_norm.inverse(net_out)
        else: return net_out

    def loss(self, x : CfdSampleBatch) -> torch.Tensor:
        pred = self.forward(x, unnorm=False)

        with torch.no_grad():
            delta_v = x.target_velocity - x.velocity
            delta_v_norm = self.out_norm(delta_v)

        residuals = (delta_v_norm - pred).sum(dim=-1)
        mask = (x.node_type == NodeType.NORMAL) \
            .logical_or(x.node_type == NodeType.OUTFLOW) \
            .squeeze()

        return residuals[mask].pow(2).mean()


class CfdData(torch.utils.data.Dataset):
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

        cells = torch.LongTensor(data['cells'][sid, ...])
        srcs, dsts = GNN.cells_to_edges(cells)
        velocity = torch.Tensor(data['velocity'][sid, ...])

        return CfdSample(
            cells=cells,
            node_type=torch.LongTensor(data['node_type'][sid, ...]).squeeze(),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            velocity=velocity,
            target_velocity=torch.Tensor(data['velocity'][sid + 1, ...]),
            pressure=torch.Tensor(data['pressure'][sid, ...]),
            srcs=srcs, dsts=dsts
        )


def collate_fn(batch): return GNN.collate_common(batch, CfdSampleBatch)
def make_model(): return CfdModel()
def make_dataset(path): return CfdModel(path)

def load_batch_npz(path : str, dtype : torch.dtype, dev : torch.device):
    np_data = np.load(path)

    return len(np_data['node_offs']), CfdSampleBatch(
        cell_offs=torch.LongTensor(np_data['cell_offs']).to(dev),
        node_offs=torch.LongTensor(np_data['node_offs']).to(dev),
        cells=torch.LongTensor(np_data['cells']).to(dev),
        node_type=torch.LongTensor(np_data['node_type']).to(dev),
        velocity=torch.Tensor(np_data['velocity']).to(dev).to(dtype),
        mesh_pos=torch.Tensor(np_data['mesh_pos']).to(dev).to(dtype),
        srcs=torch.LongTensor(np_data['srcs']).to(dev),
        dsts=torch.LongTensor(np_data['dsts']).to(dev),
        target_velocity=torch.Tensor(np_data['target_velocity']).to(dev).to(dtype)
    )

def infer(net, batch): net.loss(**batch)

