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
class IncomprNsSample(GNN.GraphNetSample):
    velocity : torch.Tensor
    target_velocity : torch.Tensor
    pressure : torch.Tensor

@dataclass
class IncomprNsSampleBatch(GNN.GraphNetSampleBatch):
    velocity : torch.Tensor
    target_velocity : torch.Tensor
    pressure : torch.Tensor

class IncomprNsModel(torch.nn.Module):
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

    def forward(self, x : IncomprNsSampleBatch, unnorm : bool = True) -> torch.Tensor:
        """Predicts Delta V"""

        node_type_oh = \
            torch.nn.functional.one_hot(x.node_type, num_classes=NodeType.SIZE) \
                .squeeze()

        node_features = torch.cat([x.velocity, node_type_oh], dim=-1)

        srcs, dsts = GNN.cells_to_edges(x.cells)
        rel_mesh_pos = x.mesh_pos[srcs, :] - x.mesh_pos[dsts, :]

        edge_features = torch.cat([
            rel_mesh_pos,
            torch.norm(rel_mesh_pos, dim=-1, keepdim=True)
        ], dim=-1)

        graph = GNN.MultiGraph(
            node_features=self.node_norm(node_features),
            edge_sets=[ GNN.EdgeSet(self.edge_norm(edge_features), srcs, dsts) ]
        )

        net_out = self.graph_net(graph)

        if unnorm: return self.out_norm.inverse(net_out)
        else: return net_out

    def loss(self, x : IncomprNsSampleBatch) -> torch.Tensor:
        pred = self.forward(x, unnorm=False)

        with torch.no_grad():
            delta_v = x.target_velocity - x.velocity
            delta_v_norm = self.out_norm(delta_v)

        residuals = (delta_v_norm - pred).sum(dim=-1)
        mask = (x.node_type == NodeType.NORMAL) \
            .logical_or(x.node_type == NodeType.OUTFLOW) \
            .squeeze()

        return residuals[mask].pow(2).mean()

    def import_numpy_weights(self, weights : dict[str, np.ndarray]):
        def hookup_norm(mod, prefix):
            mod.accum_count = GNN.make_torch_buffer(weights[f'{prefix}/num_accumulations:0'])
            mod.num_accum = GNN.make_torch_buffer(weights[f'{prefix}/acc_count:0'])
            mod.running_sum = GNN.make_torch_buffer(weights[f'{prefix}/acc_sum:0'])
            mod.running_sum_sq = GNN.make_torch_buffer(weights[f'{prefix}/acc_sum_squared:0'])

        self.graph_net.import_numpy_weights(weights, ['mesh_edges_edge_fn'])

        hookup_norm(self.out_norm, 'Model/output_normalizer')
        hookup_norm(self.node_norm, 'Model/node_normalizer')
        hookup_norm(self.edge_norm, 'Model/edge_normalizer')


class IncomprNsData(torch.utils.data.Dataset):
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

        return IncomprNsSample(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]).squeeze(),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            velocity=torch.Tensor(data['velocity'][sid, ...]),
            target_velocity=torch.Tensor(data['velocity'][sid + 1, ...]),
            pressure=torch.Tensor(data['pressure'][sid, ...])
        )

sample_type = IncomprNsSample
batch_type = IncomprNsSampleBatch
model_type = IncomprNsModel
dataset_type = IncomprNsData
