import enum
import torch
import numpy as np
import graphnet as GNN


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


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
            3, # 2D rel pos. + length
            output_dim,
            latent_size,
            num_edge_sets,
            num_layers,
            num_mp_steps)

        self.out_norm = GNN.InvertableNorm((output_dim,))
        self.node_norm = GNN.InvertableNorm((input_dim,))
        self.edge_norm = GNN.InvertableNorm((2 + 1,)) # 2D coord + length

    def forward(
        self,
        node_type : torch.LongTensor,
        velocity : torch.Tensor,
        mesh_pos : torch.Tensor,
        srcs : torch.LongTensor,
        dsts : torch.LongTensor,
        unnorm : bool = True
    ) -> torch.Tensor:
        """Predicts Delta V"""

        B = velocity.shape[0]

        node_type_oh = \
            torch.nn.functional.one_hot(node_type, num_classes=NodeType.SIZE) \
                .squeeze() \
                .expand((B, -1, -1))

        node_features = torch.cat([velocity, node_type_oh], dim=-1)
        rel_mesh_pos = mesh_pos[:, srcs, :] - mesh_pos[:, dsts, :]

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

    def loss(
        self,
        node_type : torch.Tensor,
        velocity : torch.Tensor,
        mesh_pos : torch.Tensor,
        srcs : torch.LongTensor,
        dsts : torch.LongTensor,
        target_velocity : torch.Tensor
    ) -> torch.Tensor:

        pred = self.forward(
            node_type,
            velocity,
            mesh_pos,
            srcs,
            dsts,
            unnorm=False
        )

        with torch.no_grad():
            delta_v = target_velocity - velocity
            delta_v_norm = self.out_norm(delta_v)

        residuals = (delta_v_norm - pred).sum(dim=-1)

        mask = (node_type == NodeType.NORMAL) \
            .logical_or(node_type == NodeType.OUTFLOW) \
            .squeeze()

        return residuals[:, mask].pow(2).mean()


class CylinderFlowData(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.filename = filename

        data = np.load(self.filename)
        self.num_samples = len(data['cells']) - 1

        self.cells = torch.LongTensor(data['cells'][0, ...])
        self.node_type = torch.LongTensor(data['node_type'][0, ...])
        self.srcs, self.dsts = GNN.cells_to_edges(self.cells)
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
                dsts=self.dsts
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
        'dsts': batch[0]['dsts']
    }


if __name__ == '__main__':
    net = CfdModel()
    print(net)

    loss = net.loss(
        torch.LongTensor([NodeType.NORMAL for _ in range(3)]),
        torch.randn(3, 2),
        torch.LongTensor([[0, 1, 2]]),
        torch.randn(3, 2),
        torch.randn(3, 2)
    )

    loss.backward()
