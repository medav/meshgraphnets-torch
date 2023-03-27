import enum
import torch
import json
import os
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
            [3], # 2D rel pos. + length
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

        node_type_oh = \
            torch.nn.functional.one_hot(node_type, num_classes=NodeType.SIZE) \
                .squeeze()

        node_features = torch.cat([velocity, node_type_oh], dim=-1)
        rel_mesh_pos = mesh_pos[srcs, :] - mesh_pos[dsts, :]

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

        return residuals[mask].pow(2).mean()

class CylinderFlowData(torch.utils.data.Dataset):
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

        return dict(
            node_type=torch.LongTensor(data['node_type'][sid, ...]).squeeze(),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            velocity=velocity,
            target_velocity=torch.Tensor(data['velocity'][sid + 1, ...]),
            pressure=torch.Tensor(data['pressure'][sid, ...]),
            srcs=srcs, dsts=dsts
        )


def collate_fn(batch):
    node_offs = torch.LongTensor([
        0 if i == 0 else batch[i - 1]['node_type'].shape[0]
        for i in range(len(batch))
    ]).cumsum(dim=0)

    srcss = []
    dstss = []

    for i in range(len(batch)):
        srcss.append(batch[i]['srcs'] + node_offs[i])
        dstss.append(batch[i]['dsts'] + node_offs[i])

    return dict(
        node_type=torch.cat([b['node_type'] for b in batch], dim=0),
        mesh_pos=torch.cat([b['mesh_pos'] for b in batch], dim=0),
        velocity=torch.cat([b['velocity'] for b in batch], dim=0),
        target_velocity=torch.cat([b['target_velocity'] for b in batch], dim=0),
        pressure=torch.cat([b['pressure'] for b in batch], dim=0),
        srcs=torch.cat(srcss, dim=0),
        dsts=torch.cat(dstss, dim=0),
    )

if __name__ == '__main__':
    import time

    NI = 30
    BS = 32
    dev = torch.device('cuda:0')
    net = CfdModel().to(dev)

    ds = CylinderFlowData('./data/cylinder_flow_np/train/t0.npz')

    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=True,
        batch_size=BS,
        num_workers=1,
        pin_memory=dev.type == 'cuda',
        pin_memory_device=str(dev),
        collate_fn=collate_fn)

    batch = next(iter(dl))

    with torch.amp.autocast('cuda'):
        t0 = time.perf_counter()
        for _ in range(NI):
            net.loss(
                batch['node_type'].to(dev),
                batch['velocity'].to(dev),
                batch['mesh_pos'].to(dev),
                batch['srcs'].to(dev),
                batch['dsts'].to(dev),
                batch['target_velocity'].to(dev)
            ).backward()
        t1 = time.perf_counter()

    print(f'Batch Size: {BS}')
    print(f'Num Iters: {NI}')
    print(f'Num Threads: {NI}')
    print(f'Elapsed time: {t1 - t0:.2f} seconds')
    print(f'Throughput: {NI * BS / (t1 - t0):.2f} samp/sec')

