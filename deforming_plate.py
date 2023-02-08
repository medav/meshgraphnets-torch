# The following is based on:
# https://github.com/deepmind/deepmind-research/compare/master...isabellahuang:deepmind-research-tf1:tf1
#
# Subject to Apache v2 license (see original source for details)

import torch
import random
import enum
import numpy as np
import graphnet as GNN

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


def arange2d(m, n):
    return torch.stack([
        torch.arange(m).reshape(-1, 1).repeat(1, n),
        torch.arange(n).reshape(1, -1).repeat(m, 1)
    ], dim=2)

def squared_dist(A : torch.Tensor, B : torch.Tensor):
    row_norms_A = A.pow(2).sum(dim=1).reshape(-1, 1) # N, 1
    row_norms_B = B.pow(2).sum(dim=1).reshape(1, -1) # 1, N
    return row_norms_A - 2 * (A @ B.t()) + row_norms_B

def construct_world_edges(
    world_pos : torch.Tensor,
    node_type : torch.Tensor,
    thresh : float = 0.03
) -> torch.Tensor:

    deformable = node_type != NodeType.OBSTACLE
    deformable_idx = torch.arange(node_type.shape[0])[deformable]

    actuator = node_type == NodeType.OBSTACLE
    actuator_idx = torch.arange(node_type.shape[0])[actuator]

    actuator_pos = world_pos[actuator].to(torch.float64) # M, D
    deformable_pos = world_pos[deformable].to(torch.float64) # N, D

    dists = squared_dist(actuator_pos, deformable_pos) # M, N
    M, N = dists.shape

    idxs = arange2d(M, N)
    rel_close_pair_idx = idxs[dists < (thresh ** 2)]

    srcs = actuator_idx[rel_close_pair_idx[:, 0]]
    dsts = deformable_idx[rel_close_pair_idx[:, 1]]

    return torch.cat([srcs, dsts], dim=0), torch.cat([dsts, srcs], dim=0)


class DeformingPlateModel(torch.nn.Module):
    def __init__(
        self,
        input_dim : int = 3 + NodeType.SIZE, # vx, vy, vz, one_hot(type)
        n_mesh_edge_f : int = 2 * (3 + 1), # 2x (3D coord + length)
        n_world_edge_f = 3 + 1, # 3D coord + length
        output_dim : int = 3, # vx, vy, vz
        latent_size : int = 128,
        num_layers : int = 2,
        num_mp_steps : int = 15
    ):
        super().__init__()
        self.num_edge_sets = 2
        self.graph_net = GNN.GraphNetModel(
            input_dim,
            [n_mesh_edge_f, n_world_edge_f],
            output_dim,
            latent_size,
            self.num_edge_sets,
            num_layers,
            num_mp_steps)

        self.out_norm = GNN.InvertableNorm((output_dim,))
        self.node_norm = GNN.InvertableNorm((input_dim,))
        self.mesh_edge_norm = GNN.InvertableNorm((n_mesh_edge_f,))
        self.world_edge_norm = GNN.InvertableNorm((n_world_edge_f,))

    def forward(
        self,
        node_type : torch.LongTensor,
        mesh_pos : torch.Tensor,
        world_pos : torch.Tensor,
        known_vel : torch.Tensor,
        srcs : torch.LongTensor,
        dsts : torch.LongTensor,
        wsrcs : torch.LongTensor,
        wdsts : torch.LongTensor,
        unnorm : bool = True
    ) -> torch.Tensor:
        """Predicts Velocity"""

        #
        # Node Features
        #

        node_type_oh = \
            torch.nn.functional.one_hot(node_type, num_classes=NodeType.SIZE) \
                .squeeze()

        node_features = torch.cat([known_vel, node_type_oh], dim=-1)

        #
        # Mesh Edge Features
        #

        rel_mesh_pos = mesh_pos[srcs, :] - mesh_pos[dsts, :]
        rel_world_mesh_pos = world_pos[srcs, :] - mesh_pos[dsts, :]

        mesh_edge_features = torch.cat([
            rel_mesh_pos,
            torch.norm(rel_mesh_pos, dim=-1, keepdim=True),
            rel_world_mesh_pos,
            torch.norm(rel_world_mesh_pos, dim=-1, keepdim=True)
        ], dim=-1)


        #
        # World Edge Features
        #

        rel_world_pos = world_pos[wsrcs, :] - world_pos[wdsts, :]

        world_edge_features = torch.cat([
            rel_world_pos,
            torch.norm(rel_world_pos, dim=-1, keepdim=True)
        ], dim=-1)


        graph = GNN.MultiGraph(
            node_features=self.node_norm(node_features),
            edge_sets=[
                GNN.EdgeSet(self.mesh_edge_norm(mesh_edge_features), srcs, dsts),
                GNN.EdgeSet(self.world_edge_norm(world_edge_features), wsrcs, wdsts)
            ]
        )

        net_out = self.graph_net(graph)

        if unnorm: return self.out_norm.inverse(net_out)
        else: return net_out

    def loss(
        self,
        node_type : torch.LongTensor,
        mesh_pos : torch.Tensor,
        world_pos : torch.Tensor,
        target_world_pos : torch.Tensor,
        srcs : torch.LongTensor,
        dsts : torch.LongTensor,
        wsrcs : torch.LongTensor,
        wdsts : torch.LongTensor
    ) -> torch.Tensor:

        known_vel = target_world_pos - world_pos
        known_vel[node_type != NodeType.NORMAL, :] = 0.0

        pred = self.forward(
            node_type,
            mesh_pos,
            world_pos,
            known_vel,
            srcs,
            dsts,
            wsrcs,
            wdsts,
            unnorm=False
        )

        with torch.no_grad():
            delta_x = target_world_pos - world_pos
            delta_x_norm = self.out_norm(delta_x)

        residuals = (delta_x_norm - pred).sum(dim=-1)
        mask = (node_type == NodeType.NORMAL) .squeeze()
        return residuals[mask].pow(2).mean()


class DeformingPlateData(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.filename = filename

        data = np.load(self.filename)
        self.num_samples = len(data['cells']) - 1

        self.cells = torch.LongTensor(data['cells'][0, ...])
        self.node_type = torch.LongTensor(data['node_type'][0, ...]).squeeze()
        self.srcs, self.dsts = GNN.cells_to_edges(self.cells)
        self.mesh_pos = data['mesh_pos'].copy()
        self.world_pos = data['world_pos'].copy()
        self.stress = data['stress'].copy()

    def __len__(self): return self.num_samples

    def __getitem__(self, idx : int) -> dict:
        assert idx < self.num_samples

        world_pos = torch.Tensor(self.world_pos[idx, ...])
        wsrcs, wdsts = construct_world_edges(world_pos, self.node_type)

        return dict(
            node_offs=torch.LongTensor([0]),
            node_type=self.node_type,
            mesh_pos=torch.Tensor(self.mesh_pos[idx, ...]),
            world_pos=world_pos,
            target_world_pos=torch.Tensor(self.world_pos[idx + 1, ...]),
            stress=torch.Tensor(self.stress[idx, ...]),
            srcs=self.srcs,
            dsts=self.dsts,
            wsrcs=wsrcs,
            wdsts=wdsts
        )


class TestDeformingPlateData(torch.utils.data.Dataset):
    def __init__(self, n=32):
        self.n = n

        self.node_type = torch.LongTensor(
            [NodeType.NORMAL, NodeType.NORMAL, NodeType.OBSTACLE]
        )

        self.mesh_pos = torch.randn((3, 3))
        self.world_pos = torch.randn((3, 3))
        self.target_world_pos = torch.randn((3, 3))
        self.stress = torch.randn((3, 3))

        self.srcs = torch.LongTensor([0, 0, 1, 1, 2, 2])
        self.dsts = torch.LongTensor([1, 2, 0, 2, 0, 1])

        self.wsrcs = torch.LongTensor([0, 0, 1, 2])
        self.wdsts = torch.LongTensor([1, 2, 0, 0])


    def __len__(self): return self.n

    def __getitem__(self, idx : int) -> dict:
        return dict(
            node_offs=torch.LongTensor([0]),
            node_type=self.node_type,
            mesh_pos=self.mesh_pos,
            world_pos=self.world_pos,
            target_world_pos=self.target_world_pos,
            stress=self.stress,
            srcs=self.srcs,
            dsts=self.dsts,
            wsrcs=self.wsrcs,
            wdsts=self.wdsts
        )


def collate_fn(batch):
    node_offs = torch.LongTensor([
        0 if i == 0 else batch[i - 1]['node_type'].shape[0]
        for i in range(len(batch))
    ]).cumsum(dim=0)

    srcss = []
    dstss = []

    wsrcss = []
    wdstss = []

    for i in range(len(batch)):
        srcss.append(batch[i]['srcs'] + node_offs[i])
        dstss.append(batch[i]['dsts'] + node_offs[i])
        wsrcss.append(batch[i]['wsrcs'] + node_offs[i])
        wdstss.append(batch[i]['wdsts'] + node_offs[i])

    return dict(
        node_offs=node_offs,
        node_type=torch.cat([b['node_type'] for b in batch], dim=0),
        mesh_pos=torch.cat([b['mesh_pos'] for b in batch], dim=0),
        world_pos=torch.cat([b['world_pos'] for b in batch], dim=0),
        target_world_pos=torch.cat([b['target_world_pos'] for b in batch], dim=0),
        stress=torch.cat([b['stress'] for b in batch], dim=0),
        srcs=torch.cat(srcss, dim=0),
        dsts=torch.cat(dstss, dim=0),
        wsrcs=torch.cat(wsrcss, dim=0),
        wdsts=torch.cat(wdstss, dim=0)
    )


if __name__ == '__main__':
    import time

    NI = 30
    BS = 256
    dev = torch.device('cuda:0')
    net = DeformingPlateModel().to(dev)

    ds = DeformingPlateData('./data/deforming_plate_np/train/ex0.npz')

    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=True,
        batch_size=BS,
        num_workers=1,
        pin_memory=dev.type == 'cuda',
        pin_memory_device=str(dev),
        collate_fn=collate_fn)


    batch = next(iter(dl))

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            t0 = time.perf_counter()
            for _ in range(NI):
                net.loss(
                    batch['node_type'].to(dev),
                    batch['mesh_pos'].to(dev),
                    batch['world_pos'].to(dev),
                    batch['target_world_pos'].to(dev),
                    batch['srcs'].to(dev),
                    batch['dsts'].to(dev),
                    batch['wsrcs'].to(dev),
                    batch['wdsts'].to(dev),
                ) #.backward()
            t1 = time.perf_counter()

    print(f'Batch Size: {BS}')
    print(f'Num Iters: {NI}')
    print(f'Num Threads: {NI}')
    print(f'Elapsed time: {t1 - t0:.2f} seconds')
    print(f'Throughput: {NI * BS / (t1 - t0):.2f} samp/sec')

