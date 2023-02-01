# The following is based on:
# https://github.com/deepmind/deepmind-research/compare/master...isabellahuang:deepmind-research-tf1:tf1
#
# Subject to Apache v2 license (see original source for details)

import torch
import enum
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
    thresh : float = 0.5
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

    srcs = actuator_idx[rel_close_pair_idx[:,0]]
    dsts = deformable_idx[rel_close_pair_idx[:,1]]

    return torch.stack([srcs, dsts], dim=0), torch.stack([dsts, srcs], dim=0)



class DeformingPlateModel(torch.nn.Module):
    def __init__(
        self,
        input_dim : int = 3 + NodeType.SIZE, # vx, vy, vz, one_hot(type)
        output_dim : int = 3, # vx, vy, vz
        latent_size : int = 128,
        num_layers : int = 2,
        num_mp_steps : int = 15
    ):
        super().__init__()
        self.num_edge_sets = 2
        self.graph_net = GNN.GraphNetModel(
            input_dim,
            3, # 2D rel pos. + length
            output_dim,
            latent_size,
            self.num_edge_sets,
            num_layers,
            num_mp_steps)

        self.out_norm = GNN.InvertableNorm((output_dim,))
        self.node_norm = GNN.InvertableNorm((input_dim,))

        n_mesh_edge_f = 2 * (3 + 1) # 2x (3D coord + length)
        self.mesh_edge_norm = GNN.InvertableNorm((n_mesh_edge_f,))

        n_world_edge_f = 3 + 1 # 3D coord + length
        self.world_edge_norm = GNN.InvertableNorm((n_world_edge_f,))

    def forward(
        self,
        node_type : torch.LongTensor,
        velocity : torch.Tensor,
        mesh_pos : torch.Tensor,
        world_pos : torch.Tensor,
        srcs : torch.LongTensor,
        dsts : torch.LongTensor,
        wsrcs : torch.LongTensor,
        wdsts : torch.LongTensor,
        unnorm : bool = True
    ) -> torch.Tensor:
        """Predicts Delta V"""

        B = velocity.shape[0]

        #
        # Node Features
        #

        node_type_oh = \
            torch.nn.functional.one_hot(node_type, num_classes=NodeType.SIZE) \
                .squeeze() \
                .expand((B, -1, -1))

        node_features = torch.cat([velocity, node_type_oh], dim=-1)

        #
        # Mesh Edge Features
        #

        rel_mesh_pos = mesh_pos[:, srcs, :] - mesh_pos[:, dsts, :]
        rel_world_mesh_pos = world_pos[:, srcs, :] - mesh_pos[:, dsts, :]

        mesh_edge_features = torch.cat([
            rel_mesh_pos,
            torch.norm(rel_mesh_pos, dim=-1, keepdim=True),
            rel_world_mesh_pos,
            torch.norm(rel_world_mesh_pos, dim=-1, keepdim=True)
        ], dim=-1)


        #
        # World Edge Features
        #

        rel_world_pos = world_pos[:, wsrcs, :] - world_pos[:, wdsts, :]

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
        velocity : torch.Tensor,
        mesh_pos : torch.Tensor,
        world_pos : torch.Tensor,
        srcs : torch.LongTensor,
        dsts : torch.LongTensor,
        wsrcs : torch.LongTensor,
        wdsts : torch.LongTensor,
        target_world_pos : torch.Tensor
    ) -> torch.Tensor:

        pred = self.forward(
            node_type,
            velocity,
            mesh_pos,
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
        return residuals[:, mask].pow(2).mean()
