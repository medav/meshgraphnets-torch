from dataclasses import dataclass
import enum
from typing import Optional
import torch
import numpy as np
import unsorted_segsum


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
class EdgeSet:
    features : torch.Tensor
    senders : torch.Tensor
    receivers : torch.Tensor

@dataclass
class MultiGraph:
    node_features : torch.Tensor
    edge_sets : list[EdgeSet]


class InvertableNorm(torch.nn.Module):
    def __init__(
        self,
        shape : tuple[int],
        eps: float = 1e-8,
        max_accumulations : int = 10**6
    ) -> None:
        super().__init__()
        self.shape = shape
        self.register_buffer('eps', torch.Tensor([eps]))
        self.eps : torch.Tensor
        self.max_accumulations = max_accumulations

        self.register_buffer('running_sum', torch.zeros(shape))
        self.register_buffer('running_sum_sq', torch.zeros(shape))
        self.running_sum: torch.Tensor
        self.running_sum_sq: torch.Tensor

        self.register_buffer('num_accum', torch.tensor(0, dtype=torch.long))
        self.num_accum: torch.Tensor

        self.register_buffer('accum_count', torch.tensor(0, dtype=torch.long))
        self.accum_count: torch.Tensor

    @property
    def stats(self) -> torch.Tensor:
        num_accum = max(self.num_accum.item(), 1)

        mean = self.running_sum / num_accum
        std = torch.max(
            torch.sqrt(self.running_sum_sq / num_accum - mean**2),
            self.eps)

        return mean, std

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        assert x.shape[-len(self.shape):] == self.shape
        n_batch_dims = x.ndim - len(self.shape)
        batch_dims = tuple(i for i in range(n_batch_dims))

        if self.accum_count.item() < self.max_accumulations:
            self.running_sum += x.sum(dim=batch_dims)
            self.running_sum_sq += x.pow(2).sum(dim=batch_dims)
            self.num_accum += np.prod(list(x.shape[i] for i in batch_dims))
            self.accum_count += 1

        mean, std = self.stats
        return (x - mean) / std

    def inverse(self, x : torch.Tensor) -> torch.Tensor:
        mean, std = self.stats
        return x * std + mean

class Mlp(torch.nn.Module):
    def __init__(self, input_size : int, widths : list[int], layernorm=True):
        super().__init__()
        widths = [input_size] + widths
        self.model = torch.nn.Sequential(*([
            torch.nn.Linear(widths[i], widths[i+1])
            for i in range(len(widths)-1)
        ] + ([torch.nn.LayerNorm((widths[-1],))] if layernorm else [])))

    def forward(self, x): return self.model(x)

class GraphNetBlock(torch.nn.Module):
    def __init__(
        self,
        node_feature_dim : int,
        edge_feature_dim : int,
        num_edge_sets : int,
        mlp_widths : list[int]
    ):
        super().__init__()
        self.num_edge_sets = num_edge_sets
        self.node_mlp = Mlp(
            node_feature_dim + num_edge_sets * edge_feature_dim,
            mlp_widths,
            layernorm=True)

        self.edge_mlps = torch.nn.ModuleList([
            Mlp(
                2 * node_feature_dim + edge_feature_dim,
                mlp_widths,
                layernorm=True)
            for _ in range(num_edge_sets)
        ])


    def update_node_features(
        self,
        node_features : torch.Tensor,
        edge_sets : list[EdgeSet]
    ) -> torch.Tensor:
        num_nodes = node_features.shape[-2]
        features = [node_features]

        for edge_set in edge_sets:
            features.append(unsorted_segsum.unsorted_segment_sum(
                edge_set.features, edge_set.receivers, num_nodes))

        return self.node_mlp(torch.cat(features, dim=-1))

    def update_edge_features(
        self,
        i : int,
        node_features : torch.Tensor,
        edge_set : EdgeSet
    ) -> torch.Tensor:
        srcs = node_features[:, edge_set.senders, :]
        dsts = node_features[:, edge_set.receivers, :]
        edge_features = edge_set.features
        return self.edge_mlps[i](torch.cat([srcs, dsts, edge_features], dim=-1))

    def forward(self, graph : MultiGraph) -> MultiGraph:
        node_features = graph.node_features
        edge_sets = graph.edge_sets

        assert len(edge_sets) == self.num_edge_sets

        node_features = self.update_node_features(node_features, edge_sets)

        edge_sets = [
            EdgeSet(
                features=self.update_edge_features(i, node_features, edge_set),
                senders=edge_set.senders,
                receivers=edge_set.receivers
            )
            for i, edge_set in enumerate(edge_sets)
        ]

        return MultiGraph(node_features, edge_sets)


class GraphNetEncoder(torch.nn.Module):
    def __init__(
        self,
        node_input_dim : int,
        edge_input_dim : int,
        latent_size : int,
        num_edge_sets : int,
        num_layers : int
    ):
        super().__init__()
        mlp_widths = [latent_size] * num_layers + [latent_size]
        self.node_mlp = Mlp(node_input_dim, mlp_widths, layernorm=True)
        self.edge_mlps = torch.nn.ModuleList([
            Mlp(edge_input_dim, mlp_widths, layernorm=True)
            for _ in range(num_edge_sets)
        ])

    def forward(self, graph : MultiGraph) -> MultiGraph:
        return MultiGraph(
            node_features=self.node_mlp(graph.node_features),
            edge_sets=[
                EdgeSet(
                    features=self.edge_mlps[i](edge_set.features),
                    senders=edge_set.senders,
                    receivers=edge_set.receivers
                )
                for i, edge_set in enumerate(graph.edge_sets)
            ])


class GraphNetDecoder(torch.nn.Module):
    def __init__(
        self,
        latent_size : int,
        output_size : int,
        num_layers : int
    ):
        super().__init__()
        mlp_widths = [latent_size] * num_layers + [output_size]
        self.node_mlp = Mlp(latent_size, mlp_widths, layernorm=True)

    def forward(self, graph : MultiGraph) -> torch.Tensor:
        return self.node_mlp(graph.node_features)

class GraphNetModel(torch.nn.Module):
    def __init__(
        self,
        node_input_dim : int,
        edge_input_dim : int,
        output_dim : int,
        latent_size : int,
        num_edge_sets : int,
        num_layers : int,
        num_mp_steps : int
    ):
        super().__init__()
        self.encoder = GraphNetEncoder(node_input_dim, edge_input_dim, latent_size, num_edge_sets, num_layers)
        self.decoder = GraphNetDecoder(latent_size, output_dim, num_layers)

        mp_mlp_widths = [latent_size] * num_layers + [latent_size]
        self.blocks = torch.nn.ModuleList([
            GraphNetBlock(latent_size, latent_size, num_edge_sets, mp_mlp_widths)
            for _ in range(num_mp_steps)
        ])

    def forward(self, graph : MultiGraph) -> torch.Tensor:
        graph = self.encoder(graph)
        for block in self.blocks:
            graph = block(graph)
        return self.decoder(graph)


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
        self.graph_net = GraphNetModel(
            input_dim,
            3, # 2D rel pos. + length
            output_dim,
            latent_size,
            num_edge_sets,
            num_layers,
            num_mp_steps)

        self.out_norm = InvertableNorm((output_dim,))
        self.node_norm = InvertableNorm((input_dim,))
        self.edge_norm = InvertableNorm((2 + 1,)) # 2D coord + length

    def forward(
        self,
        node_type : torch.LongTensor,
        velocity : torch.Tensor,
        mesh_pos : torch.Tensor,
        srcs : torch.LongTensor,
        dests : torch.LongTensor,
        unnorm : bool = True
    ) -> torch.Tensor:
        """Predicts Delta V"""

        B = velocity.shape[0]

        node_type_oh = \
            torch.nn.functional.one_hot(node_type, num_classes=NodeType.SIZE) \
                .squeeze() \
                .expand((B, -1, -1))

        node_features = torch.cat([velocity, node_type_oh], dim=-1)
        rel_mesh_pos = mesh_pos[:, srcs, :] - mesh_pos[:, dests, :]

        edge_features = torch.cat([
            rel_mesh_pos,
            torch.norm(rel_mesh_pos, dim=-1, keepdim=True)
        ], dim=-1)

        graph = MultiGraph(
            node_features=self.node_norm(node_features),
            edge_sets=[ EdgeSet(self.edge_norm(edge_features), srcs, dests) ]
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
        dests : torch.LongTensor,
        target_velocity : torch.Tensor
    ) -> torch.Tensor:

        pred = self.forward(
            node_type,
            velocity,
            mesh_pos,
            srcs,
            dests,
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

