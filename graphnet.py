from dataclasses import dataclass
import enum
from typing import Optional
import torch
import numpy as np
import unsorted_segsum as USS
import gather_concat as GC


def cells_to_edges(cells : torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    cells: int32[M, D]
    :ret: int32[E], int32[E]
    """

    if cells.shape[1] == 3:
        # Triangles

        raw_edges = torch.cat([
            cells[:, 0:2],
            cells[:, 1:3],
            torch.stack([cells[:, 2], cells[:, 0]], dim=1)
        ], dim=0)

    elif cells.shape[1] == 4:
        # Tetrahedrons

        raw_edges = torch.cat([
            cells[:, 0:2],
            cells[:, 1:3],
            cells[:, 2:4],
            torch.stack([cells[:, 0], cells[:, 2]], dim=1),
            torch.stack([cells[:, 0], cells[:, 3]], dim=1),
            torch.stack([cells[:, 1], cells[:, 3]], dim=1)
        ], dim=0)

    else: raise NotImplementedError('Unknown cell type')

    srcs = raw_edges.max(dim=1).values
    dsts = raw_edges.min(dim=1).values

    edges = torch.stack([srcs, dsts], dim=1)
    unique_edges = edges.unique(dim=0, sorted=False)
    srcs, dsts = unique_edges[:, 0], unique_edges[:, 1]

    return torch.cat([srcs, dsts], dim=0), torch.cat([dsts, srcs], dim=0)


@dataclass
class EdgeSet:
    features : torch.Tensor
    senders : torch.Tensor
    receivers : torch.Tensor
    offsets : Optional[torch.Tensor] = None

    def sort_(self, num_nodes):
        idxs = torch.argsort(self.receivers)
        self.features = self.features[idxs]
        self.senders = self.senders[idxs]
        self.receivers = self.receivers[idxs]
        self.offsets = GC.compute_edge_offsets(self.receivers, num_nodes)

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
        edge_sets : list[EdgeSet],
        fast_mp : bool = False
    ) -> torch.Tensor:
        num_nodes = node_features.size(0)
        num_edge_sets = len(edge_sets)
        dim = node_features.size(1)

        if fast_mp:
            node_concat = torch.zeros(
                (num_nodes, (num_edge_sets + 1) * dim),
                device=node_features.device,
                dtype=node_features.dtype)

            GC.fused_gather_concat_out(
                node_features,
                [es.features for es in edge_sets],
                [es.offsets for es in edge_sets],
                node_concat)

            return self.node_mlp(node_concat)
        else:
            features = [node_features]

            for edge_set in edge_sets:
                features.append(USS.unsorted_segment_sum(
                    edge_set.features, edge_set.receivers, num_nodes))

            return self.node_mlp(torch.cat(features, dim=-1))

    def update_edge_features(
        self,
        i : int,
        node_features : torch.Tensor,
        edge_set : EdgeSet
    ) -> torch.Tensor:
        srcs = node_features[edge_set.senders, :]
        dsts = node_features[edge_set.receivers, :]
        edge_features = edge_set.features
        return self.edge_mlps[i](torch.cat([srcs, dsts, edge_features], dim=-1))

    def forward(self, graph : MultiGraph, fast_mp : bool = False) -> MultiGraph:
        node_features = graph.node_features
        edge_sets = graph.edge_sets

        assert len(edge_sets) == self.num_edge_sets

        node_features = self.update_node_features(
            node_features, edge_sets, fast_mp)

        edge_sets = [
            EdgeSet(
                features=self.update_edge_features(i, node_features, edge_set),
                senders=edge_set.senders,
                receivers=edge_set.receivers,
                offsets=edge_set.offsets
            )
            for i, edge_set in enumerate(edge_sets)
        ]

        return MultiGraph(node_features, edge_sets)


class GraphNetEncoder(torch.nn.Module):
    def __init__(
        self,
        node_input_dim : int,
        edge_input_dims : list[int],
        latent_size : int,
        num_edge_sets : int,
        num_layers : int
    ):
        super().__init__()
        mlp_widths = [latent_size] * num_layers + [latent_size]
        self.node_mlp = Mlp(node_input_dim, mlp_widths, layernorm=True)
        self.edge_mlps = torch.nn.ModuleList([
            Mlp(edge_input_dims[i], mlp_widths, layernorm=True)
            for i in range(num_edge_sets)
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
        edge_input_dims : list[int],
        output_dim : int,
        latent_size : int,
        num_edge_sets : int,
        num_layers : int,
        num_mp_steps : int
    ):
        super().__init__()
        self.encoder = GraphNetEncoder(node_input_dim, edge_input_dims, latent_size, num_edge_sets, num_layers)
        self.decoder = GraphNetDecoder(latent_size, output_dim, num_layers)

        mp_mlp_widths = [latent_size] * num_layers + [latent_size]
        self.blocks = torch.nn.ModuleList([
            GraphNetBlock(latent_size, latent_size, num_edge_sets, mp_mlp_widths)
            for _ in range(num_mp_steps)
        ])

    def forward(self, graph : MultiGraph, fast_mp : bool = True) -> torch.Tensor:
        graph = self.encoder(graph)
        num_nodes = graph.node_features.size(0)

        if fast_mp:
            for edge_set in graph.edge_sets:
                edge_set.sort_(num_nodes)

        for block in self.blocks:
                graph = block(graph, fast_mp)

        return self.decoder(graph)

