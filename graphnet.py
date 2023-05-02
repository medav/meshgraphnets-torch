from dataclasses import dataclass
import enum
from typing import Optional
import torch
import numpy as np
from kernels import unsorted_segsum as USS
from kernels import gather_concat as GC
from kernels import scatter_concat as SC

USE_FUSED_GATHER_CONCAT = False
USE_FUSED_SCATTER_CONCAT = False
USE_FUSED_LN = False
USE_FUSED_MLP = False

def make_torch_param(data): return torch.nn.Parameter(torch.tensor(data))
def make_torch_buffer(data): return torch.tensor(data)

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
class GraphNetSample:
    cells : torch.Tensor
    node_type : torch.Tensor
    mesh_pos : torch.Tensor

    def todev(self, dev):
        def _todev(x):
            return x.to(dev)

        fields = self.__dataclass_fields__.keys()
        for field in fields:
            setattr(self, field, _todev(getattr(self, field)))

        return self

    def asdtype(self, dtype):
        def _asdtype(x):
            if torch.is_floating_point(x): return x.to(dtype)
            else: return x

        fields = self.__dataclass_fields__.keys()
        for field in fields:
            setattr(self, field, _asdtype(getattr(self, field)))

        return self

@dataclass
class GraphNetSampleBatch(GraphNetSample):
    node_offs : torch.Tensor

def collate_common(batch : list[GraphNetSample], ty):
    custom_field_names = set(batch[0].__dataclass_fields__.keys()) - \
        set(GraphNetSample.__dataclass_fields__.keys())

    node_offs = torch.LongTensor([
        0 if i == 0 else batch[i - 1].node_type.size(0)
        for i in range(len(batch))
    ]).cumsum(dim=0)

    cells = torch.cat([
        b.cells + node_offs[i]
        for i, b in enumerate(batch)
    ], dim=0)

    custom_fields = {
        k: torch.cat([getattr(b, k) for b in batch], dim=0)
        for k in custom_field_names
    }

    return ty(
        node_offs=node_offs,
        cells=cells,
        node_type=torch.cat([b.node_type for b in batch], dim=0),
        mesh_pos=torch.cat([b.mesh_pos for b in batch], dim=0),
        **custom_fields
    )

def load_npz_common(path : str, type) -> "type":
    np_data = np.load(path)

    return type(**{
        k: torch.from_numpy(v)
        for k, v in np_data.items()
        if k in type.__dataclass_fields__.keys()
    })

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
        modules = []
        for i in range(len(widths) - 1):
            if i < len(widths) - 2:
                modules.append(torch.nn.Sequential(
                    torch.nn.Linear(widths[i], widths[i + 1]), torch.nn.ReLU()))
            else:
                modules.append(torch.nn.Sequential(
                    torch.nn.Linear(widths[i], widths[i + 1])))

        if layernorm: modules.append(torch.nn.LayerNorm(widths[-1]))
        self.model = torch.nn.Sequential(*modules)

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
        num_nodes = node_features.size(0)
        num_edge_sets = len(edge_sets)
        dim = node_features.size(1)

        if USE_FUSED_GATHER_CONCAT:
            return self.node_mlp(GC.fused_gather_concat(
                node_features,
                [es.features for es in edge_sets],
                [es.offsets for es in edge_sets]))
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
        if USE_FUSED_SCATTER_CONCAT:
            return self.edge_mlps[i](
                SC.fused_scatter_concat(
                    node_features,
                    edge_set.features,
                    edge_set.senders,
                    edge_set.receivers
                ))
        else:
            srcs = node_features[edge_set.senders, :]
            dsts = node_features[edge_set.receivers, :]
            edge_features = edge_set.features
            return self.edge_mlps[i](torch.cat([srcs, dsts, edge_features], dim=-1))

    def forward(self, graph : MultiGraph) -> MultiGraph:
        node_features = graph.node_features
        edge_sets = graph.edge_sets

        assert len(edge_sets) == self.num_edge_sets

        new_edge_sets = [
            EdgeSet(
                features=self.update_edge_features(i, node_features, edge_set),
                senders=edge_set.senders,
                receivers=edge_set.receivers,
                offsets=edge_set.offsets
            )
            for i, edge_set in enumerate(edge_sets)
        ]

        new_node_features = self.update_node_features(node_features, new_edge_sets)

        for ei in range(self.num_edge_sets):
            new_edge_sets[ei].features = \
                new_edge_sets[ei].features + edge_sets[ei].features

        return MultiGraph(new_node_features + node_features, new_edge_sets)


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
        self.node_mlp = Mlp(latent_size, mlp_widths, layernorm=False)

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

    def forward(self, graph : MultiGraph) -> torch.Tensor:
        graph = self.encoder(graph)
        num_nodes = graph.node_features.size(0)

        if USE_FUSED_GATHER_CONCAT:
            for edge_set in graph.edge_sets:
                edge_set.sort_(num_nodes)

        for block in self.blocks:
            graph = block(graph)

        return self.decoder(graph)

    def import_numpy_weights(
        self,
        weights : dict[str, np.ndarray],
        es_names : list[str]
    ):
        def hookup_mlp(mod, mlp_prefix, ln_prefix):
            layer_norm_off = 0 if ln_prefix is None else 1

            for l in range(len(mod.model) - layer_norm_off):
                w = make_torch_param(weights[f'{mlp_prefix}/linear_{l}/w:0'].transpose(-1, -2))
                assert tuple(w.shape) == tuple(mod.model[l][0].weight.shape)
                mod.model[l][0].weight = w

                b = make_torch_param(weights[f'{mlp_prefix}/linear_{l}/b:0'])
                assert tuple(b.shape) == tuple(mod.model[l][0].bias.shape)
                mod.model[l][0].bias = b

            if ln_prefix is not None:
                assert isinstance(mod.model[-1], torch.nn.modules.normalization.LayerNorm)
                gamma = make_torch_param(weights[f'{ln_prefix}/gamma:0'])
                assert tuple(gamma.shape) == tuple(mod.model[-1].weight.shape)
                mod.model[-1].weight = gamma

                beta = make_torch_param(weights[f'{ln_prefix}/beta:0'])
                assert tuple(beta.shape) == tuple(mod.model[-1].bias.shape)
                mod.model[-1].bias = beta

        # Encoder
        hookup_mlp(
            self.encoder.node_mlp,
            'EncodeProcessDecode/encoder/mlp',
            'EncodeProcessDecode/encoder/layer_norm')

        for e in range(len(self.encoder.edge_mlps)):
            hookup_mlp(
                self.encoder.edge_mlps[e],
                f'EncodeProcessDecode/encoder/mlp_{e + 1}',
                f'EncodeProcessDecode/encoder/layer_norm_{e + 1}')

        # Message Passing
        for i in range(len(self.blocks)):
            block_prefix = f'EncodeProcessDecode/GraphNetBlock' if i == 0 else \
                f'EncodeProcessDecode/GraphNetBlock_{i}'

            hookup_mlp(
                self.blocks[i].node_mlp,
                f'{block_prefix}/node_fn/mlp',
                f'{block_prefix}/node_fn/layer_norm')

            for e in range(len(self.blocks[i].edge_mlps)):
                hookup_mlp(
                    self.blocks[i].edge_mlps[e],
                    f'{block_prefix}/{es_names[e]}/mlp',
                    f'{block_prefix}/{es_names[e]}/layer_norm')

        # Decoder
        hookup_mlp(
            self.decoder.node_mlp,
            'EncodeProcessDecode/decoder/mlp',
            None)


