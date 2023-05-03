
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math

cur_path = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available():
    gather_concat_cuda = load('gather_concat_cuda',
        [f'{cur_path}/gather_concat_cuda.cu'],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17'],
        extra_ldflags=['-O3'],
        verbose=False)

    import gather_concat_cuda

else:
    gather_concat_cuda = None
    print('CUDA not available, gather_concat_cuda will not be available')


class ComputeEdgeOffsets(torch.autograd.Function):
    @staticmethod
    def forward(ctx, receivers : torch.Tensor, num_nodes : int) -> torch.Tensor:
        return gather_concat_cuda.compute_edge_offsets(receivers, num_nodes)

    @staticmethod
    def backward(ctx, grad): raise NotImplementedError()


def compute_edge_offsets(receivers : torch.Tensor, num_nodes : int) -> torch.Tensor:
    return ComputeEdgeOffsets.apply(receivers, num_nodes)


class FusedGatherConcat(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_features : torch.Tensor,
        edge_features : list[torch.Tensor],
        edge_offsets : list[torch.Tensor]
    ):
        if len(edge_features) == 1:
            return gather_concat_cuda.fused_gather_concat_1e(
                node_features, edge_features[0], edge_offsets[0]
            )
        elif len(edge_features) == 2:
            return gather_concat_cuda.fused_gather_concat_2e(
                node_features,
                edge_features[0], edge_offsets[0],
                edge_features[1], edge_offsets[1]
            )
        else: raise NotImplementedError()


    @staticmethod
    def backward(ctx, grad): raise NotImplementedError()


def fused_gather_concat(
        node_features : torch.Tensor,
        edge_features : list[torch.Tensor],
        edge_offsets : list[torch.Tensor]
) -> torch.Tensor:
    return FusedGatherConcat.apply(
        node_features, edge_features, edge_offsets)


def test_compute_edge_offsets_1():
    num_nodes = 5
    receivers = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4, 4, 4]).cuda()
    expected = torch.tensor([2, 4, 6, 7, 10]).cuda()
    actual = compute_edge_offsets(receivers, num_nodes)
    assert torch.all(torch.eq(expected, actual))


def test_fused_gather_concat_1e_1():
    num_nodes = 4
    receivers = torch.tensor([0, 0, 1, 2, 2, 3]).cuda()
    edge_offs = compute_edge_offsets(receivers, num_nodes)

    node_features = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ]).half().cuda()

    edge_features = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8]
    ]).half().cuda()

    expected = torch.tensor([
        [0, 1, 2, 3, 1, 3, 5, 7],
        [1, 2, 3, 4, 2, 3, 4, 5],
        [2, 3, 4, 5, 7, 9, 11, 13],
        [3, 4, 5, 6, 5, 6, 7, 8]
    ]).half().cuda()

    print(edge_offs)

    actual = gather_concat_cuda.fused_gather_concat_1e(
        node_features, edge_features, edge_offs)

    print(expected)
    print(actual)

    assert torch.all(torch.eq(expected, actual))


if __name__ == '__main__':
    test_fused_gather_concat_1e_1()


