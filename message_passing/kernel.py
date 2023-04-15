
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math

cur_path = os.path.dirname(os.path.realpath(__file__))
cutlass_path = '/nobackup/medavies/cutlass'

if torch.cuda.is_available():
    mgn_mp_cuda = load('mgn_mp_cuda',
        [f'{cur_path}/message_passing.cu'],
        extra_include_paths=[f'{cutlass_path}/include', f'{cutlass_path}/tools/util/include'],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17', '-arch=sm_80'],
        extra_ldflags=['-O3', f'-L{cutlass_path}/build/tools/library', '-lcutlass'],
        verbose=True)

    import mgn_mp_cuda

else:
    mgn_mp_cuda = None
    print('CUDA not available, mgn_mp_cuda will not be available')

class MessagePassing(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        nf : torch.Tensor,
        efs : list[torch.Tensor],
        srcs : list[torch.Tensor],
        dsts : list[torch.Tensor],
        nw : torch.Tensor, nb : torch.Tensor,
        ews : list[torch.Tensor], ebs : list[torch.Tensor]
    ) -> torch.Tensor:
        return None


    @staticmethod
    def backward(ctx, grad): raise NotImplementedError()


def message_passing(

) -> torch.Tensor:
    raise NotImplementedError()


class ComputeEdgeOffsets(torch.autograd.Function):
    @staticmethod
    def forward(ctx, receivers : torch.Tensor, num_nodes : int) -> torch.Tensor:
        return mgn_mp_cuda.compute_edge_offsets(receivers, num_nodes)

    @staticmethod
    def backward(ctx, grad): raise NotImplementedError()


def compute_edge_offsets(receivers : torch.Tensor, num_nodes : int) -> torch.Tensor:
    return ComputeEdgeOffsets.apply(receivers, num_nodes)


class FusedGatherConcatOut(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_features : torch.Tensor,
        edge_features : list[torch.Tensor],
        edge_offsets : list[torch.Tensor],
        out : torch.Tensor
    ):
        if len(edge_features) == 2:
            mgn_mp_cuda.fused_gather_concat_2e_out(
                node_features,
                edge_features[0], edge_offsets[0],
                edge_features[1], edge_offsets[1],
                out
            )
        else: raise NotImplementedError()

    @staticmethod
    def backward(ctx, grad): raise NotImplementedError()


def fused_gather_concat_out(
        node_features : torch.Tensor,
        edge_features : list[torch.Tensor],
        edge_offsets : list[torch.Tensor],
        out : torch.Tensor
):
    return FusedGatherConcatOut.apply(
        node_features, edge_features, edge_offsets, out)


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

    actual = mgn_mp_cuda.fused_gather_concat_1e(
        node_features, edge_features, edge_offs)

    print(expected)
    print(actual)

    assert torch.all(torch.eq(expected, actual))


if __name__ == '__main__':
    test_fused_gather_concat_1e_1()


