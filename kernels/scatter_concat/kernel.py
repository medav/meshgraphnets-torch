
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
    scatter_concat_cuda = load('scatter_concat_cuda',
        [f'{cur_path}/scatter_concat.cu'],
        extra_include_paths=[f'{cutlass_path}/include', f'{cutlass_path}/tools/util/include'],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17', '-arch=sm_80'],
        extra_ldflags=['-O3', f'-L{cutlass_path}/build/tools/library', '-lcutlass'],
        verbose=False)

    import scatter_concat_cuda

else:
    scatter_concat_cuda = None
    print('CUDA not available, scatter_concat_cuda will not be available')

class FusedScatterConcatOut(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_features : torch.Tensor,
        edge_features : torch.Tensor,
        srcs : torch.Tensor,
        dsts : torch.Tensor,
        out : torch.Tensor
    ):
        scatter_concat_cuda.fused_scatter_concat_out(
            node_features, edge_features, srcs, dsts, out)

    @staticmethod
    def backward(ctx, grad): raise NotImplementedError()


def fused_scatter_concat_out(
        node_features : torch.Tensor,
        edge_features : torch.Tensor,
        srcs : torch.Tensor,
        dsts : torch.Tensor,
        out : torch.Tensor
):
    FusedScatterConcatOut.apply(
        node_features, edge_features, srcs, dsts, out)

def fused_scatter_concat(
        node_features : torch.Tensor,
        edge_features : torch.Tensor,
        srcs : torch.Tensor,
        dsts : torch.Tensor
):
    out = torch.zeros((
        srcs.size(0),
        node_features.size(1) * 2 + edge_features.size(1)
    ), dtype=node_features.dtype, device=node_features.device)

    FusedScatterConcatOut.apply(
        node_features, edge_features, srcs, dsts, out)

    return out

def test_1():
    dev = torch.device('cuda:0')

    nf = torch.Tensor([
        [1, 2],
        [3, 4],
        [5, 6],
    ]).half().to(dev)

    ef = torch.Tensor([
        [1, 1],
        [2, 2],
        [3, 3],
    ]).half().to(dev)

    srcs = torch.LongTensor([0, 1, 2]).to(dev)
    dsts = torch.LongTensor([2, 1, 0]).to(dev)

    out = fused_scatter_concat(nf, ef, srcs, dsts)

    print(out)


if __name__ == '__main__':
    test_1()


