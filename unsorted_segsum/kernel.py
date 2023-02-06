
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math

cur_path = os.path.dirname(os.path.realpath(__file__))
cpu_unsorted_segsum = load('cpu_unsorted_segsum',
    [f'{cur_path}/cpu_extension.cc', f'{cur_path}/cpu.cc'],
    extra_cflags=['-fopenmp', '-O3', '-march=native'],
    extra_ldflags=['-lgomp', '-O3', '-march=native'],
    verbose=True)

import cpu_unsorted_segsum

if torch.cuda.is_available():
    cuda_unsorted_segsum = load('cuda_unsorted_segsum',
        [f'{cur_path}/cuda_extension.cc', f'{cur_path}/cuda.cu', f'{cur_path}/cuda_kernel.cu'],
        extra_cflags=['-fopenmp', '-O3', '-march=native'],
        extra_ldflags=['-lgomp', '-O3', '-march=native'],
        verbose=True)

    import cuda_unsorted_segsum

else:
    cuda_unsorted_segsum = None
    print('CUDA not available, cuda_unsorted_segsum will not be available')

def unsorted_segment_sum_ref(
    data : torch.Tensor,
    indices : torch.Tensor,
    num_segments : int
) -> torch.Tensor:
    return torch.cat([
        data[indices == i].sum(dim=0, keepdim=True)
        for i in range(num_segments)
    ], dim=0)

class UnsortedSegmentSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data : torch.Tensor, indices  : torch.Tensor, num_segments : int) -> torch.Tensor:
        ctx.save_for_backward(indices)

        M = cuda_unsorted_segsum if data.device.type == 'cuda' else cpu_unsorted_segsum

        assert M is not None, f'No backend for {data.device}'

        if len(data.shape) == 2:
            return M.unsorted_segment_sum_fwd_fp32(data, indices, num_segments)
        elif len(data.shape) == 3:
            return M.batched_unsorted_segment_sum_fwd_fp32(data, indices, num_segments)
        else:
            raise NotImplementedError()

    @staticmethod
    def backward(ctx, grad):
        indices, = ctx.saved_tensors

        M = cuda_unsorted_segsum if grad.device.type == 'cuda' else cpu_unsorted_segsum

        assert M is not None, f'No backend for {grad.device}'

        if len(grad.shape) == 2:
            return M.unsorted_segment_sum_bwd_fp32(grad, indices), None, None
        elif len(grad.shape) == 3:
            return M.batched_unsorted_segment_sum_bwd_fp32(grad, indices), None, None
        else:
            raise NotImplementedError()

def unsorted_segment_sum(
    data : torch.Tensor,
    indices  : torch.Tensor,
    num_segments : int
) -> torch.Tensor:
    return UnsortedSegmentSum.apply(data, indices, num_segments)


def unit_test_cpu():
    print('==== Correctness Test CPU ====')
    data = torch.randn(1000, 3, requires_grad=False)
    indices = torch.randint(0, 100, (1000,), requires_grad=False)
    num_segments = 100

    d1 = data.clone().requires_grad_()
    d2 = data.clone().requires_grad_()

    ref = unsorted_segment_sum_ref(d1, indices, num_segments)
    out = UnsortedSegmentSum.apply(d2, indices, num_segments)

    print('(FWD) L2 = ', (ref - out).pow(2).sum().sqrt())

    ref.pow(2).sum().backward()
    out.pow(2).sum().backward()

    print('(BWD) L2 = ', (d1.grad - d2.grad).pow(2).sum().sqrt())

def unit_test_gpu():
    print('==== Correctness Test GPU ====')
    data = torch.randn(1000, 3, requires_grad=False)
    indices = torch.randint(0, 100, (1000,), requires_grad=False)
    num_segments = 100

    d1 = data.clone().requires_grad_()
    d2 = data.clone().cuda().requires_grad_()

    ref = UnsortedSegmentSum.apply(d1, indices, num_segments)
    out = UnsortedSegmentSum.apply(d2, indices.clone().cuda(), num_segments)

    print('(FWD) L2 = ', (ref - out.cpu()).pow(2).sum().sqrt())

    ref.pow(2).sum().backward()
    out.pow(2).sum().backward()

    print('(BWD) L2 = ', (d1.grad - d2.grad.cpu()).pow(2).sum().sqrt())

def unit_test_batched():
    print('==== Correctness Test CPU Batched ====')
    data = torch.randn(16, 1000, 3, requires_grad=False)
    indices = torch.randint(0, 100, (1000,), requires_grad=False)
    num_segments = 100

    d1 = data.clone().requires_grad_()
    d2 = data.clone().requires_grad_()


    ref = torch.stack([
        UnsortedSegmentSum.apply(d1[i, ...], indices, num_segments)
        for i in range(16)
    ])

    out = UnsortedSegmentSum.apply(d2, indices, num_segments)

    print('(FWD) L2 = ', (ref - out).pow(2).sum().sqrt())

    ref.pow(2).sum().backward()
    out.pow(2).sum().backward()

    print('(BWD) L2 = ', (d1.grad - d2.grad).pow(2).sum().sqrt())

def unit_test_batched_gpu():
    print('==== Correctness Test GPU Batched ====')
    data = torch.randn(16, 1000, 3, requires_grad=False)
    indices = torch.randint(0, 100, (1000,), requires_grad=False).cuda()
    num_segments = 100

    d1 = data.clone().cuda().requires_grad_()
    d2 = data.clone().cuda().requires_grad_()


    ref = torch.stack([
        UnsortedSegmentSum.apply(d1[i, ...], indices, num_segments)
        for i in range(16)
    ])

    out = UnsortedSegmentSum.apply(d2, indices, num_segments)

    print('(FWD) L2 = ', (ref - out).pow(2).sum().sqrt())

    ref.pow(2).sum().backward()
    out.pow(2).sum().backward()

    print('(BWD) L2 = ', (d1.grad - d2.grad).pow(2).sum().sqrt())

if __name__ == '__main__':
    unit_test_cpu()
    unit_test_batched()
    unit_test_gpu()
    unit_test_batched_gpu()

    exit(0)

    # Benchmark

    t0 = time.perf_counter()
    for _ in range(1000):
        _ = unsorted_segment_sum_ref(data, indices, num_segments)
    t1 = time.perf_counter()
    print(f'Reference (Fwd): {(t1 - t0) * 1000:.2f} ms')

    t0 = time.perf_counter()
    for _ in range(1000):
        _ = UnsortedSegmentSum.apply(data, indices, num_segments)
    t1 = time.perf_counter()
    print(f'Extension (Fwd): {(t1 - t0) * 1000:.2f} ms')

    t0 = time.perf_counter()
    for _ in range(1000):
        out = unsorted_segment_sum_ref(d1, indices, num_segments)
        out.pow(2).sum().backward()
    t1 = time.perf_counter()
    print(f'Reference (Fwd + Bwd): {(t1 - t0) * 1000:.2f} ms')

    t0 = time.perf_counter()
    for _ in range(1000):
        out = UnsortedSegmentSum.apply(d2, indices, num_segments)
        out.pow(2).sum().backward()
    t1 = time.perf_counter()
    print(f'Extension (Fwd + Bwd): {(t1 - t0) * 1000:.2f} ms')



