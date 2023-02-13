#include <torch/extension.h>
#include <ATen/ATen.h>
#include "api.hh"
#include "cuda_kernel.h"

#define CLD(N, D) ((N + D - 1) / D)

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor unsorted_segment_sum_fwd_cuda_fp32(
    at::Tensor data,
    at::Tensor indices,
    int num_segments
) {
    CHECK_INPUT(data);
    CHECK_INPUT(indices);

    const ssize_t R = data.size(0);
    const ssize_t D = data.size(1);
    at::Tensor out = at::zeros({num_segments, D}, data.options());

    dim3 blocks(num_segments);
    dim3 threads(D);

    unsorted_segment_sum_fwd_cuda_fp32_kernel<<<blocks, threads>>>(
        data.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        (int)R,
        (int)D,
        (int)num_segments
    );

    return out;
}

at::Tensor unsorted_segment_sum_fwd_cuda_fp32_v2(
    at::Tensor data,
    at::Tensor indices,
    int num_segments
) {
    CHECK_INPUT(data);
    CHECK_INPUT(indices);

    const ssize_t R = data.size(0);
    const ssize_t D = data.size(1);
    at::Tensor out = at::zeros({num_segments, data.size(1)}, data.options());

    const ssize_t tblocks = CLD(D, THREADS_PER_BLOCK);

    dim3 blocks(num_segments, tblocks);
    dim3 threads(THREADS_PER_BLOCK);

    unsorted_segment_sum_fwd_cuda_fp32_kernel_v2<<<blocks, threads>>>(
        data.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        (int)R,
        (int)D,
        (int)num_segments
    );

    return out;
}

at::Tensor unsorted_segment_sum_fwd_cuda_fp32_v3(
    at::Tensor data, // [num_rows, dim]
    at::Tensor indices, // [num_rows]
    int num_segments
) {
    CHECK_INPUT(data);
    CHECK_INPUT(indices);

    const ssize_t R = data.size(0);
    const ssize_t D = data.size(1);
    at::Tensor out = at::zeros({num_segments, D}, data.options());

    const ssize_t tblocks = CLD(D, THREADS_PER_BLOCK);

    dim3 blocks(R, tblocks);
    dim3 threads(THREADS_PER_BLOCK);

    unsorted_segment_sum_fwd_cuda_fp32_kernel_v3<<<blocks, threads>>>(
        data.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        (int)R,
        (int)D,
        (int)num_segments
    );

    return out;
}

at::Tensor unsorted_segment_sum_fwd_cuda_half_v3(
    at::Tensor data, // [num_rows, dim]
    at::Tensor indices, // [num_rows]
    int num_segments
) {
    CHECK_INPUT(data);
    CHECK_INPUT(indices);

    const ssize_t R = data.size(0);
    const ssize_t D = data.size(1);
    at::Tensor out = at::zeros({num_segments, D}, data.options());

    const ssize_t tblocks = CLD(D, THREADS_PER_BLOCK);

    dim3 blocks(R, tblocks);
    dim3 threads(THREADS_PER_BLOCK);

    unsorted_segment_sum_fwd_cuda_half_kernel_v3<<<blocks, threads>>>(
        (__half *)data.data_ptr<at::Half>(),
        indices.data_ptr<int64_t>(),
        (__half *)out.data_ptr<at::Half>(),
        (int)R,
        (int)D,
        (int)num_segments
    );

    return out;
}

at::Tensor batched_unsorted_segment_sum_fwd_cuda_fp32_v3(
    at::Tensor data, // [batch, num_rows, dim]
    at::Tensor indices, // [num_rows]
    int num_segments
) {
    CHECK_INPUT(data);
    CHECK_INPUT(indices);

    const ssize_t N = num_segments;
    const ssize_t B = data.size(0);
    const ssize_t R = data.size(1);
    const ssize_t D = data.size(2);

    at::Tensor out = at::zeros({B, N, D}, data.options());

    const ssize_t tblocks = CLD(D, THREADS_PER_BLOCK);

    dim3 blocks(B, R, tblocks);
    dim3 threads(THREADS_PER_BLOCK);

    batched_unsorted_segment_sum_fwd_cuda_fp32_kernel_v3<<<blocks, threads>>>(
        data.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        (int)R,
        (int)D,
        (int)num_segments
    );

    return out;
}

at::Tensor unsorted_segment_sum_bwd_cuda_fp32(
    at::Tensor grad,
    at::Tensor indices
) {
    CHECK_INPUT(grad);
    CHECK_INPUT(indices);

    const ssize_t R = indices.size(0);
    const ssize_t D = grad.size(1);
    at::Tensor out = at::zeros({R, D}, grad.options());

    dim3 blocks(R);
    dim3 threads(D);

    unsorted_segment_sum_bwd_cuda_fp32_kernel<<<blocks, threads>>>(
        grad.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        (int)R,
        (int)D
    );

    return out;
}

at::Tensor unsorted_segment_sum_bwd_cuda_half(
    at::Tensor grad,
    at::Tensor indices
) {
    CHECK_INPUT(grad);
    CHECK_INPUT(indices);

    const ssize_t R = indices.size(0);
    const ssize_t D = grad.size(1);
    at::Tensor out = at::zeros({R, D}, grad.options());

    dim3 blocks(R);
    dim3 threads(D);

    unsorted_segment_sum_bwd_cuda_half_kernel<<<blocks, threads>>>(
        (__half *)grad.data_ptr<at::Half>(),
        indices.data_ptr<int64_t>(),
        (__half *)out.data_ptr<at::Half>(),
        (int)R,
        (int)D
    );

    return out;
}

at::Tensor batched_unsorted_segment_sum_bwd_cuda_fp32(
    at::Tensor grad, // [batch, num_segments, dim]
    at::Tensor indices // [num_rows]
) {
    CHECK_INPUT(grad);
    CHECK_INPUT(indices);

    const ssize_t N = grad.size(1);
    const ssize_t B = grad.size(0);
    const ssize_t R = indices.size(0);
    const ssize_t D = grad.size(2);
    at::Tensor out = at::zeros({B, R, D}, grad.options());

    dim3 blocks(B, R);
    dim3 threads(D);

    batched_unsorted_segment_sum_bwd_cuda_fp32_kernel<<<blocks, threads>>>(
        grad.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        (int)R,
        (int)D,
        (int)N
    );

    return out;
}
