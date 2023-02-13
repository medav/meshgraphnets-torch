#include <torch/extension.h>
#include <ATen/ATen.h>

at::Tensor unsorted_segment_sum_fwd_cpu_fp32(
    at::Tensor data,
    at::Tensor indices,
    int num_segments
);

at::Tensor unsorted_segment_sum_bwd_cpu_fp32(
    at::Tensor grad,
    at::Tensor indices
);

at::Tensor unsorted_segment_sum_fwd_cuda_fp32(
    at::Tensor data,
    at::Tensor indices,
    int num_segments
);

at::Tensor unsorted_segment_sum_fwd_cuda_fp32_v2(
    at::Tensor data,
    at::Tensor indices,
    int num_segments
);

at::Tensor unsorted_segment_sum_fwd_cuda_fp32_v3(
    at::Tensor data,
    at::Tensor indices,
    int num_segments
);

at::Tensor unsorted_segment_sum_fwd_cuda_half_v3(
    at::Tensor data,
    at::Tensor indices,
    int num_segments
);

at::Tensor unsorted_segment_sum_bwd_cuda_fp32(
    at::Tensor grad,
    at::Tensor indices
);

at::Tensor unsorted_segment_sum_bwd_cuda_half(
    at::Tensor grad,
    at::Tensor indices
);

// == Batched ==

at::Tensor batched_unsorted_segment_sum_fwd_cpu_fp32(
    at::Tensor data,
    at::Tensor indices,
    int num_segments
);

at::Tensor batched_unsorted_segment_sum_bwd_cpu_fp32(
    at::Tensor grad,
    at::Tensor indices
);

at::Tensor batched_unsorted_segment_sum_fwd_cuda_fp32(
    at::Tensor data,
    at::Tensor indices,
    int num_segments
);

at::Tensor batched_unsorted_segment_sum_fwd_cuda_fp32_v2(
    at::Tensor data,
    at::Tensor indices,
    int num_segments
);

at::Tensor batched_unsorted_segment_sum_fwd_cuda_fp32_v3(
    at::Tensor data,
    at::Tensor indices,
    int num_segments
);

at::Tensor batched_unsorted_segment_sum_bwd_cuda_fp32(
    at::Tensor grad,
    at::Tensor indices
);

