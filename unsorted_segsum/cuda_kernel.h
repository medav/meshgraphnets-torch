#include <cuda_fp16.h>

__global__ void unsorted_segment_sum_fwd_cuda_fp32_kernel(
    float * data,
    int64_t * indices,
    float * output,
    int num_rows,
    int dim,
    int num_segments
);

#define THREADS_PER_BLOCK 32

__global__ void unsorted_segment_sum_fwd_cuda_fp32_kernel_v2(
    float * data,
    int64_t * indices,
    float * output,
    int num_rows,
    int dim,
    int num_segments
);

__global__ void unsorted_segment_sum_fwd_cuda_fp32_kernel_v3(
    float * data,
    int64_t * indices,
    float * output,
    int num_rows,
    int dim,
    int num_segments
);

__global__ void unsorted_segment_sum_fwd_cuda_half_kernel_v3(
    __half * data,
    int64_t * indices,
    __half * output,
    int num_rows,
    int dim,
    int num_segments
);

__global__ void batched_unsorted_segment_sum_fwd_cuda_fp32_kernel_v3(
    float * data,
    int64_t * indices,
    float * output,
    int num_rows,
    int dim,
    int num_segments
);

__global__ void unsorted_segment_sum_bwd_cuda_fp32_kernel(
    float * grad,
    int64_t * indices,
    float * output,
    int num_rows,
    int dim
);

__global__ void unsorted_segment_sum_bwd_cuda_half_kernel(
    __half * grad,
    int64_t * indices,
    __half * output,
    int num_rows,
    int dim
);

__global__ void batched_unsorted_segment_sum_bwd_cuda_fp32_kernel(
    float * grad,
    int64_t * indices,
    float * output,
    int num_rows,
    int dim,
    int num_segments
);
