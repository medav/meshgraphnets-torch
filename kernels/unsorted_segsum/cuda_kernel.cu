#include "cuda_kernel.h"
#include <cuda_fp16.h>

__global__ void unsorted_segment_sum_fwd_cuda_fp32_kernel(
    float * data,
    int64_t * indices,
    float * output,
    int num_rows,
    int dim,
    int num_segments
) {
    int d = threadIdx.x;
    int segment = blockIdx.x;
    int offset = segment * dim;

    for (int ii = 0; ii < num_rows; ii++) {
        if (indices[ii] == segment) {
            output[offset + d] += data[ii * dim + d];
        }
    }
}

__global__ void unsorted_segment_sum_fwd_cuda_fp32_kernel_v2(
    float * data,
    int64_t * indices,
    float * output,
    int num_rows,
    int dim,
    int num_segments
) {
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    int segment = blockIdx.x;
    int offset = segment * dim;

    if (d >= dim) return;

    float accum = 0.0f;

    for (int ii = 0; ii < num_rows; ii++) {
        if (indices[ii] == segment) {
            accum += data[ii * dim + d];
        }
    }

    output[offset + d] = accum;
}

__global__ void unsorted_segment_sum_fwd_cuda_fp32_kernel_v3(
    float * data,
    int64_t * indices,
    float * output,
    int num_rows,
    int dim,
    int num_segments
) {
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    int ii = blockIdx.x;
    int segment = indices[ii];
    int offset = segment * dim;

    if (d >= dim) return;

    atomicAdd(&output[offset + d], data[ii * dim + d]);
}

__global__ void unsorted_segment_sum_fwd_cuda_half_kernel_v3(
    __half * data,
    int64_t * indices,
    __half * output,
    int num_rows,
    int dim,
    int num_segments
) {
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    int ii = blockIdx.x;
    int segment = indices[ii];
    int offset = segment * dim;

    if (d >= dim) return;

    atomicAdd(&output[offset + d], data[ii * dim + d]);
}



__global__ void batched_unsorted_segment_sum_fwd_cuda_fp32_kernel_v3(
    float * data,
    int64_t * indices,
    float * output,
    int num_rows,
    int dim,
    int num_segments
) {
    int d = blockIdx.z * blockDim.x + threadIdx.x;
    int b = blockIdx.x;
    int r = blockIdx.y;
    int n = indices[r];
    int o_offset = b * num_segments * dim + n * dim;
    int i_offset = b * num_rows * dim + r * dim;

    if (d >= dim) return;

    atomicAdd(&output[o_offset + d], data[i_offset + d]);
}



__global__ void unsorted_segment_sum_bwd_cuda_fp32_kernel(
    float * grad,
    int64_t * indices,
    float * output,
    int num_rows,
    int dim
) {
    int d = threadIdx.x;
    int r = blockIdx.x;
    int64_t n = indices[r];

    output[r * dim + d] = grad[n * dim + d];
}

__global__ void unsorted_segment_sum_bwd_cuda_half_kernel(
    __half * grad,
    int64_t * indices,
    __half * output,
    int num_rows,
    int dim
) {
    int d = threadIdx.x;
    int r = blockIdx.x;
    int64_t n = indices[r];

    output[r * dim + d] = grad[n * dim + d];
}


__global__ void batched_unsorted_segment_sum_bwd_cuda_fp32_kernel(
    float * grad,
    int64_t * indices,
    float * output,
    int num_rows,
    int dim,
    int num_segments
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    int r = blockIdx.y;
    int64_t n = indices[r];

    int o_offset = b * num_rows * dim     + r * dim;
    int i_offset = b * num_segments * dim + n * dim;

    output[o_offset + d] = grad[i_offset + d];
}
