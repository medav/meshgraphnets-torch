#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_fp16.h>

#define CLD(N, D) ((N + D - 1) / D)

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


#define THREADS_PER_BLOCK 128

template<typename scalar_t>
__device__ scalar_t atomicAddProxy(scalar_t * address, scalar_t val) {
    return atomicAdd(address, val);
}

template<>
__device__ c10::Half atomicAddProxy<c10::Half>(c10::Half * address, c10::Half val) {
    return atomicAdd(reinterpret_cast<half *>(address), static_cast<half>(val));
}

template<typename scalar_t>
__global__ void unsorted_segment_sum_fwd_cuda_kernel(
    scalar_t * data,
    int64_t * indices,
    scalar_t * output,
    int num_rows,
    int dim,
    int num_segments
) {
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    int ii = blockIdx.x;
    int segment = indices[ii];
    int offset = segment * dim;

    if (d >= dim) return;

    atomicAddProxy<scalar_t>(&output[offset + d], data[ii * dim + d]);
}

at::Tensor unsorted_segment_sum_fwd_cuda(
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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.type(), "unsorted_segment_sum_fwd_cuda_kernel", ([&] {
        unsorted_segment_sum_fwd_cuda_kernel<scalar_t><<<blocks, threads>>>(
            (scalar_t *)data.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            (scalar_t *)out.data_ptr<scalar_t>(),
            (int)R,
            (int)D,
            (int)num_segments
        );
    }));

    return out;
}

template<typename scalar_t>
__global__ void unsorted_segment_sum_bwd_cuda_kernel(
    scalar_t * grad,
    int64_t * indices,
    scalar_t * output,
    int num_rows,
    int dim
) {
    int d = threadIdx.x;
    int r = blockIdx.x;
    int64_t n = indices[r];

    output[r * dim + d] = grad[n * dim + d];
}


at::Tensor unsorted_segment_sum_bwd_cuda(
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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "unsorted_segment_sum_bwd_cuda_kernel", ([&] {
        unsorted_segment_sum_bwd_cuda_kernel<scalar_t><<<blocks, threads>>>(
            (scalar_t *)grad.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            (scalar_t *)out.data_ptr<scalar_t>(),
            (int)R,
            (int)D
        );
    }));

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unsorted_segment_sum_fwd", &unsorted_segment_sum_fwd_cuda, "Unsorted Segment Sum");
    m.def("unsorted_segment_sum_bwd", &unsorted_segment_sum_bwd_cuda, "Unsorted Segment Sum (Grad)");
}
