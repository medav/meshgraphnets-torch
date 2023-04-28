

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CLD(N, D) ((N + D - 1) / D)

__global__ void device_fused_scatter_concat(
    half * ef, // [NE, D]
    half * nf, // [NN, D]
    half * out, // [NE, 3*D]
    int64_t * srcs, // [NE]
    int64_t * dsts, // [NE]
    const int64_t NN,
    const int64_t NE,
    const int64_t D
) {
    const int edge_i = blockIdx.x;
    const int edge_d = threadIdx.x;

    if (edge_d < D) {
        out[edge_i * 3 * D + edge_d] = ef[edge_i * D + edge_d];
    } else if (edge_d < 2 * D) {
        out[edge_i * 3 * D + edge_d] = nf[srcs[edge_i] * D + edge_d - D];
    } else {
        out[edge_i * 3 * D + edge_d] = nf[dsts[edge_i] * D + edge_d - D - D];
    }

}

void fused_scatter_concat_out(
    at::Tensor ef,
    at::Tensor nf,
    at::Tensor srcs,
    at::Tensor dsts,
    at::Tensor out
) {
    CHECK_INPUT(ef);
    CHECK_INPUT(nf);
    CHECK_INPUT(out);
    CHECK_INPUT(srcs);
    CHECK_INPUT(dsts);

    const int64_t D = ef.size(1);
    const int64_t NN = nf.size(0);
    const int64_t NE = ef.size(0);

    assert(D <= 128);
    assert(ef.size(1) == D);

    device_fused_scatter_concat<<<NE, 3*D>>>(
        (half *)ef.data_ptr<at::Half>(),
        (half *)nf.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>(),
        srcs.data_ptr<int64_t>(),
        dsts.data_ptr<int64_t>(),
        NN,
        NE,
        D
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_scatter_concat_out", &fused_scatter_concat_out, "Fused Scatter Concat Out");
}
