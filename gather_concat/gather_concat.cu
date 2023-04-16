

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CLD(N, D) ((N + D - 1) / D)

__global__ void device_compute_edge_offsets(
    int64_t * receivers,
    int64_t * out,
    int num_receivers,
    int num_nodes
) {
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= num_receivers) return;
    atomicAdd((unsigned long long *)&out[receivers[i]], (unsigned long long)1);
}

at::Tensor compute_edge_offsets(
    at::Tensor receivers,
    int64_t num_nodes
) {
    CHECK_INPUT(receivers);

    int64_t NE = receivers.size(0);
    at::Tensor out = torch::zeros({num_nodes}, receivers.options());

    device_compute_edge_offsets<<<CLD(NE, 512), 512>>>(
        receivers.data_ptr<int64_t>(),
        out.data_ptr<int64_t>(),
        (int)NE,
        (int)num_nodes
    );

    return out.cumsum(0);
}

template<size_t MAX_D>
__global__ void device_fused_gather_concat_1e(
    half * nf,
    half * ef0, int64_t * ef0_offsets,
    half * out,
    const int64_t N,
    const int64_t NE0,
    const int64_t D
) {
    __shared__ half accum[MAX_D];

    const int node_i = blockIdx.x;
    const int node_d = threadIdx.x;
    const int row_off = node_i * D;
    const int out_row_off = node_i * 2 * D;

    if (node_d < D) {
        out[out_row_off + node_d] = nf[row_off + node_d];
    } else {
        const int d_off = node_d - D;
        accum[d_off] = __float2half(0.0f);

        const int e_start = node_i == 0 ? 0 : ef0_offsets[node_i - 1];
        const int e_end = ef0_offsets[node_i];

        for (int e = e_start; e < e_end; e++) {
            accum[d_off] = __hadd(accum[d_off], ef0[e * D + d_off]);
        }

        out[out_row_off + node_d] = accum[d_off];
    }
}

at::Tensor fused_gather_concat_1e(
    at::Tensor nf,
    at::Tensor ef0,
    at::Tensor ef0_offsets
) {
    CHECK_INPUT(nf);
    CHECK_INPUT(ef0);
    CHECK_INPUT(ef0_offsets);

    const int64_t D = nf.size(1);
    const int64_t NN = nf.size(0);
    const int64_t NE0 = ef0.size(0);

    assert(ef0.size(1) == D);

    at::Tensor out = torch::zeros({NN, 2 * D}, nf.options());

    device_fused_gather_concat_1e<128><<<NN, 2*D>>>(
        (half *)nf.data_ptr<at::Half>(),
        (half *)ef0.data_ptr<at::Half>(),
        ef0_offsets.data_ptr<int64_t>(),
        (half *)out.data_ptr<at::Half>(),
        NN,
        NE0,
        D
    );

    return out;
}


at::Tensor message_passing_fwd_1e_(
    // Inputs
    at::Tensor nf,
    at::Tensor ef0,

    // Weights
    at::TensorList nw,
    at::Tensor nb,

    at::Tensor ew0,
    at::Tensor eb0
) {
    const size_t D = nf.size(1);
    const size_t NN = nf.size(0);
    const size_t NE0 = ef0.size(0);

    assert(ef0.size(1) == D);

    // at::Tensor node_concat = torch::zeros({NN, 3 * D}, nf.options());

    // at::linear(node_concat, nw, nb);
}

template<size_t MAX_D>
__global__ void device_fused_gather_concat_2e(
    half * nf,
    half * ef0, int64_t * ef0_offsets,
    half * ef1, int64_t * ef1_offsets,
    half * out,
    const int64_t N,
    const int64_t NE0,
    const int64_t NE1,
    const int64_t D
) {
    __shared__ half accum[MAX_D];

    const int node_i = blockIdx.x;
    const int node_d = threadIdx.x;
    const int row_off = node_i * D;
    const int out_row_off = node_i * 3 * D;

    if (node_d < D) {
        out[out_row_off + node_d] = nf[row_off + node_d];
    } else if (node_d < 2*D) {
        const int d_off = node_d - D;
        accum[d_off] = __float2half(0.0f);

        const int e_start = node_i == 0 ? 0 : ef0_offsets[node_i - 1];
        const int e_end = ef0_offsets[node_i];

        for (int e = e_start; e < e_end; e++) {
            accum[d_off] = __hadd(accum[d_off], ef0[e * D + d_off]);
        }

        out[out_row_off + node_d] = accum[d_off];
    } else {
        const int d_off = node_d - D - D;
        accum[d_off] = __float2half(0.0f);

        const int e_start = node_i == 0 ? 0 : ef1_offsets[node_i - 1];
        const int e_end = ef1_offsets[node_i];

        for (int e = e_start; e < e_end; e++) {
            accum[d_off] = __hadd(accum[d_off], ef1[e * D + d_off]);
        }

        out[out_row_off + node_d] = accum[d_off];
    }
}

void fused_gather_concat_2e_out(
    at::Tensor nf,
    at::Tensor ef0,
    at::Tensor eoffs0,
    at::Tensor ef1,
    at::Tensor eoffs1,
    at::Tensor node_concat
) {
    CHECK_INPUT(nf);
    CHECK_INPUT(ef0);
    CHECK_INPUT(eoffs0);
    CHECK_INPUT(ef1);
    CHECK_INPUT(eoffs1);
    CHECK_INPUT(node_concat);

    const int64_t D = nf.size(1);
    const int64_t NN = nf.size(0);
    const int64_t NE0 = ef0.size(0);
    const int64_t NE1 = ef1.size(0);

    assert(D <= 128);
    assert(ef0.size(1) == D);
    assert(ef1.size(1) == D);

    device_fused_gather_concat_2e<128><<<NN, 3*D>>>(
        (half *)nf.data_ptr<at::Half>(),
        (half *)ef0.data_ptr<at::Half>(), eoffs0.data_ptr<int64_t>(),
        (half *)ef1.data_ptr<at::Half>(), eoffs1.data_ptr<int64_t>(),
        (half *)node_concat.data_ptr<at::Half>(),
        NN,
        NE0,
        NE1,
        D
    );
}


at::Tensor test(std::vector<at::Tensor> x) {
    return torch::zeros({1});
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test", &test, "Test");
    m.def("fused_gather_concat_1e", &fused_gather_concat_1e, "Fused Gather Concat 1e");
    m.def("compute_edge_offsets", &compute_edge_offsets, "Compute Edge Offsets");
    m.def("fused_gather_concat_2e_out", &fused_gather_concat_2e_out, "Fused Gather Concat 2e Out");
}
