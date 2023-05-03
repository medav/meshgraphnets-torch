

#include <torch/extension.h>
#include <ATen/ATen.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CLD(N, D) ((N + D - 1) / D)


at::Tensor compute_edge_offsets(
    at::Tensor dsts, // [NE]
    int64_t NN
) {
    CHECK_INPUT(dsts);

    int64_t NE = dsts.size(0);
    at::Tensor out = torch::zeros({NN}, dsts.options());

    int64_t * dsts_ptr = dsts.data_ptr<int64_t>();
    int64_t * out_ptr = out.data_ptr<int64_t>();

    for (int64_t e = 0; e < NE; e++) {
        out_ptr[dsts_ptr[e]]++;
    }

    return out.cumsum(0);
}


template<typename scalar_t>
void cpu_fused_gather_concat_2e(
    const scalar_t * nf, // [N, D]
    const scalar_t * ef0, const int64_t * ef0_offsets, // [N, D], [N]
    const scalar_t * ef1, const int64_t * ef1_offsets, // [N, D], [N]
    scalar_t * out, // [N, 3D]
    const int64_t N,
    const int64_t NE0,
    const int64_t NE1,
    const int64_t D
) {
    const size_t LDD = NE1 > 0 ? D * 3 : D * 2;
    const size_t CHUNK_SIZE = 4;

    at::parallel_for(0, N, CHUNK_SIZE, [&](int64_t ss, int64_t ee) {
        for (size_t ni = ss; ni < ee; ni++) {
            #pragma GCC ivdep
            for (size_t di = 0; di < D; di++) {
                out[ni * LDD + di] = nf[ni * D + di];
            }

            const int e0_start = (ni == 0) ? 0 : ef0_offsets[ni - 1];
            const int e0_end = ef0_offsets[ni];
            for (int e = e0_start; e < e0_end; e++) {
                #pragma GCC ivdep
                for (size_t di = D; di < 2*D; di++) {
                    out[ni * LDD + di] += ef0[e * D + di - D];
                }
            }

            if (NE1 > 0) {
                const int e1_start = (ni == 0) ? 0 : ef1_offsets[ni - 1];
                const int e1_end = ef1_offsets[ni];
                for (int e = e1_start; e < e1_end; e++) {

                    #pragma GCC ivdep
                    for (size_t di = 2*D; di < 3*D; di++) {
                        out[ni * LDD + di] += ef1[e * D + di - 2*D];
                    }
                }
            }
        }
    });
}



at::Tensor fused_gather_concat_2e(
    at::Tensor nf,
    at::Tensor ef0,
    at::Tensor eoffs0,
    at::Tensor ef1,
    at::Tensor eoffs1
) {
    CHECK_INPUT(nf);
    CHECK_INPUT(ef0);
    CHECK_INPUT(eoffs0);
    CHECK_INPUT(ef1);
    CHECK_INPUT(eoffs1);

    const int64_t D = nf.size(1);
    const int64_t NN = nf.size(0);
    const int64_t NE0 = ef0.size(0);
    const int64_t NE1 = ef1.size(0);

    assert(D <= 128);
    assert(ef0.size(1) == D);
    assert(ef1.size(1) == D);

    at::Tensor out = at::zeros({NN, 3*D}, nf.options());

    AT_DISPATCH_FLOATING_TYPES(nf.scalar_type(), "cpu_fused_gather_concat", [&] {
        cpu_fused_gather_concat_2e<scalar_t>(
            nf.data_ptr<scalar_t>(),
            ef0.data_ptr<scalar_t>(), eoffs0.data_ptr<int64_t>(),
            ef1.data_ptr<scalar_t>(), eoffs1.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(),
            NN, NE0, NE1, D
        );
    });

    return out;
}


at::Tensor fused_gather_concat_1e(
    at::Tensor nf,
    at::Tensor ef0,
    at::Tensor eoffs0
) {
    CHECK_INPUT(nf);
    CHECK_INPUT(ef0);
    CHECK_INPUT(eoffs0);

    const int64_t D = nf.size(1);
    const int64_t NN = nf.size(0);
    const int64_t NE0 = ef0.size(0);

    assert(D <= 128);
    assert(ef0.size(1) == D);

    at::Tensor out = at::zeros({NN, 2*D}, nf.options());

    AT_DISPATCH_FLOATING_TYPES(nf.scalar_type(), "cpu_fused_gather_concat", [&] {
        cpu_fused_gather_concat_2e<scalar_t>(
            nf.data_ptr<scalar_t>(),
            ef0.data_ptr<scalar_t>(), eoffs0.data_ptr<int64_t>(),
            nullptr, nullptr,
            out.data_ptr<scalar_t>(),
            NN, NE0, 0, D
        );
    });

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_edge_offsets", &compute_edge_offsets, "Compute Edge Offsets");
    m.def("fused_gather_concat_1e", &fused_gather_concat_1e, "Fused Gather Concat 1e");
    m.def("fused_gather_concat_2e", &fused_gather_concat_2e, "Fused Gather Concat 2e");
}
