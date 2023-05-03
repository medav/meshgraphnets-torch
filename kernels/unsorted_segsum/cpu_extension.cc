#include <torch/extension.h>
#include <ATen/ATen.h>


at::Tensor unsorted_segment_sum_fwd(
    at::Tensor data,
    at::Tensor indices,
    int num_segments
) {
    const ssize_t N = num_segments;
    const ssize_t R = data.size(0);
    const ssize_t D = data.size(1);
    at::Tensor out = at::zeros({N, D}, data.options());

    AT_DISPATCH_FLOATING_TYPES(data.scalar_type(), "unsorted_segment_sum_fwd_cpu", [&] {
        scalar_t * data_d = data.data_ptr<scalar_t>();
        scalar_t * out_d = out.data_ptr<scalar_t>();
        int64_t * indices_d = indices.data_ptr<int64_t>();

        for (int64_t r = 0; r < R; r++) {
            const int64_t n = indices_d[r];

            // #pragma GCC ivdep
            for (int64_t d = 0; d < D; d++) {
                out_d[n * D + d] += data_d[r * D + d];
            }
        }
    });

    return out;
}

at::Tensor unsorted_segment_sum_bwd(
    at::Tensor grad,
    at::Tensor indices
) {
    const ssize_t N = grad.size(0);
    const ssize_t R = indices.size(0);
    const ssize_t D = grad.size(1);
    at::Tensor out = at::zeros({R, D}, grad.options());

    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "unsorted_segment_sum_bwd_cpu", [&] {
        scalar_t * grad_d = grad.data_ptr<scalar_t>();
        scalar_t * out_d = out.data_ptr<scalar_t>();
        int64_t * indices_d = indices.data_ptr<int64_t>();

        for (int64_t r = 0; r < R; r++) {
            const int64_t n = indices_d[r];

            // #pragma GCC ivdep
            for (int64_t d = 0; d < D; d++) {
                out_d[r * D + d] = grad_d[n * D + d];
            }
        }
    });

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unsorted_segment_sum_fwd", &unsorted_segment_sum_fwd, "Unsorted Segment Sum");
    m.def("unsorted_segment_sum_bwd", &unsorted_segment_sum_bwd, "Unsorted Segment Sum (Grad)");
}
