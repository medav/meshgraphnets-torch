#include <torch/extension.h>
#include <ATen/ATen.h>
#include "api.hh"

at::Tensor unsorted_segment_sum_fwd_cpu_fp32(
    at::Tensor data, // [num_rows, dim]
    at::Tensor indices, // [num_rows]
    int num_segments
) {
    const ssize_t N = num_segments;
    const ssize_t R = data.size(0);
    const ssize_t D = data.size(1);
    at::Tensor out = at::zeros({N, D}, data.options());

    float * data_d = data.data_ptr<float>();
    float * out_d = out.data_ptr<float>();
    int64_t * indices_d = indices.data_ptr<int64_t>();


    for (int64_t r = 0; r < R; r++) {
        const int64_t n = indices_d[r];

        #pragma GCC ivdep
        for (int64_t d = 0; d < D; d++) {
            out_d[n * D + d] += data_d[r * D + d];
        }
    }

    return out;
}

at::Tensor batched_unsorted_segment_sum_fwd_cpu_fp32(
    at::Tensor data, // [batch, num_rows, dim]
    at::Tensor indices, // [num_rows]
    int num_segments
) {
    const ssize_t N = num_segments;
    const ssize_t B = data.size(0);
    const ssize_t R = data.size(1);
    const ssize_t D = data.size(2);
    at::Tensor out = at::zeros({B, N, D}, data.options());

    float * data_d = data.data_ptr<float>();
    float * out_d = out.data_ptr<float>();
    int64_t * indices_d = indices.data_ptr<int64_t>();

    for (int64_t r = 0; r < R; r++) {
        const int64_t n = indices_d[r];

        for (int64_t b = 0; b < B; b++) {

            #pragma GCC ivdep
            for (int64_t d = 0; d < D; d++) {
                out_d[b * N * D + n * D + d] += data_d[b * R * D + r * D + d];
            }
        }
    }

    return out;
}

at::Tensor unsorted_segment_sum_bwd_cpu_fp32(
    at::Tensor grad, // [num_segments, dim]
    at::Tensor indices // [num_rows]
) {
    const ssize_t N = grad.size(0);
    const ssize_t R = indices.size(0);
    const ssize_t D = grad.size(1);
    at::Tensor out = at::zeros({R, D}, grad.options());

    float * grad_d = grad.data_ptr<float>();
    float * out_d = out.data_ptr<float>();
    int64_t * indices_d = indices.data_ptr<int64_t>();

    for (int64_t r = 0; r < R; r++) {
        const int64_t n = indices_d[r];

        #pragma GCC ivdep
        for (int64_t d = 0; d < D; d++) {
            out_d[r * D + d] = grad_d[n * D + d];
        }
    }

    return out;
}

at::Tensor batched_unsorted_segment_sum_bwd_cpu_fp32(
    at::Tensor grad, // [batch, num_segments, dim]
    at::Tensor indices // [num_rows]
) {
    const ssize_t N = grad.size(1);
    const ssize_t B = grad.size(0);
    const ssize_t R = indices.size(0);
    const ssize_t D = grad.size(2);

    at::Tensor out = at::zeros({B, R, D}, grad.options());

    float * grad_d = grad.data_ptr<float>();
    float * out_d = out.data_ptr<float>();
    int64_t * indices_d = indices.data_ptr<int64_t>();

    for (int64_t r = 0; r < R; r++) {
        const int64_t n = indices_d[r];

        for (int64_t b = 0; b < B; b++) {
            #pragma GCC ivdep
            for (int64_t d = 0; d < D; d++) {
                out_d[b * R * D + r * D + d] = grad_d[b * N * D + n * D + d];
            }
        }
    }

    return out;
}
