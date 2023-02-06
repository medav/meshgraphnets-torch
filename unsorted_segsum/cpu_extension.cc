#include <torch/extension.h>
#include <ATen/ATen.h>
#include "api.hh"

at::Tensor unsorted_segment_sum_fwd_fp32(
    at::Tensor data,
    at::Tensor indices,
    int num_segments
) {
    return unsorted_segment_sum_fwd_cpu_fp32(data, indices, num_segments);
}

at::Tensor unsorted_segment_sum_bwd_fp32(
    at::Tensor grad,
    at::Tensor indices
) {
    return unsorted_segment_sum_bwd_cpu_fp32(grad, indices);
}

at::Tensor batched_unsorted_segment_sum_fwd_fp32(
    at::Tensor data,
    at::Tensor indices,
    int num_segments
) {
    return batched_unsorted_segment_sum_fwd_cpu_fp32(data, indices, num_segments);
}

at::Tensor batched_unsorted_segment_sum_bwd_fp32(
    at::Tensor grad,
    at::Tensor indices
) {
    return batched_unsorted_segment_sum_bwd_cpu_fp32(grad, indices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unsorted_segment_sum_fwd_fp32", &unsorted_segment_sum_fwd_fp32, "Unsorted Segment Sum");
    m.def("unsorted_segment_sum_bwd_fp32", &unsorted_segment_sum_bwd_fp32, "Unsorted Segment Sum (Grad)");

    m.def("batched_unsorted_segment_sum_fwd_fp32", &batched_unsorted_segment_sum_fwd_fp32, "Batched Unsorted Segment Sum");
    m.def("batched_unsorted_segment_sum_bwd_fp32", &batched_unsorted_segment_sum_bwd_fp32, "Batched Unsorted Segment Sum (Grad)");
}
