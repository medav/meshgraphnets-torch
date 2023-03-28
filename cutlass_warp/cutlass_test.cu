#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/gemm/warp/mma.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"


#define CLD(N, D) ((N + D - 1) / D)

#define cudaErrCheck(stat)                         \
    {                                              \
        cudaErrCheck_((stat), __FILE__, __LINE__); \
    }

void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
    if (stat != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

__global__ void convertFp32ToFp16(half *out, float *in, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = in[idx];
    }
}

float cuda_time_kernel_ms(std::function<void(void)> func) {
    float time_ms;
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&stop));

    cudaErrCheck(cudaEventRecord(start));
    func();
    cudaErrCheck(cudaGetLastError());
    cudaErrCheck(cudaEventRecord(stop));

    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&time_ms, start, stop));

    return time_ms;
}



namespace cutlass {
namespace gemm {
namespace warp {

template <
    typename Shape,
    typename LayoutA,
    typename LayoutB,
    typename LayoutC>
class GemmTensorOp
{
public:
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using WarpShape = GemmShape<
        CLD(Shape::kM, InstructionShape::kM) * InstructionShape::kM,
        CLD(Shape::kN, InstructionShape::kN) * InstructionShape::kN,
        InstructionShape::kK>;

    using MmaWarp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
        WarpShape,
        InstructionShape,
        cutlass::half_t,              // Data type of A elements
        LayoutA,                      // Layout of A matrix
        cutlass::half_t,              // Data type of B elements
        LayoutB,                      // Layout of B matrix
        cutlass::half_t,              // Data type of C elements
        LayoutC                       // Layout of C matrix
        >::Type;

    // Number of 'K groups'
    int const kKgroups = CLD(Shape::kK, InstructionShape::kK);

    // Define a 'FragmentIterator' to iterate over slices of accumulators
    using FragmentIterator = typename cutlass::epilogue::warp::FragmentIteratorTensorOp<
        typename MmaWarp::Shape,
        InstructionShape,
        cutlass::half_t,
        typename MmaWarp::Policy::Operator::FragmentC,
        LayoutC>;

    // Define an epilogue 'Tile Iteterator' to iterate over slices of elements in Shared Memory
    using AccumulatorTileIterator = typename cutlass::epilogue::warp::TileIteratorTensorOpCanonical<
        typename MmaWarp::Shape,
        InstructionShape,
        cutlass::half_t,
        LayoutC>;

    using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
    using TensorRefB = typename MmaWarp::IteratorB::TensorRef;
    using TensorRefC = typename AccumulatorTileIterator::TensorRef;

public:
    CUTLASS_HOST_DEVICE
    GemmTensorOp() {}

    CUTLASS_DEVICE
    void operator()(
        TensorRefA ref_A,
        TensorRefB ref_B,
        TensorRefC ref_C,
        int lane_id) const
    {

        // Instantiate iterators pointing to slices of the A and B matrices in shared memory
        typename MmaWarp::IteratorA iter_A(ref_A, {Shape::kM, Shape::kK}, lane_id);
        typename MmaWarp::IteratorB iter_B(ref_B, {Shape::kK, Shape::kN}, lane_id);

        // Instantiate and clear accumulator tile holding the C matrix
        typename MmaWarp::FragmentC accum;
        accum.clear();

        // Instantiate the warp-level matrix multiply operator
        MmaWarp mma_op;

        // Instantiate fragments holding the slice of the matrix held by each warp
        typename MmaWarp::FragmentA frag_A[2];
        typename MmaWarp::FragmentB frag_B[2];

        // cuda::memcpy_async

        // Load fragments from shared memory
        iter_A.load(frag_A[0]);
        iter_B.load(frag_B[0]);

        ++iter_A;
        ++iter_B;

        // Load fragments from shared memory
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kKgroups; ++k)
        {

            // Load fragments from shared memory
            iter_A.load(frag_A[(k + 1) % 2]);
            iter_B.load(frag_B[(k + 1) % 2]);

            ++iter_A;
            ++iter_B;

            // Compute the matrix multiply
            mma_op(accum, frag_A[k % 2], frag_B[k % 2], accum);
            // mma_op(accum, frag_A[0], frag_B[0], accum);
        }

        // Instantiate iterators
        FragmentIterator accum_frag_it(accum);
        AccumulatorTileIterator source_tile_it(ref_C, {Shape::kM, Shape::kN}, lane_id);


        // Iterate over the epilogue components
        CUTLASS_PRAGMA_UNROLL
        for (int idx = 0; idx < FragmentIterator::kIterations; ++idx)
        {
            // Define storage for slices of the accumulators
            typename FragmentIterator::Fragment accum_fragment;

            // Select a slice of accumulators from the accumulator tile
            accum_frag_it.load(accum_fragment);

            // Store the result to shared memory
            source_tile_it.store(accum_fragment);
            ++source_tile_it;
        }
    }
};

} // namespace warp
} // namespace gemm
} // namespace cutlass




#define MATRIX_M 128
#define MATRIX_N 16
#define MATRIX_K 128

#define NUM_WARPS 8

#define NUM_ITERS 10000000


template<ssize_t NW, ssize_t M, ssize_t N, ssize_t K>
__global__ void gemm(half a[M][K], half b[K][N], float c[M][N]) {
    using Shape = cutlass::gemm::GemmShape<MATRIX_M, MATRIX_N, MATRIX_K>;

    using GemmOp = typename cutlass::gemm::warp::GemmTensorOp<
        Shape,
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor
        >;

    const ssize_t num_warps = blockDim.x / 32;
    const ssize_t warp_id = 0; //threadIdx.x / 32;
    const ssize_t lane_id = threadIdx.x % 32;

    __shared__ half sbufA[1][M][K];
    __shared__ half sbufB[1][N][K];
    __shared__ half sbufC[1][M][N];


    if (threadIdx.x == 0) {
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                sbufA[warp_id][m][k] = a[m][k];
            }
        }

        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                sbufB[0][n][k] = b[k][n];
            }
        }
    }

    __syncthreads();


    GemmOp gemm_op;

    for (int i = 0; i < NUM_ITERS; i++) {
        gemm_op(
            GemmOp::TensorRefA((GemmOp::MmaWarp::ElementA *)sbufA[warp_id], K),
            GemmOp::TensorRefB((GemmOp::MmaWarp::ElementB *)sbufB[0], N),
            GemmOp::TensorRefC((GemmOp::MmaWarp::ElementC *)sbufC[warp_id], N),
            lane_id
        );
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                c[m][n] = sbufC[warp_id][m][n];
            }
        }
    }
}

int main(int argc, char const **args)
{
    float *a_fp32;
    float *b_fp32;
    float *c_fp32;
    float *c_ref_fp32;


    printf("Matrix A (%d, %d)\n", MATRIX_M, MATRIX_K);
    printf("    Size: %lu bytes\n", MATRIX_M * MATRIX_K * sizeof(half));
    printf("    Size: %.3f KB\n", ((float)MATRIX_M * MATRIX_K * sizeof(half)) / (1024));

    printf("Matrix B (%d, %d)\n", MATRIX_K, MATRIX_N);
    printf("    Size: %lu bytes\n", MATRIX_K * MATRIX_N * sizeof(half));
    printf("    Size: %.3f KB\n", ((float)MATRIX_K * MATRIX_N * sizeof(half)) / (1024));

    printf("Matrix C (%d, %d)\n", MATRIX_M, MATRIX_N);
    printf("    Size: %lu bytes\n", MATRIX_M * MATRIX_N * sizeof(float));
    printf("    Size: %.3f KB\n", ((float)MATRIX_M * MATRIX_N * sizeof(float)) / (1024));


    a_fp32 = (float *)malloc(MATRIX_M * MATRIX_K * sizeof(float));
    b_fp32 = (float *)malloc(MATRIX_K * MATRIX_N * sizeof(float));
    c_fp32 = (float *)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    c_ref_fp32 = (float *)malloc(MATRIX_M * MATRIX_N * sizeof(float));


    // Fill A and B
    for (int i = 0; i < MATRIX_M * MATRIX_K; i++) {
        a_fp32[i] = 1.0f; //(float)rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < MATRIX_K * MATRIX_N; i++) {
        // int rK = i / MATRIX_N;
        // int cN = i % MATRIX_N;
        b_fp32[i] = (float)rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
        c_fp32[i] = 0.0f;
        c_ref_fp32[i] = 0.0f;
    }


    // Compute reference output on the CPU
    for (int m = 0; m < MATRIX_M; m++) {
        for (int n = 0; n < MATRIX_N; n++) {
            for (int k = 0; k < MATRIX_K; k++) {
                c_ref_fp32[m * MATRIX_N + n] +=
                    a_fp32[m * MATRIX_K + k] * b_fp32[k * MATRIX_N + n];
            }
        }
    }


    float *dev_a_fp32;
    float *dev_b_fp32;
    float *dev_c_fp32;

    cudaErrCheck(cudaMalloc((void **)&dev_a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&dev_b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&dev_c_fp32, MATRIX_M * MATRIX_N * sizeof(float)));

    cudaErrCheck(cudaMemcpy(dev_a_fp32, a_fp32, MATRIX_M * MATRIX_K * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(dev_b_fp32, b_fp32, MATRIX_K * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(dev_c_fp32, c_fp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));

    half *dev_a_fp16;
    half *dev_b_fp16;

    cudaErrCheck(cudaMalloc((void **)&dev_a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
    cudaErrCheck(cudaMalloc((void **)&dev_b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

    cudaErrCheck(cudaMemset(dev_a_fp16, 0, MATRIX_M * MATRIX_K * sizeof(half)));
    cudaErrCheck(cudaMemset(dev_b_fp16, 0, MATRIX_K * MATRIX_N * sizeof(half)));

    // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
    convertFp32ToFp16<<<CLD(MATRIX_M * MATRIX_K, 256), 256>>>(dev_a_fp16, dev_a_fp32, MATRIX_M * MATRIX_K);
    convertFp32ToFp16<<<CLD(MATRIX_K * MATRIX_N, 256), 256>>>(dev_b_fp16, dev_b_fp32, MATRIX_K * MATRIX_N);

    cudaDeviceSynchronize();

    //
    // Version 1
    //

    float time_ms = cuda_time_kernel_ms(
        [&]() {
            gemm<NUM_WARPS, MATRIX_M, MATRIX_N, MATRIX_K><<<1, 32 * NUM_WARPS>>>(
                reinterpret_cast<half (*)[MATRIX_K]>(dev_a_fp16),
                reinterpret_cast<half (*)[MATRIX_N]>(dev_b_fp16),
                reinterpret_cast<float (*)[MATRIX_N]>(dev_c_fp32));
        }
    );

    printf("gemm took %fms\n", time_ms);

    float flops_v1 = 2.0f * MATRIX_M * MATRIX_N * MATRIX_K * NUM_ITERS * NUM_WARPS;
    float gflops_v1 = flops_v1 / (time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v1);

    //
    // Error checking
    //

    cudaErrCheck(cudaMemcpy(c_fp32, dev_c_fp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_error = 0.0f;

    // Rewrite above but use MxN layout instead of flat array

    for (int m = 0; m < MATRIX_M; m++) {
        for (int n = 0; n < MATRIX_N; n++) {
            float error = fabs(c_fp32[m * MATRIX_N + n] - c_ref_fp32[m * MATRIX_N + n]);
            // printf("c_fp32[%d][%d] = %f, c_ref_fp32[%d][%d] = %f, error = %f\n", m, n, c_fp32[m * MATRIX_N + n], m, n, c_ref_fp32[m * MATRIX_N + n], error);
            if (error > max_error) {
                max_error = error;
            }
        }
    }

    printf("Max error: %f\n", max_error);


    cudaErrCheck(cudaFree(dev_a_fp32));
    cudaErrCheck(cudaFree(dev_b_fp32));
    cudaErrCheck(cudaFree(dev_a_fp16));
    cudaErrCheck(cudaFree(dev_b_fp16));
    cudaErrCheck(cudaFree(dev_c_fp32));

    free(a_fp32);
    free(b_fp32);
    free(c_fp32);
    free(c_ref_fp32);

    cudaErrCheck(cudaDeviceReset());
    return 0;



    return 0;
}