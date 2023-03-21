
#include <stdio.h>
#include <curand.h>
#include <functional>
#include <cublas_v2.h>

#define MATUL_VERSION wmma_v1
#define TO_STR(x) #x

#define CLD(N, D) ((N + D - 1) / D)

// Define some error checking macros.
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

#define cublasErrCheck(stat)                         \
    {                                                \
        cublasErrCheck_((stat), __FILE__, __LINE__); \
    }

void cublasErrCheck_(cublasStatus_t stat, const char *file, int line)
{
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

#define curandErrCheck(stat)                         \
    {                                                \
        curandErrCheck_((stat), __FILE__, __LINE__); \
    }

void curandErrCheck_(curandStatus_t stat, const char *file, int line)
{
    if (stat != CURAND_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
    }
}

#include <mma.h>
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
#define MATRIX_M 1024
#define MATRIX_N 128
#define MATRIX_K 384

// #define MATRIX_M 16
// #define MATRIX_N 16
// #define MATRIX_K 16

#define M_CHUNK 64

#define V2_NBLK 16

#define NUM_ITERS 1000

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

template<typename A_LAYOUT, typename B_LAYOUT>
__global__ void throughput_test(uint64_t num_iters, void * buf)
{

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, A_LAYOUT> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, B_LAYOUT> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> d_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::fill_fragment(d_frag, 0.0f);
    wmma::fill_fragment(a_frag, 1.0f);
    wmma::fill_fragment(b_frag, 1.0f);

    for (int ni = 0; ni < num_iters; ni++) {
        wmma::mma_sync(c_frag, a_frag, b_frag, d_frag);
    }

    wmma::store_matrix_sync((half *)buf, c_frag, WMMA_N, wmma::mem_row_major);
}

template<ssize_t M, ssize_t N, ssize_t K>
__global__ void wmma_v1(half a[M][K], half b[K][N], float c[M][N]) {
    int warp_m = (threadIdx.x) / warpSize;
    int warp_n = (threadIdx.y);
    int m_base = warp_m * WMMA_M;
    int m_stride = blockDim.x / warpSize * WMMA_M;
    int n = warp_n * WMMA_N;

    __syncthreads();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;


    for (int ni = 0; ni < NUM_ITERS; ni++) {
        for (int m = m_base; m < M; m += m_stride) {
            wmma::fill_fragment(c_frag, 0.0f);

            for (int k = 0; k < K; k += WMMA_K) {
                wmma::load_matrix_sync(a_frag, &a[m][k], K);
                wmma::load_matrix_sync(b_frag, &b[k][n], N);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            wmma::store_matrix_sync((float*)&c[m][n], c_frag, N, wmma::mem_row_major);
        }
    }
}

template<ssize_t M, ssize_t N, ssize_t K, ssize_t NBLK>
__global__ void wmma_v2(half a[M][K], half b[K][N], float c[M][N]) {
    int warp_id = threadIdx.x / warpSize;
    int warp_tid = threadIdx.x % warpSize;
    int num_warps = blockDim.x / warpSize;
    int warp_m = warp_id;
    int m_base = warp_m * WMMA_M;
    int m_stride = blockDim.x * WMMA_M / warpSize;

    __shared__ half bbuf[K][NBLK];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    for (int ni = 0; ni < NUM_ITERS; ni++) {
        for (int n_base = 0; n_base < N; n_base += NBLK) {
            int n_end = min(n_base + NBLK, N);

            // Load shared memory
            for (int k = 0; k < K; k += num_warps) {
                int k_idx = k + warp_id;
                if (k_idx < K) {
                    for (int n = n_base; n < n_end; n += warpSize) {
                        int n_idx = n + warp_tid;
                        if (n_idx < n_end) {
                            bbuf[k_idx][n_idx - n_base] = b[k_idx][n_idx];
                        }
                    }
                }
            }
            __syncthreads();

            for (int m = m_base; m < M; m += m_stride) {

                for (int n = n_base; n < n_end; n += WMMA_N) {
                    wmma::fill_fragment(c_frag, 0.0f);

                    for (int k = 0; k < K; k += WMMA_K) {
                        wmma::load_matrix_sync(a_frag, &a[m][k], K);
                        wmma::load_matrix_sync(b_frag, &bbuf[k][n - n_base], NBLK);
                        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }

                    wmma::store_matrix_sync((float*)&c[m][n], c_frag, N, wmma::mem_row_major);
                }
            }

            __syncthreads();
        }
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

inline bool is_close(float a, float b, float eps = 0.001)
{
    return fabs(a - b) < eps;
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

template<typename LAYOUT>
const char * name_of() { return "unknown"; }

template<>
const char * name_of<wmma::row_major>() { return "ROW"; }

template<>
const char * name_of<wmma::col_major>() { return "COL"; }

template<typename A_LAYOUT, typename B_LAYOUT>
void host_throughput_test() {
    constexpr int num_iters = 1000000;

    float * buf;
    cudaErrCheck(cudaMalloc(&buf, WMMA_M * WMMA_N * sizeof(float)));

    float wmma_ms = cuda_time_kernel_ms(
        [=]() { throughput_test<A_LAYOUT, B_LAYOUT><<<1, 32>>>(num_iters, buf); });

    float flops = (float)WMMA_M * WMMA_N * WMMA_K * 2 * num_iters;
    float gflop_s = (flops / 1.0e9) / (wmma_ms / 1.0e3);

    cudaErrCheck(cudaFree(buf));

    printf("(%s, %s): %.3f GFLOP/s\n", name_of<A_LAYOUT>(), name_of<B_LAYOUT>(), gflop_s);
}

void host_throughput_test_all() {
    host_throughput_test<wmma::row_major, wmma::row_major>();
    host_throughput_test<wmma::row_major, wmma::col_major>();
    host_throughput_test<wmma::col_major, wmma::row_major>();
    host_throughput_test<wmma::col_major, wmma::col_major>();

}

int main(int argc, char *argv[])
{
    float *a_fp32;
    float *b_fp32;
    float *c_fp32;
    float *c_ref_fp32;

    host_throughput_test_all();

    printf("Matrix A (%d, %d)\n", MATRIX_M, MATRIX_K);
    printf("    Size: %lu bytes\n", MATRIX_M * MATRIX_K * sizeof(half));
    printf("    Size: %.3f MB\n", ((float)MATRIX_M * MATRIX_K * sizeof(half)) / (1024 * 1024));

    printf("Matrix B (%d, %d)\n", MATRIX_K, MATRIX_N);
    printf("    Size: %lu bytes\n", MATRIX_K * MATRIX_N * sizeof(half));
    printf("    Size: %.3f MB\n", ((float)MATRIX_K * MATRIX_N * sizeof(half)) / (1024 * 1024));

    printf("Matrix C (%d, %d)\n", MATRIX_M, MATRIX_N);
    printf("    Size: %lu bytes\n", MATRIX_M * MATRIX_N * sizeof(float));
    printf("    Size: %.3f MB\n", ((float)MATRIX_M * MATRIX_N * sizeof(float)) / (1024 * 1024));


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

    float wmma_v1_time_ms = cuda_time_kernel_ms(
        [&]() {
            dim3 block(M_CHUNK * 32 / WMMA_M, MATRIX_N / WMMA_N);
            wmma_v1<MATRIX_M, MATRIX_N, MATRIX_K><<<1, block>>>(
                reinterpret_cast<half (*)[MATRIX_K]>(dev_a_fp16),
                reinterpret_cast<half (*)[MATRIX_N]>(dev_b_fp16),
                reinterpret_cast<float (*)[MATRIX_N]>(dev_c_fp32));
        }
    );

    printf("wmma_v1 took %fms\n", wmma_v1_time_ms);

    float flops_v1 = 2.0f * MATRIX_M * MATRIX_N * MATRIX_K * NUM_ITERS;
    float gflops_v1 = flops_v1 / (wmma_v1_time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v1);

    //
    // Version 2
    //

    float wmma_v2_time_ms = cuda_time_kernel_ms(
        [&]() {
            // dim3 block(M_CHUNK * 32 / WMMA_M);
            wmma_v2<MATRIX_M, MATRIX_N, MATRIX_K, V2_NBLK><<<1, 1024>>>(
                reinterpret_cast<half (*)[MATRIX_K]>(dev_a_fp16),
                reinterpret_cast<half (*)[MATRIX_N]>(dev_b_fp16),
                reinterpret_cast<float (*)[MATRIX_N]>(dev_c_fp32));
        }
    );

    printf("wmma_v2 took %fms\n", wmma_v2_time_ms);

    float flops_v2 = 2.0f * MATRIX_M * MATRIX_N * MATRIX_K * NUM_ITERS;
    float gflops_v2 = flops_v2 / (wmma_v2_time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v2);

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
}
