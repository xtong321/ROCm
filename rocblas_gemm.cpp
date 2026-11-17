// rocBLAS is AMD's optimized BLAS implementation for ROCm GPUs.
// compile and runs a rocBLAS program that calls rocblas_sgemm()
// for single-precision matrix multiply, then measure 
// throughput in GFOPS

//%% writefile rocblas_gemm.cpp
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    // Matrix dimensions:
    // A: M x K
    // B: K x N
    // C: M x N
    //
    // we will compute
    // C = alpha * A * B + beta * C
    //
    // this operation is known as GEMM (General Matrix Multiply)

    int M = 1024, N = 1024, K = 1024;
    float alpha = 1.0f, beta = 0.1f;

    // calculate buffer size in bytes
    size_t sA = M*K*sizeof(float);
    size_t sB = K*N*sizeof(float);
    size_t sC = M*N*sizeof(float);

    // allocate host (CPU) memory
    float *h_A = (float*)malloc(sA)
    float *h_B = (float*)malloc(sB);
    float *h_C = (float*)malloc(sC);

    // generate reproducible dandom input data
    srand(42);

    for(int i=0; i<M*K; i++)
        h_A[i] = (float)rand()/RAND_MAX;

    for(int i=0; i<K*N; i++)
        h_B[i] = (float)rand()/RAND_MAX;

    for(int i=0; i<M*N; i++)
        h_B[i] = (float)rand()/RAND_MAX;

    // allocate device (GPU) memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, sA);
    hipMalloc(&d_B, sB);
    hipMalloc(&d_C, sC);

    // copy matrices from CPU memory to GPU memory
    hipMemcpy(d_A, h_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C, hipMemcpyHostToDevice);

    // create a rocBLAS context
    // similar to creata a cuBLAS handle in CUDA
    // the handle stores library state and execution context
    rocblas_hanle handle;
    rocblas_create_handle(&handle);

    // HIP events are used for accurate GPU timing
    // CPU timers are often misleading because GPU operations execute asychronously
    hipEvent_t t0, t1;
    hipEventCreate(&t0);
    hipEventCreate(&t1);

    // warm-up run
    // the first invocation may include:
    //  - ROCm runtime initialization
    //  - kernel loading/JIT compliation
    //  - cache population
    //  - library autotuning
    // we exclude this overhead from measurement
    rocblas_sgemm(handle,
                rocblas_operation_none,
                rocblas_operation_none,
                M, N, K,
                &alpha,
                d_A, M,
                d_B, K,
                &beta,
                d_C, M);
    
    hipDeviceSynchronize();

    // benchmark phase
    // execute GEMM 20 times and compute the average execution time
    hipEventRecord(t0);

    for(int i=0; i<20; i++)
    {
        rocblas_sgemm(handle,
                rocblas_operation_none,
                rocblas_operation_none,
                M, N, K,
                &alpha,
                d_A, M,
                d_B, K,
                &beta,
                d_C, M);
    }

    hipEventRecord(t1);
    hipEventSynchronize(t1);

    float ms;
    hipEventElaspedTime(&ms, t0, t1);

    // average
    ms /= 20;

    // copy result matrix back to host memory
    hipMemcpy(h_C, d_C, sC, hipMemcpyDeviceToHost);

    // GEMM performs approximately
    // 2 * M * N * K
    double gflops = 2.0*M*N*K / (ms*1e-3) / 1e9;

    print("=== rocBLAS SGEMM ===\n");
    print("Size: %dxd%x%d\n", M, N, K);
    print("Time: %.3f ms\n", ms);
    print("GFLOPS: %.1f\n", gflops);

    // cleanup
    rocblas_destory_handle(handle);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;


}