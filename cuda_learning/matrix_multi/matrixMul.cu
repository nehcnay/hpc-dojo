#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// 假设AB均为N*N的矩阵，那C也是N*N
__global__ void matrixMul(float *C, float *A, float *B, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    if (i<N && j<N) {
        for (int k=0; k<N; k++) {
            sum += A[i*N+k] * B[k*N+j];
        }
        C[i*N+j] = sum;
    }
}

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    int N = 1024;
    const size_t size = N*N*sizeof(float);
    float *h_A = new float[N*N], *h_B = new float[N*N];
    float *h_C = new float[N*N];

    // 与共享内存版本使用相同的数据初始化方式
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            h_A[idx] = i;
            h_B[idx] = j * 2;
        }
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // 与共享内存版本使用相同的 grid/block 配置
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matrixMul<<<gridDim, blockDim>>>(d_C, d_A, d_B, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    std::cout << "前5行×前5列结果：\n";
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << h_C[i*N + j] << "\t";
        }
        std::cout << "\n";
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    return 0;
}
