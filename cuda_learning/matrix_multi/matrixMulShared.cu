// 矩阵乘法（共享内存优化版）
// 其实就是矩阵乘法的分块思想
#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16 /*定义子块大小*/

__global__ void matrixMulShared(float *C, float *A, float *B, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    // 这里是因为共享内存就是一个block的大小

    int row = blockIdx.y * TILE_SIZE + ty; // C的行
    int col = blockIdx.x * TILE_SIZE + tx; // C的列

    float sum = 0.0f;

    for (int k=0; k<N; k+=TILE_SIZE) {
        // 这里A是一个一维数组，sA是二维数组
        sA[ty][tx] = (row<N && (k+tx)<N) ? A[row*N + (k+tx)] : 0.0f; /*防止越界*/

        sB[ty][tx] = ((k+ty)<N && col<N) ? B[(k+ty)*N + col] : 0.0f;
        // 同步，等block内所有线程加载完共享内存
        __syncthreads();

        for (int t=0; t<TILE_SIZE; t++) {
            sum += sA[ty][t] * sB[t][tx];
        }
        __syncthreads();
    }

    if (row<N && col<N) {
        C[row*N+col] = sum;
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

    float *h_A = new float[N*N];
    float *h_B = new float[N*N];
    float *h_C = new float[N*N];

    for (int i = 0; i < N; i++) {  // 行
        for (int j = 0; j < N; j++) {  // 列
            int idx = i * N + j;  // 二维转一维索引
            h_A[idx] = i;         // A矩阵：第i行全为i
            h_B[idx] = j * 2;     // B矩阵：第j列全为j*2
        }
    }
    
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matrixMulShared<<<gridDim, blockDim>>>(d_C, d_A, d_B, N);
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

    // 释放内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    return 0;


}