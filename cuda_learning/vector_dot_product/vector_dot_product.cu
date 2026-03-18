// 计算两个长度为N的向量的点积，N可以是任意大小，如1M或更大
// 设置共享内存大小为S

#include <iostream>
#include <cuda_runtime.h>

#define S 256

__global__ void vectorDotProduct (float *a, float *b, float *c, int N) {
    __shared__ float s[S];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;

    if (idx < N) {
        temp = a[idx] * b[idx];
    }
    s[threadIdx.x] = temp;

    __syncthreads();

    // 归约
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s[threadIdx.x] += s[threadIdx.x+stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        c[blockIdx.x] = s[0];
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
    int N = 1024*1024;
    const size_t size = N*sizeof(float);
    int blockNum = (N+S-1)/S;
    const size_t gridSize = blockNum*sizeof(float);
    float result = 0.0f;

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[blockNum];

    for (int i=0; i<N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, gridSize));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    dim3 blockDim(S);
    dim3 gridDim(blockNum);

    vectorDotProduct<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, gridSize, cudaMemcpyDeviceToHost));

    for (int i=0; i<blockNum; i++) {
        result += h_C[i];
    }
    std::cout << result << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));


    return 0;
}
