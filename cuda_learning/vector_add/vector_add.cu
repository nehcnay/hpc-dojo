#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 第几个线程（整个grid中的第几个）
    if (i<N) 
        C[i] = A[i] + B[i];
}

int main() {
    const int N = 1024;
    const size_t size = N * sizeof(float);

    // host端分配内存并初始化
    float *h_A = new float[N], *h_B = new float[N], *h_C = new float[N];
    for (int i=0; i<N; i++) {
        h_A[i] = i;
        h_B[i] = i*2;
    }

    // device端分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 数据从host拷贝到device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 启动核函数
    dim3 blockDim(1024); // 每个block有1024个thread，能处理1024个元素
    dim3 gridDim((N+blockDim.x-1)/blockDim.x); // 向下取整

    vectorAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // 结果从device拷贝到host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 验证结果，释放内存
    for (int i=0; i<10; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << "\n";
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}