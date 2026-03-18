#include <torch/extension.h>

__global__ void square_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = input[idx] * input[idx];
    }
}

void square_cuda(torch::Tensor input, torch::Tensor output) {
    int size = input.numel();
    int threads = 1024;
    int blocks = (size + threads-1) / threads;
    square_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
}
