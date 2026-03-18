#include <stdio.h>

// 这是一个kernel函数
__global__ void printDim() {
    // 打印位于（0,0,0）的线程所在block的维度和位于（0,0,0)的block所在grid的维度
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) 
        printf("blockDim: (%d, %d. %d)\n", blockDim.x, blockDim.y, blockDim.z);
    
    // 指定grid(0,0,0)位置的block里的block(0,0,0)的线程去打印，这样才只打一次
    if ((blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) && (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0))
        printf("gridDim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

}

int main() {
    printDim<<<dim3(2,1,1), dim3(8,2,1)>>>(); // <<<Grid, Block>>> dim3可以省略，表示y，z为1，只有2/3维才需要显式

    // 等待kernel执行完成
    cudaDeviceSynchronize();
    return 0;
}

