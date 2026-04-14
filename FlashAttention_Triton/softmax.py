# Triton 实现 softmax
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,    # 行与行之间的距离
    n_cols, # 列数，在连续存储时input_row_stride=n_cols
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)  # 获取当前program在第0维的id
    row_start_ptr = input_ptr + row_idx * input_row_stride  # 指向当前行的地址
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 计算输入指针
    now_ptrs = row_start_ptr + col_offsets

    # 从显存加载数据到SRAM
    mask = col_offsets < n_cols #防止越界时加载到无效值
    value = tl.load(now_ptrs, mask=mask, other=float('-inf')) # 加载到block中，mask掩码，其他值设为-inf，因为e的负无穷近似为0

    # Softmax计算
    value_minus_max = value - tl.max(value, axis=0) 
    # 这块的求最大值值得深思，涉及到线程束和共享内存，这里的value实际上是一个向量，包含block_size个元素，分解成多个线程束，内部各自求到最大值，共享内存中求最大值
    numerator = tl.exp(value_minus_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    # 获取存放结果的指针位置
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    result_ptr = output_row_start_ptr + col_offsets

    # 将结果写回显存
    tl.store(result_ptr, result, mask=mask)


# pytorch包装函数，方便调用kernel函数
def softmax_triton(x):
    row, col = x.shape

    # 确保输入是连续的
    x = x.contiguous()

    # 创建输出张量
    output = torch.empty_like(x)

    # 计算block大小，向上取整到2的幂次（比如col=100,就取到128），Triton优化更好。BLOCK_SIZE固定了利于优化
    BLOCK_SIZE = triton.next_power_of_2(col)

    softmax_kernel[(row,)]( # [(row,)]指定启动多少个program，一行一个program
        x, output,
        x.stride(0), output.stride(0),
        col,
        BLOCK_SIZE,
    )

    return output



def test_softmax():
    print("Testing softmax ...")

    torch.manual_seed(0)

    x = torch.randn(1823, 781, device='cuda')

    # 如果用pytorch
    output_pytorch  = torch.softmax(x, axis=1)

    # triton
    output_triton = softmax_triton(x)

    print("输入形状", x.shape)
    print("PyTorch output:", output_pytorch)
    print("Triton output:", output_triton)

    max_diff = torch.max(torch.abs(output_pytorch - output_triton))
    print(f"最大差异: {max_diff:.6f}")


if __name__ == "__main__":
    test_softmax()
