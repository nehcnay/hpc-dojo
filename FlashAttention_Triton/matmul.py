import torch
import triton
import triton.language as tl


# A[M,K] x B[K,N] = C[M,N]
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    A_m_interval, A_n_interval,
    B_m_interval, B_n_interval,
    C_m_interval, C_n_interval,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    result = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    for k in range(0, K, BLOCK_SIZE_K):
        offset_k = k + tl.arange(0, BLOCK_SIZE_K)

        a_ptr = A_ptr + offset_m[:, None] * A_m_interval + offset_k[None, :] * A_n_interval
        b_ptr = B_ptr + offset_k[:, None] * B_m_interval + offset_n[None, :] * B_n_interval

        a = tl.load(a_ptr, mask=offset_k[None, :] < K, other=0.0)
        b = tl.load(b_ptr, mask=offset_k[:, None] < K, other=0.0)

        result += tl.dot(a, b)
        





    
