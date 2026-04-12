# 通过Triton实现FlashAttention的简化版（Softmax+MatMul），减少显存访问（HBM）提高性能

## 传统Attention
Attention(Q, K, V) = softmax(QK^T/√d) · V

在GPU上计算，多次读写显存（HBM）：
1. 计算 S = QK^T → 写入 HBM
2. 计算 P = softmax(S) → 写入 HBM
3. 计算 O = PV → 写入 HBM

读写显存太频繁

## FlashAttention
分块加载到SRAM，在SRAM片上完成所有计算，只写最终结果O回HBM
