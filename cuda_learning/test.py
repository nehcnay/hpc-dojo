import torch
import tensor_square

x = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')

result = tensor_square.square(x)

print(result)
