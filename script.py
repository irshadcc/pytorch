import os
import torch

# Define two tensors
x = torch.tensor([1,2], requires_grad=False, dtype=torch.float32)
w = torch.tensor([3,4], requires_grad=True, dtype=torch.float32)

# Add the tensors
result_tensor = torch.add(x, w)
sum_tensor = result_tensor.sum()
sum_tensor.backward()
print(x.grad)
print(w.grad)