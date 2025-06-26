# import os
# # os.environ["TORCH_LOGS"] = "+dynamo"
# os.environ["TORCH_COMPILE_DEBUG"]="1"
# os.environ["TORCHDYNAMO_VERBOSE"]="1"


# import torch

# @torch.compile(mode='max-autotune')
# def my_function(x, y):
#     return torch.bmm(x, x)

# # Test
# x = torch.randn(2, 3, 3)
# y = torch.randn(2, 3, 3)

# result = my_function(x, y)


import torch

class TestModel(torch.nn.Module):
    def forward(self, x, shape_params):
        return torch.ops.aten.view.default(x, shape_params)

x = torch.randn(24)
shape_params = [
    torch.tensor(2, dtype=torch.int32),
    torch.tensor(3, dtype=torch.int32),
    torch.tensor(4, dtype=torch.int32),
]

model = TestModel() 
model = torch.compile(model, backend="eager")
output = model(x, shape_params)  