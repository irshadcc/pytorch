import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)
        self.linear2 = torch.nn.Linear(10, 5)

    def forward(self, x, use_linear1: bool):
        if use_linear1:
            return self.linear1(x)
        else:
            return self.linear2(x)

model = MyModule()
example_input = torch.randn(1, 10)

# Exporting for use_linear1 = True
exported_model_linear1 = torch.export.export(model, (example_input, True))