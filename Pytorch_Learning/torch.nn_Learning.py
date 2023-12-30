import torch
import torch.nn as nn
import torch.nn.functional as f


class TestNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = input + 1
        return output


testNN = TestNN()
x = torch.tensor(1.0)
output = testNN(x)
print(output)
print("hello git")
