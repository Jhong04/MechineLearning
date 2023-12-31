import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d

input = torch.tensor([[1, 2, 0, 3, 1], [0, 1, 2, 3, 1], [1, 2, 1, 0, 0], [5, 2, 3, 1, 1], [2, 1, 0, 1, 1]],
                     dtype=torch.float)
input = torch.reshape(input, (-1, 1, 5, 5))


# 池化的作用：进行特征降维，提升训练收敛速度

class Pmaxnn(nn.Module):
    def __init__(self):
        super(Pmaxnn, self).__init__()
        # ceil_mode默认为0,当为1时卷积核超出input范围会保留在范围内的进行计算
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


pmaxnn = Pmaxnn()

output = pmaxnn(input)
print(output)
