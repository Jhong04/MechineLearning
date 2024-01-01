import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn

dataset = torchvision.datasets.CIFAR10("../Dataset-Mechine_Learning", train=False, download=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class LinearLayer(nn.Module):
    def __init__(self):
        super(LinearLayer, self).__init__()
        self.linear = LinearLayer(196608, 10)

    def forward(self, input):
        output = self.linear(input)
        return output


linear_layer = LinearLayer()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    # batchSize,channel,weight,height
    # ↓等价于这个摊平
    output = torch.flatten(imgs)
    print(output.shape)
    output = linear_layer(output)
    print(output.shape)
