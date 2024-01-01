import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../Dataset-Mechine_Learning", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

writer = SummaryWriter("logs_activate")


class Active(nn.Module):
    def __init__(self):
        super(Active, self).__init__()
        # inplace：是否在原变量上进行替换
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        output_sigmoid = self.sigmoid(input)
        return output_sigmoid


# input = torch.tensor([[1, -0.5], [-1, 3]])
# input = torch.reshape(input, (-1, 1, 2, 2))
#
active = Active()
# output = active(input)
# print(output)

step = 0
for data in dataloader:
    imgs, target = data
    writer.add_images("input", imgs, global_step=step)
    output = active(imgs)
    writer.add_images("output", output, global_step=step)
    step += 1
