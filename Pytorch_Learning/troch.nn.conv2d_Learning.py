import torch
from torch import nn
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../Dataset-Mechine_Learning", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Imgnn(nn.Module):
    def __init__(self):
        super(Imgnn, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


imgnn = Imgnn()

writer = SummaryWriter("../conv_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    outputs = imgnn(imgs)
    print("原型：", imgs.shape)
    # torch.Size([64,3,32,32])
    print("卷积后", outputs.shape)
    # torch.Size([64,6,30,30])
    outputs = torch.reshape(outputs, (-1, 3, 30, 30))
    # -1为占位符，让其自动计算维度大小
    writer.add_images("inputs", imgs, global_step=step)
    writer.add_images("outputs", outputs, global_step=step)
    step += 1
