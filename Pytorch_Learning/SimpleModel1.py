import torchvision.datasets
from torch import nn
import torch
from torch.nn import Conv2d, MaxPool2d, Linear, Softmax, Sequential, CrossEntropyLoss, Flatten
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../Dataset-Mechine_Learning", train=True, download=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class NN1(nn.Module):
    def __init__(self):
        super(NN1, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10))
        # Softmax(1))

    def forward(self, x):
        x = self.model1(x)
        return x


my_nn = NN1()
# print(my_nn)

# 损失函数
loss = CrossEntropyLoss()
# 设置优化器,使用随机梯度下降算法
optim = torch.optim.SGD(my_nn.parameters(), lr=0.01)

for epoch in range(50):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = my_nn(imgs)
        result_loss = loss(outputs, targets)
        # 记得梯度清零
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)

# # 保存整个模型
# torch.save(my_nn, "../model/myModel1")
# 只保存参数
torch.save(my_nn.state_dict(), "../model/myModel1_para")

# 模型加载
# 整个模型：
# model=torch.load("../model/myModel1")
# 参数：
# model=NN1()
# model.load_state_dict(torch.load("../model/myModel1_para"))
