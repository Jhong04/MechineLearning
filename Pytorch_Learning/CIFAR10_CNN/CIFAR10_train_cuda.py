import torch.optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import time
from model import *
from torch.utils.data import DataLoader

writer = SummaryWriter("../logs_CIFAR10NN")

train_data = torchvision.datasets.CIFAR10("../Dataset-Mechine_Learning", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10("../Dataset-Mechine_Learning", train=False, download=False,
                                         transform=torchvision.transforms.ToTensor())

# 加入cuda，可以用于模型，损失函数，数据

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集长度：{}".format(train_data_size))
print("测试集长度：{}".format(test_data_size))

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 训练模型
Cnn = CIFAR10NN()
loss_fn = nn.CrossEntropyLoss()
print(torch.cuda.is_available())
if torch.cuda.is_available():
    Cnn = Cnn.cuda()
    loss_fn.cuda()
learning_rate = 0.01
optimizer = torch.optim.SGD(Cnn.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 50

time_start = time.time()

for i in range(epoch):
    for data in test_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        output = Cnn(imgs)
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            time_end = time.time()
            print("训练100步时间：{}".format(time_start - time_end))
            print("训练次数：{},loss: {}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 边训练边测试
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            output = Cnn(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上LOSS：{}".format(total_test_loss))
    print("整体测试集上正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1
    # 保存每轮的训练模型
    torch.save(Cnn, "GPU_model/Cnn_{}.pth".format(i))

writer.close()
