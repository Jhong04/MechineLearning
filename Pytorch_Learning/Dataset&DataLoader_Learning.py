from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
import os
import torchvision


# 读取数据
# dataset获取所有数据,dataLoader将其打包为batch


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 获取目录下的文件名列表
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = cv2.imread(img_item_path)
        # 这个数据集中，标签名就是目录名
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "../Dataset-Mechine_Learning/hymenoptera/train"
ants_label_dir = "ants"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_label_dir = "bees"
bees_dataset = MyData(root_dir, bees_label_dir)

# img, label = ants_dataset[1]
# cv2.imshow("img", img)
# cv2.waitKey(0)

# img, label = bees_dataset[1]
# cv2.imshow("img", img)
# cv2.waitKey(0)

train_dataset = ants_dataset + bees_dataset

# ---------------------------------------------------
# DataLoader
test_data = torchvision.datasets.CIFAR10("../Dataset-Mechine_Learning", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
# 每次从dataset中取4个数据进行打包
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# img, target = test_data[0]
# print(img.shape)
# print(target)

writer = SummaryWriter("dataloader_logs")
step = 0
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("test_data", imgs, global_step=step)
    step += 1

writer.close()
