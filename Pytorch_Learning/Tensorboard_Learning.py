from torch.utils.tensorboard import SummaryWriter
import cv2

# 主要是展示训练变化的结果

# 创建一个logs文件
writer = SummaryWriter("logs")

# writer.add_image() 支持的数据类型为numpy和tensor
# writer.add_scalar()

# ----------------------------------------------------------------------
# for i in range(100):
#     writer.add_scalar("y=2x", 2 * i, i)


# writer.close()


# -----------------------------------------------------------------------

img_path = "../Dataset-Mechine_Learning/hymenoptera/train/ants/0013035.jpg"
img = cv2.imread(img_path)
print(type(img))
writer.add_image("test", img, 1, dataformats='HWC')
writer.close()
