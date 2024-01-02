import torch
from torch.nn import L1Loss, MSELoss, CTCLoss

inputs = torch.tensor([1, 2, 3], dtype=float)
targets = torch.tensor([1, 2, 5], dtype=float)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))
# batchsize,channel,weight,height

loss1 = L1Loss()
result1 = loss1(inputs, targets)

loss2 = MSELoss()
result2 = loss2(inputs, targets)

# 交叉熵，多用于Multiclass Classification
loss3 = CTCLoss()
result3 = loss3(inputs, targets)

print(result1)
print(result2)
print(result3)
