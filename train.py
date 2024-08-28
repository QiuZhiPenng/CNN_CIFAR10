import torch
import torchvision.datasets
from model import ConvModel
from torch import nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('TkAgg')  # 设置为交互式后端
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 1.准备数据集
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean, std=std)
])

# 训练集
CIFAR10_train = torchvision.datasets.CIFAR10(root="./dataset/CIFAR10", train=True, transform=transform, download=True)

# 训练集长度
CIFAR10_train_size = len(CIFAR10_train)
print(f"训练集长度为：{CIFAR10_train_size}")

# 2.加载数据集
CIFAR10_train_dataloder = DataLoader(dataset=CIFAR10_train, batch_size=32)
# print(next(enumerate(CIFAR10_train_dataloder)))


# 3.搭建网络模型
model = ConvModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练的设备：{device}")
model.to(device)

# 4.创建损失函数
loss_fc = nn.CrossEntropyLoss()
loss_fc.to(device)

# 5.创建优化器
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr)

num_epochs = 5
losses = []
# 6.训练网络模型
for epoch in range(num_epochs):
    print(f"------第{epoch+1}轮训练开始------")
    model.train()
    for i, data in enumerate(CIFAR10_train_dataloder):
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fc(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.data.item())

        if (i+1) % 100 == 0:
            print(f"epoch:{epoch+1}/{num_epochs},iter:{i+1}/{CIFAR10_train_size//32},loss:{loss.data.item()}")

plt.xlabel("total_train_step")
plt.ylabel("train_loss")
plt.plot(losses)
plt.show()

# 7.保存模型参数
torch.save(model.state_dict(), "conv_model_params.pth")
print("模型参数已保存")




