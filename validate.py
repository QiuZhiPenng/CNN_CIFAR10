import torch
import torchvision.datasets

from torch.utils.data import DataLoader
from model import ConvModel

# 1.加载数据集
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=mean, std=std)])

# 验证集
CIFAR10_val = torchvision.datasets.CIFAR10(root="./dataset/CIFAR10", train=False, transform=transform, download=True)

CIFAR10_val_size = len(CIFAR10_val)
print(f"验证集长度：{CIFAR10_val_size}")

CIFAR10_val_dataloder = DataLoader(CIFAR10_val, 32)

# 2.加载预训练模型
model = ConvModel()
model.load_state_dict(torch.load("conv_model_params.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备：{device}")
model.to(device)

# 3.模型验证
model.eval()
total_accuracy = 0
with torch.no_grad():
    for data in CIFAR10_val_dataloder:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        predict = model(imgs)
        # 计算准确率
        accuracy = (predict.argmax(1) == targets).sum()
        total_accuracy += accuracy

print(f"accuracy/total : {total_accuracy}/{CIFAR10_val_size}")
print(f"正确率：{total_accuracy / CIFAR10_val_size * 100}%")

# accuracy/total : 7368/10000
# 正确率：73.67999267578125%
