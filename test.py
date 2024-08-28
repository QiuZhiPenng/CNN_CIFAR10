import os.path

import torch
import torchvision
from PIL import Image
from model import ConvModel

# 1.加载图片数据
img_path = "./images/cat.png"
img = Image.open(img_path)
img = img.convert("RGB")
print(img)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((32, 32))
])

img = transform(img)
print(img.shape)

img = torch.reshape(img, (1, 3, 32, 32))
print(img.shape)

# 2.加载预训练模型
model = ConvModel()
model.load_state_dict(torch.load("conv_model_params.pth"))

# 3.使用模型预测
with torch.no_grad():
    output = model(img)

print(output)
print(output.argmax(1).item())

class_to_idx = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print(f"图片为：{os.path.basename(img_path)}")
print(f"对应的分类为：{class_to_idx[output.argmax(1).item()]}")
