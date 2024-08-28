import torch
from torch import nn


class ConvModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # (3,32,32) -> (16,28,28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        # (16,28,28) -> (16,14,14)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # (16,14,14) -> (32,12,12)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # (32,12,12) -> (64,10,10)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # (64,10,10) -> (64,5,5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(in_features=64*5*5, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        # 展平
        x = x.view(x.size(0), -1)
        # print(x.shape) torch.Size([32, 1600])
        out = self.fc(x)
        return out


if __name__ == '__main__':
    model = ConvModel()
    input = torch.ones((32, 3, 32, 32))
    output = model(input)
    # print(output.shape) torch.Size([32, 10])
