import torch
import torch.nn as nn


class conv2d_batchnorm(nn.Module):
    """Khối cơ bản: Conv2D -> BatchNorm -> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class small_basic_block(nn.Module):
    """Khối chia nhỏ bộ lọc tăng tốc độ"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        inner_channels = out_channels // 4
        self.c1 = conv2d_batchnorm(in_channels, inner_channels, kernel_size=(1, 1))
        self.c2 = conv2d_batchnorm(inner_channels, inner_channels, kernel_size=(3, 1), padding=(1, 0))
        self.c3 = conv2d_batchnorm(inner_channels, inner_channels, kernel_size=(1, 3), padding=(0, 1))
        self.c4 = conv2d_batchnorm(inner_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.c4(self.c3(self.c2(self.c1(x))))


class LPRNet(nn.Module):
    def __init__(self, num_classes=32):
        super().__init__()
        # 1. Khối Input
        self.block1 = nn.Sequential(
            conv2d_batchnorm(3, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            small_basic_block(64, 128),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=(1, 1))
        )

        # 2. Khối Basic và Conv
        self.block2 = nn.Sequential(
            small_basic_block(128, 256),
            small_basic_block(256, 256),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=(1, 1))
        )

        # 3. Khối Dropout chống học vẹt
        self.dropout1 = nn.Dropout(0.5)
        self.conv_valid = conv2d_batchnorm(256, 256, kernel_size=(4, 1), padding=0)
        self.dropout2 = nn.Dropout(0.5)

        # 4. Khối Head + Global Context
        self.classes_conv = conv2d_batchnorm(256, num_classes, kernel_size=(1, 13), padding=(0, 6))
        self.pattern_conv = nn.Conv2d(num_classes, 128, kernel_size=1)
        self.final_conv = conv2d_batchnorm(num_classes + 128, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        x = self.dropout1(x)
        x = self.conv_valid(x)
        x = self.dropout2(x)

        classes = self.classes_conv(x)
        pattern = self.pattern_conv(classes)

        # Gộp đặc trưng
        x = torch.cat([classes, pattern], dim=1)
        x = self.final_conv(x)

        # 5. Ép dẹt chiều cao (Reduce Mean theo dim=2 tức là Height)
        x = torch.mean(x, dim=2)

        # Chuẩn bị shape (Thời gian, Batch, Classes) cho CTC Loss
        x = x.permute(2, 0, 1)

        return x  # Trả về Logits thô


if __name__ == '__main__':
    model = LPRNet()
    print(model)