import math
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # 二维卷积层，输入通道数为channels，输出通道数为channels，卷积核大小为3，填充为1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)  # 二维批归一化层
        self.prelu = nn.PReLU()  # Parametric ReLU激活函数
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)  # 二维批归一化层

    def forward(self, x):
        # 应用对应的layer得到前向传播的输出(残差项)
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual  # 将输入x与残差项相加，得到最终输出


# 上采样块
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        # 卷积层，输入通道数为in_channels，输出通道数为in_channels * 2 ** 2，卷积核大小为3，填充为1
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        # 像素重排操作，上采样因子为up_scale
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x



# 生成器模型
class Generator(nn.Module):
    def __init__(self, scale_factor):
        # 计算需要进行上采样的块的数量
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        # 二维卷积层，输入通道数为3，输出通道数为64，卷积核大小为9，填充为4
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()  # Parametric ReLU激活函数
        )
        self.block2 = ResidualBlock(64) # 定义(残差)ResidualBlock模块
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        # 由多个UpsampleBlock模块组成的列表
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)  # 由block8列表中的模块组成的序列模块

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        # 将输出限制在0到1之间，通过tanh激活函数和缩放操作得到最终生成的图像
        return (torch.tanh(block8) + 1) / 2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # 二维卷积层，输入通道数为3，输出通道数为64，卷积核大小为3，填充为1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),  # LeakyReLU激活函数

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # 自适应平均池化层，将输入特征图转换为大小为1x1的特征图
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        # 输入批次的大小
        batch_size = x.size(0)
        # 使用torch.sigmoid函数将特征图映射到0到1之间，表示输入图像为真实图像的概率。
        return torch.sigmoid(self.net(x).view(batch_size))


if __name__ == '__main__':
    input = torch.rand([2, 3, 200, 200])
    model = Generator(4)
    out = model(input)
    print(out.shape)