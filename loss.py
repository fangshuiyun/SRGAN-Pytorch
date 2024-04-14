import torch
from torch import nn
from torchvision.models.vgg import vgg16

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        # 使用预训练的 VGG16 模型来构建特征提取网络
        vgg = vgg16(pretrained=True)
        # 选择 VGG16 模型的前 31 层作为损失网络，并将其设置为评估模式（不进行梯度更新）
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # 冻结其参数，不进行梯度更新
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        # 定义均方误差损失函数： 计算生成器生成图像与目标图像之间的均方误差损失
        self.mse_loss = nn.MSELoss()
        # 定义总变差损失函数： 计算生成器生成图像的总变差损失，用于平滑生成的图像
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss（对抗损失）：使生成的图像更接近真实图像，目标是最小化生成器对图像的判别结果的平均值与 1 的差距
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss（感知损失）：计算生成图像和目标图像在特征提取网络中提取的特征之间的均方误差损失
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss（图像损失）：计算生成图像和目标图像之间的均方误差损失
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss（总变差损失）：计算生成图像的总变差损失，用于平滑生成的图像
        tv_loss = self.tv_loss(out_images)
        # 返回生成器的总损失，四个损失项加权求和
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        # 计算水平方向上的总变差损失
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        # 计算垂直方向上的总变差损失
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        # 返回总变差损失
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        # 返回张量的尺寸大小，即通道数乘以高度乘以宽度
        return t.size()[1] * t.size()[2] * t.size()[3]


