import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from PIL import Image

transform = transforms.Compose([
            transforms.Resize(400),
            transforms.CenterCrop(400),    #显存不足就缩小图像尺寸
            transforms.ToTensor()        #显存不足就缩小图像尺寸
])

class PreprocessDataset(Dataset):
    """预处理数据集类"""

    def __init__(self, HRPath, scale_factor):
        """初始化预处理数据集类"""
        self.scale_factor = scale_factor
        img_names = os.listdir(HRPath)
        self.HR_imgs = [HRPath + "/" + img_name for img_name in img_names]

    def __len__(self):
        """获取数据长度"""
        return len(self.HR_imgs)

    def __getitem__(self, index):
        """获取数据"""
        HR_img = self.HR_imgs[index]

        HR_img = Image.open(HR_img)

        HR_img = transform(HR_img)
        LR_img = torch.nn.MaxPool2d(self.scale_factor, stride=self.scale_factor)(HR_img)   #将高分辨率下采样4倍，形成低分辨率

        return LR_img, HR_img     #返回低和高分辨率图片


class testPreprocessDataset(Dataset):
    """预处理数测试据集类，不进行Resize操作，进行原图的指标验证"""

    def __init__(self, HRPath, scale_factor):
        """初始化预处理数据集类"""
        self.scale_factor = scale_factor
        img_names = os.listdir(HRPath)
        self.HR_imgs = [HRPath + "/" + img_name for img_name in img_names]

    def __len__(self):
        """获取数据长度"""
        return len(self.HR_imgs)

    def __getitem__(self, index):
        """获取数据"""
        HR_img = self.HR_imgs[index]

        HR_img = Image.open(HR_img)

        HR_img = transforms.ToTensor()(HR_img)
        LR_img = torch.nn.MaxPool2d(self.scale_factor, stride=self.scale_factor)(HR_img)   #将高分辨率下采样4倍，形成低分辨率

        return LR_img, HR_img     #返回低和高分辨率图片

