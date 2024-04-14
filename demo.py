import argparse
from model import Generator
import torch
from PIL import Image
import os
from torchvision.transforms import ToTensor, ToPILImage

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='用于指定超分辨率的放大因子，默认为4')
parser.add_argument('--image_path', default='./image/2.jpg', type=str,
                    help='图片路径')
parser.add_argument('--model_checkpoint', default='./save_checkpoint/netG_epoch_4_100.pth', type=str,
                    help='模型参数')

args = parser.parse_args()

device = "cpu"

# 加载训练好的模型参数
model = Generator(args.upscale_factor).eval().to(device)
model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))

image = Image.open(args.image_path)

with torch.no_grad():
    image = ToTensor()(image).unsqueeze(0).to(device)
    print(image.shape)
    out = model(image)
    print(out.shape)

    out_img = ToPILImage()(out[0].data.cpu())

    out_img.show()
    save_path = "./demo_result/"
    file_name = os.path.basename(args.image_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_save_path = save_path+file_name
    out_img.save(img_save_path)
    print("图像已保存到文件夹中。")