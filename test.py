import argparse
from model import Generator
import torch
from tqdm import tqdm
from process_dataset import testPreprocessDataset
from torch.utils.data import DataLoader
from math import log10
from ssim import SSIM
import pandas as pd
import os


parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--test_dataset', default="./datasets/DIV2K_valid_HR", type=str,
                    help='测试集的图片路径')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='用于指定超分辨率的放大因子，默认为4')
parser.add_argument('--model_checkpoint', default='./save_checkpoint/netG_epoch_4_100.pth', type=str,
                    help='模型参数')


if __name__ == '__main__':
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # 加载训练好的模型参数
    model = Generator(args.upscale_factor).eval().to(device)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    Ssim = SSIM()
    # 加载测试数据集
    test_dataset = testPreprocessDataset(args.test_dataset, args.upscale_factor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

    # 创建一个用于 test_loader 的 tqdm 进度条
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    # 保存每个测试数据集的结果
    results = {'psnr': [], 'ssim': []}

    num_img = len(test_dataset)

    total_psnr = 0
    total_ssim = 0

    for test_lr, test_hr in test_bar:
        test_lr, test_hr = test_lr.to(device), test_hr.to(device)
        # 生成超分变率图像
        test_fake = model(test_lr)
        mse = ((test_hr - test_fake) ** 2).data.mean()
        # 计算峰值信噪比（Peak Signal-to-Noise Ratio）
        psnr = 10 * log10(255 ** 2 / mse)
        # 计算结构相似性指数（Structural Similarity Index）
        ssim = Ssim(test_fake, test_hr).item()
        #
        results['psnr'].append(psnr)
        results['ssim'].append(ssim)
        #
        total_psnr += psnr
        total_ssim += ssim
    #每张图片的平均性能指标
    avg_psnr = total_psnr/num_img
    avg_ssim = total_ssim/num_img

    data_frame = pd.DataFrame(data={'PSNR': results['psnr'], 'SSIM': results['ssim']},
                              index=range(1, num_img + 1))

    # 在DataFrame的底部添加一行，仅包含平均的PSNR和SSIM值
    avg_data_frame = pd.DataFrame(data={'PSNR': [avg_psnr], 'SSIM': [avg_ssim]},
                                  index=["Average"])

    # 将平均值的DataFrame追加至原来的DataFrame
    final_data_frame = pd.concat([data_frame, avg_data_frame])
    # 将DataFrame对象保存为CSV文件

    out_path = './statistics'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    final_data_frame.to_csv(out_path + '/test_results.csv', index_label='Image')
