import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def draw_train(path):
    """
    :param path: train-csv路径
    :return: null
    """
    # 读取CSV文件
    data = pd.read_csv(path)

    # 设置图像大小
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # 绘制Loss_D和Loss_G
    axs[0, 0].plot(data['Epoch'], data['Loss_D'], label='Loss_D')
    axs[0, 0].plot(data['Epoch'], data['Loss_G'], label='Loss_G')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Loss_D and Loss_G')
    axs[0, 0].legend()

    # 绘制Score_D和Score_G
    axs[0, 1].plot(data['Epoch'], data['Score_D'], label='Score_D')
    axs[0, 1].plot(data['Epoch'], data['Score_G'], label='Score_G')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Score')
    axs[0, 1].set_title('Score_D and Score_G')
    axs[0, 1].legend()

    # 绘制PSNR
    axs[1, 0].plot(data['Epoch'], data['PSNR'], label='PSNR')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('PSNR')
    axs[1, 0].set_title('PSNR')
    axs[1, 0].legend()

    # 绘制SSIM
    axs[1, 1].plot(data['Epoch'], data['SSIM'], label='SSIM')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('SSIM')
    axs[1, 1].set_title('SSIM')
    axs[1, 1].legend()

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.5)

    # 保存图像
    plt.savefig('./image/train_results.png', dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()

def draw_test(path):
    """
    :param path: test-csv路径
    :return: null
    """
    # 读取CSV文件
    data = pd.read_csv(path)

    # 剔除Average行
    data = data[data['Image'] != 'Average']

    # 重置索引
    data = data.reset_index(drop=True)

    # 创建一个新的图像
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # 绘制PSNR
    axs[0].plot(data['Image'], data['PSNR'], color='y', label='PSNR')
    axs[0].set_xlabel('Image')
    axs[0].set_ylabel('PSNR')
    axs[0].set_title('PSNR')
    axs[0].set_xticks(np.arange(len(data))[::5])  # 每隔一个标记显示一次
    axs[0].legend()

    # 绘制SSIM
    axs[1].plot(data['Image'], data['SSIM'], color='b', label='SSIM')
    axs[1].set_xlabel('Image')
    axs[1].set_ylabel('SSIM')
    axs[1].set_title('SSIM')
    axs[1].set_xticks(np.arange(len(data))[::5])  # 每隔一个标记显示一次
    axs[1].legend()

    # 保存图像
    plt.savefig('./image/test_results.png', dpi=300, bbox_inches='tight')

    plt.show()

draw_train("./statistics/train_results.csv")
draw_test("./statistics/test_results.csv")