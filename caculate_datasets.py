import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ====== 设置部分 ======

# 定义训练集的路径，请将其替换为您的实际路径
train_data_path = '/home/wangdongzhihan/datasets/KiC4R/dataset/sequences/02/image_0/'

# 定义图像尺寸，请根据您的模型输入调整
image_height = 512
image_width = 1392

# 定义批次大小，根据您的显存大小进行调整
batch_size = 64

# 定义用于计算的图像转换（不进行归一化）
transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
])

# ====== 加载数据集 ======

# 使用 ImageFolder 加载训练数据集
train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ====== 计算均值和标准差 ======

# 初始化累加器
mean = torch.zeros(3)
std = torch.zeros(3)
total_images_count = 0

print('开始计算均值和标准差...')

for images, _ in train_loader:
    # 获取当前批次的图像数量
    batch_samples = images.size(0)
    total_images_count += batch_samples

    # 将图像展平成 [batch_size, channels, height * width]
    images = images.view(batch_samples, images.size(1), -1)

    # 计算每个通道的均值和标准差，并累加
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

# 计算总体均值和标准差
mean /= total_images_count
std /= total_images_count

print('计算完成！')
print('均值（Mean）：{}'.format(mean))
print('标准差（Std）：{}'.format(std))

# ====== 在预处理步骤中使用计算得到的均值和标准差 ======

# 定义新的预处理步骤，包含标准化
preprocess = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[mean[0].item(), mean[1].item(), mean[2].item()],
        std=[std[0].item(), std[1].item(), std[2].item()]),
])

# 您可以在训练和验证过程中使用新的 preprocess 进行数据处理