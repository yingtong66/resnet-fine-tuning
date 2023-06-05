import logging
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split
from PIL import Image


logger = logging.getLogger(__name__)

# 定义数据集的路径和类别
txt_path = './data/labels.txt'

# 定义自定义数据集的类，继承Dataset类
class CustomDataset(Dataset):
    # 初始化方法，读取txt文件中的图片路径和label，并保存为列表
    def __init__(self, txt_path, transform=None):
        self.img_paths = [] # 用于存储图片路径的列表
        self.labels = [] # 用于存储label的列表
        self.transform = transform # 用于对图片进行变换的函数
        with open(txt_path, 'r') as f: # 打开txt文件
            for line in f: # 遍历每一行
                img_path, label = line.strip().split() # 分割图片路径和label
                self.img_paths.append(img_path) # 将图片路径添加到列表中
                self.labels.append(int(label)) # 将label转换为整数并添加到列表中
    
    # 返回数据集的大小，即len(dataset)
    def __len__(self):
        return len(self.img_paths)
    
    # 支持索引，即dataset[i]可以返回第i个样本和对应的label
    def __getitem__(self, index):
        img_path = self.img_paths[index] # 根据索引获取图片路径
        label = self.labels[index] # 根据索引获取label
        image = Image.open(img_path) # 用PIL库打开图片
        if self.transform is not None: # 如果有变换函数，则对图片进行变换
            image = self.transform(image)
        return image, label # 返回图片和label


def get_loader(args):
    # 定义数据预处理的变换
    transform = transforms.Compose([
        transforms.Resize(256), # 将图片缩放到256x256
        transforms.CenterCrop(224), # 从中心裁剪出224x224的区域
        transforms.ToTensor(), # 将图片转换为张量
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # 对张量进行归一化
    ])

    # 创建自定义数据集对象，指定txt文件路径和变换函数
    dataset = CustomDataset(txt_path, transform=transform)

    # 划分训练集、验证集和测试集，按照7:2:1的比例，使用torch.utils.data.random_split函数
    train_size = int(0.7 * len(dataset)) # 训练集大小
    eval_size = int(0.2 * len(dataset)) # 验证集大小
    test_size = len(dataset) - train_size - eval_size # 测试集大小
    train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size]) # 划分数据集

    train_sampler = RandomSampler(train_dataset)  # 随机采样
    eval_sampler = SequentialSampler(eval_dataset)  # 按顺序采样
    test_sampler = SequentialSampler(test_dataset)

    train_loader = DataLoader(train_dataset, 
                            #   sampler=train_sampler,
                              batch_size=args.train_batch_size, 
                              shuffle=True) # 训练集加载器
    val_loader = DataLoader(eval_dataset, 
                            # sampler=eval_sampler,
                            batch_size=args.eval_batch_size, 
                            shuffle=False) # 验证集加载器
    test_loader = DataLoader(test_dataset, 
                            #  sampler=test_sampler,
                             batch_size=args.eval_batch_size, 
                             shuffle=False) # 测试集加载器

    return train_loader, val_loader, test_loader