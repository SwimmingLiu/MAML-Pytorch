import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random

class MiniImagenet(Dataset):
    # MiniImagenet类：处理MiniImagenet数据集，用于元学习和少样本学习任务。

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        # 初始化函数。
        # root: 数据集根目录；mode: 模式（'train', 'val', 'test'）；
        # batchsz: 批次大小；n_way: 每个任务的类别数；
        # k_shot: 每类的样本数；k_query: 查询集的样本数；
        # resize: 图像调整大小；startidx: 标签起始索引。

        self.batchsz = batchsz  # 批次大小
        self.n_way = n_way  # 每个任务的类别数
        self.k_shot = k_shot  # 每类的样本数
        self.k_query = k_query  # 查询集的样本数
        self.setsz = self.n_way * self.k_shot  # 支持集的总样本数
        self.querysz = self.n_way * self.k_query  # 查询集的总样本数
        self.resize = resize  # 图像调整大小
        self.startidx = startidx  # 标签起始索引

        # 根据模式设置图像变换
        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.path = os.path.join(root, 'images')  # 图像路径
        csvdata = self.loadCSV(os.path.join(root, mode + '.csv'))  # 加载CSV数据
        self.data = []  # 数据列表
        self.img2label = {}  # 图像到标签的映射
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # 图像文件名列表
            self.img2label[k] = i + self.startidx  # 创建图像名到标签的映射

        self.cls_num = len(self.data)  # 类别数量
        self.create_batch(self.batchsz)  # 创建批次

    def loadCSV(self, csvf):
        # 加载CSV文件，返回标签到图像文件名的映射。
        # csvf: CSV文件路径。
        # 返回：标签到图像文件名列表的字典。
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # 跳过表头
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def create_batch(self, batchsz):
        # 为元学习创建批次。
        # batchsz: 批次大小。
        self.support_x_batch = []  # 支持集批次
        self.query_x_batch = []  # 查询集批次
        for b in range(batchsz):  # 每个批次
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # 随机选择n_way类别
            np.random.shuffle(selected_cls)
            support_x = []  # 支持集
            query_x = []  # 查询集
            for cls in selected_cls:
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # 选择用于Dtrain的索引
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # 选择用于Dtest的索引
                support_x.append(np.array(self.data[cls])[indexDtrain].tolist())  # 获取当前Dtrain的所有图像文件名
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())  # 获取当前Dtest的所有图像文件名

            # 打乱支持集和查询集之间的对应关系
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # 将当前集添加到支持集批次
            self.query_x_batch.append(query_x)  # 将当前集添加到查询集批次

    def __getitem__(self, index):
        # 根据索引获取数据批次。
        # :param index: 批次的索引
        # :return: 支持集和查询集（图像和标签）

        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)  # 支持集图像
        support_y = np.zeros((self.setsz), dtype=np.int)  # 支持集标签
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)  # 查询集图像
        query_y = np.zeros((self.querysz), dtype=np.int)  # 查询集标签

        flatten_support_x = [os.path.join(self.path, item) for sublist in self.support_x_batch[index] for item in
                             sublist]
        support_y = np.array(
            [self.img2label[item[:9]] for sublist in self.support_x_batch[index] for item in sublist]).astype(
            np.int32)
        flatten_query_x = [os.path.join(self.path, item) for sublist in self.query_x_batch[index] for item in
                           sublist]
        query_y = np.array(
            [self.img2label[item[:9]] for sublist in self.query_x_batch[index] for item in sublist]).astype(
            np.int32)

        unique = np.unique(support_y)  # 支持集中的唯一标签
        random.shuffle(unique)  # 打乱标签
        support_y_relative = np.zeros(self.setsz)  # 支持集的相对标签
        query_y_relative = np.zeros(self.querysz)  # 查询集的相对标签
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx  # 分配相对标签
            query_y_relative[query_y == l] = idx  # 分配相对标签

        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)  # 转换支持集图像

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)  # 转换查询集图像

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        # 返回数据集中的批次总数
        return self.batchsz

if __name__ == '__main__':
    # 测试代码，通过可视化查看一组图像。

    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time

    plt.ion()  # 交互模式

    tb = SummaryWriter('runs', 'mini-imagenet')
    mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=5, k_shot=1, k_query=1, batchsz=1000,
                        resize=168)

    for i, set_ in enumerate(mini):
        support_x, support_y, query_x, query_y = set_  # 提取支持集和查询集

        support_x = make_grid(support_x, nrow=2)  # 支持集图像网格
        query_x = make_grid(query_x, nrow=2)  # 查询集图像网格

        plt.figure(1)
        plt.imshow(support_x.transpose(2, 0).numpy())  # 显示支持集图像
        plt.pause(0.5)  # 暂停半秒钟
        plt.figure(2)
        plt.imshow(query_x.transpose(2, 0).numpy())  # 显示查询集图像
        plt.pause(0.5)

        tb.add_image('support_x', support_x)
        tb.add_image('query_x', query_x)

        time.sleep(5)

    tb.close()
