import torch, os
import numpy as np
from MiniImagenet import MiniImagenet  # 导入MiniImagenet类处理数据集
import scipy.stats
from torch.utils.data import DataLoader  # PyTorch中用于高效数据加载的DataLoader
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse

from meta import Meta  # 导入Meta模型

def mean_confidence_interval(accs, confidence=0.95):
    # 计算准确率的均值和置信区间
    n = len(accs)
    m, se = np.mean(accs), scipy.stats.sem(accs)  # 均值和标准误差
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)  # 置信区间
    return m, h

def main():
    # 主函数，用于训练元学习模型
    torch.manual_seed(222)  # 设置PyTorch的随机种子
    torch.cuda.manual_seed_all(222)  # 设置CUDA的随机种子
    np.random.seed(222)  # 设置NumPy的随机种子

    print(args)  # 打印参数

    # 配置用于元学习的卷积神经网络结构
    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda')  # 设置设备为CUDA (GPU)
    maml = Meta(args, config).to(device)  # 初始化Meta模型

    tmp = filter(lambda x: x.requires_grad, maml.parameters())  # 过滤出可训练的参数
    num = sum(map(lambda x: np.prod(x.shape), tmp))  # 计算可训练参数的总数
    print(maml)  # 打印模型架构
    print('Total trainable tensors:', num)

    # 加载MiniImagenet数据集
    mini = MiniImagenet(r'mini-imagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)  # 训练集
    mini_test = MiniImagenet(r'mini-imagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)  # 测试集

    for epoch in range(args.epoch // 10000):
        # 训练循环
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=0, pin_memory=True)  # 训练数据的DataLoader

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            # 遍历训练数据集
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)  # 训练模型

            if step % 30 == 0:
                print('step:', step, '\\ttraining acc:', accs)  # 打印训练准确率

            if step % 500 == 0:  # 评估
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)  # 测试数据的DataLoader
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # 迭代次数
    argparser.add_argument('--epoch', type=int, help='epoch number', default=10000)
    # 每个任务的类别数
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    # 每类的样本数
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    # 查询集的样本数
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    # 图片尺寸大小
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    # 图片通道数
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
