"""
https://www.jianshu.com/p/723af68beb2e
see spatial transformations
"""

import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

img = Image.open('../transform_img/cat.jpg')
img = transforms.ToTensor()(img)


def plt_torch_img(img):
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.show()


def py_trans(img):
    theta = np.array([  # 仿射变换矩阵
        [1, 0, 80],  # x
        [0, 1, 100]  # y
    ])
    R = theta[:, [0, 1]]  # 旋转
    T = theta[:, [2]]  # 平移 (2, 1)，加上 [2] shape 增加一维

    new_img = torch.zeros_like(img)
    _, h, w = new_img.shape

    for x in range(w):
        for y in range(h):
            # input pos
            pos = np.array([[x], [y]])  # 2x1
            # trans pos 转换后的坐标位置
            new_pos = R @ pos + T
            new_x, new_y = new_pos[0][0], new_pos[1][0]
            if 0 <= new_x < w and 0 <= new_y < h:
                new_img[:, new_y, new_x] = img[:, y, x]

    plt_torch_img(new_img)


def torch_trans(img):
    # 0.5   放大2倍
    # 2     缩小为1/2，都是自图像中心变换的；解释了平移范围为 [-1,1]
    # 放缩系数为 -1，表示了反射，和水平翻转一致
    theta = torch.tensor([
        [-1., 0, 0],  # - x 向右；值域为 [-2,2]，-1 向右平移一半；
        [0, 1, 0],  # - y 向下
    ])

    # 旋转矩阵
    # angle = 0 * math.pi / 180
    # theta = torch.tensor([
    #     [math.cos(angle), math.sin(-angle), 0],
    #     [math.sin(angle), math.cos(angle), 0],
    # ])
    # grid = F.affine_grid(theta.unsqueeze(0), size=[1, 3, 4, 4])  # 假设图向尺寸 2x2
    # print(grid)  # 对应 [-1, -1/3, 1/3, 1] 4个位置，值大小由放缩因子确定

    # 支持 N 张图片，每张图 采用不同的 trans matrix，同时完成 1个 batch 的转换; 适合 STN 训练过程，batch 图片转换
    # 参数2 size 可以设置 转换后图片的 大小
    grid = F.affine_grid(theta.unsqueeze(0), img.unsqueeze(0).size(), align_corners=True)  # 转换后图像 size
    new_img = F.grid_sample(img.unsqueeze(0), grid, align_corners=True, padding_mode='zeros')[0]

    plt_torch_img(new_img)


torch_trans(img)
