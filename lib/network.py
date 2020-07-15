import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import config as cfg


class Network(nn.Module):
    def __init__(self, mode='stn'):
        assert mode in ['stn', 'cnn']

        super(Network, self).__init__()
        self.mode = mode
        self.local_net = LocalNetwork()

        self.conv = nn.Sequential(  # input = 1*40*40
            nn.Conv2d(in_channels=cfg.channel, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=cfg.height // 4 * cfg.width // 4 * 16, out_features=1024),
            nn.ReLU(),
            nn.Dropout(cfg.drop_prob),
            nn.Linear(in_features=1024, out_features=10)
        )

    def forward(self, img):
        '''
        :param img: (b, c, h, w)
        :return: (b, c, h, w), (b,)
        '''
        batch_size = img.size(0)
        if self.mode == 'stn':
            transform_img = self.local_net(img)
            img = transform_img  # 矫正图像后 再传给分类器；训练模型将 input 转化为适合 cnn 判断的图片
        else:
            transform_img = None

        conv_output = self.conv(img).view(batch_size, -1)
        predict = self.fc(conv_output)
        return transform_img, predict


class LocalNetwork(nn.Module):
    def __init__(self):
        super(LocalNetwork, self).__init__()
        self.fc = nn.Sequential(  # whole img input, FC
            nn.Linear(in_features=cfg.channel * cfg.height * cfg.width, out_features=20),
            nn.Tanh(),  # [-1, 1]
            nn.Dropout(cfg.drop_prob),
            nn.Linear(in_features=20, out_features=6),  # affine transformation parameters
            nn.Tanh(),
        )
        nn.init.constant_(self.fc[3].weight, 0)
        self.fc[3].bias.data.copy_(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0])))
        # reshape 为 affine 矩阵后; 单位矩阵，identical transformation 没有发生变化
        # [[1, 0, 0]  # 旋转矩阵 属于 正交矩阵，向量之间正交关系不变，坐标系转换
        #  [0, 1, 0]]

    def forward(self, img):
        '''

        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)

        theta = self.fc(img.view(batch_size, -1)).view(batch_size, 2, 3)
        # theta 限制范围为 [-1, 1.]
        # 平移变换：不会移出 1/4 边界
        # 旋转变换：

        # torch1.4 后，默认将 align_corners=False, 与 interpolate 函数一致
        grid = F.affine_grid(theta, size=[batch_size, cfg.channel, cfg.height, cfg.width])  # torch.Size([1, 40, 40, 2])
        img_transform = F.grid_sample(img, grid)

        return img_transform


if __name__ == '__main__':
    net = LocalNetwork()

    x = torch.randn(1, 1, 40, 40) + 1
    net(x)
