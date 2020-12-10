# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.graphunet import GraphUNet, GraphNet, GraphUHandNet
from models.resnet import resnet50, resnet10
from models.hourglass import Net_HM_HG


class HopeNet(nn.Module):

    def __init__(self):
        super(HopeNet, self).__init__()
        # self.resnet = resnet50(pretrained=True, num_classes=21*2)
        # self.graphnet = GraphNet(in_features=2050, out_features=2)
        # self.graphunet = GraphUNet(in_features=2, out_features=3)

        self.hourglass = Net_HM_HG(21)
        # self.graphnet = GraphNet(in_features=56*56, out_features=2)
        # self.graphnet = GraphNet(in_features=512+2, out_features=2)
        self.graphnet = GraphNet(in_features=32*32+512, out_features=2)
        # self.graphunet = GraphUNet(in_features=2, out_features=3)
        self.graph_uhand_net = GraphUHandNet(in_features=2, out_features=3)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) #64*64 -> 32*32

    def forward(self, x, freeze_hg=True):
    # def forward(self, x):
        # points2D_init, features = self.resnet(x)
        # features = features.unsqueeze(1).repeat(1, 21, 1)
        # in_features = torch.cat([points2D_init, features], dim=2)
        # points2D = self.graphnet(in_features)
        # points3D = self.graphunet(points2D)
        # return points2D_init, points2D, points3D


        # heatmaps, _ = self.hourglass(x) # (256,256) -> (64,64)
        # points2D_init, heatmaps, features = self.hourglass(x)
        heatmaps, features = self.hourglass(x)
        features = features.unsqueeze(1).repeat(1, 21, 1)
        hm = self.maxpool(heatmaps[-1])
        hm_flatten = torch.flatten(hm, start_dim=2) # (B,21,32,32) -> (B, 21, 32*32)
        # hm_features = heatmaps[-1]
        # in_features = self.conv(hm_features)
        in_features = torch.cat([hm_flatten, features], dim=2)
        if freeze_hg:
            in_features = in_features.detach()
        points2D = self.graphnet(in_features)
        points3D = self.graph_uhand_net(points2D)
        return heatmaps, points2D, points3D
        # return points2D_init, heatmaps, points2D, points3D
