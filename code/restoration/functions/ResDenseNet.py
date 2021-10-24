#!/usr/bin/python
from __future__ import print_function

import torch
import torch.nn as nn


class ResDenseBlock(nn.Module):
    def __init__(self, num_convs, rdb_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d((i+1) * rdb_channels, rdb_channels, kernel_size=3, stride=1, padding=1) for i in range(num_convs)])
        self.relus = nn.ModuleList([nn.ReLU() for i in range(num_convs)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(rdb_channels) for i in range(num_convs)])
        self.conv1 = nn.Conv2d((num_convs+1) * rdb_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d()

    def forward(self, x):
        y = x.clone().detach()
        for i in range(len(self.convs)):
            out = self.relus[i](self.bns[i](self.convs[i](x)))
            x = torch.cat((x, out), dim=1)

        return y + self.relu1(self.conv1(x))


class ResDenseNet(nn.Module):
    def __init__(self, num_rdbs, num_convs, in_channels, rdb_channels, inter_channels, out_channels):
        super().__init__()
        self.convin = nn.Conv2d(in_channels, rdb_channels, kernel_size=3, stride=1, padding=1)
        self.reluin = nn.ReLU()

        self.rdbs = nn.ModuleList([ResDenseBlock(num_convs, rdb_channels, rdb_channels) for i in range(num_rdbs)])

        self.conv1 = nn.Conv2d(num_rdbs * rdb_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inter_channels)

        self.conv3 = nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.reluin(self.convin(x))
        for i in range(len(self.rdbs)):
            x = self.rdbs[i](x)
            if i == 0:
                y = x.clone().detach()
            else:
                y = torch.cat((y, x), dim=1)

        return self.conv3(self.relu1(self.bn1(self.conv1(y))))