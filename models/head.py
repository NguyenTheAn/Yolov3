import torch
import torch.nn as nn
import numpy as np

class YoloHead(nn.Module):
    def __init__(self, opt, num_classes = 6, num_anchors = None, num_layers = None, in_channels = [128, 256, 512]):
        super(YoloHead, self).__init__()
        self.opt = opt
        self.num_classes = num_classes
        self.num_out = num_classes + 5
        self.num_anchors = num_anchors
        self.num_layers = num_layers

        self.modules_list = nn.ModuleList(nn.Conv2d(x, self.num_out * self.num_anchors, 1) for x in in_channels)

    def forward(self, x):
        for i in range(self.num_layers):
            x[i] = self.modules_list[i](x[i])
            bs, c, h, w = x[i].shape
            x[i] = x[i].view(bs, self.num_anchors, self.num_out, h, w).permute(0, 1, 3, 4, 2).contiguous()
        return x
