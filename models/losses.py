from typing import IO
import torch.nn as nn
import numpy as np
import torch

class IOULoss(nn.Module):
    def __init__(self, losstype, reduction):
        super(IOULoss, self).__init__()
        self.eps = 1e-8
        self.loss_type = losstype
        self.reduction = reduction

    def forward(self, pxy, pwh, txy, twh):

        pred_tl = pxy - 0.5*pwh
        pred_br = pxy + 0.5*pwh
        pred_area = pwh[...,0]*pwh[...,1]

        true_tl = txy - 0.5*twh
        true_br = txy + 0.5*twh
        true_area = twh[...,0]*twh[...,1]

        intersect_tl = torch.max(pred_tl, true_tl)
        intersect_br = torch.min(pred_br, true_br)
        intersect_wh = intersect_br - intersect_tl
        intersect_area = intersect_wh[...,0]*intersect_wh[...,1]

        iou = intersect_area/(pred_area + true_area - intersect_area + self.eps)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pxy - pwh / 2), (txy - twh / 2)
            )
            c_br = torch.max(
                (pxy + pwh / 2), (txy + twh / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - intersect_area) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        
        return loss