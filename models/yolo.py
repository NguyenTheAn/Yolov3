from .fpn import YoloFPN
from .head import YoloHead
import torch.nn as nn
import torch
import os
import numpy as np
from utils.metrics import wh_iou, square_error

class Yolo(nn.Module):
    strides = {
        21 : [8, 16, 32]
    }
    out_channels = {
        21 : [128, 256, 512]
    }
    def __init__(self, opt, depth=21, anchors = [], num_classes = 6):
        super(Yolo, self).__init__()
        self.opt = opt
        
        self.strides = Yolo.strides[depth]
        self.num_anchors = len(anchors[0])
        num_layers = len(Yolo.strides[depth])
        self.num_classes = num_classes

        self.anchors = torch.FloatTensor(anchors).to(self.opt.device)
        for i, stride in enumerate(self.strides):
            self.anchors[i] /= stride

        self.backbone = YoloFPN(depth)
        self.head = YoloHead(opt, num_classes, self.num_anchors, num_layers, Yolo.out_channels[depth])
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return x

class YoloWithLoss(nn.Module):
    def __init__(self, model):
        super(YoloWithLoss, self).__init__()
        self.model = model
    
    def forward(self, x, targets):
        x = self.model(x)

        target_tensors = self.bboxes2tensor(x, targets)
        total_loss, coord_loss, obj_loss, cls_loss = self.compute_loss(x, target_tensors)
        loss = {
            "total_loss": total_loss,
            "coord_loss": coord_loss.item(),
            "obj_loss": obj_loss.item(),
            "cls_loss": cls_loss.item()
        }

        return x, loss

    def bboxes2tensor(self, p, targets):
        num_layers = len(p)
        out = []
        for i in range(num_layers):
            bs, num_anchors, gridy, gridx, num_out = p[i].shape
            target_tensors = []
            for target in targets:
                target_tensor = torch.zeros((1, num_anchors, gridy, gridx, num_out)).to(self.model.opt.device)
                anchors = self.model.anchors[i] * self.model.strides[i]
                for bbox in target:
                    bbox = bbox.astype(np.float)
                    cellx, celly = int(bbox[0] / self.model.strides[i]), int(bbox[1] / self.model.strides[i])
                    bbox[0] = (bbox[0] / self.model.strides[i]) - cellx
                    bbox[1] = (bbox[1] / self.model.strides[i]) - celly
                    w, h = bbox[2], bbox[3]
                    bbox[2] /= self.model.strides[i]
                    bbox[3] /= self.model.strides[i]
                    iou = wh_iou(anchors, torch.tensor([[w, h]]).to(self.model.opt.device))
                    indices = torch.where(iou > self.model.opt.iou_t)[0]
                    if len(indices) == 0:
                        indices = torch.argmax(iou)
                        continue

                    label = np.zeros(self.model.num_classes)
                    label[int(bbox[-1])] = 1

                    target_tensor[:, indices, celly, cellx, :4] = torch.tensor(len(indices) * [bbox[:-1]], dtype=torch.float32).to(self.model.opt.device)
                    target_tensor[:, indices, celly, cellx, 4] = torch.tensor(len(indices) * [1], dtype=torch.float32).to(self.model.opt.device)
                    target_tensor[:, indices, celly, cellx, 5:] = torch.tensor(len(indices) * [label], dtype=torch.float32).to(self.model.opt.device)
                target_tensors.append(target_tensor)
            target_tensors = torch.cat(target_tensors, 0)
            out.append(target_tensors)
        return out

    def compute_loss(self, pred_tensors, target_tensors):
        for pred_tensor, target_tensor, anchor in zip(pred_tensors, target_tensors, self.model.anchors):
            
            pxy = pred_tensor[..., :2].sigmoid()
            pwh = torch.exp(pred_tensor[..., 2:4]) * anchor.view(1, -1, 1, 1, 2)
            pobj = pred_tensor[..., 4:5].sigmoid()
            pcls = pred_tensor[..., 5:].sigmoid()
            
            txy = target_tensor[..., :2]
            twh = target_tensor[..., 2:4]
            tobj = target_tensor[..., 4:5]
            tcls = target_tensor[..., 5:]

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

            iou = intersect_area/(pred_area + true_area - intersect_area)
            max_iou = torch.max(iou, dim=1, keepdim=True)[0]
            best_box_index =  torch.unsqueeze(torch.eq(iou, max_iou).float(), dim = -1)
            true_box_conf = best_box_index*tobj
            
            lambda_coord = 5.0
            lambda_noobj = 0.5

            coord_loss = lambda_coord * true_box_conf * (square_error(pxy, txy) + square_error(torch.sqrt(pwh), torch.sqrt(twh)))
            obj_loss = true_box_conf * square_error(pobj, tobj) + lambda_noobj * (1 - true_box_conf) * square_error(pobj, tobj)
            cls_loss = true_box_conf * square_error(pcls, tcls)

            # NOOB_W, CONF_W, XY_W, PROB_W, WH_W = 2.0, 10.0, 0.5, 1.0, 0.1
            # xy_loss =  (square_error(pxy, txy)*true_box_conf*XY_W).sum()
            # wh_loss =  (square_error(pwh, twh)*true_box_conf*WH_W).sum()
            # obj_loss = (square_error(pobj, tobj)*(CONF_W*true_box_conf + NOOB_W*(1-true_box_conf))).sum()
            # cls_loss = (square_error(pcls, tcls)*true_box_conf*PROB_W).sum()            
            # total_loss = xy_loss + wh_loss + obj_loss + cls_loss

            coord_loss = coord_loss.sum()
            obj_loss = obj_loss.sum()
            cls_loss = cls_loss.sum()

            total_loss = coord_loss + obj_loss + cls_loss
            return total_loss, coord_loss, obj_loss, cls_loss