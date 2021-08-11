import torch
import os
import numpy as np
from models.yolo import Yolo
from tqdm import tqdm
import argparse
import cv2
import torchvision
from utils.decoce import decode_output

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--img_path', type=str)
        self.parser.add_argument('--weight', type=str)
    
    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        return opt
    
    def init(self, args=''):
        opt = self.parse(args)
        return opt

def preprocess(img_path):
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                dtype=np.float32).reshape(1, 1, 3)
    origin = cv2.imread(img_path)
    img = origin.copy()
    img = (img.astype(np.float32) / 255.)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, ...]
    img = np.ascontiguousarray(img, dtype=np.float32)

    return torch.from_numpy(img), origin

def post_process(dets, num_classes, conf_thre, nms_thre):
    box_corner = dets.new(dets.shape)
    box_corner[:, :, 0] = dets[:, :, 0] - dets[:, :, 2] / 2
    box_corner[:, :, 1] = dets[:, :, 1] - dets[:, :, 3] / 2
    box_corner[:, :, 2] = dets[:, :, 0] + dets[:, :, 2] / 2
    box_corner[:, :, 3] = dets[:, :, 1] + dets[:, :, 3] / 2
    dets[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in range(len(dets))]
    for i, image_pred in enumerate(dets):
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
        
        return output

if __name__ == "__main__":
    opt = opts().init()
    opt.num_classes = 13
    opt.device = torch.device('cuda')
    state_dict = torch.load(opt.weight, map_location=opt.device)
    anchors = state_dict['anchors']
    model = Yolo(opt, 21, anchors, opt.num_classes).to(opt.device)
    model.load_state_dict(state_dict['model'])
    model.eval()

    inp, origin = preprocess(opt.img_path)
    with torch.no_grad():
        outputs = model(inp.to(opt.device))

    dets = decode_output(outputs, anchors, model.strides)
    dets = post_process(dets, opt.num_classes, 0.3, 0.3)
    for det in dets[0]:
        det = det.cpu().numpy()
        bbox = [int(round(i)) for i in det[:4]]
        cv2.rectangle(origin, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.imshow("img", origin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        