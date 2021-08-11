import torch.utils.data as data
import os
import cv2
import numpy as np
from utils.image import get_affine_transform, affine_transform
import torch

class Chess_Dataset(data.Dataset):
    def __init__(self, opt, split):
        super(Chess_Dataset, self).__init__()
        self.mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.28863828, 0.27408164, 0.27809835],
                    dtype=np.float32).reshape(1, 1, 3)
        self.split = split
        self.opt = opt
        if self.split == "train":
            self.img_dir = "./datasets/data/chess_dataset/train"
            self.label_path = "./datasets/data/chess_dataset/train/_annotations.txt"
        if self.split == "val":
            self.img_dir = "./datasets/data/chess_dataset/valid"
            self.label_path = "./datasets/data/chess_dataset/valid/_annotations.txt"
        if self.split == "test":
            self.img_dir = "./datasets/data/chess_dataset/test"
            self.label_path = "./datasets/data/chess_dataset/test/_annotations.txt"

        self.bboxes = dict()
        self.img_names = []
        with open(self.label_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split(" ")
                name = line[0]
                bboxes = []
                for box in line[1:]:
                    x1,y1,x2,y2,class_id = list(map(int, box.split(",")))[:]
                    xc, yc = (x1+x2)/2, (y1+y2)/2
                    w, h = x2-x1, y2-y1
                    bboxes.append(np.array([xc,yc,w,h,class_id]))
                self.bboxes[name] = np.array(bboxes)
                self.img_names.append(name)
    def __len__(self):
        return len(self.img_names)
    
    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_names[index])
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

        if self.opt.keep_res:
            input_h = (height | 31) + 1
            input_w = (width | 31) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_w, input_h = self.opt.input_w, self.opt.input_h

        flipped = False
        if self.split == "train":
            if np.random.random() < self.opt.rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(
                    low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(
                    low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1
        
        trans_input = get_affine_transform(
                c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                            (input_w, input_h),
                            flags=cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        trans_output = get_affine_transform(c, s, 0, [input_w, input_h])

        bboxes = self.bboxes[self.img_names[index]]
        new_bboxes = []
        for bbox in bboxes:
            if flipped:
                bbox[0] = width - bbox[0] - 1
            bbox[:2] = affine_transform(bbox, trans_output)
            if bbox[0] < 0 or bbox[0] >= input_w or bbox[1] < 0 or bbox[1] >= input_h:
                continue
            new_bboxes.append(bbox)
            
        #     cv2.rectangle(inp, (int(bbox[0] - 0.5*bbox[2]), int(bbox[1] - 0.5*bbox[3])), (int(bbox[0] + 0.5*bbox[2]), int(bbox[1] + 0.5*bbox[3])), (0, 255, 0), 2)
        # cv2.imshow("img", inp)
        # cv2.waitKey(0)

        return torch.from_numpy(inp), np.array(new_bboxes)

    def collate_fn(self, batch):
        images = list()
        boxes = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])

        images = torch.stack(images, dim=0)

        return images, boxes
    