import argparse
import os
import sys

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--dataset', default='fire',
                                 help='chess | fire')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true')

        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')

        self.parser.add_argument('--input_res', type=int, default=-1,
                                 help='input height and width. -1 for default from '
                                 'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1,
                                 help='input width. -1 for default from dataset.')
        self.parser.add_argument('--keep_res', action='store_true')

        self.parser.add_argument('--lr', type=float, default=1e-3,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=int, default=50,
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)
        self.parser.add_argument('--num_epochs', type=int, default=100,
                                 help='total training epochs.')
        self.parser.add_argument('--warming_epochs', type=int, default=2,
                                 help='number of warming epochs.')
        self.parser.add_argument('--batch_size', type=int, default=8,
                                 help='batch size')
        self.parser.add_argument('--num_workers', type=int, default=1,
                                 help='num workers')

        self.parser.add_argument('--rand_crop', type=float, default=0.5)
        self.parser.add_argument('--shift', type=float, default=0.1)
        self.parser.add_argument('--scale', type=float, default=0.1)
        self.parser.add_argument('--flip', type=float, default=0.5)
        self.parser.add_argument('--iou_t', type=float, default=0.3)


        self.parser.add_argument('--output_dir', type=str, default="checkpoints")
    
    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]

        if opt.dataset == "chess":
            opt.num_classes = 13
        if opt.dataset == "fire":
            opt.num_classes = 1

        if opt.input_res != -1:
            opt.input_w = opt.input_res 
            opt.input_h = opt.input_res

        return opt
    
    def init(self, args=''):
        opt = self.parse(args)
        return opt