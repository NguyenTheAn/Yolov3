import os
from typing import Text
import xml.etree.ElementTree as ET
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', default='./')    
parser.add_argument('--output_folder', default='./')    
args = parser.parse_args()

xml_files = glob.glob(args.input_folder + "/*.xml")

for path in xml_files:
    tree = ET.parse(path)
    root = tree.getroot()
    filename = root[1].text.split(".")[0]
    width = int(root[4][0].text)
    height = int(root[4][1].text)
    bboxes = [[int(bbox.text) for bbox in leaf[4][:]] for leaf in root[6:]]

    with open(f"{args.output_folder}/{filename}.txt", 'w') as file:
        for bbox in bboxes:
            x, y, x2, y2 = bbox[:]
            w, h = x2-x, y2-y
            xc, yc = x + w//2, y+h//2
            xc, yc, w, h = xc/width, yc/height, w/width, h/height
            file.write(f"{0} {xc} {yc} {w} {h}\n")