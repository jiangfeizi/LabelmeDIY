

import os
import json
import math
import sys
sys.path.append(r'D:\Code\project\LabelmeDIY\tools')

import numpy as np
import cv2
from tqdm import tqdm

from utils import is_image

source_dir = r'\\192.168.18.213\data\jiangwei\项目数据\扫码枪\二维码'
dest_dir = r'\\192.168.18.213\data\jiangwei\项目数据\扫码枪\二维码txt'
label_list = ['qr', 'dm']

image_list = [item for item in os.listdir(source_dir) if is_image(item)]


for image_name in tqdm(image_list):
    json_name = os.path.splitext(image_name)[0] + '.json'
    text_name = os.path.splitext(image_name)[0] + '.txt'
    data = json.load(open(os.path.join(source_dir, json_name), encoding='utf8'))

    width = data['imageWidth']
    height = data['imageHeight']
    all_label = [label_list.index(shape['label']) for shape in data['shapes']]
    points_array = [np.array(shape['points'], dtype=np.float32) for shape in data['shapes']]
    rotate_rects = [cv2.minAreaRect(points) for points in points_array]
    all_x = [rect[0][0] / width for rect in rotate_rects]
    all_y = [rect[0][1] / height for rect in rotate_rects]
    all_w = [rect[1][0] / width for rect in rotate_rects]
    all_h = [rect[1][1] / height for rect in rotate_rects]
    all_angle = [rect[2] / 180 * math.pi for rect in rotate_rects]
    lines = ['{} {} {} {} {} {} {}\n'.format(label, x, y, w, h, math.cos(angle), math.sin(angle)) 
                for label, x, y, w, h, angle in zip(all_label, all_x, all_y, all_w, all_h, all_angle)]
    
    open(os.path.join(dest_dir, text_name), 'w', encoding='utf8').writelines(lines)