import json
import os
import shutil
from pathlib import Path
import numpy as np
import cv2
from numpy.lib.function_base import angle
from tqdm import tqdm

def is_image(image_name):
    return image_name.endswith('.png') or image_name.endswith('.jpg') or image_name.endswith('.bmp') or image_name.endswith('.jpeg') or image_name.endswith('.tiff')


images_dir = r'\\192.168.18.213\data\jiangwei\项目数据\扫码枪\训练'
jsons_dir = r'\\192.168.18.213\data\jiangwei\项目数据\扫码枪\训练'
dest_dir = r'\\192.168.10.186\test'

for item in tqdm(os.listdir(jsons_dir)):
    if item.endswith('json'):
        json_path = os.path.join(jsons_dir,item)
        with open(json_path, encoding='utf8') as fp:
            data = json.load(fp)
        data['imageData'] = None

        image_name = data['imagePath']
        height = data['imageHeight']
        width = data['imageWidth']


        image_path = os.path.join(images_dir, image_name)
        image_name_no_ext, image_ext = os.path.splitext(image_name)
        
        barcode_shape = [item for item in data['shapes'] if item['label'] == 'barcode']
        other_shape = [item for item in data['shapes'] if item['label'] != 'barcode']

        if len(barcode_shape):
            print(image_name)
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flags=-1)

            data['shapes'] = barcode_shape

            cv2.imencode('.' + image_ext, image)[1].tofile(os.path.join(dest_dir, image_name))
            json.dump(data, open(os.path.join(dest_dir, item), 'w', encoding='utf8'))
    


            
