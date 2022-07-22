import os
import json
import argparse
import glob
from tkinter import image_names

import numpy as np
import cv2
import onnxruntime
import shapely
import shapely.geometry
from tqdm import tqdm
        

def is_image(image_name):
    return image_name.endswith('.png') or image_name.endswith('.jpg') or image_name.endswith('.bmp') or image_name.endswith('.jpeg') or image_name.endswith('.tiff')

def gen_shape(label, points, shape_type):
    shape = {
        "label": label,
        "points": points,
        "group_id": None,
        "shape_type": shape_type,
        "flags": {}
    }
    return shape

def gen_data(image, image_name, shapes):
    height, width = image.shape[:2]
    data = {
        'version': '4.2.10',
        'imageHeight': height,
        'imageWidth': width,
        'imagePath': image_name,
        'imageData': None,
        'flags': {},
        'shapes': shapes
    }
    return data
 
def xywhrm2xyxyxyxy(xywhrm):
    """
        xywhrm : shape (N, 6)
        Transform x,y,w,h,re,im to x1,y1,x2,y2,x3,y3,x4,y4
        Suitable for both pixel-level and normalized
    """
    x0, x1, y0, y1 = -xywhrm[:, 2:3]/2, xywhrm[:, 2:3]/2, -xywhrm[:, 3:4]/2, xywhrm[:, 3:4]/2
    xyxyxyxy = np.concatenate((x0, y0, x1, y0, x1, y1, x0, y1), axis=-1).reshape(-1, 4, 2)
    R = np.zeros((xyxyxyxy.shape[0], 2, 2), dtype=xyxyxyxy.dtype)
    R[:, 0, 0], R[:, 1, 1] = xywhrm[:, 4], xywhrm[:, 4]
    R[:, 0, 1], R[:, 1, 0] = xywhrm[:, 5], -xywhrm[:, 5]
    
    xyxyxyxy = np.matmul(xyxyxyxy, R).reshape(-1, 8)+xywhrm[:, [0, 1, 0, 1, 0, 1, 0, 1]]
    return xyxyxyxy

def polygon_inter_union_cpu(box1, box2):
    polygon1 = shapely.geometry.Polygon(np.array(box1).reshape(4,2)).convex_hull
    polygon2 = shapely.geometry.Polygon(np.array(box2).reshape(4,2)).convex_hull
    inter = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    return inter/union

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, top_num=1):
    output = []

    prediction = prediction[0]   
    prediction[:, 7:] *= prediction[:, 6:7]  
    for i in range(prediction.shape[1] - 7):
        class_prediction = prediction[:, 7+i]

        xc = class_prediction > conf_thres
        x = prediction[xc]

        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue

        if n > max_det:  # excess boxes
            x = x[:max_det]  # sort by confidence

        # Batched NMS
        boxes = xywhrm2xyxyxyxy(x[:, :6]).tolist()

        while boxes:
            box1 = boxes.pop(0)
            output.append((box1, i))

            pop_indexes = []
            num = 0
            for index, box2 in enumerate(boxes):
                if polygon_inter_union_cpu(box1, box2) > iou_thres:
                    pop_indexes.append(index)
                    if num < top_num - 1:
                        output.append((boxes[index], i))
                        num += 1
            for index in pop_indexes[::-1]:
                boxes.pop(index)

    return output

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate labelme's file.")
    parser.add_argument("--root_dir", default=r'\\192.168.18.213\data\jiangwei\项目数据\扫码枪\训练', help="The directory of images that need to be predicted.")
    parser.add_argument("--save_dir", default=r'\\192.168.18.213\data\jiangwei\项目数据\扫码枪\resample', help="The directory of images that need to be predicted.")
    args = parser.parse_args()

    sample_ratio = 0.5
    file_list = glob.glob(f'{args.root_dir}/**/*', recursive=True)
    image_list = [item for item in file_list if is_image(item)]
    
    for image_path in tqdm(image_list):
        image_name = os.path.basename(image_path)
        ext = os.path.splitext(image_name)[1]
        json_name = os.path.splitext(image_name)[0] + '.json'
        json_path = os.path.join(args.root_dir, json_name)
        ori_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flags=cv2.IMREAD_UNCHANGED)

        resize_image = cv2.resize(ori_image, None, fx=sample_ratio, fy=sample_ratio)

        cv2.imencode(ext, resize_image)[1].tofile(os.path.join(args.save_dir, image_name))

        data = json.load(open(json_path, encoding='utf8'))
        data['imageData'] = None
        for shape in data['shapes']:
            for point in shape['points']:
                point[0] = point[0] * sample_ratio
                point[1] = point[1] * sample_ratio

        json.dump(data, open(os.path.join(args.save_dir, json_name), 'w', encoding='utf8'))


