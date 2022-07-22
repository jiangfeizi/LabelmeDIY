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

# def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, top_num=1):
#     output = []
#     max_nms = 30000

#     prediction = prediction[0]     
#     prediction[:, 7:] *= prediction[:, 6:7]
#     score_best = np.max(prediction[:, 7:], axis=1)
#     xc = score_best > conf_thres
#     x = prediction[xc]

#     n = x.shape[0]  # number of boxes
#     if not n:  # no boxes
#         return output

#     score_best = np.max(x[:, 7:], axis=1)
#     x = x[np.argsort(-score_best)]
#     arg_best = np.argmax(x[:, 7:], axis=1)

#     if n > max_nms:  # excess boxes
#         x = x[:max_nms]  # sort by confidence

#     # Batched NMS
#     boxes = xywhrm2xyxyxyxy(x[:, :6]).tolist()
#     class_best = arg_best.tolist()

#     while boxes:
#         box1 = boxes.pop(0)
#         cl = class_best.pop(0)
#         output.append((box1, cl))

#         pop_indexes = []
#         num = 0
#         for index, box2 in enumerate(boxes):
#             if polygon_inter_union_cpu(box1, box2) > iou_thres:
#                 pop_indexes.append(index)
#                 if num < top_num - 1:
#                     output.append((boxes[index], class_best[index]))
#                     num += 1
#         for index in pop_indexes[::-1]:
#             boxes.pop(index)
#             class_best.pop(index)

#     return output

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, top_num=1):
    output = []

    prediction = prediction[0]   
    prediction[:, 7:] *= prediction[:, 6:7]  
    for i in range(prediction.shape[1] - 7):
        class_prediction = prediction[:, 7+i]

        xc = class_prediction > conf_thres
        x = prediction[xc]
        class_prediction = class_prediction[xc]

        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue

        x = x[np.argsort(-class_prediction)]

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
    parser.add_argument("root_dir", help="The directory of images that need to be predicted.")
    parser.add_argument("-w", "--weights", default="./rotate_last.onnx",help="The path of weights.")
    parser.add_argument("-c", "--classes", default="./classes.txt",help="The text of classes.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-xs", "--xsmall", action="store_true")
    group.add_argument("-s", "--small", action="store_true")
    group.add_argument("-m", "--medium", action="store_true")
    group.add_argument("-l", "--large", action="store_true")
    args = parser.parse_args()

    file_list = glob.glob(f'{args.root_dir}/**/*', recursive=True)
    image_list = [item for item in file_list if is_image(item)]
    session = onnxruntime.InferenceSession(args.weights, providers=['CPUExecutionProvider'])
    class_list = [line.strip() for line in open(args.classes, encoding='utf8')]

    if args.xsmall:
        input_size = 320
    if args.small:
        input_size = 640
    elif args.medium:
        input_size = 1280
    elif args.large:
        input_size = 1920
    
    for image_path in tqdm(image_list):
        image_dir, image_name = os.path.split(image_path)
        json_name = os.path.splitext(image_name)[0] + '.json'
        json_path = os.path.join(image_dir, json_name)
        ori_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flags=cv2.IMREAD_COLOR)

        image, ratio, (dw, dh) = letterbox(ori_image, (input_size, input_size), (0, 0, 0), scaleup=False)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., None]
        image = image.transpose((2, 0, 1))
        image = np.ascontiguousarray(image)
        image = np.array(image, dtype=np.float32) / 255
        if len(image.shape) == 3:
            image = image[None]  # expand for batch dim
        prediction = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: image})[0]

        output = non_max_suppression(prediction, 0.1, 0.05, top_num=1)

        shapes = []
        for box, cl in output:
            box[0] = (box[0] -dw) / ratio[0]
            box[1] = (box[1] -dh) / ratio[1]
            box[2] = (box[2] -dw) / ratio[0]
            box[3] = (box[3] -dh) / ratio[1]
            box[4] = (box[4] -dw) / ratio[0]
            box[5] = (box[5] -dh) / ratio[1]
            box[6] = (box[6] -dw) / ratio[0]
            box[7] = (box[7] -dh) / ratio[1]
            shapes.append(gen_shape(class_list[cl], [[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]], 'polygon'))

        data = gen_data(ori_image, image_name, shapes)
        json.dump(data, open(json_path, 'w', encoding='utf8'))


