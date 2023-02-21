import json
import logging
import math
import os
import shutil
import traceback
from collections import Counter
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

extensions = {'.bmp', '.gif', '.jpeg', '.jpg', '.pbm', '.png', '.tif', '.tiff'}

def parse_xml(path):
    with open(path, 'r') as file:
        print(123)

def parse_car_csv(path):
    with open(path, 'r') as file:
        print(123)

def compute_ious(rects, bndboxs):
    print(123)


def get_circle(p1, p2, p3):
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return None, -1

    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    cx, cy, radius = cx, cy, radius

    return (cx, cy), radius


def cv_simplify_polygon(points_list, simplify_level: int = 1) -> np.ndarray:
    """
    :param simplify_level: 简化等级
    :param points_list: list类型，多边形的各个顶点,值越大，简化出来的点越少
    :return:points_list
    """
    polygon_vertex_2d = np.array(points_list, np.float32)
    x, y = polygon_vertex_2d.min(axis=0)
    w, h = polygon_vertex_2d.max(axis=0) - [x, y]
    w2, h2 = 500, 500
    polygon_vertex_2d -= [[x, y]]
    polygon_vertex_2d /= [[w / w2, h / h2]]
    polygon_vertex_2d = polygon_vertex_2d.astype(np.int32)

    img = np.zeros((h2, w2), dtype=np.uint8)

    cv2.fillPoly(img, [polygon_vertex_2d], 255)

    contours, hierarchy = cv2.findContours(img, 1, 2)
    cnt = contours[-1]

    approx = cv2.approxPolyDP(cnt, simplify_level, True)
    approx = approx[:, 0]

    approx = approx.astype(np.float32)
    approx *= [[w / w2, h / h2]]
    approx += [[x, y]]
    approx = approx.astype(np.int32)
    return approx


def shape_to_points(shape):
    new_points = []
    try:
        # 从json中获取数据
        shape_type = shape['shape_type']
        points = shape['points']
    except:
        # 直接从Labelme中获取数据
        shape_type = shape.shape_type
        points = shape.getPointsPos()
    if shape_type == 'polygon':
        new_points = points
        if len(points) < 3:
            new_points = []
            print('polygon 异常，少于三个点', shape)
    elif shape_type == 'rectangle':
        (x1, y1), (x2, y2) = points
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        new_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    elif shape_type == "circle":
        # Create polygon shaped based on connecting lines from/to following degress
        bearing_angles = [
            0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210,
            225, 240, 255, 270, 285, 300, 315, 330, 345, 360
        ]

        orig_x1 = points[0][0]
        orig_y1 = points[0][1]

        orig_x2 = points[1][0]
        orig_y2 = points[1][1]

        # Calculate radius of circle
        radius = math.sqrt((orig_x2 - orig_x1)**2 + (orig_y2 - orig_y1)**2)

        circle_polygon = []

        for i in range(0, len(bearing_angles) - 1):
            ad1 = math.radians(bearing_angles[i])
            x1 = radius * math.cos(ad1)
            y1 = radius * math.sin(ad1)
            circle_polygon.append((orig_x1 + x1, orig_y1 + y1))

            ad2 = math.radians(bearing_angles[i + 1])
            x2 = radius * math.cos(ad2)
            y2 = radius * math.sin(ad2)
            circle_polygon.append((orig_x1 + x2, orig_y1 + y2))

        new_points = circle_polygon
    elif shape_type == 'point':
        new_points = points
    else:
        print('未知 shape_type', shape['shape_type'])

    new_points = np.asarray(new_points, dtype=np.int32)
    return new_points


def labelme2coco(
        json_path_list,
        new_img_dir,
        category_map,
        category_list,
        copy=True,
        queue=None,
        prograss_start=0,
        prograss_end=100,
        overwrite=False,
        add_subdir=True):
    """
    :param json_path_list: json 文件名列表
    :param new_img_dir: 目标图片文件夹路径
    :param category_map: labelme 分类列表（abcdefg）
    :param category_list: coco 分类列表（脏污、黑点等）
    :param copy:
    :return:
    """
    annotations = []
    images = []
    obj_count = 0
    with tqdm(json_path_list, desc='labelme2coco') as pbar:
        bar_stepcount = len(pbar)
        bar_step = prograss_end - prograss_start
        bar_step *= 1.00
        bar_step /= bar_stepcount
        for idx, json_path in enumerate(pbar):
            if queue is not None:
                queue.put(
                    {
                        'func': 'progress_update',
                        'color': 2,
                        'val': prograss_start + idx * bar_step
                    })

            if not os.path.exists(json_path):  # OK 样本
                continue
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)

            old_dir = os.path.split(json_path)[0]
            img_path = f"{old_dir}/{data['imagePath']}"
            if not os.path.exists(img_path):
                logging.error(f'图片文件不存在：{img_path}')
                continue

            if add_subdir:
                subdir = os.path.split(old_dir)[-1]
                new_img_path = f"{new_img_dir}/{subdir}_{data['imagePath']}"
            else:
                new_img_path = f"{new_img_dir}/{data['imagePath']}"
            if copy:
                try:
                    if overwrite or not os.path.exists(new_img_path):
                        new_img_path.replace('\\', '/')
                        shutil.copy(img_path, new_img_path)
                except:
                    traceback.print_exc()
                    print(f'拷贝文件失败：{img_path}')
                    continue

            img = cv_imread(new_img_path)
            img_filename = os.path.split(new_img_path)[-1]
            height, width = img.shape[:2]
            images.append(
                dict(
                    id=idx, file_name=img_filename, height=height,
                    width=width))

            for shape in data['shapes']:
                if shape['label'] not in category_map:
                    print('发现未知标签', json_path, shape)
                    continue

                new_points = []
                try:
                    new_points = shape_to_points(shape)
                except:
                    logging.error(traceback.format_exc())

                if len(new_points) == 0:
                    print('解析 shape 失败', json_path, shape)
                    continue

                px = [x[0] for x in new_points]
                py = [x[1] for x in new_points]
                poly = new_points.flatten().tolist()
                x_min, y_min, x_max, y_max = (
                    min(px), min(py), max(px), max(py))

                # 处理越界的 bbox
                x_max = min(x_max, width - 1)
                y_max = min(y_max, height - 1)

                category_id = category_list.index(category_map[shape['label']])
                data_anno = dict(
                    image_id=idx,
                    id=obj_count,
                    category_id=category_id,
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    area=(x_max - x_min) * (y_max - y_min),
                    segmentation=[poly],
                    iscrowd=0)

                annotations.append(data_anno)
                obj_count += 1

    categories = [{'id': i, 'name': x} for i, x in enumerate(category_list)]
    coco_format_json = dict(
        images=images, annotations=annotations, categories=categories)

    # 统计分类
    category_counter = Counter([x['category_id'] for x in annotations])
    for k, v in category_counter.most_common():
        print(category_list[k], v)
    return coco_format_json


def cv_imread(file_path):
    # 支持读
    cv_img = cv2.imdecode(
        np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv_img


def cv_imread2(file_path):
    # 支持看
    if os.path.exists(file_path):
        if Path(file_path).suffix == '.png':
            cv_img = cv2.imdecode(
                np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            cv_img = cv2.imdecode(
                np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        print(f'{file_path}文件不存在')
        cv_img = np.zeros((3, 3, 3))
    return cv_img


def sorted_path_list(path_list):
    return sorted(path_list, key=lambda path: os.path.normpath(path))


def glob_extensions(directory: str, ext_names: list = extensions):
    path_list = []
    new_list = []
    if directory:
        path_list += glob(f'{directory}/**/*', recursive=True)
        for ext in ext_names:
            for x in path_list:
                if x.endswith(ext):
                    new_list.append(x)
    return sorted_path_list(new_list)


def make_dirs(path):
    path = Path(path)
    if path.suffix:
        path_dir = path.parent
    else:
        path_dir = path
    if not os.path.exists(path_dir):
        os.makedirs(path_dir, exist_ok=True)


def normalize_16bit_image(img_16):
    if img_16.dtype == np.uint16:
        img_16 = img_16.astype(np.float32)
        img_16 -= img_16[img_16 > 0].min()
        img_16[img_16 < 0] = 0
        img_16 /= (img_16.max() / 255)
        img_16 = img_16.astype(np.uint8)
        if len(img_16.shape) == 2:
            img_16 = cv2.cvtColor(img_16, cv2.COLOR_GRAY2RGB)
    return img_16


def show_img(img):
    if len(img.shape) < 3:
        new_img = Image.fromarray(img, 'L')
    else:
        new_img = Image.fromarray(img, 'RGB')
    new_img.show()


def show_qt_pixmap(pixmap):
    img = Image.fromqpixmap(pixmap)
    img.show()


def save_img(img: np.ndarray, img_path):
    suffix = Path(img_path).suffix
    cv2.imencode(suffix, img)[1].tofile(img_path)


def save_json(json_path, data):
    try:
        make_dirs(Path(json_path).parent)
        with open(json_path, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except:
        raise ValueError(f'保存Json失败:{json_path}')


def load_json(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'json文件不存在!\n'
                                f'{json_path}')
    try:
        with open(json_path, encoding='utf8') as f:
            data = json.load(f)
    except:
        try:
            with open(json_path) as f:
                data = json.load(f)
        except:
            raise Exception(f'打开{json_path}失败，请检查是否编写正确')
    return data


def save_yaml(yaml_path, data):
    try:
        make_dirs(Path(yaml_path).parent)
        with open(yaml_path, 'w', encoding='utf8') as f:
            yaml.safe_dump(data, f, allow_unicode=True)
    except:
        raise ValueError(f'保存Yaml失败:{yaml_path}')


def load_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f'yaml文件不存在!\n'
                                f'{yaml_path}')
    try:
        with open(yaml_path, encoding='utf8') as f:
            data = yaml.safe_load(f)
    except:
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        except:
            raise Exception(f'打开{yaml_path}失败，请检查是否编写正确')
    return data


def pixmap_to_ndarray(pixmap):
    image = pixmap.toImage()
    size = image.size()
    s = image.bits().asstring(
        size.width() * (image.depth() // 8) * size.height())
    ndarray = np.frombuffer(
        s, dtype=np.uint8).reshape(
            (size.height(), size.width(), image.depth() // 8))
    return ndarray


def crop_json(json_data, x, y, w, h):
    try:
        json_data["imageHeight"] = h
        json_data["imageWidth"] = w
        for i, shape in enumerate(json_data["shapes"]):
            points = np.array(shape['points'])
            points[:, 0] = points[:, 0] - x
            points[:, 1] = points[:, 1] - y
            points[points[:, 0] >= w, 0] = w
            points[points[:, 0] <= 0, 0] = 0
            points[points[:, 1] >= h, 1] = h
            points[points[:, 1] <= 0, 1] = 0
            json_data["shapes"][i]['points'] = points.tolist()
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        return {}
    return json_data


def crop_img(img, x, y, w, h):
    try:
        res_img = img[max(0, int(y)):min(int(y), h),
                      max(0, int(x)):min(int(x), w)].copy()
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        return np.zeros((3, 3, 3))
    return res_img
