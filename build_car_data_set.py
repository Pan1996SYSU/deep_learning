import os
import random
import shutil
from pathlib import Path

import numpy as np

from utils_func import make_dirs

suffix_xml = '.xml'
suffix_jpg = '.jpg'

car_train_path = './DATA/VOCdevkit/VOC2007/ImageSets/Main/car_train.txt'
car_val_path = './DATA/VOCdevkit/VOC2007/ImageSets/Main/car_val.txt'

voc_annotation_dir = './DATA/VOCdevkit/VOC2007/Annotations/'
voc_jpg_dir = './DATA/VOCdevkit/VOC2007/JPEGImages'

car_root_dir = './DATA/voc_car/'


def parse_train_val(data_path):
    """
    提取指定类别图像
    """
    samples = []

    with open(data_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            res = line.strip().split(' ')
            if len(res) == 3 and int(res[2]) == 1:
                samples.append(res[0])

    return np.array(samples)


def sample_train_val(samples):
    """
    随机采样样本，减少数据集个数（留下1/10）
    """
    for name in ['train', 'val']:
        dataset = samples[name]
        length = len(dataset)

        random_samples = random.sample(range(length), int(length / 10))
        # print(random_samples)
        new_dataset = dataset[random_samples]
        samples[name] = new_dataset

    return samples


def save_car(car_samples, data_root_dir, data_annotation_dir, data_jpeg_dir):
    """
    保存类别Car的样本图片和标注文件
    """
    for sample_name in car_samples:
        src_annotation_path = os.path.join(voc_annotation_dir, sample_name + suffix_xml)
        dst_annotation_path = os.path.join(data_annotation_dir, sample_name + suffix_xml)
        make_dirs(Path(dst_annotation_path).parent)
        shutil.copyfile(src_annotation_path, dst_annotation_path)

        src_jpeg_path = os.path.join(voc_jpg_dir, sample_name + suffix_jpg)
        dst_jpeg_path = os.path.join(data_jpeg_dir, sample_name + suffix_jpg)
        make_dirs(Path(dst_jpeg_path).parent)
        shutil.copyfile(src_jpeg_path, dst_jpeg_path)

    csv_path = os.path.join(data_root_dir, 'car.csv')
    np.savetxt(csv_path, np.array(car_samples), fmt='%s')


if __name__ == '__main__':
    # 构建训练集和测试集的文件stem的ndarray
    samples = {'train': parse_train_val(car_train_path), 'val': parse_train_val(car_val_path)}

    for name in ['train', 'val']:
        data_root_dir = os.path.join(car_root_dir, name)
        data_annotation_dir = os.path.join(data_root_dir, 'Annotations')
        data_jpg_dir = os.path.join(data_root_dir, 'JPEGImages')

        save_car(samples[name], data_root_dir, data_annotation_dir, data_jpg_dir)

    print('done')
