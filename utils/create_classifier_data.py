from pathlib import Path

import numpy as np
import shutil
import time
import cv2
import os
from utils import selective_search

# train
# positive num: 625
# negative num: 366028
# val
# positive num: 625
# negative num: 321474
from utils.utils_func import make_dirs, parse_xml, parse_car_csv, compute_ious


def parse_annotation_jpeg(annotation_path, jpeg_path, gs):
    """
    获取正负样本（注：忽略属性difficult为True的标注边界框）
    正样本：标注边界框
    负样本：IoU大于0，小于等于0.3。为了进一步限制负样本数目，其大小必须大于标注框的1/5
    """
    img = cv2.imread(jpeg_path)

    selective_search.config_get_img_method(gs, img, strategy='q')
    # 计算候选建议
    rects = selective_search.get_rect(gs)
    # 获取标注边界框
    bndboxs = parse_xml(annotation_path)

    bbox_list = np.array(bndboxs['car'])

    # 标注框大小
    maximum_bndbox_size = 0
    for bndbox in bbox_list:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_size = (ymax - ymin) * (xmax - xmin)
        if bndbox_size > maximum_bndbox_size:
            maximum_bndbox_size = bndbox_size

    # 获取候选建议和标注边界框的IoU
    iou_list = compute_ious(rects, bbox_list)

    negative_list = list()
    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)

        iou_score = iou_list[i]
        if 0 < iou_score <= 0.3 and rect_size > maximum_bndbox_size / 5.0:
            # 负样本
            negative_list.append(rects[i])
        else:
            pass

    return bbox_list, negative_list


if __name__ == '__main__':
    car_root_dir = '../DATA/voc_car/'
    classifier_root_dir = '../DATA/voc_car/classifier_car/'

    gs = selective_search.get_selective_search()
    for name in ['train', 'val']:
        src_root_dir = os.path.join(car_root_dir, name)
        src_annotation_dir = os.path.join(src_root_dir, 'Annotations')
        src_jpeg_dir = os.path.join(src_root_dir, 'JPEGImages')

        dst_root_dir = os.path.join(classifier_root_dir, name)
        dst_annotation_dir = os.path.join(dst_root_dir, 'Annotations')
        dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')
        make_dirs(Path(dst_root_dir).parent)
        make_dirs(Path(dst_annotation_dir).parent)
        make_dirs(Path(dst_jpeg_dir).parent)

        total_num_positive = 0
        total_num_negative = 0

        samples = parse_car_csv(src_root_dir)
        # 复制csv文件
        src_csv_path = os.path.join(src_root_dir, 'car.csv')
        dst_csv_path = os.path.join(dst_root_dir, 'car.csv')
        shutil.copyfile(src_csv_path, dst_csv_path)
        for sample_name in samples:
            since = time.time()

            src_annotation_path = os.path.join(src_annotation_dir, sample_name + '.xml')
            src_jpeg_path = os.path.join(src_jpeg_dir, sample_name + '.jpg')
            # 获取正负样本
            positive_list, negative_list = parse_annotation_jpeg(src_annotation_path, src_jpeg_path, gs)
            total_num_positive += len(positive_list)
            total_num_negative += len(negative_list)

            dst_annotation_positive_path = os.path.join(dst_annotation_dir, sample_name + '_1' + '.csv')
            dst_annotation_negative_path = os.path.join(dst_annotation_dir, sample_name + '_0' + '.csv')
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + '.jpg')
            make_dirs(Path(dst_annotation_positive_path).parent)
            make_dirs(Path(dst_annotation_negative_path).parent)
            make_dirs(Path(dst_jpeg_path).parent)
            # 保存图片
            shutil.copyfile(src_jpeg_path, dst_jpeg_path)
            # 保存正负样本标注
            np.savetxt(dst_annotation_positive_path, np.array(positive_list), fmt='%d', delimiter=' ')
            np.savetxt(dst_annotation_negative_path, np.array(negative_list), fmt='%d', delimiter=' ')

            time_elapsed = time.time() - since
            print('parse {}.png in {:.0f}m {:.0f}s'.format(sample_name, time_elapsed // 60, time_elapsed % 60))
        print('%s positive num: %d' % (name, total_num_positive))
        print('%s negative num: %d' % (name, total_num_negative))
    print('done')
