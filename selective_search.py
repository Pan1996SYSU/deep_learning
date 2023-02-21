import time

import cv2
from utils_func import cv_imread


def get_selective_search():
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return ss


def config(ss, image, strategy='q'):
    ss.setBaseImage(image)
    if strategy == 's':
        ss.switchToSingleStrategy()
    elif strategy == 'f':
        ss.switchToSelectiveSearchFast()
    elif strategy == 'q':
        ss.switchToSelectiveSearchQuality()
    else:
        print('模式选择错误')


def get_rect(ss):
    rectangle = ss.process()
    rectangle[:, 2] += rectangle[:, 0]
    rectangle[:, 3] += rectangle[:, 1]
    return rectangle


def rect_img(image, rectangle, color=(0, 0, 255)):
    for x1, y1, x2, y2 in rectangle[:200]:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=1)
    cv2.imshow('img', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    """
    选择性搜索算法操作
    """
    gs = get_selective_search()
    img = cv_imread(r"D:\桌面\sth\hdasdjasjdasldjkal.jpg")
    config(gs, img, strategy='f')
    star = time.time()
    rect = get_rect(gs)
    print(time.time() - star)
    rect_img(img, rect)
