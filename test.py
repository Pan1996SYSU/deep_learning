import cv2
import numpy as np
from sonic.utils_func import cv2_read_img, glob_extensions


def getStat(train_path):
    img_path_list = glob_extensions(train_path)
    mean = [0, 0, 0]
    std = [0, 0, 0]
    for img_path in img_path_list:
        img = cv2_read_img(img_path)
        img = cv2.resize(img, (401, 401))
        for i in range(3):
            mean[i] += img[:, :, i].mean()
            std[i] += img[:, :, i].std()
    mean = np.array(mean)/len(img_path_list)
    std = np.array(std)/len(img_path_list)
    return mean, std


if __name__ == '__main__':
    train_path = "./DATA/cat-dog-all-data/test-dataset/train"
    print(getStat(train_path))
