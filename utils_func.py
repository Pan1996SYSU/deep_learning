import inspect
import json
import logging
import os
import platform
import sys
import time
import warnings
from collections import defaultdict
from datetime import date
from enum import Enum
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageFont, ImageDraw
from PyQt5 import QtWidgets, QtCore, Qt
from PyQt5.QtCore import QFileInfo
from natsort import os_sorted

from sonic.lib.get_img_size import get_img_size

extensions = {'.bmp', '.gif', '.jpeg', '.jpg', '.pbm', '.png', '.tif', '.tiff'}


def cv2_read_img(file_path):
    """
    读图函数
    :param file_path: 图片路径
    :return: 图片变量-array
    """
    cv_img = cv2.imdecode(
        np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv_img


cv_img_read = cv2_read_img


def sorted_path_list(path_list):
    """
    文件名排序
    :param path_list: 路径list
    :return: 排序好的路径list
    """
    return sorted(path_list, key=lambda path: os.path.normpath(path))


def glob_extensions(directory: str, ext_names: list = extensions):
    """
    获取文件夹下所有图片路径
    :param directory: 文件夹路径
    :return: 所以图片路径 type-list
    """
    path_list = []
    new_list = []
    if directory:
        path_list += glob(f'{directory}/**/*', recursive=True)
        for ext in ext_names:
            for x in path_list:
                if x.endswith(ext):
                    new_list.append(x)
    return sorted_path_list(new_list)


class glob_model(Enum):
    quickly = 0,
    quickly_filter = 1,
    normal = 2
    safe = 3


def glob_current_extensions(
        directory: str,
        ext_names: set = None,
        model=glob_model.quickly_filter):
    """
    获取当前文件夹下的所有文件夹
    """
    if ext_names is None:
        ext_names = extensions
    new_extensions = ext_names

    path_list = glob(f'{directory}/*', recursive=False)
    path_list = sorted_path_list(path_list)
    dir_list = list()
    ext_list = list()

    # 根据后缀名判断文件夹 这个模式最快 但是文件名中有  .  时就寄了  实测 1秒(5000 文件)
    if model == glob_model.quickly:
        for x in path_list:
            pth_x = Path(x)
            suffix = pth_x.suffix
            if suffix == "":
                dir_list.append(x)
                continue
            if suffix in new_extensions:
                ext_list.append(x)
                continue

    # 根据 后缀名-文件格式过滤 判断文件夹 这个模式快且有一定的保障 实测 1秒(5000 文件)
    if model == glob_model.quickly_filter:
        file = QFileInfo()
        for x in path_list:
            file.setFile(x)
            suffix = file.suffix()

            if "." + suffix in new_extensions:
                ext_list.append(x)
                continue

            if suffix == "" or not suffix.isalnum() or file.isDir():
                dir_list.append(x)
                continue

    # QT提供的文件夹判断 实测 7秒(5000 文件)
    if model == glob_model.normal:
        info = QFileInfo()
        for x in path_list:
            info.setFile(x)
            suffix = info.suffix()
            if info.isDir():
                dir_list.append(x)
                continue
            if "." + suffix in new_extensions:
                ext_list.append(x)
                continue

    # python提供的文件夹判断 实测 10秒(5000 文件)
    if model == glob_model.safe:
        for x in path_list:
            pth_x = Path(x)
            suffix = pth_x.suffix
            if Path.is_dir(pth_x):
                dir_list.append(x)
                continue
            if suffix in new_extensions:
                ext_list.append(x)
                continue

    return {"path_list": path_list, "dir_list": dir_list, "ext_list": ext_list}


def make_dirs(path):
    """
    :path:需要创建的文件夹路径,当path是文件路径时，创建其父文件夹
    :return:
    """
    path = Path(path)
    if path.suffix:
        warnings.warn(f'调用make_dirs函数不要传入文件路径!')
        path_file = path
        path_dir = path.parent
    else:
        path_dir = path
    if not os.path.exists(path_dir):
        os.makedirs(path_dir, exist_ok=True)


def make_dirs_2_(path_dir):
    """
    :path:只支持文件夹路径
    :return:
    """
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


# debug函数
def show_img(img):
    """
    显示array类型的图片
    """
    if len(img.shape) < 3:
        new_img = Image.fromarray(img, 'L')
    else:
        new_img = Image.fromarray(img, 'RGB')
    new_img.show()


def show_qpixmap(pixmap):
    """
    显示qpixmap图片
    """
    img = Image.fromqpixmap(pixmap)
    img.show()


def save_img(img: np.ndarray, img_path, src_img_format="rgb"):
    """
    保存图片为指定格式 转化BGR为RGB
    """
    suffix = Path(img_path).suffix
    img_channels = img.shape[-1]

    if img.dtype != np.uint8 or src_img_format != 'rgb':
        cv2.imencode(suffix, img)[1].tofile(img_path)
    elif len(img.shape) == 3 and img_channels >= 3:
        # 判断是否是3通道 或者以上的图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imencode(suffix, img)[1].tofile(img_path)
    else:
        cv2.imencode(suffix, img)[1].tofile(img_path)


def save_json(json_path, data):
    """
    保存json
    :param json_path: 保存路径
    :param data: 数据
    """
    try:
        make_dirs_2_(Path(json_path).parent)
        with open(json_path, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except:
        raise ValueError(f'保存Json失败:{json_path}')


def load_json(json_path):
    """
    读取json数据
    :param json_path: json路径
    :return: json数据
    """
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
    """
    保存yaml
    :param yaml_path: 保存路径
    :param data: 数据
    """
    try:
        make_dirs_2_(Path(yaml_path).parent)
        with open(yaml_path, 'w', encoding='utf8') as f:
            yaml.safe_dump(data, f, allow_unicode=True)
    except:
        raise ValueError(f'保存Yaml失败:{yaml_path}')


def load_yaml(yaml_path):
    """
    读取yaml数据
    :param yaml_path: yaml路径
    :return: yaml内数据
    """
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


def get_right_value_dialog(
        accept_condition_func,
        default_text="",
        prompt_text="提示: 请输入正确的值",
        window_title="请输入正确的值",
        parent=None,
        default_return=''):
    """
    一个输入对话框, 根据 accept_condition_func来进行判断输入是否正确
    accept_condition_func: 当 raise 与 return Flase时表示输入错误 当不抛出异常与 return True表示输入正确
    例:
        def func(value):
            if value == "zzz":
                raise "111"
            else:
                return True
    """

    import types
    if not isinstance(accept_condition_func, types.FunctionType):
        raise "弹窗的第一个参数不为函数"

    ret_text = default_return
    dialog_input = Qt.QDialog()
    dialog_input.setWindowTitle(window_title)

    def check_text():
        result = False
        try:
            if not accept_condition_func(lineedit_input.text()):
                raise "error"
            result = True
        except:
            pass

        button_accept.setEnabled(result)

    def accept_text():
        nonlocal ret_text
        ret_text = lineedit_input.text()
        dialog_input.close()

    label_prompt = Qt.QLabel(prompt_text)
    lineedit_input = Qt.QLineEdit(default_text)
    lineedit_input.textChanged.connect(check_text)
    button_accept = Qt.QPushButton("确定")
    button_accept.setEnabled(False)
    button_accept.clicked.connect(accept_text)
    button_cancel = Qt.QPushButton("取消")
    button_cancel.clicked.connect(dialog_input.close)

    layout = Qt.QGridLayout()
    layout.addWidget(label_prompt, 0, 0, 1, 4)
    layout.addWidget(lineedit_input, 1, 0, 1, 4)
    layout.addWidget(button_accept, 2, 2, 1, 1)
    layout.addWidget(button_cancel, 2, 3, 1, 1)
    dialog_input.setLayout(layout)
    dialog_input.exec_()

    return ret_text