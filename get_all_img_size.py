from utils_func import cv_imread, glob_extensions

train_path = r"C:\Users\MasterZ\Desktop\cat-dog-all-data\test-dataset\train"
test_path = r"C:\Users\MasterZ\Desktop\cat-dog-all-data\test-dataset\test"

train_path_list = glob_extensions(train_path)
test_path_list = glob_extensions(test_path)

max_h = 0
max_w = 0
min_h = 9999
min_w = 9999
for path in train_path_list:
    img = cv_imread(path)
    h, w = img.shape[:2]
    if h > max_h:
        max_h = h
    if w > max_w:
        max_w = w
    if h < min_h:
        min_h = h
    if w < min_w:
        min_w = w
print(f'train_max_h:{max_h} train_min_h{min_h}')
print(f'train_max_w:{max_w} train_min_w{min_w}')
max_h = 0
max_w = 0
min_h = 9999
min_w = 9999
for path in test_path_list:
    img = cv_imread(path)
    h, w = img.shape[:2]
    if h > max_h:
        max_h = h
    if w > max_w:
        max_w = w
    if h < min_h:
        min_h = h
    if w < min_w:
        min_w = w
print(f'test_max_h:{max_h} test_min_h{min_h}')
print(f'test_max_w:{max_w} test_min_w{min_w}')