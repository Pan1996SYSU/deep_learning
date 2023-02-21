import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from utils.utils_func import glob_extensions, cv_imread
'''
猫狗分类
训练集：22500张
测试集：2500张
50张/批
训练20代
准确率：83.52%
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(r'.\pth\CNN_11.pth')
accuracy_sum = []

test_path = r".\DATA\\cat-dog-all-data\test-dataset\test"
test_path_list = glob_extensions(test_path)
random.shuffle(test_path_list)

for i, img_path in enumerate(test_path_list):
    if i % 50 != 0:
        continue
    else:
        img = cv_imread(img_path)
        img = cv2.resize(img, (401, 401))
        img = np.transpose(img, (2, 0, 1))
        img = torch.unsqueeze(torch.as_tensor(img, device=device), dim=0)
        img = img.to(torch.float32)
        batch_img = img
        if Path(img_path).parent.name == 'cat':
            y = torch.tensor([1, 0], device=device)
        else:
            y = torch.tensor([0, 1], device=device)
        y = torch.unsqueeze(torch.as_tensor(y), dim=0)
        y = y.to(torch.float32)
        b_y = y
        for j in range(1, 50):
            img = cv_imread(test_path_list[i + j])
            img = cv2.resize(img, (401, 401))
            img = np.transpose(img, (2, 0, 1))
            img = torch.unsqueeze(torch.as_tensor(img, device=device), dim=0)
            img = img.to(torch.float32)
            if Path(test_path_list[i + j]).parent.name == 'cat':
                y = torch.tensor([1, 0])
            else:
                y = torch.tensor([0, 1])
            y = torch.unsqueeze(torch.as_tensor(y, device=device), dim=0)
            y = y.to(torch.float32)
            batch_img = torch.cat((batch_img, img))
            b_y = torch.cat((b_y, y))
    test_x = Variable(batch_img)
    test_y = Variable(b_y)
    out = model(test_x)
    accuracy = torch.max(out.cpu(),
                         1)[1].numpy() == torch.max(test_y.cpu(),
                                                    1)[1].numpy()
    accuracy_sum.append(accuracy.mean())
    print(f'Iteration: {i}\nAccuracy: {round(accuracy.mean() * 100, 2)}%')
    print(f'-------------------------')

print(
    f'Final Accuracy: {round(sum(accuracy_sum) / len(accuracy_sum) * 100, 2)}%'
)

plt.figure('Accuracy')
plt.plot(accuracy_sum, 'o', label='accuracy')
plt.title('Pytorch_CNN_Accuracy')
plt.legend()
plt.show()
