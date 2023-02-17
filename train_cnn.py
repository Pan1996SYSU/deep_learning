from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torch.utils import data
from torchvision.datasets import mnist

from CNN import CNNNet

# data_tf = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize([0.5], [0.5])
#     ])
#
# data_path = r'.\DATA'
#
# train_data = mnist.MNIST(
#     data_path, train=True, transform=data_tf, download=False)
# test_data = mnist.MNIST(
#     data_path, train=False, transform=data_tf, download=False)
#
# train_loader = data.DataLoader(train_data, batch_size=300, shuffle=True)
# test_loader = data.DataLoader(test_data, batch_size=100, shuffle=True)
from utils_func import glob_extensions, cv_imread

model = CNNNet()
# model = torch.load(r'.\pth\4.pth')

loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = []
test_loader = []
train_path = r"C:\Users\MasterZ\Desktop\cat-dog-all-data\test-dataset\train"
test_path = r"C:\Users\MasterZ\Desktop\cat-dog-all-data\test-dataset\test"
train_path_list = glob_extensions(train_path)
test_path_list = glob_extensions(test_path)
for path in train_path_list[:100]:
    path = Path(path)
    img = cv_imread(path)
    img = cv2.resize(img, (401, 401))
    img = np.transpose(img, (2, 0, 1))
    if path.parent.name == 'cat':
        train_loader.append([torch.tensor(img), torch.tensor([1, 0])])
    else:
        train_loader.append([torch.tensor(img), torch.tensor([0, 1])])

for path in test_path_list[:100]:
    path = Path(path)
    img = cv_imread(path)
    img = cv2.resize(img, (401, 401))
    mg = np.transpose(img, (2, 0, 1))
    if path.parent.name == 'cat':
        test_loader.append([torch.tensor(img), torch.tensor([1, 0])])
    else:
        test_loader.append([torch.tensor(img), torch.tensor([0, 1])])

loss_count = []
for epoch in range(5):
    for i, (x, y) in enumerate(train_loader):
        # [128, 1, 28, 28]->[128, 16, 14, 14]->[128, 32, 7, 7]->[128, 64, 4, 4]->
        # [128, 64, 2, 2]->[128, 256]->[128, 100]->[128, 10]
        batch_x = Variable(x)
        # [128]
        batch_y = Variable(y)
        # [128,10]
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        # 清空上一步残余更新参数值
        opt.zero_grad()
        loss.backward()
        # 将参数更新值施加到net的parameters上
        opt.step()
        if i % 10 == 0:
            loss_count.append(loss)
        if i % 100 == 0:
            for a, b in test_loader:
                test_x = Variable(a)
                test_y = Variable(b)
                out = model(test_x)
                accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
                print(
                    f'Epoch: {epoch}\nIteration: {i}\nAccuracy: {accuracy.mean() * 100}%'
                )
                print(f'-------------------------')
                break
    torch.save(model, rf'.\pth\CNN_{epoch}.pth')
plt.figure('PyTorch_CNN_Loss')
loss = [l.detach().numpy() for l in loss_count]
plt.plot(loss, label='Loss')
plt.legend()
plt.show()
