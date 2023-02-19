
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

# from CNN import CNNNet
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = CNNNet()
model = torch.load(r'.\pth\CNN_11.pth')
model.cuda(0)
# model = torch.load(r'.\pth\4.pth')

loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = []

train_path = r".\DATA\cat-dog-all-data\test-dataset\train"
train_path_list = glob_extensions(train_path)
random.shuffle(train_path_list)

num = len(train_path_list)

loss_count = []
for epoch in range(4):
    for i, img_path in enumerate(train_path_list):
        if i % 1000 == 0:
            print(f'epoch:{epoch}, {round((i / num) * 100, 2)}%')
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
                img = cv_imread(train_path_list[i + j])
                img = cv2.resize(img, (401, 401))
                img = np.transpose(img, (2, 0, 1))
                img = torch.unsqueeze(
                    torch.as_tensor(img, device=device), dim=0)
                img = img.to(torch.float32)
                if Path(train_path_list[i + j]).parent.name == 'cat':
                    y = torch.tensor([1, 0])
                else:
                    y = torch.tensor([0, 1])
                y = torch.unsqueeze(torch.as_tensor(y, device=device), dim=0)
                y = y.to(torch.float32)
                batch_img = torch.cat((batch_img, img))
                b_y = torch.cat((b_y, y))
        batch_x = Variable(batch_img)
        batch_y = Variable(b_y)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        # 清空上一步残余更新参数值
        opt.zero_grad()
        loss.backward()
        # 将参数更新值施加到net的parameters上
        opt.step()
        if i % 500 == 0:
            loss_count.append(loss)
    torch.save(model, rf'.\pth\CNN_{epoch}.pth')
plt.figure('PyTorch_CNN_Loss')
loss = [l.cpu().detach().numpy() for l in loss_count]
plt.plot(loss, label='Loss')
plt.legend()
plt.show()
