import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models

from my_dataset import CatDogDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet152()
# num_features = model.fc.out_features
model.fc = torch.nn.Linear(512, 2)
model.cuda(0)

loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = []

train_path = r".\DATA\dogs-vs-cats\train\train"
dataset = CatDogDataset(root_dir=train_path)
dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
loss_count = []
for epoch in range(12):
    for i, (images, annotations) in enumerate(dataloader):
        images = images.permute(0, 3, 2, 1).to(torch.float32)
        images = images.to(device)
        annotations = annotations.to(device)
        batch_x = Variable(images)
        batch_y = Variable(annotations)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        # 梯度清零
        opt.zero_grad()
        # 反向传播
        loss.backward()
        # 根据梯度更新网络参数
        opt.step()
        if i % 500 == 0:
            loss_count.append(loss)
    torch.save(model, rf'.\pth\ResNet_{epoch}.pth')
plt.figure('PyTorch_ResNet_Loss')
loss = [l.cpu().detach().numpy() for l in loss_count]
plt.plot(loss, label='Loss')
plt.legend()
plt.show()
