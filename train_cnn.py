import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from torch.utils import data
from torchvision.datasets import mnist

from CNN import CNNnet

# 数据集的预处理
data_tf = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])
    ])

data_path = r'.\DATA'
# 获取数据集
train_data = mnist.MNIST(
    data_path, train=True, transform=data_tf, download=False)
test_data = mnist.MNIST(
    data_path, train=False, transform=data_tf, download=False)

train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=100, shuffle=True)

model = CNNnet()

loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

loss_count = []
for epoch in range(5):
    for i, (x, y) in enumerate(train_loader):
        batch_x = Variable(x)  # torch.Size([128, 1, 28, 28])
        batch_y = Variable(y)  # torch.Size([128])
        # 获取最后输出
        out = model(batch_x)  # torch.Size([128,10])
        # 获取损失
        loss = loss_func(out, batch_y)
        # 使用优化器优化损失
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward()  # 误差反向传播，计算参数更新值
        opt.step()  # 将参数更新值施加到net的parmeters上
        if i % 20 == 0:
            loss_count.append(loss)
            print('{}:\t'.format(i), loss.item())
        if i % 100 == 0:
            for a, b in test_loader:
                test_x = Variable(a)
                test_y = Variable(b)
                out = model(test_x)
                # print('test_out:\t',torch.max(out,1)[1])
                # print('test_y:\t',test_y)
                accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
                print('accuracy:\t', accuracy.mean())
                break
    torch.save(model, rf'.\pth\{epoch}.pth')
plt.figure('PyTorch_CNN_Loss')
loss = [l.detach().numpy() for l in loss_count]
plt.plot(loss, label='Loss')
plt.legend()
plt.show()
