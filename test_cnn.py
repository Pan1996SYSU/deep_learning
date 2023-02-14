import torch
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量
import torchvision
from torchvision.datasets import mnist # 获取数据集
import matplotlib.pyplot as plt

data_tf = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])
    ])

data_path = r'.\DATA'
# 获取数据集
test_data = mnist.MNIST(
    data_path, train=False, transform=data_tf, download=False)

test_loader = data.DataLoader(test_data, batch_size=100, shuffle=True)
# 测试网络
model = torch.load(r'.\pth\4.pth')
accuracy_sum = []
for i,(test_x,test_y) in enumerate(test_loader):
    test_x = Variable(test_x)
    test_y = Variable(test_y)
    out = model(test_x)
    accuracy = torch.max(out,1)[1].numpy() == test_y.numpy()
    accuracy_sum.append(accuracy.mean())
    print('accuracy:\t',accuracy.mean())

print('总准确率：\t',sum(accuracy_sum)/len(accuracy_sum))
# 精确率图
print('总准确率：\t',sum(accuracy_sum)/len(accuracy_sum))
plt.figure('Accuracy')
plt.plot(accuracy_sum,'o',label='accuracy')
plt.title('Pytorch_CNN_Accuracy')
plt.legend()
plt.show()

