import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from torch.utils import data
from torchvision.datasets import mnist

data_tf = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])
    ])

data_path = r'.\DATA'

test_data = mnist.MNIST(
    data_path, train=False, transform=data_tf, download=False)

test_loader = data.DataLoader(test_data, batch_size=1000, shuffle=True)

model = torch.load(r'.\pth\4.pth')
accuracy_sum = []

for i, (test_x, test_y) in enumerate(test_loader):
    test_x = Variable(test_x.cuda[0])
    test_y = Variable(test_y.cuda[0])
    out = model(test_x)
    accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
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
