import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from my_dataset import CustomDataset
'''
猫狗分类
训练集：22500张
测试集：2500张
40张/批
训练12代
准确率：92.92%
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(r'.\pth\ResNet_11.pth')
accuracy_sum = []

test_path = r".\DATA\\cat-dog-all-data\test-dataset\test"
dataset = CustomDataset(root_dir=test_path)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

for i, (images, annotations) in enumerate(dataloader):
    images = images.permute(0, 3, 2, 1).to(torch.float32)
    images = images.to(device)
    annotations = annotations.to(device)
    test_x = Variable(images)
    test_y = Variable(annotations)
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
