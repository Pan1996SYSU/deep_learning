import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models, transforms
from my_dataset import CatDogDataset
'''
猫狗分类
训练集：22500张
测试集：2500张
30张/批
训练12代
准确率：93.96%
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('./pth/ResNet_15.pth')
model.cuda(0)
accuracy_sum = []
# normalize = transforms.Normalize(
#     mean=[106.35824316, 116.09900846, 124.61032364], std=[57.35260147, 57.33807308, 58.44982434])
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((401, 401)),
        # normalize,
    ])
test_path = "./DATA/cat-dog-all-data/cat-dog-all-data/test-dataset/test"
dataset = CatDogDataset(root_dir=test_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

for i, (images, annotations) in enumerate(dataloader):
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
