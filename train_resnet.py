import copy

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models, transforms

from my_dataset import CatDogDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet152(pretrained=True)
model.fc = torch.nn.Sequential(
                                torch.nn.Linear(2048, 512, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Linear(512, 64, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Linear(64, 8, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Linear(8, 2, bias=True),
)
model.cuda(0)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
train_loader = []

normalize = transforms.Normalize(
    mean=[106.35824316, 116.09900846, 124.61032364], std=[57.35260147, 57.33807308, 58.44982434])
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((401, 401)),
        normalize,
    ])

train_path = "./DATA/cat-dog-all-data/cat-dog-all-data/test-dataset/train"
dataset = CatDogDataset(root_dir=train_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=15, shuffle=True)

val_path = "./DATA/cat-dog-all-data/cat-dog-all-data/test-dataset/test"
val_dataset = CatDogDataset(root_dir=val_path, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=5, shuffle=True)

loss_count = []

best_model_weights = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(20):
    best_model_weights = copy.deepcopy(model.state_dict())
    for i, (images, annotations) in enumerate(dataloader):
        images = images.to(device)
        annotations = annotations.to(device)
        batch_x = Variable(images)
        batch_y = Variable(annotations)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        # ????????????
        optimizer.zero_grad()
        # ????????????
        loss.backward()
        # ??????????????????????????????
        optimizer.step()
        if i % 300 == 0:
            loss_count.append(loss)
            print(f'epoch: {epoch}')
            print(f'Iteration: {i}')
            print(f'loss: {loss}')
            print('-----------------------')
    print(f'????????????ResNet_{epoch}.pth...')
    torch.save(model, f'./pth/ResNet_{epoch}.pth')
    print(f'ResNet_{epoch}.pth????????????')
    print('-----------------------')
    accuracy_sum = []
    for j, (val_images, val_annotations) in enumerate(val_dataloader):
        val_images = val_images.to(device)
        val_annotations = val_annotations.to(device)
        val_test_x = Variable(val_images)
        val_test_y = Variable(val_annotations)
        val_out = model(val_test_x)
        accuracy = torch.max(val_out.cpu(),
                             1)[1].numpy() == torch.max(val_test_y.cpu(),
                                                        1)[1].numpy()
        accuracy_sum.append(accuracy.mean())
    accuracy = round(sum(accuracy_sum) / len(accuracy_sum) * 100, 2)
    print(f'accuracy: {accuracy}%')
    print('-----------------------')
    if accuracy > best_acc:
        best_acc = accuracy
        best_model_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_weights)

plt.figure('PyTorch_ResNet_Loss')
loss = [l.cpu().detach().numpy() for l in loss_count]
plt.plot(loss, label='Loss')
plt.legend()
plt.show()
