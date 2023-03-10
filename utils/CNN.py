import torch


class CNNNet(torch.nn.Module):

    def __init__(self):
        super(CNNNet, self).__init__()
        # [50, 3, 401, 401]->[50, 15, 201, 201]->[50, 30, 101, 101]->[50, 30, 51, 51]->
        # [50, 30, 26, 26]->[50, 30, 13, 13]->[50, 30, 7, 7]->[50, 1470]->
        # [50, 500]->[50, 50]->[50, 2]
        # in_channels, out_channels, kernel_size, stride, padding
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 15, 3, 2, 1), torch.nn.BatchNorm2d(15),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(15, 30, 3, 2, 1), torch.nn.BatchNorm2d(30),
            torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(30, 30, 3, 2, 1), torch.nn.BatchNorm2d(30),
            torch.nn.ReLU())
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(30, 30, 3, 2, 1), torch.nn.BatchNorm2d(30),
            torch.nn.ReLU())
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(30, 30, 3, 2, 1), torch.nn.BatchNorm2d(30),
            torch.nn.ReLU())
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(30, 30, 3, 2, 1), torch.nn.BatchNorm2d(30),
            torch.nn.ReLU())
        # in_features, out_features
        self.mlp1 = torch.nn.Linear(1470, 500, torch.nn.Sigmoid())
        self.mlp2 = torch.nn.Linear(500, 50, torch.nn.Sigmoid())
        self.mlp3 = torch.nn.Linear(50, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x
