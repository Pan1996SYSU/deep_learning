import torch


class CNNNet(torch.nn.Module):

    def __init__(self):
        super(CNNNet, self).__init__()
        # [32, 3, 401, 401]->[32, 15, 200, 200]->[32, 30, 99, 99]->[32, 30, 49, 49]->
        # [32, 30, 24, 24]->[32, 30, 11, 11]->[32, 30, 5, 5]->[32, 750]->
        # [32, 300]->[32, 50]->[32, 2]
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
        self.mlp1 = torch.nn.Linear(1470, 500)
        self.mlp2 = torch.nn.Linear(500, 50)
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
