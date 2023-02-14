import torch


class CNNnet(torch.nn.Module):

    def __init__(self):
        super(CNNnet, self).__init__()
        # in_channels, out_channels, kernel_size, stride, padding
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, 2, 1), torch.nn.BatchNorm2d(16),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1), torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1), torch.nn.BatchNorm2d(64),
            torch.nn.ReLU())
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0), torch.nn.BatchNorm2d(64),
            torch.nn.ReLU())
        # in_features, out_features
        self.mlp1 = torch.nn.Linear(2 * 2 * 64, 100)
        self.mlp2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x