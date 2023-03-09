import torch

from my_dataset import CatDogDataset


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += torch.tensor(X[:, d, :, :], dtype=torch.float32).mean()
            std[d] += torch.tensor(X[:, d, :, :], dtype=torch.float32).std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    train_path = "./DATA/cat-dog-all-data/test-dataset/train"
    dataset = CatDogDataset(root_dir=train_path)
    print(getStat(dataset))
