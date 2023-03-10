from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.utils_func import cv_imread, glob_extensions


class CatDogDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.img_list = glob_extensions(self.root_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        try:
            img_path = Path(self.img_list[idx])
            image = cv_imread(img_path)
            if 'cat' in img_path.stem:
                annotation = np.array([1.0, 0.0])
            else:
                annotation = np.array([0.0, 1.0])

            if self.transform:
                image = self.transform(image)

            return image, annotation
        except:
            print(img_path)


if __name__ == "__main__":
    dataset = CatDogDataset(
        root_dir=r'./DATA/cat-dog-all-data/test-dataset/train')
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    for images, annotations in dataloader:
        print(images.shape, annotations)
