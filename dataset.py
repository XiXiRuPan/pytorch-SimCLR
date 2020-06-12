import os

import cv2
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transforms import get_transform

def get_data_loaders(image_shape, train_path, test_path, train_batch_size, test_batch_size):

    train_transform = get_transform(image_shape)
    test_transform = get_transform(image_shape)

    # define train and test datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

    # define train and test loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size)

    return train_loader, test_loader

class SimCLRDataset(Dataset):
    """Dataset for loading raw and transformed images."""

    def __init__(self, root_dir, image_shape):

        class_dirs = os.listdir(root_dir)
        self.num_classes = len(class_dirs)
        self.classes = sorted(class_dirs)
        self.image_paths = []

        # save image paths
        for class_label in self.classes:
            images = os.listdir(os.path.join(root_dir, class_label))
            for image in images:
                self.image_paths.append(os.path.join(root_dir, class_label, image))

        height, width = image_shape[:2]
        self.transform = transforms.Compose(
                    [transforms.Resize((width, height)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        raw_image = cv2.imread(image_path)
        transformed_image = self.transform(raw_image)

        return raw_image, transformed_image
