from torchvision import datasets
from transforms import get_transform
from torch.utils.data import DataLoader

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