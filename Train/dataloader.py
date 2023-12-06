import torch
from torch import nn
import torchvision
import time 
import numpy as np
import scipy.io as scio
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class MNIST_vor(Dataset):

    base_folder = ""
    # 
    file_name = "mnist_vor.mat"
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        download: bool = False,
    ) -> None:

        super().__init__()

        self.train = train  # training set or test set
        self.transform = transform
        self.root = root
        file_path = os.path.join(self.root, self.base_folder, self.file_name)
        print(file_path)
        data = scio.loadmat(file_path)
        train_data = data['train_x_vor']
        test_data = data['test_x_vor']
        train_label = data['train_y']
        test_label = data['test_y']

        self.data: Any = []
        self.targets = []

        if train:
            self.data = train_data
            self.targets = train_label
        else:
            self.data = test_data
            self.targets = test_label

        self.data = np.vstack(self.data).reshape(-1, 1, 28, 28)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.argmax(self.targets, axis=1)


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.data[index], self.targets[index]
        # print(img.shape)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img.astype(np.uint8)
        img = np.squeeze(img)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


    def __len__(self) -> int:
        return len(self.data)

class Fashion_vor(Dataset):

    base_folder = ""
    file_name = "fashion_vor.mat"
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        download: bool = False,
    ) -> None:

        super().__init__()

        self.train = train  # training set or test set
        self.transform = transform
        self.root = root
        file_path = os.path.join(self.root, self.base_folder, self.file_name)
        print(file_path)
        data = scio.loadmat(file_path)
        train_data = data['train_x_vor']
        test_data = data['test_x_vor']
        train_label = data['train_y']
        test_label = data['test_y']

        self.data: Any = []
        self.targets = []

        if train:
            self.data = train_data
            self.targets = train_label
        else:
            self.data = test_data
            self.targets = test_label

        self.data = np.vstack(self.data).reshape(-1, 1, 28, 28)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.argmax(self.targets, axis=1)
        # print(self.data.shape)


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.data[index], self.targets[index]
        # print(img.shape)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img.astype(np.uint8)
        img = np.squeeze(img)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


    def __len__(self) -> int:
        return len(self.data)

class MNIST_random(Dataset):
    base_folder = ""
    file_list = {
        "train": ["mnist_train_random_v1.mat"],
        "test":  ["mnist_test_random_v1.mat"]
    }
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        download: bool = False,
    ) -> None:

        super().__init__()

        self.train = train  # training set or test set
        self.transform = transform
        self.root = root
        if self.train:
            downloaded_list = self.file_list['train']
        else:
            downloaded_list = self.file_list['test']

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            print(file_path)
            entry = scio.loadmat(file_path)

            if self.train: 
                self.data.append(entry["train_x_vor"])
                self.targets.extend(entry["train_y"])
            else:
                self.data.append(entry["test_x_vor"])
                self.targets.extend(entry["test_y"])

        self.data = np.vstack(self.data).reshape(-1, 1, 28, 28)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.argmax(self.targets, axis=1)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img.astype(np.uint8)
        img = np.squeeze(img)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


    def __len__(self) -> int:
        return len(self.data)

class Fashion_random(Dataset):
    base_folder = ""
    file_list = {
        "train": ["fashion_train_random_v1.mat"],
        "test":  ["fashion_test_random_v1.mat"]
    }
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        download: bool = False,
    ) -> None:

        super().__init__()

        self.train = train  # training set or test set
        self.transform = transform
        self.root = root
        if self.train:
            downloaded_list = self.file_list['train']
        else:
            downloaded_list = self.file_list['test']

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            print(file_path)
            entry = scio.loadmat(file_path)

            if self.train: 
                self.data.append(entry["train_x_vor"])
                self.targets.extend(entry["train_y"])
            else:
                self.data.append(entry["test_x_vor"])
                self.targets.extend(entry["test_y"])

        self.data = np.vstack(self.data).reshape(-1, 1, 28, 28)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.argmax(self.targets, axis=1)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img.astype(np.uint8)
        img = np.squeeze(img)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


    def __len__(self) -> int:
        return len(self.data)
    
class CIFAR10_vor(Dataset):
    base_folder = ""
    file_list = {
        "train": ["cifar_vor_train.mat"],
        "test":  ["cifar_vor_test.mat"]
    }
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        download: bool = False,
    ) -> None:

        super().__init__()

        self.train = train  # training set or test set
        self.transform = transform
        self.root = root
        if self.train:
            downloaded_list = self.file_list['train']
        else:
            downloaded_list = self.file_list['test']

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            print(file_path)
            entry = scio.loadmat(file_path)
            # print(entry.keys())
            if self.train: 
                self.data.append(entry["train_x_vor"])
                self.targets.extend(entry["labels"])
            else:
                self.data.append(entry["test_x_vor"])
                self.targets.extend(entry["labels_test"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = torch.tensor(self.targets)
        self.targets = self.targets.reshape(self.targets.size(0))


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


    def __len__(self) -> int:
        return len(self.data)

class CIFAR10_random(Dataset):
    base_folder = ""
    file_list = {
        "train": ["cifar_train_random_v1.mat"],
        "test":  ["cifar_test_random_v1.mat"]
    }
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        download: bool = False,
    ) -> None:

        super().__init__()

        self.train = train  # training set or test set
        self.transform = transform
        self.root = root
        if self.train:
            downloaded_list = self.file_list['train']
        else:
            downloaded_list = self.file_list['test']

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            print(file_path)
            entry = scio.loadmat(file_path)
            # print(entry.keys())
            if self.train: 
                self.data.append(entry["train_x_vor"])
                self.targets.extend(entry["labels"])
            else:
                self.data.append(entry["test_x_vor"])
                self.targets.extend(entry["labels_test"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = torch.tensor(self.targets)
        self.targets = self.targets.reshape(self.targets.size(0))


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        # target = target.resize(target.size(0))

        return img, target


    def __len__(self) -> int:
        return len(self.data)


def dataloader(name, batch_size=128, img_size=(28, 28), num_workers=32):
    BATCH_SIZE = batch_size

    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])

    train_transform_cifar = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform_cifar = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if name == "mnist_vor":
        train_set = MNIST_vor(root="../DataProcess/dataset/mnist/vor",
                            train=True,
                            transform=train_transform)
        test_set = MNIST_vor(root="../DataProcess/dataset/mnist/vor",
                            train=False,
                            transform=test_transform)
    elif name == "mnist":
        train_set = torchvision.datasets.MNIST(root='../DataProcess/dataset/mnist/origin', train=True, transform=train_transform, download=True)
        test_set = torchvision.datasets.MNIST(root='../DataProcess/dataset/mnist/origin', train=False, transform=test_transform, download=True)
    elif name == "mnist_random":
        train_set = MNIST_random(root="../DataProcess/dataset/mnist/random",
                            train=True,
                            transform=train_transform)
        test_set = MNIST_random(root="../DataProcess/dataset/mnist/random",
                            train=False,
                            transform=test_transform)

    elif name == "fashion_vor":
        train_set = Fashion_vor(root="../DataProcess/dataset/fashion/vor",
                            train=True,
                            transform=train_transform)
        test_set = Fashion_vor(root="../DataProcess/dataset/fashion/vor",
                            train=False,
                            transform=test_transform)
    elif name == "fashion":
        train_set = torchvision.datasets.FashionMNIST(root='../DataProcess/dataset/fashion/origin', train=True, transform=train_transform, download=True)
        test_set = torchvision.datasets.FashionMNIST(root='../DataProcess/dataset/fashion/origin', train=False, transform=test_transform, download=True)
    elif name == "fashion_random":
        train_set = Fashion_random(root="../DataProcess/dataset/fashion/random",
                            train=True,
                            transform=train_transform)
        test_set = Fashion_random(root="../DataProcess/dataset/fashion/random",
                            train=False,
                            transform=test_transform)
    
    elif name == "cifar10":
        train_set = torchvision.datasets.CIFAR10(root='../DataProcess/dataset/cifar/origin', train=True, transform=train_transform_cifar, download=True)
        test_set = torchvision.datasets.CIFAR10(root='../DataProcess/dataset/cifar/origin', train=False, transform=test_transform_cifar, download=True)
    elif name == "cifar10_vor":
        train_set = CIFAR10_vor(root="../DataProcess/dataset/cifar/vor",
                            train=True,
                            transform=train_transform_cifar)
        test_set = CIFAR10_vor(root="../DataProcess/dataset/cifar/vor",
                            train=False,
                            transform=test_transform_cifar)
    elif name == "cifar10_random":
        train_set = CIFAR10_random(root="../DataProcess/dataset/cifar/random",
                            train=True,
                            transform=train_transform_cifar)
        test_set = CIFAR10_random(root="../DataProcess/dataset/cifar/random",
                            train=False,
                            transform=test_transform_cifar)
    
    else:
        raise Exception("could not find the dataset!")

    # dataloader
    train_loader = DataLoader(train_set,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=num_workers)
    test_loader = DataLoader(test_set,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=num_workers)
    return train_loader, test_loader
    
if __name__ == "__main__":
    train, test = dataloader()
    
