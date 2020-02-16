from torch.utils.data import DataLoader
# from torchvision.transforms import Compose, ToTensor, Normalize
# from torchvision.datasets import MNIST

from mssd.data.transforms import build_transforms, build_target_transform
from mssd.data.datasets import build_dataset


def get_data_loaders(cfg):
    train_transform = build_transforms(cfg, is_train=True)
    validate_transform = build_transforms(cfg, is_train=False)
    target_transform = build_target_transform(cfg)

    train_dataset_list = cfg["datasets"]["train"]
    validate_dataset_list = cfg["datasets"]["test"]

    train_dataset = build_dataset(cfg, train_dataset_list, transform=train_transform, target_transform=target_transform, is_train=True)
    validate_dataset = build_dataset(cfg, validate_dataset_list, transform=validate_transform, target_transform=None, is_train=False)

    train_loader = DataLoader(train_dataset,
                                batch_size=cfg["training"]["batch_size"],
                                shuffle=True,
                                num_workers=cfg["data_loader"]["num_workers"])
    
    validate_loader = DataLoader(validate_dataset,
                                batch_size=cfg["validate"]["batch_size"],
                                shuffle=False,
                                num_workers=cfg["data_loader"]["num_workers"])

    # data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    # train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
    #                             batch_size=cfg["training"]["batch_size"], shuffle=True)
    # validate_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
    #                             batch_size=cfg["validate"]["batch_size"], shuffle=False)

    return train_loader, validate_loader
