from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
# from torchvision.transforms import Compose, ToTensor, Normalize
# from torchvision.datasets import MNIST

from mssd.data.transforms import build_transforms, build_target_transform
from mssd.data.datasets import build_dataset
from mssd.structures.container import Container

class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])

        if self.is_train:
            list_targets = transposed_batch[1]
            targets = Container(
                {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
            )
        else:
            targets = None
        return images, targets, img_ids

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
                                num_workers=cfg["data_loader"]["num_workers"],
                                collate_fn=BatchCollator(is_train=True))
    
    validate_loader = DataLoader(validate_dataset,
                                batch_size=cfg["validate"]["batch_size"],
                                shuffle=False,
                                num_workers=cfg["data_loader"]["num_workers"],
                                collate_fn=BatchCollator(is_train=False))

    # data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    # train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
    #                             batch_size=cfg["training"]["batch_size"], shuffle=True)
    # validate_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
    #                             batch_size=cfg["validate"]["batch_size"], shuffle=False)

    return train_loader, validate_loader
