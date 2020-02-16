from mssd.modeling.anchors.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand([123, 117, 104]),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg["input"]["size"]),
            SubtractMeans([123, 117, 104]),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg["input"]["size"]),
            SubtractMeans([123, 117, 104]),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(), center_variance=0.1, size_variance=0.2, iou_threshold=0.5)
    return transform
