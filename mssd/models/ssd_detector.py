from torch import nn

from mssd.models.vgg_ssd import VGGSSD
from mssd.models.ssd_box_head import SSDBoxHead
from mssd.utils.model_zoo import load_state_dict_from_url

model_urls = {
    'vgg': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
}

class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = VGGSSD(cfg)
        self.boxhead = SSDBoxHead(cfg)

        if cfg["model"]["pretrain"]:
            self.backbone.init_from_pretrain(load_state_dict_from_url(model_urls['vgg']))

    def forward(self, images, targets=None):
        features = self.backbone(images)
        out = self.boxhead(features)
        return out
