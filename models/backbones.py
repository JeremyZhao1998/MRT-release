import torch
import torchvision.ops.misc as misc
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
from torchvision.models._utils import IntermediateLayerGetter

from utils import is_main_process


class ResNetMultiScale(torch.nn.Module):

    def __init__(self):
        super().__init__()
        backbone, self.num_channels = self.get_backbone()
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        self.num_outputs = 3
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.strides = [8, 16, 32]
        self.intermediate_getter = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def get_backbone(self):
        raise NotImplementedError('This method should be implemented by subclasses.')

    def forward(self, tensor):
        out = self.intermediate_getter(tensor)
        return [out['0'], out['1'], out['2']]


class ResNet18MultiScale(ResNetMultiScale):

    def __init__(self):
        super().__init__()

    def get_backbone(self):
        return resnet18(
            replace_stride_with_dilation=[False, False, False],
            weights=ResNet18_Weights.DEFAULT if is_main_process() else None,
            norm_layer=misc.FrozenBatchNorm2d
        ), [128, 256, 512]


class ResNet50MultiScale(ResNetMultiScale):

    def __init__(self):
        super().__init__()

    def get_backbone(self):
        return resnet50(
            replace_stride_with_dilation=[False, False, False],
            weights=ResNet50_Weights.IMAGENET1K_V1 if is_main_process() else None,
            norm_layer=misc.FrozenBatchNorm2d
        ), [512, 1024, 2048]


class ResNet101MultiScale(ResNetMultiScale):

    def __init__(self):
        super().__init__()

    def get_backbone(self):
        return resnet101(
            replace_stride_with_dilation=[False, False, False],
            weights=ResNet101_Weights.DEFAULT if is_main_process() else None,
            norm_layer=misc.FrozenBatchNorm2d
        ), [512, 1024, 2048]
