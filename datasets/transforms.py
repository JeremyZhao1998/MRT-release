import random

import torch
import torchvision.transforms as tv_trans
import torchvision.transforms.functional as tv_f
from PIL import ImageFilter

from utils import box_xyxy_to_cxcywh


class ResizeImgAnno(tv_trans.Resize):
    """
    Resize the image for the shortest edge to be a fixed size(size).
    If longest edge is longer than max_size, than resize the image for the longest size to be max_size.
    When doing resize, resize the boxes at the same time.
    """
    def __init__(self, size=800, max_size=1333):
        super(ResizeImgAnno, self).__init__(size, max_size=max_size)

    def forward(self, image, annotation=None):
        width, height = image.size
        image = super(ResizeImgAnno, self).forward(image)
        new_width, new_height = image.size
        if annotation is None:
            return image, annotation
        new_annotation = annotation.copy()
        ratio_w, ratio_h = new_width / width, new_height / height
        boxes = new_annotation['boxes']
        new_annotation['boxes'] = boxes * torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h])
        new_annotation['size'] = torch.tensor([new_height, new_width])
        return image, new_annotation


class RandomResizeImgAnno:
    """
    Randomly choose a size from sizes to resize the image and boxes
    """
    def __init__(self, sizes, max_size=1333):
        self.resize = [
            ResizeImgAnno(size=s, max_size=max_size) for s in sizes
        ]

    def __call__(self, image, annotation=None):
        resize = random.choice(self.resize)
        return resize(image, annotation)


class RandomSizeCropImgAnno(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    @staticmethod
    def get_region(image, th, tw):
        w, h = image.size
        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th + 1)
        j = random.randint(0, w - tw + 1)
        # i = torch.randint(0, h - th + 1, size=(1,)).item()
        # j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __call__(self, image, annotation=None):
        w = random.randint(self.min_size, min(image.width, self.max_size))
        h = random.randint(self.min_size, min(image.height, self.max_size))
        region = self.get_region(image, h, w)
        image = tv_f.crop(image, *region)
        if annotation is None:
            return image, annotation
        new_annotation = annotation.copy()
        i, j, h, w = region
        boxes = new_annotation["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0).reshape(-1, 4)
        tmp = cropped_boxes.reshape(-1, 2, 2)
        keep = torch.all(torch.gt(tmp[:, 1, :], tmp[:, 0, :]), dim=1)
        new_annotation['boxes'] = cropped_boxes[keep]
        new_annotation['labels'] = new_annotation['labels'][keep]
        new_annotation['size'] = torch.tensor([h, w])
        return image, new_annotation


class RandomHorizontalFlipImgAnno(tv_trans.RandomHorizontalFlip):
    """
    Random horizontal flip. When doing flip, flip the boxes at the same time.
    """
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipImgAnno, self).__init__(p)

    def forward(self, image, annotation=None):
        new_annotation = annotation.copy()
        if random.random() < self.p:
            image = tv_f.hflip(image)
            if annotation is not None:
                width, height = image.size
                boxes = new_annotation["boxes"]
                boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([width, 0, width, 0])
                new_annotation["boxes"] = boxes
        return image, new_annotation


class RandomApplyImgAnno(tv_trans.RandomApply):

    def __init__(self, transforms, p=0.5):
        super(RandomApplyImgAnno, self).__init__(transforms, p)

    def forward(self, image, annotation=None):
        if self.p < torch.rand(1):
            return image, annotation
        for t in self.transforms:
            image, annotation = t(image, annotation)
        return image, annotation


class RandomSelectImgAnno:
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, image, annotation):
        if random.random() < self.p:
            return self.transforms1(image, annotation)
        return self.transforms2(image, annotation)


class ColorJitterImgAnno(tv_trans.ColorJitter):
    """
    Color jitter, keep annotation
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(ColorJitterImgAnno, self).__init__(brightness, contrast, saturation, hue)

    def forward(self, image, annotation=None):
        return super(ColorJitterImgAnno, self).forward(image), annotation


class RandomGrayScaleImgAnno(tv_trans.RandomGrayscale):
    """
    Random grayscale, keep annotation
    """
    def __init__(self, p=0.1):
        super(RandomGrayScaleImgAnno, self).__init__(p)

    def forward(self, image, annotation=None):
        return super(RandomGrayScaleImgAnno, self).forward(image), annotation


class GaussianBlurImgAnno:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as described in SimCLR.
    """
    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [0.1, 2.0]
        self.sigma = sigma

    def __call__(self, image, annotation=None):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return image, annotation


class RandomErasingImgAnno(tv_trans.RandomErasing):
    """
    Random erasing, keep annotation
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super(RandomErasingImgAnno, self).__init__(p, scale, ratio, value, inplace)

    def forward(self, image, annotation=None):
        return super(RandomErasingImgAnno, self).forward(image), annotation


class ToTensorImgAnno(tv_trans.ToTensor):
    """
    Convert PIL image to Tensor and keep annotation.
    """
    def __call__(self, image, annotation=None):
        return super(ToTensorImgAnno, self).__call__(image), annotation


class ToPILImgAnno(tv_trans.ToPILImage):
    """
    Convert Tensor to PIL image and keep annotation.
    """
    def __call__(self, image, annotation=None):
        return super(ToPILImgAnno, self).__call__(image), annotation


class NormalizeImgAnno(tv_trans.Normalize):
    """
    Normalize image with mean and std
    and convert box from [x, y, x, y] to [cx, cy, w, h]
    """
    def __init__(self, mean=None, std=None, inplace=False, norm_image=True):
        super(NormalizeImgAnno, self).__init__(mean, std, inplace)
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.mean = mean
        self.std = std
        self.norm_image = norm_image

    def forward(self, image, annotation=None):
        if self.norm_image:
            image = super(NormalizeImgAnno, self).forward(image)
        if annotation is None:
            return image, None
        h, w = image.shape[-2:]
        new_annotation = annotation.copy()
        boxes = new_annotation["boxes"]
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        new_annotation["boxes"] = boxes
        return image, new_annotation


class ComposeImgAnno(tv_trans.Compose):
    """
    Compose multiple transforms on image and annotation.
    """
    def __init__(self, transforms):
        super(ComposeImgAnno, self).__init__(transforms)

    def __call__(self, image, annotation=None):
        for t in self.transforms:
            image, annotation = t(image, annotation)
        return image, annotation
