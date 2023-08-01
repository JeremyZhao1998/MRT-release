from .transforms import *


base_trans = ComposeImgAnno([
    ToTensorImgAnno(),
    NormalizeImgAnno()
])


val_trans = ComposeImgAnno([
    ResizeImgAnno(size=800, max_size=1332),
    base_trans
])


train_trans = ComposeImgAnno([
    RandomHorizontalFlipImgAnno(),
    RandomSelectImgAnno(
        RandomResizeImgAnno(sizes=[480, 512, 544, 576, 608, 640, 672], max_size=1333),
        ComposeImgAnno([
            RandomResizeImgAnno(sizes=[400, 500, 600], max_size=1333),
            RandomSizeCropImgAnno(256, 600), # 384
            RandomResizeImgAnno(sizes=[480, 512, 544, 576, 608, 640, 672], max_size=1333),
        ])
    ),
    base_trans
])


weak_aug = ComposeImgAnno([
    RandomHorizontalFlipImgAnno(p=0.5),
    ResizeImgAnno(size=800, max_size=1333)
])


weak_trans = ComposeImgAnno([
    weak_aug,
    base_trans
])


strong_aug = ComposeImgAnno([
    RandomApplyImgAnno(
        [ColorJitterImgAnno(0.4, 0.4, 0.4, 0.1)], p=0.8
    ),
    RandomGrayScaleImgAnno(p=0.2),
    RandomApplyImgAnno(
        [GaussianBlurImgAnno([0.1, 2.0])], p=0.5
    )
])


strong_trans = ComposeImgAnno([
    weak_aug,
    RandomApplyImgAnno(
        [ColorJitterImgAnno(0.4, 0.4, 0.4, 0.1)], p=0.8
    ),
    RandomGrayScaleImgAnno(p=0.2),
    RandomApplyImgAnno(
        [GaussianBlurImgAnno([0.1, 2.0])], p=0.5
    ),
    ToTensorImgAnno(),
    NormalizeImgAnno()
])
