import os

import torch
from torchvision.datasets.coco import CocoDetection
from typing import Optional, Callable


class CocoStyleDataset(CocoDetection):

    img_dirs = {
        'cityscapes': {
            'train': 'cityscapes/leftImg8bit/train', 'val': 'cityscapes/leftImg8bit/val'
        },
        'foggy_cityscapes': {
            'train': 'foggy_cityscapes/leftImg8bit_foggy/train', 'val': 'foggy_cityscapes/leftImg8bit_foggy/val'
        },
        'bdd100k': {
            'train': 'bdd100k/images/100k/train', 'val': 'bdd100k/images/100k/val',
        },
        'sim10k': {
            'train': 'sim10k/JPEGImages'
        },
    }
    anno_files = {
        'cityscapes': {
            'source': {
                'train': 'cityscapes/annotations/cityscapes_train_cocostyle.json',
                'val': 'cityscapes/annotations/cityscapes_val_cocostyle.json',
            },
            'target': {
                'train': 'cityscapes/annotations/cityscapes_train_caronly_cocostyle.json',
                'val': 'cityscapes/annotations/cityscapes_val_caronly_cocostyle.json'
            }
        },
        'foggy_cityscapes': {
            'target': {
                'train': 'foggy_cityscapes/annotations/foggy_cityscapes_train_cocostyle.json',
                'val': 'foggy_cityscapes/annotations/foggy_cityscapes_val_cocostyle.json'
            }
        },
        'bdd100k': {
            'target': {
                'train': 'bdd100k/annotations/bdd100k_daytime_train_cocostyle.json',
                'val': 'bdd100k/annotations/bdd100k_daytime_val_cocostyle.json'
            },
        },
        'sim10k': {
            'source': {
                'train': 'sim10k/annotations/sim10k_train_cocostyle.json',
            },
        },
    }

    def __init__(self,
                 root_dir: str,
                 dataset_name: str,
                 domain: str,
                 split: str,
                 transforms: Optional[Callable] = None):
        # dataset_root = os.path.join(root_dir, dataset_name)
        img_dir = os.path.join(root_dir, self.img_dirs[dataset_name][split])
        self.anno_file = os.path.join(root_dir, self.anno_files[dataset_name][domain][split])
        super(CocoStyleDataset, self).__init__(root=img_dir, annFile=self.anno_file, transforms=transforms)
        self.split = split

    @staticmethod
    def convert(image_id, image, annotation):
        w, h = image.size
        anno = [obj for obj in annotation if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        new_annotation = {
            'boxes': boxes[keep],
            'labels': classes[keep],
            'image_id': torch.as_tensor([image_id]),
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)])
        }
        return image, new_annotation

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image = self._load_image(image_id)
        annotation = self._load_target(image_id)
        image, annotation = self.convert(image_id, image, annotation)
        if self.transforms is not None:
            image, annotation = self.transforms(image, annotation)
        return image, annotation

    @staticmethod
    def pad_mask(tensor_list):
        assert len(tensor_list[0].shape) == 3
        shapes = [list(img.shape) for img in tensor_list]
        max_h, max_w = shapes[0][1], shapes[0][2]
        for shape in shapes[1:]:
            max_h = max(max_h, shape[1])
            max_w = max(max_w, shape[2])
        batch_shape = [len(tensor_list), tensor_list[0].shape[0], max_h, max_w]
        tensor = torch.zeros(batch_shape, dtype=tensor_list[0].dtype, device=tensor_list[0].device)
        mask = torch.ones((len(tensor_list), max_h, max_w), dtype=torch.bool, device=tensor_list[0].device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
        return tensor, mask

    @staticmethod
    def collate_fn(batch):
        """
        Function used in dataloader.
        batch: [sample_{i} for i in range(batch_size)]
            sample_{i}: (image, annotation)
        """
        image_list = [sample[0] for sample in batch]
        batched_images, masks = CocoStyleDataset.pad_mask(image_list)
        annotations = [sample[1] for sample in batch]
        return batched_images, masks, annotations


class CocoStyleDatasetTeaching(CocoStyleDataset):

    def __init__(self,
                 root_dir: str,
                 dataset_name: str,
                 domain: str,
                 split: str,
                 weak_aug: Callable,
                 strong_aug: Callable,
                 final_trans: Callable):
        super(CocoStyleDatasetTeaching, self).__init__(root_dir, dataset_name, domain, split, None)
        self.weak_aug = weak_aug
        self.strong_aug = strong_aug
        self.final_trans = final_trans

    def __getitem__(self, idx):
        image, annotation = super(CocoStyleDatasetTeaching, self).__getitem__(idx)
        teacher_image, annotation = self.weak_aug(image, annotation)
        student_image, _ = self.strong_aug(teacher_image, None)
        teacher_image, annotation = self.final_trans(teacher_image, annotation)
        student_image, _ = self.final_trans(student_image, None)
        return teacher_image, student_image, annotation

    @staticmethod
    def collate_fn_teaching(batch):
        """
        Function used in dataloader.
        batch: [sample_{i} for i in range(batch_size)]
            sample_{i}: (teacher_image, student_image, annotation)
        """
        teacher_image_list = [sample[0] for sample in batch]
        batched_teacher_images, teacher_masks = CocoStyleDataset.pad_mask(teacher_image_list)
        student_image_list = [sample[1] for sample in batch]
        batched_student_images, student_masks = CocoStyleDataset.pad_mask(student_image_list)
        annotations = [sample[2] for sample in batch]
        batched_images = torch.stack([batched_teacher_images, batched_student_images], dim=0)
        assert teacher_masks.equal(student_masks)
        return batched_images, teacher_masks, annotations


class DataPreFetcher:

    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.next_images = None
        self.next_masks = None
        self.next_annotations = None
        self.stream = torch.cuda.Stream()
        self.preload()

    def to_cuda(self):
        self.next_images = self.next_images.to(self.device, non_blocking=True)
        self.next_masks = self.next_masks.to(self.device, non_blocking=True)
        self.next_annotations = [
            {k: v.to(self.device, non_blocking=True) for k, v in t.items()}
            for t in self.next_annotations
        ]

    def preload(self):
        try:
            self.next_images, self.next_masks, self.next_annotations = next(self.loader)
        except StopIteration:
            self.next_images = None
            self.next_masks = None
            self.next_annotations = None
            return
        with torch.cuda.stream(self.stream):
            self.to_cuda()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        images, masks, annotations = self.next_images, self.next_masks, self.next_annotations
        if images is not None:
            self.next_images.record_stream(torch.cuda.current_stream())
        if masks is not None:
            self.next_masks.record_stream(torch.cuda.current_stream())
        if annotations is not None:
            for anno in self.next_annotations:
                for k, v in anno.items():
                    v.record_stream(torch.cuda.current_stream())
        self.preload()
        return images, masks, annotations

