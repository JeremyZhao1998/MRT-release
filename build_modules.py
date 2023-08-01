import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler
from datasets.coco_style_dataset import CocoStyleDataset, CocoStyleDatasetTeaching
from models.backbones import ResNet50MultiScale, ResNet18MultiScale, ResNet101MultiScale
from models.positional_encoding import PositionEncodingSine
from models.deformable_detr import DeformableDETR
from models.deformable_transformer import DeformableTransformer
from models.criterion import SetCriterion
from datasets.augmentations import weak_aug, strong_aug, base_trans


def build_sampler(args, dataset, split):
    if split == 'train':
        if args.distributed:
            sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        else:
            sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    else:
        if args.distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, args.eval_batch_size, drop_last=False)
    return batch_sampler


def build_dataloader(args, dataset_name, domain, split, trans):
    dataset = CocoStyleDataset(root_dir=args.data_root,
                               dataset_name=dataset_name,
                               domain=domain,
                               split=split,
                               transforms=trans)
    batch_sampler = build_sampler(args, dataset, split)
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=CocoStyleDataset.collate_fn,
                             num_workers=args.num_workers)
    return data_loader


def build_dataloader_teaching(args, dataset_name, domain, split):
    dataset = CocoStyleDatasetTeaching(root_dir=args.data_root,
                                       dataset_name=dataset_name,
                                       domain=domain,
                                       split=split,
                                       weak_aug=weak_aug,
                                       strong_aug=strong_aug,
                                       final_trans=base_trans)
    batch_sampler = build_sampler(args, dataset, split)
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=CocoStyleDatasetTeaching.collate_fn_teaching,
                             num_workers=args.num_workers)
    return data_loader


def build_model(args, device):
    if args.backbone == 'resnet50':
        backbone = ResNet50MultiScale()
    elif args.backbone == 'resnet18':
        backbone = ResNet18MultiScale()
    elif args.backbone == 'resnet101':
        backbone = ResNet101MultiScale()
    else:
        raise ValueError('Invalid args.backbone name: ' + args.backbone)
    position_encoding = PositionEncodingSine()
    transformer = DeformableTransformer(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout
    )
    model = DeformableDETR(
        backbone=backbone,
        position_encoding=position_encoding,
        transformer=transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels
    )
    model.to(device)
    return model


def build_criterion(args, device, box_loss=True):
    criterion = SetCriterion(
        num_classes=args.num_classes,
        coef_class=args.coef_class,
        coef_boxes=args.coef_boxes if box_loss else 0.0,
        coef_giou=args.coef_giou if box_loss else 0.0,
        coef_domain=args.coef_domain,
        coef_domain_bac=args.coef_domain_bac,
        coef_mae=args.coef_mae,
        alpha_focal=args.alpha_focal,
        alpha_dt=args.alpha_dt,
        gamma_dt=args.gamma_dt,
        max_dt=args.max_dt,
        device=device
    )
    criterion.to(device)
    return criterion


def build_optimizer(args, model, enable_mae=False):
    params_backbone = [param for name, param in model.named_parameters()
                       if 'backbone' in name]
    params_linear_proj = [param for name, param in model.named_parameters()
                          if 'reference_points' in name or 'sampling_offsets' in name]
    params = [param for name, param in model.named_parameters()
              if 'backbone' not in name and 'reference_points' not in name and 'sampling_offsets' not in name]
    param_dicts = [
        {'params': params, 'lr': args.lr},
        {'params': params_backbone, 'lr': 0.0 if enable_mae else args.lr_backbone},
        {'params': params_linear_proj, 'lr': args.lr_linear_proj},
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def build_teacher(args, student_model, device):
    teacher_model = build_model(args, device)
    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
    for key, value in state_dict.items():
        state_dict[key] = student_state_dict[key].clone().detach()
    teacher_model.load_state_dict(state_dict)
    return teacher_model
