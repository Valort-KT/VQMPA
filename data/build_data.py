# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import queue
from typing import Dict, Sequence
import warnings
import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from data.folder import ImageFolder
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform


from .samplers import SubsetRandomSampler

def build_loader(config):

    config.defrost()
    dataset_train, _ = build_dataset(is_train=True, config=config)
    config.freeze()
    le = len(dataset_train)
    # print(f"global rank {dist.get_rank()} successfully build train dataset // len = {le}")
    print(f"successfully build train dataset // len = {le}")
    


    # sampler_train = torch.utils.data.DistributedSampler(
    #     dataset_train,  shuffle=True
    # )

    data_loader_train = torch.utils.data.DataLoader(
        # dataset_train, sampler=sampler_train,
        dataset_train,
        shuffle=True,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        persistent_workers=True,
        # prefetch_factor=24,
    )

    #-----------------------------------Val Dataset-----------------------------------

    dataset_val, _ = build_dataset(is_train=False, config=config)
    lev = len(dataset_val)
    print(f"global ranksuccessfully build val dataset // len = {lev}")
    #  {dist.get_rank()} 

    # indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    # sampler_val = SubsetRandomSampler(indices)

    data_loader_val = torch.utils.data.DataLoader(
        # dataset_val, sampler=sampler_val,
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        persistent_workers=True
    )
    
    # setup mixup / cutmix
    # mixup_fn = None
    # mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    # if mixup_active:
    #     mixup_fn = Mixup(
    #         mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
    #         prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
    #         label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    #     return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn

    return dataset_train, dataset_val, data_loader_train, data_loader_val

def buildval_loader(config):
    dataset_val, _ = build_dataset(is_train=False, config=config)
    lev = len(dataset_val)
    print(f"global ranksuccessfully build val dataset // len = {lev}")
    #  {dist.get_rank()} 

    # indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    # sampler_val = SubsetRandomSampler(indices)

    data_loader_val = torch.utils.data.DataLoader(
        # dataset_val, sampler=sampler_val,
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True
    )
    
    # setup mixup / cutmix
    # mixup_fn = None
    # mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    # if mixup_active:
    #     mixup_fn = Mixup(
    #         mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
    #         prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
    #         label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    #     return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn

    return dataset_val, data_loader_val


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22K':
        if is_train:
            root = config.DATA.DATA_PATH
        else:
            root = config.DATA.EVAL_DATA_PATH
        dataset = ImageFolder(root, transform=transform)
        nb_classes = 21841
    elif config.DATA.DATASET == "two_plas":
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = ImageFolder(root, transform=transform)
        nb_classes = config.MODEL.NUM_CLASSES
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    # if is_train:
    #     # this should always dispatch to transforms_imagenet_train
    #     transform = create_transform(
    #         input_size=config.DATA.IMG_SIZE,
    #         is_training=True,
    #         color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
    #         auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
    #         re_prob=config.AUG.REPROB,
    #         re_mode=config.AUG.REMODE,
    #         re_count=config.AUG.RECOUNT,
    #         interpolation=config.DATA.INTERPOLATION,
    #     )
    #     if not resize_im:
    #         # replace RandomResizedCropAndInterpolation with
    #         # RandomCrop
    #         transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
    #     return transform

    t = []
    if resize_im:
        if config.DATA.IMG_SIZE > 224:  
            t.append(
            transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {config.DATA.IMG_SIZE} size input images...")
        elif config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=transforms.InterpolationMode.BICUBIC)
            )

    t.append(transforms.ToTensor())
    # t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
