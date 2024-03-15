import os
import yaml
from yacs.config import CfgNode as CN
from configs.imagenet_config import get_image_config
from configs.microplas.two_plas_config import get_two_plas_config


def select_config(args):
    """Select config file from model name."""
    if args.data == "imagenet":
        config = get_image_config(args)
    elif args.data == "two_plas":
        config = get_two_plas_config(args)
    else:
        raise NotImplementedError(f"Unkown model: {args.model}")

    # config = get_image_config(args)
    return config