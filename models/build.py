import torch
from models.vq2v2_de import *


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'vq2v2_de':
        model = Model_v2de(config)
    else :
        raise NotImplementedError(f"Unkown model: {model_type}")
    return model
