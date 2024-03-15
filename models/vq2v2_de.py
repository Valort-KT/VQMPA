import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import math

from torch import nn, Tensor
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, List, Sequence, Tuple, Union


import torchvision.transforms as transforms
from torchvision.ops import StochasticDepth
import torchvision.datasets as datasets
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import _log_api_usage_once
from functools import partial
# from model.eff2d import efficientenc_b0
from models.effi_nocla import efficientnet_b3

from torchvision.ops.misc import Conv2dNormActivation, FrozenBatchNorm2d,SqueezeExcitation
from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param, _make_divisible


@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))
    
class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

class FusedMBConv(nn.Module):
    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                Conv2dNormActivation(
                    expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

class Fusion_de(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1) -> None:
        super().__init__()
        layer : List[nn.Module] = []
        norm_layer = nn.BatchNorm2d

        # layer.append()
        self.block = Conv2dNormActivation(in_channels,out_channels,kernel_size=kernel_size,activation_layer=nn.SiLU)
        # self.stochastic_depth = StochasticDepth(0.2, "row")
    def forward(self,x):
        result = self.block(x)
        # result = self.stochastic_depth(result)
        result += x
        return result

class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result




## %%
inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 2),
        ]
last_channel = 1280

## %%


class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
        in_cha:int = 3,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        # if "block" in kwargs:
        #     warnings.warn(
        #         "The parameter 'block' is deprecated since 0.13 and will be removed 0.15. "
        #         "Please pass this information on 'MBConvConfig.block' instead."
        #     )
        #     if kwargs["block"] is not None:
        #         for s in inverted_residual_setting:
        #             if isinstance(s, MBConvConfig):
        #                 s.block = kwargs["block"]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []
        self.in_channels = in_cha
        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                in_cha, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        if last_channel:
            lastconv_input_channels = inverted_residual_setting[-1].out_channels
            lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
            layers.append(
                Conv2dNormActivation(
                    lastconv_input_channels,
                    lastconv_output_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=nn.SiLU,
                )
            )
        else:
            lastconv_output_channels = inverted_residual_setting[-1].out_channels
        en_bot = layers[:4]
        en_top = layers[4:]
        self.fea_bot = nn.Sequential(*en_bot)
        self.fea_top = nn.Sequential(*en_top)
        # self.features = nn.Sequential(*layers)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=dropout, inplace=True),
        #     nn.Linear(lastconv_output_channels, num_classes),
        # )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out")
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Linear):
        #         init_range = 1.0 / math.sqrt(m.out_features)
        #         nn.init.uniform_(m.weight, -init_range, init_range)
        #         nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # x = self.features(x)
        enb = self.fea_bot(x)
        ent = self.fea_top(enb)
        

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        # x = self.classifier(x)

        return enb,ent

    def forward(self, x: Tensor) -> Tensor:
        # return self._forward_impl(x)
        enb = self.fea_bot(x)
        ent = self.fea_top(enb)
        

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        # x = self.classifier(x)

        return enb,ent
    


class de_top(nn.Module):
    def __init__(self,inverted_residual_setting) -> None:
        super().__init__()

        layer : List[nn.Module] = []


        now_cha0 = inverted_residual_setting[2].out_channels
        stage :List[nn.Module] = []
        stage.append(nn.Conv2d(now_cha0,now_cha0*2,1,bias=False))
        stage.append(nn.PixelShuffle(2))
        stage.append(Fusion_de(now_cha0//2,now_cha0//2))

        now_cha1 = now_cha0//2
        stage.append(nn.Conv2d(now_cha1,now_cha1*2,1,bias=False))
        stage.append(nn.PixelShuffle(2))
        stage.append(nn.UpsamplingNearest2d(size=(38,38)))
        stage.append(Fusion_de(now_cha1//2,now_cha1//2))

        # layer.append(nn.Sequential(*stage))
        self.det = nn.Sequential(*stage)
    def forward(self,x):
        result = self.det(x)
        return result
    
class de_bot(nn.Module):
    def __init__(self,inverted_residual_setting,in_cha) -> None:
        super().__init__()
        layer : List[nn.Module] = []
        cha0 = inverted_residual_setting[2].input_channels
        in_cha0 = inverted_residual_setting[2].out_channels
        layer.append(nn.Conv2d(in_cha,cha0*4,1,bias=False))
        layer.append(nn.PixelShuffle(2))
        layer.append(nn.UpsamplingNearest2d(size=(75,75)))
        layer.append(Fusion_de(cha0,cha0))

        cha1 = inverted_residual_setting[1].input_channels
        in_cha1 = inverted_residual_setting[1].out_channels
        layer.append(nn.Conv2d(in_cha1,cha1*4,1,bias=False))
        layer.append(nn.PixelShuffle(2))
        layer.append(Fusion_de(cha1,cha1))

        cha2 = inverted_residual_setting[0].input_channels
        in_cha2 = inverted_residual_setting[0].out_channels
        layer.append(nn.Conv2d(in_cha2,cha2*4,1,bias=False))
        layer.append(nn.PixelShuffle(2))
        layer.append(Fusion_de(cha2,cha2))

        layer.append(nn.Conv2d(cha2,1,1,bias=False))
        self.deb = nn.Sequential(*layer)

    def forward(self,x):
        result = self.deb(x)
        return result
        
    
# x = torch.randn(1,256,10,10)
# model = de_top(inverted_residual_setting[3:])
# y = model(x)
# print(y.shape)

# x1 = torch.randn(1,320,38,38)
# model = de_bot(inverted_residual_setting[:3],in_cha = x1.shape[1])
# y1 = model(x1)
# print(y1.shape)
                
class VectorQuantizer(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_embeddings: int,
                 initialization = 'uniform',

                 ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        if initialization == 'uniform':
            self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self._commitment_cost = 0.25

    def forward(self, input):
        in_re = input.permute(0, 2, 3, 1).contiguous()
        input_shape = in_re.shape
        #flatten in_re
        x_fla = in_re.view(-1, self.embedding_dim)

        dist = (torch.sum(x_fla**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2 * torch.matmul(x_fla, self.embedding.weight.t()))
        
        self.encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        encoding = torch.zeros(self.encoding_indices.shape[0], self.num_embeddings, device=input.device)
        encoding.scatter_(1, self.encoding_indices, 1)

        softmax_histogram = torch.sum(nn.Softmax(-1)(-dist).view((input.shape[0], -1, self.num_embeddings)), dim=1)

        #quantize and unflatten
        quantized = torch.matmul(encoding, self.embedding.weight).view(input_shape)

        #loss
        e_latent_loss = F.mse_loss(quantized.detach(), in_re)
        q_latent_loss = F.mse_loss(quantized, in_re.detach())
        # softmax_loss = nn.CrossEntropyLoss()(-dist, self.encoding_indices)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss 

        quantized = in_re + (quantized - in_re).detach()
        avg_probs = torch.mean(encoding, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # encoding_indice = self.encoding_indices.view((-1,1) + input.shape[2:])
        encoding_indice = self.encoding_indices.view(input.shape[0],-1)
        index_program = torch.stack(
            list(
                map(
                    lambda x: torch.histc(x,bins=self.num_embeddings,min=0,max=self.num_embeddings-1),
                    self.encoding_indices.view((input.shape[0], -1)).float(),)
            )
        )
        return quantized,loss,perplexity,encoding_indice,index_program,softmax_histogram


class fct(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.last = Conv2dNormActivation(256,512,1,activation_layer=nn.SiLU)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(512, config.MODEL.NUM_CLASSES),
        )
    def forward(self,x):
        x = self.last(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class fcb(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.last = Conv2dNormActivation(128,256,1,activation_layer=nn.SiLU)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, config.MODEL.NUM_CLASSES),
        )
    def forward(self,x):
        x = self.last(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

## %%
class Model_v2de(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()

        self.encoder = EfficientNet(inverted_residual_setting, dropout=0.2,in_cha=1)
        self.vqt = VectorQuantizer(128,512)
        self.vqb = VectorQuantizer(128,512)
        self.fct = fct(config)
        self.fcb = fcb(config)

        self.de_top = de_top(inverted_residual_setting[3:])
        self.ent_up = nn.Sequential(nn.Conv2d(256,512,1,bias=False),nn.PixelShuffle(4),nn.Upsample(size=(38,38),mode='bicubic'))

        self.de_bot = de_bot(inverted_residual_setting[:3],in_cha = 160)

    def forward(self,x,mode=None):
        enb,ent = self.encoder(x)
        vqt,tloss,v2per,v2enco_indice,v2index_program,v2softmax_histogram = self.vqt(ent)
        fct = self.fct(vqt)

        det = self.de_top(vqt)
        in_vqb = torch.cat([enb,det],dim=1)
        vqb,bloss,v1per,v1enco_indice,v1index_program,v1softmax_histogram=self.vqb(in_vqb)
        fcb = self.fcb(vqb)

        vqt_up = self.ent_up(vqt)

        in_deb = torch.cat([vqt_up,vqb],dim=1)
        deb = self.de_bot(in_deb)

        msel = F.mse_loss(det,enb)
        if mode == "ent":
            return ent,fct
        elif mode == "vqb":
            return vqb,fcb,v1index_program
        elif mode == "vqt":
            return vqt,fct,v2index_program
        elif mode == "vqindhist":
            return v2index_program
        return deb,msel,fct,fcb,tloss,bloss
    

# print(inverted_residual_setting[0].out_channels)
# x = torch.randn(1,1,300,300)
# model = Model_v2de(1)
# from torchinfo import summary
# # summary(model,(1,1,300,300),depth=6,col_names=("input_size","output_size","num_params","kernel_size","mult_adds"))
# y = model(x)
# print(y[0].shape,y[1],y[2].shape,y[3].shape)
