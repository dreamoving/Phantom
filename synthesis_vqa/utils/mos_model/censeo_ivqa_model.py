# -*- coding: utf-8 -*-
# @Time    : 2021/5/12 2:06 下午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: censeo_ivqa_model.py
# @Software: PyCharm

import torch
from torch import nn
from . import heads
from . import backbones


class CenseoIVQAModel(nn.Module):
    """
    censeo的一个strong baseline model
    """

    def __init__(self, pretrained=True):
        super(CenseoIVQAModel, self).__init__()
        # self.config = config
        input_channels = 3
        model_name="resnet18"
        # pretrained=True
        # out_indices=(3,)
        # strides=(2, 2, 2)

        self.backbone = getattr(backbones, model_name)(input_channels=input_channels,
                                                       pretrained=pretrained,
                                                       out_indices=(3,),
                                                       strides=(2, 2, 2))
        self.head = getattr(heads, 'SimpleHead')(self.backbone.ouput_dims,
                                                                out_num=1)

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)
        out = torch.sigmoid(out)
        return out

