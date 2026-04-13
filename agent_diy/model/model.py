#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, state_shape=None, action_shape=0, softmax=False):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, *args, **kwargs):
        return None
