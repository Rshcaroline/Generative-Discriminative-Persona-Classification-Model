#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/1 下午9:03
# @Author  : Shihan Ran
# @Site    : 
# @File    : discriminative.py
# @Software: PyCharm
# @Description:

import torch.nn as nn
import torch.nn.functional as F

class Discriminative(nn.Module):
    def __init__(self, encoder, MLP):
        super(Discriminative, self).__init__()
        self.encoder = encoder
        self.MLP = MLP

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, teacher_forcing_ratio=0):
        _, encoder_hidden = self.encoder(input_variable, input_lengths)
        results = self.MLP(encoder_hidden)
        return results