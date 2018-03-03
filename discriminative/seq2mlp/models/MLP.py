#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/1 下午9:36
# @Author  : Shihan Ran
# @Site    : 
# @File    : MLP.py
# @Software: PyCharm
# @Description:

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Seq2MLP(
#   (encoder): EncoderRNN(
#     (input_dropout): Dropout(p=0)
#     (embedding): Embedding(21853, 256)
#     (rnn): GRU(256, 256, batch_first=True, bidirectional=True)
#   )
#   (MLP): MLP(
#     (fc1): Linear(in_features=256, out_features=128)
#     (fc2): Linear(in_features=128, out_features=6)
#     (fc3): LogSoftmax()
#   )
# )

class MLP(nn.Module):
    def __init__(self, input_size, h1, num_classes, h2=None):
        super(MLP, self).__init__()
        self.h2 = h2
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, h1)
        if h2 is None:
            self.fc2 = nn.Linear(h1, num_classes)
        else:
            self.fc2 = nn.Linear(h1, h2)
            self.fc3 = nn.Linear(h2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):

        fc1 = F.relu(self.fc1(x))
        fc2 = F.relu(self.fc2(fc1))
        if self.h2 is None:
            res = self.log_softmax(fc2)
        else:
            fc3 = F.relu(self.fc3(fc2))
            res = self.log_softmax(fc3)

        return res