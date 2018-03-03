#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/1 下午9:03
# @Author  : Shihan Ran
# @Site    : 
# @File    : Seq2MLP.py
# @Software: PyCharm
# @Description:

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2MLP(nn.Module):
    def __init__(self, encoder, MLP):
        super(Seq2MLP, self).__init__()
        self.encoder = encoder
        self.MLP = MLP

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, teacher_forcing_ratio=0):
        _, encoder_hidden = self.encoder(input_variable[0], input_lengths[0])

        results = self.MLP(encoder_hidden)
        return results

class Seq2MLP_cr(nn.Module):
    def __init__(self, encoder_c, encoder_r, MLP):
        super(Seq2MLP_cr, self).__init__()
        self.encoder_c = encoder_c
        self.encoder_r = encoder_r
        self.MLP = MLP

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, teacher_forcing_ratio=0):

        _, encoder_hidden_c = self.encoder_c(input_variable[0], input_lengths[0])

        """sort"""
        input_lengths = np.array(input_lengths)
        sort_idx = np.argsort(-input_lengths[1])
        unsort_idx = torch.LongTensor(np.argsort(sort_idx)).cuda() if torch.cuda.is_available() else torch.LongTensor(np.argsort(sort_idx))
        sort_length = input_lengths[1][sort_idx]
        sort_input = input_variable[1][torch.LongTensor(sort_idx).cuda() if torch.cuda.is_available() else torch.LongTensor(sort_idx)]

        _, encoder_hidden_r = self.encoder_r(sort_input, sort_length.tolist())

        encoder_hidden_r = torch.transpose(encoder_hidden_r, 0, 1)[unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        encoder_hidden_r = torch.transpose(encoder_hidden_r, 0, 1)

        encoder_hidden = torch.cat((encoder_hidden_c,encoder_hidden_r), 2)

        results = self.MLP(encoder_hidden)
        return results