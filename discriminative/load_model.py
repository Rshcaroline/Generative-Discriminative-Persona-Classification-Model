#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/1 下午5:09
# @Author  : Shihan Ran
# @Site    : 
# @File    : loamisc/perlunipropsd_model.py
# @Software: PyCharm
# @Description:

from seq2mlp.util.checkpoint import Checkpoint
from seq2mlp.evaluator import Predictor

import string

Check = Checkpoint.load(Checkpoint.get_latest_checkpoint('./experiment'))

predictor = Predictor(Check.model, Check.input_vocab, Check.output_vocab)

while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print "The speaking person is:", predictor.predict(seq)