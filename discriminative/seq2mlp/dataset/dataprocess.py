import os
import numpy as np
from torchtext.data import TabularDataset, Field, Iterator

##################################################
# SpeakerDataset
# ----------------
# Inherit torchtext.data.TabularDataset to split
# dialogs by split_num_sentence() and load data
#
#
# split_num_sentence()
# ----------------------
# num: number of sentences in a dialog
# src: .csv format raw data filename
# dst: .csv format processed data filename

MIN_CHARACTER = 10

class SpeakerDataset(TabularDataset):

    @classmethod
    def splits(cls, num, path = None, train = None, validation = None, test = None, root='.data', **kwargs):
        # override the splits method to pass the parameter num
        if path is None:
            raise NotImplementedError
        if train == None:
            train, validation, test = cls.split_num_sentence(num, path)
        train_data = None if train is None else cls(
            train, **kwargs)
        val_data = None if validation is None else cls(
            validation, **kwargs)
        test_data = None if test is None else cls(
            test, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    @staticmethod
    def split_num_sentence(num, src):
        # split the data as "character, sentence, character, sentence ..." for num times
        dst = src+".temp"

        dst_train = src + ".train" + str(num)
        dst_valid = src + ".valid" + str(num)
        dst_test = src + ".test" + str(num)

        f_src = open(src, 'r')
        f_dst = open(dst, 'w')

        src_lines = f_src.readlines()

        np.random.seed(55)

        length = len(src_lines)
        shuffled_idx = np.random.permutation(length)
        train_idx, valid_idx, test_idx = \
            shuffled_idx[:int(0.8*length)], shuffled_idx[int(0.8*length):int(0.9*length)], shuffled_idx[int(0.9*length):]

        for i, line in enumerate(src_lines):
            if i % num == 0:
                if i:
                    print >> f_dst, dst_line

                dst_line = line[:-1]
            else:
                dst_line += line[:-1]


        f_dst.close()
        f_dst = open(dst, 'r')
        dst_lines = f_dst.readlines()

        f_train = open(dst_train, 'w')
        f_valid = open(dst_valid, 'w')
        f_test = open(dst_test, 'w')

        for i, line in enumerate(dst_lines):
            if i in train_idx:
                print >> f_train, line,
            elif i in valid_idx:
                print >> f_valid, line,
            elif i in test_idx:
                print >> f_test, line,

        return dst_train, dst_valid, dst_test

    # TODO: setattr()
    @staticmethod
    def concat(num, datasets, mode = "t"):
        # concat data of **fields**  eg. ('src1', 'src2', 'src3')
        # output as data of output_field  eg. 'src1'  (must be one of fields above)
        if (mode == "t" and num == 1) or (mode=="cr" and num == 2):
            return
        count = num - 2 if mode == "t" else num - 3

        for dataset in datasets:
            for eg in dataset:
                for i in range(count):
                    eg.src += getattr(eg, "src"+str(i))

#######################################################################################
# data_loader
# ------------
# description:
#  API to get  glove vocabulary batch iterator
#
# input:
#  num: num of sentence, including the response
#  col_name: the attribute tuple of every column such as ("A", "text", "B", "Response")

# return:
#  vocab: index and vectors (from glove.100d) formthese content which can be use as
#   vocab.vector[int], vocab.itos[int], vocab.stoi[str]
#  iters: parse as train_iter, val_iter, test_iter if these paths defined
#   which has __iter__, and can be use as iter.col_name to get the torch Variable
#


def data_loader(num, col_name, path = './seq2mlp/dataset/data/', train = 'data.csv', valid = None, test = 'data.csv'):

    # parse arguments
    fix_length = 150
    batch_size = (32,32)
    fields_name = col_name

    # define the txt field
    TEXT = Field(sequential=True, include_lengths=True, batch_first=True, lower=True, fix_length=fix_length)
    RESPONSE = Field(sequential=True, batch_first=True, lower=True, fix_length=fix_length)
    SPEAKER = Field(sequential=True, lower=True, fix_length=1, pad_token=None, unk_token=None)

    # prepare the fields for tabular dataset by column
    fields = [((name, TEXT) if i % 2 else (name, SPEAKER)) for i, name in enumerate(fields_name)]
    fields[-1] = (fields_name[-1], RESPONSE)

    # load the dataset
    datasets = SpeakerDataset.splits(
            num=num, path=path,train=train,
            test=test, validation=valid, format = 'csv',
            fields=fields)

    # build the vocab
    TEXT.build_vocab(datasets[0], vectors="glove.6B.100d")
    RESPONSE.build_vocab(datasets[0], vectors="glove.6B.100d")
    SPEAKER.build_vocab(datasets[0], vectors="glove.6B.100d")

    vocab = (TEXT.vocab, RESPONSE.vocab)

    # get the iters for dataset by batch_size
    # iters = Iterator.splits(datasets, shuffle=True, batch_sizes=batch_size, device=-1)

    return vocab, datasets

###############################
# example for use data_loader

if __name__ == "__main__":

    # Use the data_loader API
    vocab, iters = data_loader(num=2, col_name=("A", "text", "B", "Response"))

    # parse the iters
    train_batchiter, test_batchiter = iters

    # use the iter every batch
    for i in train_batchiter:
        print i.A
        print i.text
        print i.B
        print i.Response
        break