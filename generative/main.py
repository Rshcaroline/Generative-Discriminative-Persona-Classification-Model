import os
import argparse
import logging

import torch
from torch import optim

from seq2seq.trainer import SpkTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq, SpkSeq2seq, SpkDecoderRNN
from seq2seq.loss import Perplexity
from seq2seq.dataset import SourceField, TargetField, SpeakerField, SpeakerDataset
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.optim import Optimizer

from torchtext.data.dataset import TabularDataset


try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


############  parse arguments  ##############

parser = argparse.ArgumentParser()

# load data
parser.add_argument('--path', default='../data/',
                    help='Input data path.')
parser.add_argument('--num_sentence', type=int, default=4,
                    help='Number of sentences in every dialog')
parser.add_argument('--format', default='csv',
                    help='The format of data file.')


parser.add_argument('--train_data', default="bt_aug_clip_train4.csv",
                    help='Input train data filename.')
parser.add_argument('--dev_data', default="bt_aug_clip_dev4.csv",
                    help='Input dev data filename.')
parser.add_argument('--test_data', default="bt_aug_clip_test4.csv",
                    help='Input train data filename.')

# load model
parser.add_argument('--encoder', default='EncoderRNN',
                    help='Choose Encoder')
parser.add_argument('--decoder', default='SpkDeconderRNN',
                    help='Choose Decoder')
parser.add_argument('--bidirectional', type=bool, default=True,
                    help='Use bidirectional or not')
parser.add_argument('--embedding', type=int, default=0,
                    help='Use bidirectional or not')
parser.add_argument('--max_len', type=int, default=150,
                    help='Choose max_len of Encoder')
parser.add_argument('--hidden_size', type=int, default=100,
                    help='Choose the hidden size')
parser.add_argument('--num_spk', type=int, default=7,  # only six main roles
                    help='Choose the speaker size')
parser.add_argument('--spk_embed_size', type=int, default=100,
                    help='Define the vocab size')
parser.add_argument('--vocab_size', type=int, default=50000,
                    help='Define the vocab size')
# train model
parser.add_argument('--cuda', type=bool, default=True,
                    help='Use cuda or not')
parser.add_argument('--lr', type=float, default=0.002,
                    help='Choose the learning rate')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Choose the batch_size tuple for (train, valid, test)')
parser.add_argument('--epochs', type=int, default=6,
                    help='Choose the number of epochs')
parser.add_argument('--num_steps', type=int, default=48,
                    help='Choose the num steps')
parser.add_argument('--freeze_embedding', type=bool, default=False,
                    help='Freeze embedding or not')
parser.add_argument('--max_grad_norm', type=int, default=5,
                    help='max gradients norm for clipping')
parser.add_argument('--seed', type=int, default=55,
                    help='Random seed for initialization')
parser.add_argument('--dropout_p', type=float, default=0.5,
                    help='define the probability of dropout')
parser.add_argument('--input_dropout_p', type=float, default=0.5,
                    help='define the probability of dropout')
# define log
parser.add_argument('--log-level', dest='log_level', default='info',
                    help='Logging level.')
parser.add_argument('--verbose', type=int, default=50,
                    help='Output the evaluation result every X time')
parser.add_argument('--ckpt_every', type=int, default=50,
                    help='Save the model every X time')
# checkpoint
parser.add_argument('--resume', action='store_true', dest='resume',default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')

args = parser.parse_args()
args.cuda = args.cuda & torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(3)

############  define some tools  ##############

def init_log():
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, args.log_level.upper()))
    logging.info(args)

def init_model():
    if args.load_checkpoint is not None:
        logging.info("loading checkpoint from {}".format(os.path.join(args.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, args.load_checkpoint)))
        checkpoint_path = os.path.join(args.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, args.load_checkpoint)
        checkpoint = Checkpoint.load(checkpoint_path)
        model = checkpoint.model
        input_vocab = checkpoint.input_vocab
        output_vocab = checkpoint.output_vocab
    else:
        # build the vocabulary index and embedding
        spk.build_vocab(train, vectors="glove.6B.100d")
        src.build_vocab(train, max_size=args.vocab_size, vectors="glove.6B.100d")
        tgt.build_vocab(train, max_size=args.vocab_size, vectors="glove.6B.100d")
        input_vocab, output_vocab = src.vocab, tgt.vocab

        # Initialize model
        encoder = EncoderRNN(vocab_size=len(input_vocab),
                             max_len=args.max_len,
                             vectors=input_vocab.vectors if args.embedding else None,
                             input_dropout_p=args.input_dropout_p,
                             dropout_p=args.dropout_p,
                             hidden_size=args.hidden_size,
                             bidirectional=args.bidirectional,
                             variable_lengths=True)

        decoder = SpkDecoderRNN( num_spk = args.num_spk,
                                 spk_embed_size = args.spk_embed_size,
                                 vocab_size=len(output_vocab),
                                 max_len=args.max_len,
                                 hidden_size=args.hidden_size * 2 if args.bidirectional else args.hidden_size,
                                 dropout_p=args.dropout_p,
                                 input_dropout_p=args.input_dropout_p,
                                 vectors=input_vocab.vectors if args.embedding else None,
                                 use_attention=True,
                                 bidirectional=args.bidirectional,
                                 eos_id=tgt.eos_id,
                                 sos_id=tgt.sos_id)
        model = SpkSeq2seq(encoder, decoder)
        if torch.cuda.is_available():
            model.cuda()

        for param in model.parameters():
            param.data.uniform_(-0.08, 0.08)

    return model, input_vocab, output_vocab


init_log()

################  create dataset  #################

# Prepare dataset fields
src = SourceField()
spk = SpeakerField()
tgt = TargetField()

# load the dataset

# fields = [('0', spk), ('src', src), ('1', spk), ('src0', src), ('2', spk), ('src1', src), ('3', spk), ('tgt', src)]
#
# train, validation, dev = SpeakerDataset.splits(
#     num=args.num_sentence, format=args.format,
#     path=args.path,
#     fields=fields)

# load the dataset
if args.num_sentence > 1:
    fields = [('0', spk), ('src', src)]
    for i in range(args.num_sentence-2):
        fields.append((str(i+1), spk))
        fields.append(('src'+str(i), src))
    fields.append((str(args.num_sentence-1), spk))
    fields.append(('tgt', tgt))
else:
    raise NotImplementedError

train, dev, test = TabularDataset.splits(format=args.format,
                                         path=args.path,
                                         fields=fields,
                                         train=args.train_data,
                                         validation=args.dev_data,
                                         test=args.test_data)

SpeakerDataset.concat(args.num_sentence, (train, dev, test))

################  define model ##################

model, input_vocab, output_vocab = init_model()

# Define loss
weight = torch.ones(len(output_vocab))
pad = output_vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)
if torch.cuda.is_available():
    loss.cuda()

# Define Optimizer
optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=args.max_grad_norm)

###############  train model ################

t = SpkTrainer(args = args,
                      loss = loss,
                      batch_size = args.batch_size,
                      checkpoint_every = args.ckpt_every,
                      random_seed = args.seed,
                      print_every = args.verbose,
                      expt_dir=args.expt_dir
                      )

discrim = t.train(model = model,
                  data = train,
                  num_epochs = args.epochs,
                  dev_data=dev,
                  optimizer=optimizer,
                  teacher_forcing_ratio=0.5,
                  resume=args.resume
                  )

################ initialize predictor ###################

predictor = Predictor(model=model,
                      src_vocab=input_vocab,
                      tgt_vocab=output_vocab
                      )

while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))
