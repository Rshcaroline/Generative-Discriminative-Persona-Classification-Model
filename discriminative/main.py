import os
import argparse
import logging

import torch
from torch import optim
from seq2mlp.trainer import SupervisedTrainer
from seq2mlp.models import EncoderRNN,  MLP, Seq2MLP_cr
from seq2mlp.loss import NLLLoss
from seq2mlp.dataset import TargetField, SourceField, SpeakerField, SpeakerDataset
from seq2mlp.evaluator import Predictor
from seq2mlp.util.checkpoint import Checkpoint
from seq2mlp.optim import Optimizer

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
                    help='Input test data filename.')

# load model
parser.add_argument('--encoder', default='EncoderRNN',
                    help='Choose Encoder')
parser.add_argument('--decoder', default='MLP',
                    help='Choose Decoder')
parser.add_argument('--bidirectional', type=bool, default=False,
                    help='Use bidirectional or not')
parser.add_argument('--embedding', type=int, default=0,
                    help='Use pretrained word embedding or not')
parser.add_argument('--max_len', type=int, default=150,
                    help='Choose max_len of Encoder')
parser.add_argument('--hidden_size', type=int, default=100,
                    help='Choose the hidden size')
parser.add_argument('--num_speaker', type=int, default=7,  # only seven main roles
                    help='Choose the speaker size')
parser.add_argument('--layer_size', type=int, default=[256, 128],
                    help='Choose the layer size')
parser.add_argument('--vocab_size', type=int, default=20000,
                    help='Define the vocab size')
# train model
parser.add_argument('--cuda', type=bool, default=True,
                    help='Use cuda or not')
parser.add_argument('--lr', type=float, default=0.05,
                    help='Choose the learning rate')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Choose the batch_size tuple for (train, valid, test)')
parser.add_argument('--epochs', type=int, default=30,
                    help='Choose the number of epochs')
parser.add_argument('--num_steps', type=int, default=48,
                    help='Choose the num steps')
parser.add_argument('--freeze_embedding', type=bool, default=False,
                    help='Freeze embedding or not')
parser.add_argument('--max_grad_norm', type=int, default=5,
                    help='max gradients norm for clipping')
parser.add_argument('--seed', type=int, default=55,
                    help='Random seed for initialization')
parser.add_argument('--dropout_p', type=float, default=0.7,
                    help='Choose dropout prob')
parser.add_argument('--input_dropout_p', type=float, default=0.7,
                    help='Choose input dropout prob')
# define log
parser.add_argument('--log-level', dest='log_level', default='info',
                    help='Logging level.')
parser.add_argument('--verbose', type=int, default=1000,
                    help='Output the evaluation result every X time')
parser.add_argument('--ckpt_every', type=int, default=1000,
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

        input_vocab, output_vocab= src.vocab, spk.vocab

        # Initialize model
        encoder_c = EncoderRNN(vocab_size = len(input_vocab),
                               max_len = args.max_len,
                               hidden_size = args.hidden_size,
                               vectors=input_vocab.vectors if args.embedding else None,
                               bidirectional=args.bidirectional,
                               dropout_p=args.dropout_p,
                               input_dropout_p=args.input_dropout_p,
                               variable_lengths=True)

        encoder_r = EncoderRNN(vocab_size = len(input_vocab),
                               max_len = args.max_len,
                               hidden_size = args.hidden_size,
                               vectors=input_vocab.vectors if args.embedding else None,
                               bidirectional=args.bidirectional,
                               dropout_p=args.dropout_p,
                               input_dropout_p=args.input_dropout_p,
                               variable_lengths=True)

        mlp = MLP(input_size = 2 * args.hidden_size,
                  h1 = args.hidden_size,
                  num_classes = args.num_speaker)

        model = Seq2MLP_cr(encoder_c, encoder_r, mlp)
        print model
        if args.cuda:
            model.cuda()

        # initialize the weights
        for param in model.parameters():
            param.data.uniform_(-0.08, 0.08)

    return model, input_vocab, output_vocab

init_log()

################ create dataset ##################

# Prepare dataset fields
spk = SpeakerField()
src = SourceField()

# load the dataset
if args.num_sentence > 1:
    fields = [('spk0', src), ('src', src)]
    for i in range(args.num_sentence-2):
        fields.append(('spk'+str(i+1), src))
        fields.append(('src'+str(i), src))
    fields.append(('spk', spk))
    fields.append(('tgt', src))
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

# Prepare loss
weight = torch.ones(len(output_vocab))
pad = input_vocab.stoi[spk.pad_token]
loss = NLLLoss(weight, pad)  # Q: what is pad?
if args.cuda:
    loss.cuda()

optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=args.max_grad_norm)

# begin Train
t = SupervisedTrainer(args = args,
                      loss = loss,
                      batch_size = args.batch_size,
                      checkpoint_every = args.ckpt_every,
                      random_seed = args.seed,
                      print_every = args.verbose,
                      expt_dir=args.expt_dir
                      )

print "evaluate before training:"
t.evaluator.evaluate(model, train)
t.evaluator.evaluate(model, dev)

t.train(  model = model,
          data = train,
          num_epochs = args.epochs ,
          dev_data=dev,
          optimizer=optimizer,
          teacher_forcing_ratio=0.5,
          resume=args.resume
          )

