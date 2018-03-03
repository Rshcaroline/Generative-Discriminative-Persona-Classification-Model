import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .attention import Attention
from DecoderRNN import DecoderRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device

class SpkDecoderRNN(DecoderRNN):
    def __init__(self, num_spk, spk_embed_size,
                 vocab_size, max_len, hidden_size,
                 sos_id, eos_id, word_embed_size=None, vectors=None,
                 n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False):
        word_embed_size = hidden_size + spk_embed_size
        super(SpkDecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                            sos_id, eos_id, word_embed_size=word_embed_size, vectors=vectors,
                                            n_layers=n_layers, rnn_cell=rnn_cell, bidirectional=bidirectional,
                                            input_dropout_p=input_dropout_p, dropout_p=dropout_p, use_attention=use_attention)
        self.spk_embedding = nn.Embedding(
            num_embeddings=num_spk,
            embedding_dim=spk_embed_size
        )


    def forward_step(self, input_spk, input_var, hidden, encoder_outputs, function):
        # difference from DecoderRNN
        # Get the embedding of the current input word (last output word)]
        size = input_var.size(1)
        input_spk_ = torch.t(input_spk).repeat(1,input_var.size(1))
        embedded_spk = self.spk_embedding(input_spk_)
        embedded_var = self.embedding(input_var)
        # cat
        if not(embedded_var.size(0) == embedded_spk.size(0) and embedded_var.size(1) == embedded_spk.size(1)):
            print "a"
        embedded = torch.cat((embedded_spk, embedded_var), 2)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        # TODO
        batch_size = input_var.size(0)
        output_size = input_var.size(1)

        # Calculate attention weights and apply to encoder outputs
        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(output.view(-1, self.hidden_size))).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, input_spk, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(input_spk, decoder_input, decoder_hidden, encoder_outputs,
                                                                     function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)

            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(input_spk, decoder_input, decoder_hidden, encoder_outputs,
                                                                         function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict