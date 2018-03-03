import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio, volatile
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, input_spk=None):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
class SpkSeq2seq(Seq2seq):
    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, input_spk=None):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        result = self.decoder(input_spk=input_spk[-1],
                              inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result

class HierSeq2seq(Seq2seq):
    def __init__(self, encoder, context, decoder, decode_function=F.log_softmax):
        super(HierSeq2seq, self).__init__(encoder, decoder, decode_function)
        self.context = context

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.context.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, input_spk=None, context_hidden=None):
        for idx, (input, length) in enumerate(zip(input_variable, input_lengths)):
            length = np.array(length)
            sort_idx = np.argsort(-length)
            unsort_idx = torch.LongTensor(np.argsort(sort_idx)).cuda() if torch.cuda.is_available() else torch.LongTensor(np.argsort(sort_idx))
            sort_length = length[sort_idx]
            sort_input = input[torch.LongTensor(sort_idx).cuda() if torch.cuda.is_available() else torch.LongTensor(sort_idx)]

            _, encoder_hidden = self.encoder(sort_input, sort_length.tolist())

            # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
            encoder_hidden = torch.transpose(encoder_hidden, 0, 1)[unsort_idx]
            encoder_hidden = torch.transpose(encoder_hidden, 0, 1)

            # encoder_hiddens: [num_sentence, batch_size, hidden_size]
            if idx == 0:
                encoder_hiddens = encoder_hidden
            else:
                encoder_hiddens = torch.cat(encoder_hiddens, encoder_hidden, dim=0)

        # for hred, train should take the context of the previous turn
        # should return current loss as well as context representation
        # input_variable [32, 8], input_lengths list
        # encoder_outputs [32, 8, 400], encoder_hidden [2, 32, 200]

        # encoder_hiddens: [batch_size, num_sentence, hidden_size]
        encoder_hiddens = torch.transpose(encoder_hiddens, 0, 1)

        context_outputs, context_hidden = self.context(encoder_hiddens)

        result = self.decoder(input_spk=input_spk[-1],
                              inputs=target_variable,
                              encoder_hidden=context_hidden,
                              encoder_outputs=context_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result