from __future__ import print_function, division

import torch
import torchtext
from torch.autograd import Variable

import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score

import seq2seq
from seq2seq.loss import NLLLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2mlp.loss, optional): loss for evaluator (default: seq2mlp.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, args = None, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.args = args
        self.batch_size = batch_size

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2mlp.models): model to evaluate
            data (seq2mlp.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0
        correct_spk = 0
        total_response = 0
        y_true = np.array([])
        y_pred = np.array([])

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, device=device, train=False)
        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]

        for batch in batch_iterator:
            input_temp = [getattr(batch, "src" + str(i)) for i in range(self.args.num_sentence - 1)]
            input_variables = [i[0] for i in input_temp]
            input_lengths = [i[1].tolist() for i in input_temp]
            target_variables, target_lengths = getattr(batch, seq2seq.tgt_field_name)
            spk_inputs = [getattr(batch, str(i)) for i in range(self.args.num_sentence)]

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths,
                                                           target_variables, input_spk=spk_inputs)

            # Predict Speaker:
            target_spk = spk_inputs[-1]
            target_lengths = target_lengths.cpu().numpy()
            prob_per_spk = []
            for i in range(self.args.num_spk):
                spk_inputs_predict = Variable(torch.LongTensor(np.full((1,batch.batch_size), i)).clone())
                if self.args.cuda: spk_inputs_predict = spk_inputs_predict.cuda()
                spk_inputs_ = [getattr(batch, str(idx)) for idx in range(self.args.num_sentence-1)]
                spk_inputs_.append(spk_inputs_predict)
                decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths,
                                                               target_variables, input_spk=spk_inputs_)
                prob_per_responce = []
                target = target_variables.transpose(0,1)
                for word, prob in zip(target[1:], decoder_outputs):
                    word, prob = word.data.cpu().numpy(), prob.data.cpu().numpy()
                    indices = zip(np.arange(len(word)), word)
                    prob_per_word = [prob[i] for i in indices] # select the prob for *sentence_len* times
                    prob_per_responce.append(prob_per_word)
                prob_per_spk.append([np.sum(prob_per_responce[:i]) for i in target_lengths])
            prob_per_spk = np.transpose(prob_per_spk) # output: (batch_size, num_spk), every element is the log prob for responce

            predicts = np.argmax(prob_per_spk, axis=1)
            correct_spk += (predicts == target_spk.data.cpu().numpy()).sum()
            total_response += len(batch.tgt[0])

            # Evaluation
            seqlist = other['sequence']
            for step, step_output in enumerate(decoder_outputs):
                target = target_variables[:, step + 1]
                loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                non_padding = target.ne(pad)
                correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().data[0]
                match += correct
                total += non_padding.sum().data[0]

            y_pred = np.append(y_pred, predicts)
            y_true = np.append(y_true, target_spk.data.cpu().numpy())

        print(y_pred.shape)

        print(accuracy_score(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        if total_response == 0:
            accuracy_spk = float('nan')
        else:
            accuracy_spk = correct_spk / total_response

        return loss.get_loss(), accuracy, accuracy_spk
