import torch
from torch.autograd import Variable
import numpy as np

class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2mlp.util.checkpoint.load`
            src_vocab (seq2mlp.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2mlp.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab


    def predict(self, src_seq):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        src_id_seq = Variable(torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]),
                              volatile=True).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        softmax_list, _, other = self.model(src_id_seq, [len(src_seq)])
        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return ' '.join(tgt_seq)




class SpkPredictor(Predictor):
    def __init__(self,  model, src_vocab, tgt_vocab, spk_vocab=None):
        super(SpkPredictor, self).__init__(model, src_vocab, tgt_vocab)
        self.spk_vocab = spk_vocab

    # calculate score log(P(y|x,s)*P(s)) and store them in the list
    def _get_prob(self, src_seq, tgt_seq, spk):
        self.model.eval()
        decode_outputs = self.model(input_variable=src_seq,
                                   input_lengths=len(src_seq),
                                   target_variable=tgt_seq,
                                   input_spk=spk
                                   )
        probs = torch.sum(torch.gather(decode_outputs, index=tgt_seq))
        return probs

    def predict(self, src_seq, tgt_seq=None, input_spk=None):
        prob = []

        for spk in range(len(self.spk_vocab)):
            prob.append(self._get_prob(src_seq, tgt_seq, spk))

        return np.argmax(np.array(prob))


