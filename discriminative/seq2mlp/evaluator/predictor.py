import torch
from torch.autograd import Variable

class Predictor(object):

    def __init__(self, model, src_vocab, spk_vocab):   # src_vocab
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2mlp.util.checkpoint.load`
            src_vocab (seq2mlp.dataset.vocabulary.Vocabulary): source sequence vocabulary
            src_vocab (seq2mlp.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.spk_vocab = spk_vocab


    def predict(self, src_seq):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            speaker

        """
        src_id_seq = Variable(torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]),
                              volatile=True).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        results = self.model(src_id_seq, [len(src_seq)])

        result = torch.sum(results, 0)
        _, predicted = torch.max(result.data, 1)

        speaker = predicted.cpu().numpy()[0]

        return self.spk_vocab.itos[speaker]

        # length = other['length'][0]
        #
        # src_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        # src_seq = [self.src_vocab.itos[tok] for tok in src_id_seq]
        # return ' '.join(src_seq)
