from __future__ import print_function, division

import torch
import torchtext
from torch.autograd import Variable

from ..loss import NLLLoss

from ..dataset import  spk_field_name, src_field_name, tgt_field_name

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2mlp.loss, optional): loss for evaluator (default: seq2mlp.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
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


        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)

        correct = 0
        total = 0
        y_true = np.array([])
        y_pred = np.array([])

        for batch in batch_iterator:
            input_variables, input_lengths = getattr(batch, src_field_name)
            input_variables, input_lengths = [input_variables], [input_lengths.tolist()]
            if hasattr(batch, tgt_field_name):
                input_variable, input_length = getattr(batch, tgt_field_name)
                input_variables.append(input_variable)
                input_lengths.append(input_length.tolist())
            target_variables = getattr(batch, spk_field_name)

            results = model(input_variables, input_lengths, target_variables)

            for step, step_output in enumerate(results):
                _, predicted = torch.max(step_output.data, 1)
                total += target_variables.size(1)
                correct += (predicted == target_variables.data).sum()
                prediction = step_output.view(target_variables.size(1), -1)
                loss.eval_batch(prediction, Variable(target_variables.data[step,:]))

            y_pred = np.append(y_pred, predicted)
            y_true = np.append(y_true,  target_variables.data.cpu().numpy())

        print(y_pred.shape)

        print(accuracy_score(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))

        print('Accuracy of the network: %d %%' % (
            100 * correct / total))

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = 100 * correct / total

        return loss.get_loss(), accuracy
