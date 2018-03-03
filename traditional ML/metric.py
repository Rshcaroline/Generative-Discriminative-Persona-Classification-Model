from utils import global_config
import numpy as np
import sklearn.metrics

class Metric():
    def __init__(self,logits,log_probas,labels, label_num):
        self.logits = logits
        self.labels = labels
        self.log_probas = log_probas
        self.label_num = label_num

    def k_instance_accuracy(self,k):
        predicted, truth = [],[]
        for l in range(self.label_num):
            buffer_k = np.zeros(self.label_num)
            cnt = 0
            for i in range(self.log_probas.shape[0]):
                if self.labels[i] == l:
                    buffer_k += self.log_probas[i]
                    cnt += 1
                if cnt == k:
                    predicted.append(np.argmax(buffer_k))
                    truth.append(l)
                    buffer_k = np.zeros(self.label_num)
                    cnt = 0
        accr = sklearn.metrics.accuracy_score(truth,predicted)
        f1 = sklearn.metrics.f1_score(truth,predicted,average='micro')
        print('accr',accr,'f1',f1)
        print(sklearn.metrics.confusion_matrix(truth,predicted))
        return accr
