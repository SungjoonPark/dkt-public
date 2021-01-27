import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

"""
Computing metrics -> CPU computations
"""

class Metrics():

    def __init__(self, args):
        self.args = args


    def flatten(self, tp, tc, tm):
        """
        flatten batch predictions
        
        move tensors to cpu for metric computation
        """

        tp = tp.reshape(-1)
        tc = tc.reshape(-1)
        tm = tm.reshape(-1)

        predict = []
        correct = []
        for p, c, m in zip(tp, tc, tm):
            if m == 1:
                predict.append(p)
                correct.append(c)

        return predict, correct


    def acc(self, predictions, gts):
        acc = accuracy_score(gts, predictions)
        return acc
    

    def auc(self, predictions, gts):
        auc = roc_auc_score(gts, predictions)
        return auc

