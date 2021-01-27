import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop

from .base import TrainerBase
from .assistment import AssistTrainer


class SimulatedTrainer(AssistTrainer):

    def __init__(self, args):
        super(SimulatedTrainer, self).__init__(args)
        self.args = args
        self.data = args.data
        self.loader = args.loader
        print(self.data)
        print(self.loader)
