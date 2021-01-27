import torch
import torch.nn as nn

from .rnn import RNN


class LSTM(RNN):

    def __init__(self, args):
        super(LSTM, self).__init__(args)
        self.args = args
        self.device = args.device

        self.rnn = nn.LSTM(
            self.input_size, 
            self.hidden_dim, 
            self.n_layers, 
            batch_first=True)


    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers, 
            batch_size, 
            self.hidden_dim)
        h = h.to(self.device)
        
        c = torch.zeros(
            self.n_layers, 
            batch_size, 
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)
        