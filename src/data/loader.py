import csv
import torch
import random

"""
load synth / real data as:
[
    {
        "questionId": tensor([ 0,  1,  2.,..]) # id 1~n, torch.LongTensor
        "correct": tensor([ 0,  1,  0.,..]) # correct 1, incorrect 0, torch.LongTensor
        "n_answers": 4 # seq_len, int
    },
    {
        "questionId": tensor([ 0,  1,  2,..])
        "correct": tensor([ 0,  1,  0,..])
        "n_answers": 4
    },
]

Usage:
    # real data
    loader = AssistDataLoader(args)
    train_data = loader.get_train_data()

    # synth data
    loader = SyntheticDataLoader(args)
    train_data = loader.get_train_data()

"""

class Loader():

    def __init__(self, args):
        self.args = args
        
        self.train_data = None
        self.test_data = None


    def get_train_data(self):
        return self.train_data


    def get_test_data(self):
        return self.test_data


    def read_data(self):
        NotImplementedError


    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2






