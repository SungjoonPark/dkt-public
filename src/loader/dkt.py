"""
Build Pytorch dataloader: (train, dev, test)
from loaded dictionary. 

[
    (questionId) 
        tensor([[99, 99, 99,  ...,  0,  0,  0],
                [ 1,  1,  1,  ..., 87, 87, 87],
                [31, 31, 31,  ...,  0,  0,  0],
                ...,
                [31, 31, 31,  ...,  0,  0,  0],
                [33, 33, 33,  ...,  0,  0,  0],
                [14, 14, 14,  ...,  0,  0,  0]]), 
    (correct)
        tensor([[0, 0, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 1, 1],
                [1, 1, 1,  ..., 0, 0, 0],
                ...,
                [1, 1, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0]]), 
    (sequence_mask)
        tensor([[ True,  True,  True,  ..., False, False, False],
                [ True,  True,  True,  ...,  True,  True,  True],
                [ True,  True,  True,  ..., False, False, False],
                ...,
                [ True,  True,  True,  ..., False, False, False],
                [ True,  True,  True,  ..., False, False, False],
                [ True,  True,  True,  ..., False, False, False]]), 
    (sequence_length)
        tensor([ 42, 512,  22, 113, 512,   2,  75,   7,  59,   5,  26,  33,  29,   3,
                14, 128,  13,   5,  23,  29,  17, 285, 512,  14,   5, 127,  40, 115,
                7,   6,   7,  11])]
]

"""


import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler, 
    SequentialSampler, 
    TensorDataset
    )
from torch.nn.utils.rnn import pad_sequence



class KTDataLoader():

    def __init__(self, args):
        self.args = args
        # 3 features
        self.sequence_features = ['questionId', 'correct']
        self.meta_features = ['n_answers'] # n_answers = actual sequence length
        self.features = self.sequence_features + self.meta_features


    def _gather_features(self, data):
        """
        gather examples by features

        returns:
        data = {
            "questionID": [[], [], .. []] # sequence-level : list of torch.tensors (ids, ids,..)
            "correct": [[], [], .. []] # sequence-level : list of torch.tensors (1, 0, ...)
            "n_answers": [3, 4, 1, 5..] # seq_len : list of ints
        }
        """
        data_dict = {}
        for f in self.features:
            data_dict[f] = []

        for d in data:
            for f in self.features:
                data_dict[f].append(d[f])
        
        return data_dict


    def _truncate_sequences(self, sequences, max_seq_len):
        truncated_sequences = []
        for s in sequences:
            if len(s) > max_seq_len:
                s = s [:max_seq_len + 1]
            truncated_sequences.append(s)
        return truncated_sequences


    def _pad_and_trunc_sequence(self, data_dict, max_seq_len):
        """
        for sequence features, 
            1) truncate (to args.max_seq_len)
            2) pad sequence (pad: 0)

            "sequence_mask" : build bool mask
            "sequence_length" : change "n_answers" to "sequence_length"
        """
        
        for sf in self.sequence_features:
                        
            # 1. trunc sequence to args.max_seq_len & add seq_len
            data_dict[sf] = self._truncate_sequences(data_dict[sf], max_seq_len)
            #print(data_dict[sf][:10])

            # 2. pad sequence (to max_seq_len)
            data_dict[sf] = pad_sequence(
                data_dict[sf], 
                batch_first=True)
            #print(data_dict[sf][:10])

        # 3. build mask
        assert 'n_answers' in self.meta_features
        #print(data_dict['n_answers'][:10])
        mask = torch.arange(max_seq_len).expand(
            len(data_dict['n_answers']), 
            max_seq_len) < torch.Tensor(
                [x-1 for x in data_dict['n_answers']] # to have (n_answers -1)
                ).unsqueeze(1)
        data_dict['sequence_mask'] = mask
        
        # change feature name 'n_answers' to 'sequence_length'
        n_answers = data_dict.pop('n_answers')
        n_answers = [max_seq_len if x > max_seq_len else x for x in n_answers]    
        data_dict['sequence_length'] = n_answers

        #print(data_dict['sequence_mask'][:10])
        #print(data_dict['sequence_length'][:10])
        #print(data_dict.keys())
        return data_dict


    def build_dataset(self, data, max_seq_len):
        """
        wrapper of gathering features / pad+truc sequence.
        
        returns:
            torch.dataset:
                questionID, 
                correct, 
                mask, 
                sequence_length
        """
        data_dict = self._gather_features(data)
        data_dict = self._pad_and_trunc_sequence(data_dict, max_seq_len)
        
        # contruct dataset
        feature_list = []
        for f in data_dict.keys():
            if not isinstance(data_dict[f], torch.Tensor):
                feature_list.append(torch.Tensor(data_dict[f]))
            else:
                feature_list.append(data_dict[f])
        dataset = TensorDataset(*feature_list)
        # questionID, correct, mask, sequence_length

        return dataset


    def build_dataloader(self, dataset, split_name):
        """
        build dataloader based on dataset.
        
        returns:
            torch.dataloader:
                batch_size * 
                    questionID, 
                    correct, 
                    mask, 
                    sequence_length
        """
        assert split_name in ['train', 'dev', 'test']

        if split_name == 'train':
            #sampler = RandomSampler
            sampler = SequentialSampler
            batch_size = self.args.train_batch_size
        else:
            sampler = SequentialSampler
            batch_size = self.args.eval_batch_size

        dataloader = DataLoader(
            dataset,
            sampler = sampler(dataset),
            batch_size = batch_size,
            shuffle = False, # sampler option is mutually exclusive with shuffle
        )

        return dataloader