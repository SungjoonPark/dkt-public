import csv
import random
import torch

from .loader import Loader


class SyntheticDataLoader(Loader):

    def __init__(self, args):
        super(SyntheticDataLoader, self).__init__(args)
        
        self.args = args
        self.read_data()


    def get_name(self):
        prefix = 'naive'
        c = 'c' + str(self.args.syn_c)
        q = 'q' + str(self.args.syn_q)
        n = 's' + str(self.args.syn_n)
        v = 'v' + str(self.args.syn_v)
        extension = '.csv'
        file_name = "_".join([prefix, c, q, n, v]) + extension
        return file_name


    def read_data(self):
        file_name = self.get_name()
        print('Loading:', file_name)
        file_path = self.args.syn_path + file_name
        data = []
        with open(file_path, newline='') as csvfile:
            for line in csvfile:
                seq = [int(x) for x in line.strip().split(',')]
                data.append(seq)
        data = torch.LongTensor(data) # all -> int 
        # id 0~49 (no need of padding because all seq_len is the same)
        total_students, n_questions = data.shape
        n_steps = n_questions - 1
        n_students = int(total_students / 2)

        self.n_questions = n_questions
        self.n_students = n_students
        self.n_steps = n_steps

        self.n_test = self.n_students
        self.n_train = self.n_students

        train_data = torch.narrow(data, 0, 0, n_students)
        test_data = torch.narrow(data, 0, n_students, n_students)

        self.train_data = self.compress_data(train_data)
        self.test_data = self.compress_data(test_data)

        # load training data
        total_students = len(self.train_data) 
        n_questions = len(self.train_data[0]['questionId'])
        print('training data:')
        print(' n_students:', len(self.train_data))
        print(' n_questions:', n_questions)
        print(' total answers:', total_students * n_questions)
        print(' longest:', n_questions)
        
        # load test data
        total_students = len(self.test_data) 
        n_questions = len(self.test_data[0]['questionId'])
        print('test data:')
        print(' n_students:', len(self.test_data))
        print(' n_questions:', n_questions)
        print(' total answers:', total_students * n_questions)
        print(' longest:', n_questions)


    def compress_data(self, dataset):
        new_dataset = []
        for i in range(self.n_students):
            answers = self.compress_answers(dataset[i])
            new_dataset.append(answers)

        return new_dataset


    def compress_answers(self, answers):
        new_answers = {}
        new_answers['questionId'] = torch.zeros(self.n_questions, dtype=torch.int64)
        # 'time' is not used
        #new_answers['time'] = torch.zeros(self.n_questions, dtype=torch.int64)
        new_answers['correct'] = torch.zeros(self.n_questions, dtype=torch.int64)
        new_answers['n_answers'] = self.n_questions

        for i in range(self.n_questions):
            new_answers['questionId'][i] = i # no padding
            new_answers['correct'][i] = answers[i]     

        return new_answers
