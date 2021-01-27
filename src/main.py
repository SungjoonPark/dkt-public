import os
import argparse
import csv
import torch

from data.simulated import SyntheticDataLoader
from data.assistment import AssistDataLoader

from loader.dkt import KTDataLoader

from trainer.assistment import AssistTrainer
from trainer.simulated import SimulatedTrainer

from model.lstm import LSTM
from model.rnn import RNN



def main(args):
    # dataset
    assert args.dataset in ['syn', 'assist']
    if args.dataset == 'assist':
        args.n_questions = 124 # 0-123
        args.data = AssistDataLoader(args)
        args.loader = KTDataLoader(args)
        trainer = AssistTrainer(args)
    elif args.dataset == 'syn':
        args.n_questions = 50 # 0-49
        args.data = SyntheticDataLoader(args)
        args.loader = KTDataLoader(args)
        trainer = SimulatedTrainer(args)
    else:
        NotImplementedError
      
    # model
    if args.model_type == "RNN":
        args.model = RNN
    elif args.model_type == "LSTM":
        args.model = LSTM
    else:
        raise NotImplementedError

    trainer.train()

    return args


def check_args(args):
    # dataset args
    assert args.dataset in ['syn', 'assist']
    
    # dataset type (for synthetic-2 or -5)
    assert args.syn_c in [2, 5]
    assert args.syn_q in [50]
    assert args.syn_n in [4000]
    assert args.syn_v in list(range(20))

    # dataset length
    if args.dataset == 'syn':
        assert args.train_max_seq_len < 50
        assert args.eval_max_seq_len < 50
    elif args.dataset == 'assist':
        assert args.train_max_seq_len < 4290
        assert args.eval_max_seq_len < 8214
    elif args.dataset == 'iscream_public':
        assert args.train_max_seq_len < 100000 # check
        assert args.eval_max_seq_len < 100000 # check

    # device
    assert args.device in ['cpu', 'gpu']
    if args.device == 'gpu':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    # model
    assert args.model_type in ['RNN', 'LSTM']
    
    # input
    assert args.input_type in ["onehot", "dense"]

    return args


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--syn_path', default='../data/synthetic/', type=str, help='path to synthetic data directory')
    parser.add_argument('--ast_path', default='../data/assistments/', type=str, help='path to assistment data directory')
    
    # data : synthtic-2 / 5
    parser.add_argument('--dataset', default='ast_path', type=str, help='type ot the dataset')
    parser.add_argument('--syn_c', default=5, type=int, help='concept_num')
    parser.add_argument('--syn_q', default=50, type=int, help='number_of_questions')
    parser.add_argument('--syn_n', default=4000, type=int, help='number_of_students')
    parser.add_argument('--syn_v', default=0, type=int, help='version')

    # preprocessing args
    parser.add_argument('--train_max_seq_len', default=500, type=int, help='max sequence length in train batch')
    parser.add_argument('--eval_max_seq_len', default=500, type=int, help='max sequence length in dev/test batch')
    parser.add_argument('--use_dev', default=False, type=bool, help='if True, use dev set')

    # model type
    parser.add_argument('--model_type', default="LSTM", type=str, help='model type. RNN/LSTM available')
    # model args (RNN)
    parser.add_argument('--input_type', default='onehot', type=str, help='input vector type.')
    parser.add_argument('--n_input_dim', default=100, type=int, help='number of input one-hot vector dimensions')
    parser.add_argument('--embedding_dim', default=100, type=int, help='number of input embedding vector dimensions')
    parser.add_argument('--hidden_dim', default=200, type=int, help='number of RNN/LSTM hidden dimensions')
    parser.add_argument('--n_layers', default=1, type=int, help='number of RNN/LSTM layers')

    # train : device args
    parser.add_argument('--device', default='gpu', type=str, help='Use CPU or GPU')
    parser.add_argument('--device_id', default=0, type=int, help='GPU device id')

    # train : loss / opimt args
    parser.add_argument('--loss', default='BCE', type=str, help='loss type. BCE recommended.')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer type.')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum parameter.')
    parser.add_argument('--n_epochs', default=10, type=int, help='total number of epochs to train.')
    parser.add_argument('--clip_grad', default=100.0, type=float, help='gradient clipping')
    parser.add_argument('--log_steps', default=10, type=int, help='train log steps.')

    # train : batch size
    parser.add_argument('--train_batch_size', default=100, type=int, help='batch size for training')
    parser.add_argument('--eval_batch_size', default=100, type=int, help='batch size for dev/test')

    # train : save/load args
    parser.add_argument('--task', default='dkt', type=str, help='dkt available.')
    parser.add_argument('--save_model', default=False, type=bool, help='whether to save model')
    parser.add_argument('--save_dir', default="./../ckpt/", type=str, help='save model path.')
    parser.add_argument('--save_epochs', default=10, type=int, help='save moodel for every n epochs')
    parser.add_argument('--load_dir', default="./../ckpt/", type=str, help='load model path')
    parser.add_argument('--load_model', default=False, type=bool, help='whether to load model')
    parser.add_argument('--load_epoch', default=30, type=int, help='what epoch should be loaded.')
    parser.add_argument('--refresh_optim', default=False, type=bool, help='if loading model, do not load optimizer state.')

    args = parser.parse_args()
    args = check_args(args)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)    
