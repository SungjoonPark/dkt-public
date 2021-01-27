import os, sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop

from .metrics import Metrics


class TrainerBase():

    def __init__(self, args):
        self.args = args

        self.n_updates = 0 # current number of updates
        self.n_epochs = 0 # current number of epochs

        # set device
        self.args.device = self.set_device()

        # metrics
        self.metrics = Metrics(self.args)


    def set_device(self):
        if torch.cuda.is_available() and self.args.device == 'gpu':
            device = "cuda"
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            device = "cpu"
            print('Using the CPU instead.')
        return device


    def set_optimizer(self, params):
        if self.args.optim == "adam":
            self.optimizer = Adam(
                params, 
                lr=self.args.lr, 
                betas=(0.9, 0.999), 
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)

        elif self.args.optim == "adamW":
            self.optimizer = AdamW(
                params, 
                lr=self.args.lr,
                betas=(0.9, 0.999), 
                eps=1e-08,
                weight_decay=1e-2, 
                amsgrad=False)

        elif self.args.optim == "sgd":
            self.optimizer = SGD(
                params,
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=0, 
                nesterov=False)
        
        elif self.args.optim == "rmsprop":
            self.optimizer = RMSprop(
                params,
                lr=self.args.lr,
                alpha=0.99,
                eps=1e-08, 
                weight_decay=0, 
                momentum=self.args.momentum)
        
        else:
            raise NotImplementedError


    def set_loss(self):
        if self.args.loss == "NLL":
            loss = nn.NLLLoss()
        elif self.args.loss == "CE": # logit, class
            loss = torch.nn.CrossEntropyLoss()
        elif self.args.loss == "BCE": # logit, class # default
            loss = torch.nn.BCELoss(
                reduction="none"
            )
        elif self.args.loss == "MSE":
            loss = torch.nn.MSELoss()
        return loss


    def _get_model_path(self):
        ckpt_name = "-".join([
            self.args.dataset, 
            self.args.task,
            self.args.model_type,
            str(self.args.n_layers),
            str(self.n_epochs)]) + ".ckpt"
        save_path = self.args.save_dir + ckpt_name

        return save_path


    def save_model(self, model, optimizer):
        save_path = self._get_model_path()
        print("Saving Model to:", save_path)
        save_state = {
            'n_updates': self.n_updates,
            'epoch': self.n_epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if not os.path.isfile(save_path):
            f = open(save_path, "x")
        torch.save(save_state, save_path)
        print("Saving Model to:", save_path, "...Finished.")


    def load_model(self):
        save_path = self._get_model_path()
        print("Loading Model from:", save_path)
        load_state = torch.load(save_path)

        # 1. load model state
        self.model.load_state_dict(
            load_state['state_dict'], 
            strict=False)

        # 2. load optim state
        if self.args.refresh_optim:
            self.n_updates = 0 # renewed
            self.n_epochs = 0 # renewed
        else:
            self.optimizer.load_state_dict(load_state['optimizer'])
            self.n_updates = load_state['n_updates']
            self.n_epochs = load_state['epoch']
        
        print("Loading Model from:", save_path, "...Finished.")


    def dataloader(self):
        raise NotImplementedError


    def train(self):
        raise NotImplementedError

    
    def eval(self):
        raise NotImplementedError





