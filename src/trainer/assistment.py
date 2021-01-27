import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop

from .base import TrainerBase


class AssistTrainer(TrainerBase):

    def __init__(self, args):
        super(AssistTrainer, self).__init__(args)
        self.args = args
        self.data = args.data
        self.loader = args.loader
        self.model = None


    def dataloader(self):
        """
        load data and build dataset -> loaders
        """

        # load / split data
        train_data = self.data.get_train_data()
        if self.args.use_dev:
            train_data, dev_data = self.data.split_data(train_data)
        test_data = self.data.get_test_data()

        #print(train_data[0])
        #print(dev_data[0])
        #print(test_data[0])

        # build dataset
        train_dataset = self.loader.build_dataset(
            train_data, 
            self.args.train_max_seq_len)
        train_loader = self.loader.build_dataloader(
            train_dataset, 'train')

        test_dataset = self.loader.build_dataset(
            test_data,
            self.args.eval_max_seq_len)
        test_loader = self.loader.build_dataloader(
            test_dataset, 'test')

        if self.args.use_dev:
            dev_dataset = self.loader.build_dataset(
                dev_data,
                self.args.eval_max_seq_len)
            dev_loader = self.loader.build_dataloader(
                dev_dataset, 'dev')
            return train_loader, dev_loader, test_loader
        else:
            return train_loader, test_loader


    def process_batch(self, batch, max_seq_len):
        qid, correct, mask, n_questions = batch
        # qid           : [args.train_batch_size, args.max_seq_len], float
        # correct       : [args.train_batch_size, args.max_seq_len], float
        # mask          : [args.train_batch_size, args.max_seq_len], bool (for each example, mask length is n_questions -1)
        # n_questions   : [args.max_seq_len], int

        # 0. actual sequence -1 masks
        mask = mask.type(torch.FloatTensor)

        # 1. build input_X (~2q dim)
        # [incorrect 0 0 0 0 0 correct 0 0 0 0 0]
        input_index = correct * self.args.n_questions + qid
        max_input_index = 2 * self.args.n_questions
        inputX = F.one_hot(
            input_index, 
            max_input_index)[:,:max_seq_len,:]
        inputX = inputX.type(torch.FloatTensor) * mask.unsqueeze(2)
        #print(inputX.size())

        # 2. build inputY (q dim) : answer one-hot ids
        inputY = F.one_hot(
            qid, 
            self.args.n_questions)[:,1:,:]
        inputY = inputY.type(torch.FloatTensor) * mask.unsqueeze(2)
        #print(inputY.size())
        
        # 3. build correct (q dim) : answer 1/0
        correct = inputY * correct[:,1:].unsqueeze(2) * mask.unsqueeze(2)
        #print(correct.size())
        
        #print(mask)
        #print(inputX[0][:4])
        #print(inputY[0][:4])
        #print(correct[0][:4])
        
        inputX = inputX.to(self.args.device)
        inputY = inputY.to(self.args.device)
        correct = correct.to(self.args.device)
        mask = mask.to(self.args.device)

        return inputX, inputY, correct, mask


    def compute_loss(self, logit, correct, mask):
        """
        input :
            logit   : (batch_size, max_seq_len, n_questions)
            correct : (batch_size, max_seq_len, n_questions)
            mask    : (batch_size, max_seq_len)
        output :
            loss
        """
        loss = self.loss_function(
            logit, 
            correct) * mask.unsqueeze(2) #
        loss = torch.sum(loss, dim=2) # masked dim
        loss = torch.mean(loss)
        return loss


    def update_params(self, loss):
        """
        update self.model.parameters()
        
        # TODO : gradient accumulation
        """
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.args.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.n_updates += 1


    def set_model(self):
        self.model = self.args.model(self.args)
        self.model.to(self.args.device) # load model to target device


    def train(self):
        # load dataloadrers
        if self.args.use_dev:
            train_loader, dev_loader, test_loader = self.dataloader()
        else:
            train_loader, test_loader = self.dataloader()

        # set trainer components
        if self.model is None: 
            self.set_model()

        # set loss / optim
        self.loss_function = self.set_loss()
        self.set_optimizer(self.model.parameters())
        self.optimizer.zero_grad()

        # train model
        for i in range(self.args.n_epochs):
            print("Start Training: Epoch", str(i+1))
            self.model.train()

            for train_batch in train_loader:
                # process inputs
                inputX, inputY, correct, mask = self.process_batch(
                    train_batch,
                    self.args.train_max_seq_len)
                
                # forward step
                logits = self.model(inputX, inputY)

                # compute loss
                loss = self.compute_loss(logits, correct, mask)
                
                # backward params
                self.update_params(loss)

                if self.n_updates % self.args.log_steps == 0:
                    print("Training steps:", str(self.n_updates), "Loss:", str(loss.item()))
                
            self.n_epochs += 1

            if self.args.save_model:
                if self.n_epochs % self.args.save_epochs == 0 and self.n_epochs != 0:
                    self.save_model(self.model, self.optimizer)

            # eval model
            if self.args.use_dev:
                dev_tp, dev_tc, dev_tm = self.predict(dev_loader)

                dev_predict, dev_correct = self.metrics.flatten(dev_tp, dev_tc, dev_tm)
                dev_auc = self.metrics.auc(dev_predict, dev_correct)
                
                print("Dev AUC:", dev_auc)

            tst_tp, tst_tc, tst_tm = self.predict(test_loader)
            
            tst_predict, tst_correct = self.metrics.flatten(tst_tp, tst_tc, tst_tm)
            tst_auc = self.metrics.auc(tst_predict, tst_correct)
            
            print("Test AUC:", tst_auc)


    def predict(self, eval_loader):
        total_predictions = []
        total_correct = []
        total_mask = []

        if self.model is None: self.set_model()

        self.model.eval()
        
        for eval_batch in eval_loader:
            # process inputs
            inputX, inputY, correct, mask = self.process_batch(
                eval_batch,
                self.args.eval_max_seq_len)
            
            # forward step
            logits = self.model(inputX, inputY)

            # predictions
            # product with mask because of logit=0 is sigm(0) != 0
            predictions = torch.mean(
                self.model.activation(logits) * mask.unsqueeze(2), 
                2)
            correct = torch.sum(correct, 2)

            # if using GPU, move every batch predictions to CPU 
            # to not consume GPU memory for all prediction results
            if self.args.device == 'cuda':
                predictions = predictions.to('cpu').detach().numpy()
                correct = correct.to('cpu').detach().numpy()
                mask = mask.to('cpu').detach().numpy()
            else: # cpu
                predictions = predictions.detach().numpy()
                correct = correct.detach().numpy()
                mask = mask.detach().numpy()

            total_predictions.append(predictions)
            total_correct.append(correct)
            total_mask.append(mask)
        
        total_predictions = np.concatenate(total_predictions) # (total eval examples, max_sequence_length)
        total_correct = np.concatenate(total_correct) # (total eval examples, max_sequence_length)
        total_mask = np.concatenate(total_mask) # (total eval examples, max_sequence_length)

        return total_predictions, total_correct, total_mask

        
        