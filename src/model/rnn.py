import torch
import torch.nn as nn


class RNN(nn.Module):
    
    def __init__(self, args):
        super(RNN, self).__init__()
        self.args = args
        self.device = args.device
        
        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        if self.args.input_type == "onehot":
            self.input_size = 2 * self.args.n_questions
        else : # self.args.input_type == "dense":
            self.input_fc = nn.Linear(
                2 * self.args.n_questions,
                self.args.n_input_dim
            )
            self.input_size = self.args.n_input_dim
        
        # Defining the layers
        # Embedding Layer
        self.embedding_layer = nn.Embedding(
            num_embeddings=self.args.n_questions * 2, # input_index = correct * self.args.n_questions + qid
            embedding_dim=self.args.embedding_dim)

        # RNN Layer
        self.rnn = nn.RNN(
            self.input_size, 
            self.hidden_dim, 
            self.n_layers, 
            batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(
            self.hidden_dim, 
            self.args.n_questions)
    
        self.activation = nn.Sigmoid()


    def forward(self, inputX, inputY):
        batch_size = inputX.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # use input layer if using compressedSensing
        if self.args.input_type == "dense":
            inputX = self.embedding_layer(inputX)
            
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(inputX, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        #print(out.size())

        out = self.fc(out)
        out = out.contiguous().view(
            batch_size, 
            -1, 
            self.args.n_questions)
        prob = self.activation(out) * inputY
        #print(out.size())
        #print(inputY.size())
        #print(logit.size())

        return prob
    

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(
            self.n_layers, 
            batch_size, 
            self.hidden_dim)
        hidden = hidden.to(self.device)
        
        return hidden

