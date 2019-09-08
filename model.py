import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def init_hidden(layer):
    '''
    Initialize NN layers
    '''
    input_size = layer.weight.data.size()[0]
    lim = 1.0/np.sqrt(input_size)
    return (-lim, lim)

class ActorNN(nn.Module):
    '''
    Actor Class
    '''
    def __init__(self,state_size,action_size, hidden_size1=512,hidden_size2=256):
        '''
        Initialize parameters
        '''
        super(ActorNN, self).__init__()
        self.state_size = state_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2= hidden_size2
        self.action_size = action_size

        self.FC1 = nn.Linear(self.state_size,self.hidden_size1)
        self.FC2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.FC3 = nn.Linear(self.hidden_size2, self.action_size)
        self.reset_parameters()

    def forward(self, state):
        #print(state.size)
        x = F.relu(self.FC1(state))
        #print(state.size)
        x = F.relu(self.FC2(x))
        x = torch.tanh(self.FC3(x))
        return x

    def reset_parameters(self):
        self.FC1.weight.data.uniform_(*init_hidden(self.FC1))
        self.FC2.weight.data.uniform_(*init_hidden(self.FC2))
        #self.FC3.weight.data.uniform_(*init_hidden(self.FC3))
        self.FC3.weight.data.uniform_(-3e-3,3e-3)

class CriticNN(nn.Module):
    '''
    Critic class
    '''
    def __init__(self,state_size,action_size, hidden_size1=512,hidden_size2=256):
        '''
        Initialize parameters
        '''
        super(CriticNN, self).__init__()
        self.state_size = state_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2= hidden_size2
        self.action_size = action_size

        self.FC1 = nn.Linear(self.state_size, self.hidden_size1)
        self.FC2 = nn.Linear(self.hidden_size1+ self.action_size,self.hidden_size2)
        self.FC3 = nn.Linear(self.hidden_size2, 1)
        self.reset_parameters()

    def forward(self, state, action):
        x = F.relu(self.FC1(state))
        x = torch.cat((x,action),dim =1)
        x = F.relu(self.FC2(x))
        x = self.FC3(x)
        return x
    def reset_parameters(self):
        self.FC1.weight.data.uniform_(*init_hidden(self.FC1))
        self.FC2.weight.data.uniform_(*init_hidden(self.FC2))
        #self.FC3.weight.data.uniform_(*init_hidden(self.FC3))
        self.FC3.weight.data.uniform_(-3e-3,3e-3)
