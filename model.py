import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchviz import make_dot

#def show_network(network):
#    x = Variable(torch.randn(1,network.state_size))
#    y = network(x)
#    make_dot(y, params=dict(list(network.named_parameters())))

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, hidden_sizes = [64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (list): List of number of nodes in each hidden layer
        """
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(self.state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], self.action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def show_network(self):
        x = Variable(torch.randn(1,self.state_size))
        y = self(x)
        return make_dot(y, params=dict(list(self.named_parameters())))
    
class DuelingQNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed, hidden_sizes = [64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (list): List of number of nodes in each hidden layer
            
        """
        super(DuelingQNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1 = nn.Linear(self.state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3adv = nn.Linear(hidden_sizes[1], self.action_size)
        self.fc3val = nn.Linear(hidden_sizes[1], 1)
     
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        adv = F.relu(self.fc3adv(x))
        val = F.relu(self.fc3val(x))
        return val + (adv - adv.mean(dim=1, keepdim=True))
    
    def show_network(self):
        x = Variable(torch.randn(1,self.state_size))
        y = self(x)
        return make_dot(y, params=dict(list(self.named_parameters())))
