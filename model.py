import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.layers = nn.ModuleList([nn.Linear(self.state_size, hidden_sizes[0])])
        self.layers.extend([nn.Linear(s1, s2) for s1, s2 in 
                            zip(hidden_sizes[:-1], hidden_sizes[1:])])
        self.out = nn.Linear(hidden_sizes[-1], self.action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)
    
    
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
        
        self.layers = nn.ModuleList([nn.Linear(self.state_size, hidden_sizes[0])])
        self.layers.extend([nn.Linear(s1, s2) for s1, s2 in 
                            zip(hidden_sizes[:-1], hidden_sizes[1:])])
        self.adv = nn.Linear(hidden_sizes[-1], self.action_size)
        self.val = nn.Linear(hidden_sizes[-1], 1)
     
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        adv = F.relu(self.adv(x))
        val = F.relu(self.val(x))    
        #return val + (adv - adv.mean(dim=1, keepdim=True))
        return val + adv
