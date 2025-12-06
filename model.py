import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from typing import Literal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, 128)
        self.output_layer = nn.Linear(128,128)
        self.actor = nn.Linear(128,action_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.output_layer(x))
        actor = F.softmax(self.actor(x), dim = -1)
        return actor

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, 64)
        self.output_layer = nn.Linear(64,64)
        self.critic = nn.Linear(64,1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.output_layer(x))
        critic = self.critic(x)
        return critic


class A2C():
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 device: Literal['cuda', 'cpu'] = 'cuda'
                 ):
        print(f'Using device: {device}')
        self.device = torch.device(device)
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        




