import torch
import random
from collections import deque
class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen = int(capacity))

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, sample_size):
        batch = random.sample(self.buffer, sample_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states, dim = 0),
            torch.tensor(actions, dtype = torch.long),
            torch.tensor(rewards, dtype = torch.float32),
            torch.stack(next_states, dim = 0),
            torch.tensor(dones, dtype = torch.float32),
        )
    
    def __len__(self):
        return len(self.buffer)