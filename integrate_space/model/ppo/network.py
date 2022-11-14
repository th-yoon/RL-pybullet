import torch
import torch.nn as nn
import numpy as np

H1_SIZE = 64
H2_SIZE = 64

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class _Actor_con(nn.Module):
    def __init__(self, state_size, action_size, seed=1,
                 h1_size=H1_SIZE, h2_size=H2_SIZE, init_w=3e-3):
        super(_Actor_con, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, action_size)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        out = torch.tanh(self.fc1(state))
        out = torch.tanh(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out

class _Actor_dis(nn.Module):
    def __init__(self, state_size, action_size, seed=1,
                 h1_size=H1_SIZE, h2_size=H2_SIZE, init_w=3e-3):
        super(_Actor_dis, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, action_size)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        out = torch.relu(self.fc1(state))
        out = torch.relu(self.fc2(out))
        out = torch.softmax(self.fc3(out), dim=-1)
        return out

class _Critic(nn.Module):
    def __init__(self, state_size, seed=1,
                 h1_size=H1_SIZE, h2_size=H2_SIZE, init_w=3e-3):
        super(_Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, 1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        out = torch.relu(self.fc1(state))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out

