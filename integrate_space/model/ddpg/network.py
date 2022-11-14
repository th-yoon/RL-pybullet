import torch
import numpy as np

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(torch.nn.Module):
    def __init__(self, num_states, num_actions, upper_bound):
        super(Actor, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.upper_bound = upper_bound

        self.fc1 = torch.nn.Linear(self.num_states, 400)
        self.fc2 = torch.nn.Linear(400, 300)
        self.fc3 = torch.nn.Linear(300, self.num_actions)

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-0.003, 0.003)
    
    def forward(self, x):
        out = self.fc1(x)
        out = torch.nn.LayerNorm(400)(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.nn.LayerNorm(300)(out)
        out = torch.relu(out)
        out = self.fc3(out)
        out = torch.tanh(out)

        outputs = out * self.upper_bound
        return outputs

class Critic(torch.nn.Module):
    def __init__(self, num_states, num_actions):
        super(Critic, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = torch.nn.Linear(self.num_states, 400)
        self.fc2 = torch.nn.Linear(400 + self.num_actions, 300)
        self.fc3 = torch.nn.Linear(300, 1)

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-0.0003, 0.0003)

    def forward(self, state_input, action_input):
        state_out = self.fc1(state_input)
        state_out = torch.relu(state_out)
        state_out = torch.nn.LayerNorm(400)(state_out)

        concat = torch.cat([state_out, action_input], 1)

        out =self.fc2(concat)
        out = torch.relu(out)
        out = torch.nn.LayerNorm(300)(out)
        outputs = self.fc3(out)

        return outputs