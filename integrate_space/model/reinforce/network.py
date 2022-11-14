import torch
import torchvision.transforms
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(torch.nn.Module):

    def __init__(self, observation_space_shape, action_space_shape, is_continuous):
        super(Policy, self).__init__()
        self.observation_space_shape = observation_space_shape
        self.action_space_shape = action_space_shape
        if self.action_space_shape == ():
            self.action_space_shape = (1,)
        self.affine1 = torch.nn.Linear(self.observation_space_shape[0], 128)
        self.affine2 = torch.nn.Linear(128, 256)
        self.affine3 = torch.nn.Linear(256, 128)
        if is_continuous:
            self.value_interval = 3 if self.action_space_shape[0] == 1 else 21 
            self.affine4 = torch.nn.Linear(128, self.action_space_shape[0] * self.value_interval)
        else:
            self.value_interval = 2
            self.affine4 = torch.nn.Linear(128, 1 * self.value_interval)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.leaky_relu(x)
        x = self.affine2(x)
        x = self.leaky_relu(x)
        x = self.affine3(x)
        x = self.leaky_relu(x)
        x = self.affine4(x)
        action_scores = x.view(-1, self.value_interval)
        # return action_scores probabilities
        return self.softmax(action_scores)