from .model import REINFORCE
import torch
import torchvision
import numpy as np
from .network import Policy
from .util import save_plot, save_variance_plot
import matplotlib.pyplot as plt
import os
import time

class AgentREINFORCE(REINFORCE):

    def __init__(self, observation_space_size, action_space_size, lr=1e-3, gamma=0.99, checkpoint=None):
        self.checkpoint = checkpoint
        super().__init__(observation_space_size, action_space_size, lr, gamma)
        if self.checkpoint is not None:
            self.policy.load_state_dict(checkpoint['policy_net_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])

    def reset(self):
        self.__init__(self.observation_space_size, self.action_space_size, self.lr, self.gamma, self.checkpoint)

