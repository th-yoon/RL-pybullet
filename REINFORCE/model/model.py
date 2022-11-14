import torch
import torchvision
import numpy as np
from model.network import Policy
from .util import save_plot, save_variance_plot
import matplotlib.pyplot as plt
import os
import time


class REINFORCE():

    def __init__(self, observation_space_shape, action_space_shape, is_continuous, lr=1e-3, gamma=0.99):
        self.policy = None
        self.policy_optimizer = None
        self.policy = Policy(observation_space_shape=observation_space_shape,
                            action_space_shape=action_space_shape,
                            is_continuous=is_continuous)

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr)
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = gamma
        self.observation_space_shape = observation_space_shape
        self.action_space_shape = action_space_shape
        self.state = None
        self.lr = lr
        self.is_continuous = is_continuous

    def get_action(self, state):
        self.state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(self.state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        dim = len(action)
        self.policy.saved_log_probs.append(m.log_prob(action))
        if self.is_continuous:
            action = torch.true_divide(action, int(len(probs[0]) / 2)) - torch.ones(dim)
        else:
            action = int(action)
        return action

    def finish_episode(self):
        G = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, G in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()
        # initialization
        self.policy.rewards = []
        self.policy.saved_log_probs = []
        return policy_loss

    def reset(self):
        pass