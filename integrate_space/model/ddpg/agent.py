import numpy as np

from .noise import OUActionNoise
from .model import DDPG
from .util import *

class AgentDDPG(DDPG):
    def __init__(self, std_dev, buffer_capacity=100000, batch_size=64, gamma=0.99, tau=0.005, num_states=1, num_actions=1, lower_bound=-10, upper_bound=10):
        super().__init__(buffer_capacity, batch_size, gamma, tau, num_states, num_actions, lower_bound, upper_bound)
        self.noise = OUActionNoise(mean=np.zeros(self.num_actions), std_deviation = float(std_dev) * np.ones(self.num_actions))
    
    def policy(self, state):
        sampled_actions = self.actor_model(to_tensor(state))
        noise = self.noise()
        sampled_actions = to_numpy(sampled_actions) + noise
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return legal_action