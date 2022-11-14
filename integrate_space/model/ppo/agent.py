import torch
import numpy as np
from .model import PPO

class MemoryBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.values = []
        self.logprobs = []
        self.rewards = []
        self.is_done = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        del self.values[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_done[:]


class AgentPPO(PPO):
    def __init__(self, state_size, action_size, config):
        super().__init__(state_size, action_size, config)

        self.save_path = config.save_path
        self.load_path = config.model_load_path
        self.model_load_num = config.model_load_num

        self.memory = MemoryBuffer()

    def update(self, next_state):
        super().update(self.memory, next_state)

    def save_model(self, episode_num, train_name):
        torch.save(
            self.policy.state_dict(),
            self.save_path + "/" + train_name
            + "/model{}.pkl".format(str(episode_num).zfill(4))
        )

    def load_model(self):
        self.policy.load_state_dict(
            torch.load(self.load_path +
                       "/model{}.pkl".format(str(self.model_load_num).zfill(4)))
        )
        self.policy_old.load_state_dict(self.policy.state_dict())

    def memory_clear(self):
        self.memory.clear_memory()
