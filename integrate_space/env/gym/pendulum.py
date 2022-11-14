import gym
import numpy as np
import random

class Pendulum():
    def __init__(self, config):
        self._env_name = 'pendulum'
        self.env = gym.make('Pendulum-v0')
        self.action_size = 1
        self.state_size = 3
        self.lower_bound = -2.0
        self.upper_bound = 2.0
        self.is_render = config.is_render

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if self.is_render:
            self.env.render()
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done
