import gym
import numpy as np
import random

class CartPole():
    def __init__(self, config):
        self._env_name = 'cartpole'
        self.env = gym.make('CartPole-v1')
        self.action_size = 2
        self.state_size = 4
        self.is_render = config.is_render

    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        if self.is_render:
            self.env.render()
        print(type(action))
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done
