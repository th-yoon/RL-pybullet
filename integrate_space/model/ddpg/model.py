import numpy as np
import torch

from .network import Critic, Actor
from .util import *

class DDPG:
    def __init__(self, buffer_capacity=100000, batch_size=64, gamma=0.99, tau=0.005, num_states=1, num_actions=1, lower_bound=-10, upper_bound=10):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.buffer_counter = 0

        self.num_states = num_states
        self.num_actions = num_actions

        self.gamma = gamma
        self.tau = tau

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.actor_model = Actor(self.num_states, self.num_actions, self.upper_bound)
        self.critic_model = Critic(self.num_states, self.num_actions)
        
        self.target_actor = Actor(self.num_states, self.num_actions, self.upper_bound)
        self.target_critic = Critic(self.num_states, self.num_actions)

        hard_update(self.target_actor, self.actor_model)
        hard_update(self.target_critic, self.critic_model)

        self.critic_lr = 0.005
        self.actor_lr = 0.001

        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=self.critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)

        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
    
    def update_target(self):
        soft_update(self.target_critic, self.critic_model, self.tau)
        soft_update(self.target_actor, self.actor_model, self.tau)

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def learn(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = np.array(self.state_buffer[batch_indices])
        action_batch = np.array(self.action_buffer[batch_indices])
        reward_batch = np.array(self.reward_buffer[batch_indices])
        reward_batch = to_tensor(reward_batch.astype(dtype=np.float32))
        next_state_batch = np.array(self.next_state_buffer[batch_indices])

        target_actions = self.target_actor(to_tensor(next_state_batch))
        y = reward_batch + self.gamma * self.target_critic(to_tensor(next_state_batch), target_actions)

        self.critic_model.zero_grad()
        critic_value = self.critic_model(to_tensor(state_batch), to_tensor(action_batch))
        critic_loss = torch.nn.MSELoss()(critic_value, y)
        critic_loss.backward()
        self.critic_optimizer.step()

        q = to_numpy(critic_loss)
        v = np.mean(to_numpy(critic_value))

        self.actor_model.zero_grad()
        actions = self.actor_model(to_tensor(state_batch))
        critic_value = self.critic_model(to_tensor(state_batch), actions)
        actor_loss = -critic_value.mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        a = to_numpy(actor_loss)

        return q, v, a