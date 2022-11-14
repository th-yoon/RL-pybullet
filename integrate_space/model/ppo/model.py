import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical, MultivariateNormal

from .network import _Actor_con, _Actor_dis, _Critic


class ActorCritic_con(nn.Module):
    def __init__(self, state_size, action_size, action_std, device):
        super(ActorCritic_con, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.actor = _Actor_con(self.state_size, self.action_size)
        self.critic = _Critic(self.state_size, self.action_size)
        self.action_var = torch.full((action_size,),
                                     action_std * action_std).to(self.device)

    def forward(self, state):
        prob = self.actor(state)
        value = self.critic(state)
        return prob, value

    def get_action(self, state, action=None):
        prob, value = self.forward(state)
        cov_mat = torch.diag(self.action_var).to(self.device)
        # print('prob:',prob)
        dist = MultivariateNormal(prob, cov_mat)
        if action == None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        # print(log_prob)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze()

class ActorCritic_dis(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(ActorCritic_dis, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.actor = _Actor_dis(self.state_size, self.action_size)
        self.critic = _Critic(self.state_size, self.action_size)

    def forward(self, state):
        prob = self.actor(state)
        value = self.critic(state)
        return prob, value

    def get_action(self, state, action=None):
        prob, value = self.forward(state)
        # print('prob:',prob)
        dist = Categorical(prob)
        if action == None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        # print(log_prob)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze()


class PPO:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.continuous_action = config.continuous_action

        self.eps_clip = config.eps_clip
        self.k_epochs = config.k_epochs
        self.action_std = config.action_std

        self.reward_norm = config.reward_norm
        self.advantage_norm = config.advantage_norm
        self.lr = config.lr
        self.gae_param = config.gae_param
        self.betas = config.betas
        self.gamma = config.gamma

        self.update_timestep = config.update_timestep
        self.update_batch_size = config.update_batch_size

        if config.gpu_num == None:
            self.device = torch.device(
                "cuda:{}".format(config.train_num % torch.cuda.device_count())
                if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:{}".format(config.gpu_num)
                                       if torch.cuda.is_available() else "cpu")

        if self.continuous_action:
            self.policy = ActorCritic_con(state_size, action_size, self.action_std, self.device).to(self.device)
            self.policy_old = ActorCritic_con(state_size, action_size, self.action_std, self.device).to(self.device)
        else:
            self.policy = ActorCritic_dis(state_size, action_size, self.device).to(self.device)
            self.policy_old = ActorCritic_dis(state_size, action_size, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)

    def get_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        return self.policy_old.get_action(state)

    def update(self, memory, last_state):
        # convert list to tensor
        states = torch.FloatTensor(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        _, _, _, last_value = self.policy.get_action(
            torch.FloatTensor(last_state).to(self.device))

        rewards = []
        advantages = []
        advantage = 0
        discounted_reward = 0
        if self.gae_param == None:
            for t, (reward, value, done) in enumerate(
                    zip(reversed(memory.rewards),
                        reversed(memory.values[:-1]),
                        reversed(memory.is_done))):
                if done:
                    discounted_reward = 0

                if t == 0:
                    discounted_reward = reward + (self.gamma * last_value)
                else:
                    discounted_reward = reward + (self.gamma * discounted_reward)
                advantage = discounted_reward - value

                advantages.insert(0, advantage)
                rewards.insert(0, discounted_reward)
        else:
            for t, (reward, value, next_value, done) in enumerate(
                    zip(reversed(memory.rewards),
                        reversed(memory.values[:-1]),
                        reversed(memory.values),
                        reversed(memory.is_done))):
                if done:
                    discounted_reward = 0

                if t == 0:
                    discounted_reward = reward + (self.gamma * last_value)
                    td_error = reward - value
                else:
                    discounted_reward = reward + (
                                self.gamma * discounted_reward)
                    td_error = reward + self.gamma * next_value - value
                advantage = advantage * self.gae_param * self.gamma + td_error
                advantages.insert(0, advantage)
                rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).to(self.device).detach()
        advantages = torch.tensor(advantages).to(self.device).detach()

        # Normalizing the rewards and advantages
        if self.reward_norm:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        if self.advantage_norm:
            advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-5)

        # Optimize policy for K epochs
        for k_itr in range(self.k_epochs):
            sampler = self.random_sample(self.update_timestep - 1, self.update_batch_size)
            for sample_idx in sampler:
                mb_states = states[1:][sample_idx]
                mb_old_actions = old_actions[1:][sample_idx]
                mb_old_logprobs = old_logprobs[1:][sample_idx]
                mb_rewards = rewards[sample_idx]
                mb_advantages = advantages[sample_idx]

                total_loss, clip, vf, s = self.update_network(mb_states,
                                                              mb_old_actions,
                                                              mb_old_logprobs,
                                                              mb_rewards,
                                                              mb_advantages)
        print('total_loss:', total_loss, 'clip:', clip, 'vf:', vf, 's:', s)
        # print(old_logprobs)

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def update_network(self, state, action, old_logprob, reward, advantage):

        # Evaluating old actions and values
        _, logprob, entropy, value = self.policy.get_action(state, action)
        ratio = torch.exp(logprob - old_logprob.detach())
        # Surrogate
        surr1 = ratio * advantage.detach()
        surr2 = torch.clamp(ratio, 1. - self.eps_clip,
                            1. + self.eps_clip) * advantage.detach()

        clip = torch.min(surr1, surr2).mean()
        vf = (reward - value).pow(2).mean()
        s = entropy.mean()

        # Objective
        loss = -clip + 0.5 * vf - 0.01 * s
        # loss = 0.5*vf

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data.cpu().numpy(), clip.data.cpu().numpy(), vf.data.cpu().numpy(), s.data.cpu().numpy()

    def random_sample(self, inds, minibatch_size):
        inds = np.random.permutation(inds)
        batches = inds[:len(inds) // minibatch_size * minibatch_size].reshape(
            -1, minibatch_size)
        for batch in batches:
            yield torch.from_numpy(batch).long()
        r = len(inds) % minibatch_size
        if r:
            yield torch.from_numpy(inds[-r:]).long()
