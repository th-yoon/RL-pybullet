import torch
import numpy as np
import os
from model.util import save_plot, save_variance_plot
from model.network import Policy
from model.agent import AgentREINFORCE

class Runner():

    def __init__(self, env, num_iterations, num_episodes, num_timesteps, 
        log_interval, save_freq, save_path, is_render, is_pid, 
        is_continuous, checkpoint=None):

        self.env = env
        self.agent = AgentREINFORCE(observation_space_shape=env.observation_space.shape,
                                action_space_shape=env.action_space.shape,
                                is_continuous=is_continuous,
                                lr=1e-3, gamma=0.99, checkpoint=checkpoint) 
                        
        self.num_iterations = num_iterations
        self.num_episodes = num_episodes
        self.num_timesteps = num_timesteps
        self.log_interval = log_interval
        self.save_freq = save_freq
        self.save_path = save_path
        self.is_render = is_render
        self.is_pid = is_pid
        self.checkpoint = checkpoint

        self.running_reward = []
        self.average_reward = []
        self.cumulative_reward = 0
        self.policy_loss = []
        self.iteration_average_reward = [[] for _ in range(num_iterations)]
        self.iteration_policy_loss = [[] for _ in range(num_iterations)]

    def reset(self):
        self.running_reward = []
        self.average_reward = []
        self.cumulative_reward = 0
        self.policy_loss = []
        self.agent.reset()

    def load_model(self, checkpoint):
        if checkpoint is not None:
            self.running_reward = checkpoint['running_reward']
            self.average_reward = checkpoint['average_reward']
            self.cumulative_reward = checkpoint['cumulative_reward']
            self.policy_loss = checkpoint['policy_loss']
            self.episode_start_num = checkpoint['epoch']
            self.iteration_start_num = checkpoint['iteration']
            self.iteration_average_reward = checkpoint['iteration_average_reward']
            self.iteration_policy_loss = checkpoint['iteration_policy_loss']

    def save_model(self, iteration, epoch):
        model_save_path = os.path.join(self.save_path, 
            'model', 'model_iter{}_ep{}.pth'.format(iteration, epoch))
        torch.save({
            'iteration': iteration,
            'epoch': epoch + 1,
            'policy_net_state_dict': self.agent.policy.state_dict(),
            'policy_optimizer_state_dict': self.agent.policy_optimizer.state_dict(),
            'running_reward': self.running_reward,
            'average_reward': self.average_reward,
            'cumulative_reward': self.cumulative_reward,
            'policy_loss': self.policy_loss,
            'iteration_average_reward': self.iteration_average_reward,
            'iteration_policy_loss': self.iteration_policy_loss
        }, model_save_path)

    def save_episodic_plots(self, episode_counter):
        episodic_png_path = os.path.join(self.save_path, self.env.unwrapped.spec.id + '_episodic_reward.png')
        average_png_path = os.path.join(self.save_path, self.env.unwrapped.spec.id + '_average_reward.png')
        policy_png_path = os.path.join(self.save_path, self.env.unwrapped.spec.id + '_policy_loss.png')
        x = range(0, episode_counter)
        save_plot(x=x, y=self.running_reward, x_label='# Episode', y_label='Episodic rewards', save_path=episodic_png_path)
        save_plot(x=x, y=self.policy_loss, x_label='# Episode', y_label='Policy loss per episode', save_path=policy_png_path)
        save_plot(x=x, y=self.average_reward, x_label='# Episode', y_label='Average reward', save_path=average_png_path)

    def save_iterations_plots(self):
        iteration_average_png_path = os.path.join(self.save_path, self.env.unwrapped.spec.id + '_iter_average_reward.png')
        iteration_policy_png_path = os.path.join(self.save_path, self.env.unwrapped.spec.id + '_iter_policy_loss.png')
        x = range(0, int(self.num_episodes / self.log_interval))
        iteration_average_reward_np = np.array(self.iteration_average_reward)
        iteration_policy_loss_np = np.array(self.iteration_policy_loss)
        average_reward_mean = np.mean(iteration_average_reward_np, axis=0)
        average_reward_sigma = 2 * np.std(iteration_average_reward_np, axis=0)
        average_policy_loss_mean = np.mean(iteration_policy_loss_np, axis=0)
        average_policy_loss_sigma = 2 * np.std(iteration_policy_loss_np, axis=0)
        save_variance_plot(x=x, y=average_reward_mean, variance=average_reward_sigma,
                        x_label='# Episode', y_label='Average reward', interval=self.log_interval, save_path=iteration_average_png_path) 
        save_variance_plot(x=x, y=average_policy_loss_mean, variance=average_policy_loss_sigma,
                        x_label='# Episode', y_label='Policy loss', interval=self.log_interval, save_path=iteration_policy_png_path) 

    def rollout(self, episode_counter, iteration):
        state = self.env.reset(fs=self.num_timesteps) if self.is_pid else self.env.reset()
        episodic_reward = 0
        for _ in range(self.num_timesteps):
            probs = self.agent.get_action(state)
            state, reward, done, _ = self.env.step(probs, is_pid=self.is_pid) if self.is_pid else self.env.step(probs)
            # alive check needs to be done after env.step only if discrete action space
            if done == 1:
                break
            self.agent.policy.rewards.append(reward)
            episodic_reward += reward
        self.running_reward.append(episodic_reward)
        policy_loss_item = self.agent.finish_episode()
        self.policy_loss.append(policy_loss_item)
        self.cumulative_reward += episodic_reward
        cumulative_mean_reward = self.cumulative_reward / episode_counter
        self.average_reward.append(cumulative_mean_reward)
        # print rewards
        if episode_counter % self.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                episode_counter, episodic_reward, cumulative_mean_reward))
            self.iteration_average_reward[iteration].append(cumulative_mean_reward)
            self.iteration_policy_loss[iteration].append(policy_loss_item.detach().numpy())
        # save parameters
        if episode_counter % self.save_freq == 0:
            self.save_model(iteration=iteration, epoch=episode_counter)
            self.save_episodic_plots(episode_counter=episode_counter)

    def train(self):
        episode_start_num = 1
        iteration_start_num = 0
        self.load_model(self.checkpoint)
        for i in range(iteration_start_num, self.num_iterations):
            for episode_counter in range(episode_start_num, self.num_episodes + 1):
                if self.is_render:
                    self.env.render()
                self.rollout(episode_counter=episode_counter, iteration=i)
            # initialize after iteration
            self.reset()
            print('Iteration #{} completed'.format(i + 1))
        if self.num_iterations > 1:
            self.save_iterations_plots()
        self.env.close()
        print('Training completed.')