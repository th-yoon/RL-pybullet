import gym
from env import HalfCheetah, Ant, Pendulum, CartPole
from model.ppo import Runner as ppo_runner
from model.reinforce import Runner as reinforce_runner
from model.ddpg import Runner as ddpg_runner
from collections import deque
from utils import make_directory
import numpy as np


class Trainer:
    def __init__(self, config):
        self.seed = config.seed
        self.set_env(config)
        self.n_train = config.train_num

        self.save_freq = config.save_freq
        self.save_path = config.save_path
        self.log_save_path = config.log_save_path
        self.train_name = self.train_name_make(config)

        # make_directory(self.model_save_path + "/" + self.train_name)
        # make_directory(self.log_save_path + "/" + self.train_name)

        if config.algorithm_type == "ppo":
            print("running with PPO Agent")
            self.runner = ppo_runner(config, self.env)
        elif config.algorithm_type == "reinforce":
            print("running with REINFORCE Agent")
            self.runner = reinforce_runner(config, self.env)
        elif config.algorithm_type == "ddpg":
            print("running with DDPG Agent")
            self.runner = ddpg_runner(config, self.env)

    def train(self):
        self.runner.train()

    def set_env(self, config):
        if config.env_type == "ant":
            print("env is ANT")
            self.env = Ant(config)

        elif config.env_type == "halfcheetah":
            print("env is HalfCheetah")
            self.env = HalfCheetah(config)

        elif config.env_type == "cartpole":
            print("env is CartPole-v1")
            self.env = CartPole(config)

        elif config.env_type == "pendulum":
            print("env is Pendulum-v0")
            self.env = Pendulum(config)

    def train_name_make(self, config):
        name = "env_{}-step_tick_{}-clip_{}-k_epoch_{}-update_step_{}-max_epi_{}-max_step_{}-train_{}".format(
            config.env_type, config.step_tick, config.eps_clip, config.k_epochs,
            config.update_timestep, config.max_episode, config.max_step, self.n_train
        )
        return name
