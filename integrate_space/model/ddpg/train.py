import numpy as np
from .agent import AgentDDPG


class Runner:
    def __init__(self, config, env):

        self.max_episode = config.max_episode
        self.max_step = config.max_step
        self.gamma = config.gamma
        self.tau = config.tau
        self.action_std = config.action_std
        self.upper_bound = config.env_upper_bound
        self.lower_bound = config.env_lower_bound

        self.env = env
        self.agent = AgentDDPG(config.action_std, 1000000, 64,
                               gamma=self.gamma, tau=self.tau,
                               num_states=self.env.state_size,
                               num_actions=self.env.action_size,
                               lower_bound=self.lower_bound,
                               upper_bound=self.upper_bound)
    
    def train(self):
        epr = []
        avg = []
        ql, vl, al = [], [], []
        ll = []
        for ep in range(self.max_episode):
            episodic_reward = 0
            state = self.env.reset()
            for step in range(self.max_step):
                action = self.agent.policy(state)
                next_state, reward, done = self.env.step(action)
                episodic_reward += reward
                self.agent.record((state, action, reward, next_state))
                q, v, a = self.agent.learn()
                ql.append(q)
                vl.append(v)
                al.append(a)
                ll.append(self.agent.update_target())
                if done == True:
                    break
                state = next_state
            epr.append(episodic_reward)
            avg.append(np.mean(epr[-40:]))
            print("Episode * {} * Avg Reward is ==> {}".format(ep, np.mean(epr[-40:])))

        self.env.env.close()

        # return epr, avg
        