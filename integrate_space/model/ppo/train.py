from .agent import AgentPPO
from collections import deque
import numpy as np


class Runner():
    def __init__(self, config, env):
        self.env = env
        self.agent = AgentPPO(self.env.state_size, self.env.action_size, config)

        self.max_episode = config.max_episode
        self.max_step = config.max_step
        self.update_timestep = config.update_timestep
        self.model_save_period = config.save_freq
        self.model_save_path = config.save_path
        self.log_save_path = config.log_save_path
        self.train_num = config.train_num
        self.upper_bound = config.env_upper_bound
        self.lower_bound = config.env_lower_bound

    def train(self):
        time_step = 0
        for episode in range(self.max_episode):
            state = self.env.reset()
            log_buffer = deque(maxlen=self.max_step)
            done = False

            cumulative_reward = 0
            for step in range(self.max_step):
                time_step += 1

                action, log_prob, entropy, value = self.agent.get_action(state)
                next_state, reward, done = self.env.step(action.data.cpu().numpy())
                if done:
                    reward = 0

                cumulative_reward += reward

                # log_buffer.append([*state, *next_state, action, reward, done])

                self.agent.memory.actions.append(action)
                self.agent.memory.states.append(state)
                self.agent.memory.values.append(value)
                self.agent.memory.logprobs.append(log_prob)
                self.agent.memory.rewards.append(reward)
                self.agent.memory.is_done.append(done)

                if time_step % self.update_timestep == 0:
                    print('episode:', episode, 'cumulative reward:',
                          cumulative_reward, end=' ')
                    self.agent.update(next_state)
                    self.agent.memory_clear()
                    time_step = 0
                if done:
                    break
                state = next_state
            print('episode:', episode, 'cumulative reward:', cumulative_reward)
            # log_buffer = np.array(log_buffer)
            # np.save(self.log_save_path + '/' + str(self.train_num) +
            #         '/log{}'.format(str(episode).zfill(5)), log_buffer)
