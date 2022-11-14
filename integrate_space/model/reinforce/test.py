import torch
import numpy as np
import os
import time

from model.agent import AgentREINFORCE

def test(env, num_timesteps, num_episodes, is_render, is_pid, is_continuous, checkpoint):
    if checkpoint is None:
        raise ValueError('checkpoint argument is required')
    agent = AgentREINFORCE(observation_space_shape=env.observation_space.shape,
                        action_space_shape=env.action_space.shape,
                        is_continuous=is_continuous,
                        lr=1e-3, gamma=0.99, checkpoint=checkpoint)
    for e in range(num_episodes):
        if is_render:
            env.render()
        state = env.reset(fs=num_timesteps) if is_pid else env.reset()
        print('Episode #', e)
        for t in range(num_timesteps):
            if is_render:
                env.render()
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action, is_pid=is_pid) if is_pid else env.step(action)
            time.sleep(1./240.)
            # adjust camera
            # alive check needs to be done after env.step
            if done:
                break
    env.close()
    print('Test completed')