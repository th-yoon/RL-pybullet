import argparse
import torch
import gym
import os
import random

from gym.envs.registration import registry, make, spec
from model.train import Runner 
from model.test import test
#from env.integrated_env.ant_env import AntEnv
#from env.integrated_env.half_cheetah_env import HalfCheetahEnv 

def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)

def make_path(algo, env, num_episodes):
    reinforce_path = os.path.join(os.getcwd(), algo)
    if not os.path.exists(reinforce_path):
        os.mkdir(reinforce_path)
    env_path = os.path.join(reinforce_path, env)
    if not os.path.exists(env_path):
        os.mkdir(env_path)
    save_path = os.path.join(env_path, str(num_episodes) + 'iters')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model_path = os.path.join(save_path, 'model')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    return save_path

'''
def select_env(name, render):
    if name == 'halfcheetah':
        return HalfCheetahEnv(render=render)
    if name == 'ant':
        return AntEnv(render=render)
    raise ValueError('No such environment is found')'''
    


#from train import train

def main():
    # ------------bullet-------------
    register(id='AntBulletEnv-v0',
             entry_point='env.locomotion_env:AntBulletEnv'
             )

    register(id='AntPentapodBulletEnv-v0',
             entry_point='env.locomotion_env:AntPentapodBulletEnv'
             )

    register(id='HalfCheetahBulletEnv-v0',
             entry_point='env.locomotion_env:HalfCheetahBulletEnv'
             )

    register(id='SwimmerBulletEnv-v0',
             entry_point='env.locomotion_env:SwimmerBulletEnv',
             )

    register(id='HumanoidBulletEnv-v0',
             entry_point='env.locomotion_env:HumanoidBulletEnv',
             )
             
    register(id='InvertedPendulumBulletEnv-v0',
             entry_point='env.pendulum_env:InvertedPendulumBulletEnv',
             )

    register(id='InvertedDoublePendulumBulletEnv-v0',
             entry_point='env.pendulum_env:InvertedDoublePendulumBulletEnv',
             )

    register(id='InvertedPendulumSwingupBulletEnv-v0',
             entry_point='env.pendulum_env:InvertedPendulumSwingupBulletEnv',
             )

    register(id='CartPoleBulletEnv-v0',
             entry_point='env.pendulum_env:CartPoleBulletEnv',
             )

    parser = argparse.ArgumentParser('RL trainer with pytorch')
    parser.add_argument('--algo', help='RL algorithm (default: REINFORCE)', default='REINFORCE',
                        type=str, required=False)  
    parser.add_argument('--env', help='environment ID (default: HalfCheetahBulletEnv-v0)',
                        default='HalfCheetahBulletEnv-v0', type=str)
    parser.add_argument('--test', help='Test the trained model (default: False)', default=bool(False),
                        type=bool)
    parser.add_argument('-n', '--num-iterations', help='Number of iterations (default: 1)', default=1,
                        type=int)
    parser.add_argument('--train-episodes', help='Number of episodes to train(default: 1e3)', default=int(1e3),
                        type=int)
    parser.add_argument('--train-timesteps', help='Timesteps per episode to train (default: 1000)', default=1000,
                        type=int)
    parser.add_argument('--save-freq', help='Save the model every n steps (default: 50)',
                        default=50, type=int)
    parser.add_argument('--seed', help='Random seed (default: 1)',
                        default=1, type=int)
    parser.add_argument('--log-interval', help='Interval between training status log (default: 10)',
                        default=10, type=int)
    parser.add_argument('--load', help='Load the trained model (default: None)',
                        default=None, type=str)
    parser.add_argument('--render', help='Render the training process (default: False)',
                        default=False, type=bool)
    parser.add_argument('--pid', help='Use PID controller (default: False)',
                        default=False, type=bool)
    parser.add_argument('--test-timesteps', help='Number of timesteps per episode to test (default: 10000)',
                        default=int(1e3), type=int)
    parser.add_argument('--test-episodes', help='Number of episodes to test (default: 5)',
                        default=5, type=int)
    parser.add_argument('--continuous-action', help='Is action type continuous? (default: True)',
                        default=1, type=int)
    args = parser.parse_args()

    save_path = make_path(algo=args.algo, env=args.env, num_episodes=args.train_episodes)

    # Instantiate the environment
    train_env = gym.make(args.env)
    test_env = gym.make(args.env)
    #train_env = select_env(name=args.env, render=args.render)
    #train_env.reset()
    #test_env = select_env(name=args.env, render=args.render)

    # Seed the environments
    train_env.seed(args.seed) if args.num_iterations == 1 else train_env.seed(random.randrange(100))
    test_env.seed(args.seed)

    # Seed the torch
    torch.manual_seed(args.seed) if args.num_iterations == 1 else torch.manual_seed(random.randrange(100))

    # Load the checkpoint if any
    checkpoint = torch.load(args.load) if args.load is not None else None

    # Train or test
    runner = Runner(env=train_env,
                    num_iterations=args.num_iterations,
                    num_episodes=args.train_episodes,
                    num_timesteps=args.train_timesteps,
                    log_interval=args.log_interval,
                    save_freq=args.save_freq,
                    save_path=save_path,
                    is_render=args.render,
                    is_pid=args.pid,
                    is_continuous=args.continuous_action,
                    checkpoint=checkpoint)
                    

    if args.test is True:
        test(env=test_env,
            num_timesteps=args.test_timesteps,
            num_episodes=args.test_episodes,
            is_render=True,
            is_pid=args.pid,
            is_continuous=args.continuous_action,
            checkpoint=checkpoint)
    else:
        runner.train()


if __name__ == '__main__':
    main()
