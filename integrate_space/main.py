from train import Trainer
from utils import ObjectView, make_directory
import click


@click.command()
@click.option("--work_type", type=str, default="train", help="choose work type 'train' or 'test'")
@click.option("--seed", type=int, default=1, help="choose seed number")
@click.option("--env_type", type=str, default="pendulum", help="choose environment type 'Ant' and 'HalfCheetah'")
@click.option("--algorithm_type", type=str, default="ppo", help="choose algorithm type 'PPO', 'REINFORCE' and 'DDPG'")
#env
@click.option("--is_render", type=bool, default=False, help="render the environment or not (default: False)")
@click.option("--is_pid", type=bool, default=False, help="Use PID controller (default: False)")
@click.option("--gravity", type=tuple, default=(0, 0, -10), help="gravity of simulation")
@click.option("--step_tick", type=int, default=15, help="number of call the stepSimulation() function in one step")
@click.option("--env_upper_bound", type=int, default=1, help="upper bound value of actor network output")
@click.option("--env_lower_bound", type=int, default=-1, help="lower bound value of actor network output")
#algorithms
@click.option("--lr", type=float, default=1e-4, help="learning rate")
@click.option("--betas", type=tuple, default=(0.9, 0.999), help="betas for optimizer")
@click.option("--gamma", type=float, default=0.99, help="discount factor")
@click.option("--action_std", type=float, default=0.5, help="constant std for action distribution (Multivariate Normal)")
#ppo
@click.option("--continuous_action", type=bool, default=True, help="if True, the action of model is continuous")
@click.option("--reward_norm", type=bool, default=False, help="nomalize the set of returns(rewards)")
@click.option("--advantage_norm", type=bool, default=False, help="nomalize the set of advantages")
@click.option("--gae_param", type=float, default=0.95, help="GAE parameter")
@click.option("--eps_clip", type=float, default=0.2, help="clipping parameter for PPO")
@click.option("--k_epochs", type=int, default=80, help="number of epoch of update policy")
@click.option("--update_batch_size", type=int, default=50, help="number of epoch of update policy")
@click.option("--update_timestep", type=int, default=800, help="number of epoch of update policy")
#reinforce
@click.option("--log_interval", type=int, default=10, help="Interval between training status log")
#ddpg
@click.option("--tau", type=float, default=0.001, help="Interval between training status log")
#train
@click.option("--num_iterations", type=int, default=1, help="number of iterations")
@click.option("--max_episode", type=int, default=100000, help="max episode number")
@click.option("--max_step", type=int, default=100, help="max step number of one episode")
@click.option("--train_num", type=int, default=1, help="number of train")
@click.option("--gpu_num", type=int, default=None, help="number of gpu to use")
@click.option("--save_freq", type=int, default=100, help="Save the model every n steps")
@click.option("--save_path", type=str, default="save_model", help="model save path")
@click.option("--model_load_path", type=str, default=None, help="model load path")
@click.option("--model_load_num", type=int, default=None, help="epoch number of saved model")
@click.option("--log_save_path", type=str, default="save_log", help="log save path")
def main(*args, **kwargs):
    config = ObjectView(kwargs)
    # make_directory(config.save_path)
    # make_directory(config.log_save_path)

    t = Trainer(config)
    t.train()


if __name__ == "__main__":
    main()
