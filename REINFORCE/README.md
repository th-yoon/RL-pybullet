# REINFORCE with pytorch and pybullet



## How to train
* Run `__main__.py` with training options  
e.g., `python __main__.py  --env=AntBulletEnv-v0 -n=1 --num-episodes=1000 --render=True --pid=True`  
if you want to train `AntBulletEnv-v0` environment with 1000 episodes with PID and render the process.

## How to test
* Run `__main__.py` with testing options  
e.g., `python __main__.py  --env=AntBulletEnv-v0 --test=True --load=path/to/model.pth --pid=True`

## Results

### Demo (InvertedPendulumBulletEnv)
![invertedpendulum](img/InvertedPendulumBulletEnv.gif)  
Tested at 700th episode w/o PID

### Average rewards (InvertedPendulumBulletEnv)
![invertedpendulum](img/InvertedPendulumBulletEnv-v0_iter_average_reward.png)

### Policy loss (InvertedPendulumBulletEnv)
![invertedpendulum](img/InvertedPendulumBulletEnv-v0_iter_policy_loss.png)

### Demo (HalfCheetahBulletEnv)
![halfcheeatah](img/HalfCheetahBulletEnv.gif)  
Tested at 250th episode with PID

### Average rewards (HalfCheetahBulletEnv)
![halfcheeatah](img/HalfCheetahBulletEnv-v0_iter_average_reward.png)

### Policy loss (HalfCheetahBulletEnv)
![halfcheeatah](img/HalfCheetahBulletEnv-v0_iter_policy_loss.png)

### Demo (AntBulletEnv)
![ant](img/AntBulletEnv.gif)  
Tested at 250th episode with PID

### Average reward (AntBulletEnv)
![ant](img/AntBulletEnv-v0_iter_average_reward.png)

### Policy loss (AntBulletEnv)
![ant](img/AntBulletEnv-v0_iter_policy_loss.png)

## Reference
https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py  
https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs  
https://github.com/NaleRaphael/cartpole-control/blob/master/cpc/agent.py  
Williams, R. J. (1992a). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8:229–256.
