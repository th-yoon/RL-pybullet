from .base_env import _Env
import numpy as np


class Ant(_Env):
    def __init__(self, config):
        super().__init__(config)

        self.robotId = self.p.loadMJCF('mjcf/ant.xml')[0]
        self._env_name = 'ant'
        self.torque_gamma = 600
        print("r-id:", self.robotId)
        print("ant-env created!")

        self.env_init()

    def reward(self, state: np.ndarray, pre_state: np.ndarray) -> (float, bool):
        # object: move 100 to x axis direction.
        cur_x = state[0]
        cur_y = state[1]
        pre_x = pre_state[0]
        pre_y = pre_state[1]
        done = False

        if cur_x == 100:
            done = True

        reward = (cur_x - pre_x) - abs(cur_y - pre_y)*0.5 + cur_x*0.6 - abs(cur_y)*0.3

        return reward, done

    def reset(self):
        self.p.resetBasePositionAndOrientation(self.robotId, (0, 0, 0.75), (0, 0, 0, 1.))
        state = self.get_state()
        self.previous_state = state
        return state