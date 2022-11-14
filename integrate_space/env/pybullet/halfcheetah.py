from .base_env import _Env
import numpy as np


class HalfCheetah(_Env):
    def __init__(self, config):
        super().__init__(config)

        self.robotId = self.p.loadMJCF('mjcf/half_cheetah.xml')[0]
        self._env_name = 'half_cheetah'
        self.torque_gamma = 300
        print("r-id:", self.robotId)
        print("halfcheetah-env created!")

        self.env_init()

    def reward(self, state: np.ndarray, pre_state: np.ndarray) -> (float, bool):
        # object: move 100 to x axis direction.
        cur_x = state[0]
        pre_x = pre_state[0]
        done = False

        if cur_x == 100:
            done = True

        reward = (cur_x - pre_x)

        return reward, done

    def reset(self):
        self.p.resetBasePositionAndOrientation(self.robotId, (0, 1, 0.75), (0, 0, 0, 1))
        state = self.get_state()
        self.previous_state = state
        return state
