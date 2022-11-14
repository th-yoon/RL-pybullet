from .pid_control import PID
import pybullet as p
import pybullet_data
import numpy as np

class _Env:
    def __init__(self, config):
        self.p = p
        self.render_type = None
        self.planeId = None
        self.robotId = None
        self._env_name = None
        self.gravity = config.gravity
        self.step_tick = config.step_tick
        self.torque_gamma = 1

        self.action_size = None
        self.state_size = None

        self.pid_control = PID(self.step_tick)

        if config.is_render == True:
            self.render_type = p.GUI
        else:
            self.render_type = p.DIRECT

        self.physicsClient = self.p.connect(self.render_type)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        self.planeId = self.p.loadURDF('plane.urdf')
        self.previous_state = None

    def env_init(self):
        p.setGravity(*self.gravity)

        self.joint_dict = {}
        for j in range(p.getNumJoints(self.robotId)):
            info = p.getJointInfo(self.robotId, j)

            if info[2] != p.JOINT_REVOLUTE: continue
            joint_name = info[1].decode('ascii')
            self.joint_dict[joint_name] = j

        self.action_size = len(self.joint_dict)
        # position and velocity of each joint + base position and orientation
        self.state_size = self.action_size * 2 + 7

    def get_state(self):
        # getJointStates return
        # (jointPosition, jointVelocity, (jointReactionForces), appliedJointMotorTorque)
        # (float, float, list of 6 floats, float)
        joint_state = p.getJointStates(self.robotId, list(self.joint_dict.values()))
        joint_state = np.array(joint_state)[:,:2].flatten()

        # getBasePositionAndOrientation return
        # position list of 3 floats and orientation as list of 4 floats in [x,y,z,w] order
        # ((x,y,z), (x,y,z,w))
        pos, ori = p.getBasePositionAndOrientation(self.robotId)
        base_state = np.array([*pos, *ori])
        # print('base_state:', base_state)
        return np.concatenate((base_state, joint_state)).astype(np.float)

    def step(self, action):
        for itr in range(self.step_tick):
            self._torque_action(action)
            p.stepSimulation()

        next_state = self.get_state()
        reward, done = self.reward(next_state, self.previous_state)
        self.previous_state = next_state

        return next_state, reward, done

    def _torque_action(self, torque_val):
        for joint_idx, t_val in zip(self.joint_dict.values(), torque_val):
            p.setJointMotorControl2(self.robotId, joint_idx,
                                    controlMode=p.TORQUE_CONTROL,
                                    force=t_val * self.torque_gamma)

    def pid_action(self, target_angle, current_angle):

        self.pid_control.update(target_angle, current_angle)
        output = self.pid_control.output
        # print('t:', target_angle,'c:', current_angle, 'o:', output)
        joint_torque = 10 * output
        return joint_torque

    def reward(self, state: np.ndarray, pre_state: np.ndarray) -> (float, bool):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
