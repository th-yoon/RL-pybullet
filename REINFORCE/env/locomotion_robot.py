from .base_robot import XmlBasedRobot, MJCFBasedRobot, URDFBasedRobot, BodyPart
import numpy as np
import pybullet
import os
import pybullet_data
import torch


class WalkerBase(MJCFBasedRobot):

    # velocity parameter is added
    def __init__(self, fn, robot_name, action_dim, obs_dim, power, velocity):
        MJCFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim)
        # power for torque control
        self.power = power
        # velocity for velocity control
        self.velocity = velocity
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.walk_target_x = 1e3  # a kilometer away
        self.walk_target_y = 0  # goal is to walk towards x-direction
        self.body_xyz = [0, 0, 0]

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        for j in self.ordered_joints:
            j.reset_current_position(
                self.np_random.uniform(low=-0.1, high=0.1), 0)

        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array(
            [0.0 for f in self.foot_list], dtype=np.float32)
        self.scene.actor_introduce(self)
        self.initial_z = None

    def apply_action(self, action, is_pid=False):
        #assert (np.isfinite(action_probs).all())
        if is_pid:
            for n, j in enumerate(self.ordered_joints):
                if hasattr(j, 'velocity_coef'):
                    j.set_velocity(j.velocity_coef * action[n])
                else:
                    j.set_velocity(action[n])
        else:
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_torque(self.power * j.power_coef * action[n])

    def calc_state(self):
        j = np.array([j.current_relative_position() for j in self.ordered_joints],
                     dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz()
                              for p in self.parts.values()]).flatten()
        self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2]
                         )  # torso z is more informative than mean z
        self.body_real_xyz = body_pose.xyz()
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        if self.initial_z == None:
            self.initial_z = z
        r, p, yaw = self.body_rpy
        # this calculates the angle that vector(target - body) forms on xy-plane
        self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                            self.walk_target_x - self.body_xyz[0])
        # this calculates the distance between the target and the body
        self.walk_target_dist = np.linalg.norm(
            [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
        angle_to_target = self.walk_target_theta - yaw  # the closer to 0 the better

        rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw),
                                                                 np.cos(-yaw), 0], [0, 0, 1]])
        vx, vy, vz = np.dot(rot_speed,
                            self.robot_body.speed())  # rotate speed back to body point of view

        more = np.array(
            [
                z - self.initial_z,
                np.sin(angle_to_target),
                np.cos(angle_to_target),
                0.3 * vx,
                0.3 * vy,
                0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
                r,
                p
            ],
            dtype=np.float32)
        return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second,
        # this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        # calc_state needs to precede to call this function as self.walk_target_dist is calculated in cal_state
        debugmode = 0
        if (debugmode):
            print("calc_potential: self.walk_target_dist")
            print(self.walk_target_dist)
            print("self.scene.dt")
            print(self.scene.dt)
            print("self.scene.frame_skip")
            print(self.scene.frame_skip)
            print("self.scene.timestep")
            print(self.scene.timestep)
        # return -(speed) as potential
        return -self.walk_target_dist / self.scene.dt
