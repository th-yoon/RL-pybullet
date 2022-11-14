from .locomotion_robot import WalkerBase
import numpy as np

class HalfCheetah(WalkerBase):
    foot_list = ["ffoot", "fshin", "fthigh", "bfoot", "bshin",
                 "bthigh"]  # track these contacts with ground

    def __init__(self):
        WalkerBase.__init__(self, "half_cheetah.xml", "torso",
                            #action_dim=6, obs_dim=26, power=0.90, velocity=5)
                            action_dim=6, obs_dim=26, power=1.80, velocity=1)

    def alive_bonus(self, z, pitch):
        # Use contact other than feet to terminate episode: due to a lot of strange walks using knees
        return +1.0 if np.abs(pitch) < 1.0 and not self.feet_contact[1] and not self.feet_contact[
            2] and not self.feet_contact[4] and not self.feet_contact[5] else -1.0

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        # power coefficient
        self.jdict["bthigh"].power_coef = 120.0
        self.jdict["bshin"].power_coef = 90.0
        self.jdict["bfoot"].power_coef = 60.0
        self.jdict["fthigh"].power_coef = 140.0
        self.jdict["fshin"].power_coef = 60.0
        self.jdict["ffoot"].power_coef = 30.0
        # velocity coefficient
        self.jdict["bthigh"].velocity_coef = 2.0
        self.jdict["bshin"].velocity_coef = 1.5 
        self.jdict["bfoot"].velocity_coef = 1.0
        self.jdict["fthigh"].velocity_coef = 2.3 
        self.jdict["fshin"].velocity_coef = 1.0
        self.jdict["ffoot"].velocity_coef = 0.5 
