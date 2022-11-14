from .locomotion_robot import WalkerBase

class Swimmer(WalkerBase):
    foot_list = ['torso', 'mid', 'back']

    def __init__(self):
        WalkerBase.__init__(self, "swimmer.xml", "torso",
                            action_dim=5, obs_dim=21, power=2.5, velocity=4)

    def alive_bonus(self, z, pitch):
        # 0.25 is central sphere rad, die if it scrapes the ground
        #return +1 if z > 0.26 else -1
        return 0

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        # power coefficient
        self.jdict["rot2"].power_coef = 2 
        self.jdict["rot3"].power_coef = 2 
        # velocity coefficient
        self.jdict["rot2"].velocity_coef = 1
        self.jdict["rot3"].velocity_coef = 1