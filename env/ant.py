from .locomotion_robot import WalkerBase

class Ant(WalkerBase):
    foot_list = ['front_left_foot', 'front_right_foot',
                 'left_back_foot', 'right_back_foot']

    def __init__(self):
        WalkerBase.__init__(self, "ant.xml", "torso",
                            action_dim=8, obs_dim=28, power=2.5, velocity=4)

    def alive_bonus(self, z, pitch):
        # 0.25 is central sphere rad, die if it scrapes the ground
        return +1 if z > 0.26 else -1

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        # power coefficient
        self.jdict["hip_1"].power_coef = 50
        self.jdict["hip_2"].power_coef = 50
        self.jdict["hip_3"].power_coef = 50
        self.jdict["hip_4"].power_coef = 50
        self.jdict["ankle_1"].power_coef = 100
        self.jdict["ankle_2"].power_coef = 100
        self.jdict["ankle_3"].power_coef = 100
        self.jdict["ankle_4"].power_coef = 100
        # velocity coefficient
        self.jdict["hip_1"].velocity_coef = 1
        self.jdict["hip_2"].velocity_coef = 1
        self.jdict["hip_3"].velocity_coef = 1
        self.jdict["hip_4"].velocity_coef = 1
        self.jdict["ankle_1"].velocity_coef = 2
        self.jdict["ankle_2"].velocity_coef = 2
        self.jdict["ankle_3"].velocity_coef = 2
        self.jdict["ankle_4"].velocity_coef = 2