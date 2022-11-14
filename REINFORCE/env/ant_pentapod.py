from .locomotion_robot import WalkerBase

class AntPentapod(WalkerBase):
    foot_list = ['top_foot', 'front_left_foot',
                 'front_right_foot', 'back_left_foot', 'back_right_foot']

    def __init__(self):
        WalkerBase.__init__(self, "ant_pentapod.xml", "torso",
                            action_dim=10, obs_dim=33, power=2.5, velocity=4)

    def alive_bonus(self, z, pitch):
        # 0.25 is central sphere rad, die if it scrapes the ground
        return +1 if z > 0.26 else -1