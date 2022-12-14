from .base_robot import URDFBasedRobot
import numpy as np

class Cartpole(URDFBasedRobot):

    swingup = False

    def __init__(self):
        URDFBasedRobot.__init__(
            self, 'cartpole.urdf', 'cart', action_dim=1, obs_dim=5)

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.pole = self.parts["pole"]
        self.slider = self.jdict["slider_to_cart"]
        self.j1 = self.jdict["cart_to_pole"]
        u = self.np_random.uniform(low=-.1, high=.1)
        self.j1.reset_current_position(
            u if not self.swingup else 3.1415 + u, 0)
        self.j1.set_motor_torque(0)

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        if not np.isfinite(a).all():
            print("a is inf")
            a[0] = 0
        self.slider.set_motor_torque(200 * float(np.clip(a[0], -1, +1)))

    def calc_state(self):
        self.theta, theta_dot = self.j1.current_position()
        x, vx = self.slider.current_position()
        assert (np.isfinite(x))

        if not np.isfinite(x):
            print("x is inf")
            x = 0

        if not np.isfinite(vx):
            print("vx is inf")
            vx = 0

        if not np.isfinite(self.theta):
            print("theta is inf")
            self.theta = 0

        if not np.isfinite(theta_dot):
            print("theta_dot is inf")
            theta_dot = 0

        return np.array([x, vx, np.cos(self.theta), np.sin(self.theta), theta_dot])