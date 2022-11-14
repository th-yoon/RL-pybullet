import numpy as np


class PID:
    def __init__(self, step_tick):
        self.kp = 2.0
        self.ki = 1.0
        self.kd = 0.1
        self.P = 0
        self.I = 0
        self.D = 0
        self.tau = 1.0 / step_tick
        self.previous_error = 0
        self.output = 0


    def update(self, target_angle, current_angle):
        current_error = target_angle - current_angle
        delta_error = current_error - self.previous_error

        self.P = self.kp * current_error
        self.I += current_error * self.tau
        self.D = delta_error / self.tau

        self.previous_error = current_error
        self.output = self.P + (self.ki * self.I) + (self.kd * self.D)

        # print((self.kp, self.ki, self.kp), (self.P, self.I, self.D))


    def clear(self):
        self.P = 0
        self.I = 0
        self.D = 0

        self.previous_error = 0
        self.output = 0