import numpy as np

class PID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, setpoint=0.0, integrator_limit=1e6):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self._integral = 0.0
        self._last_error = None
        self.integrator_limit = integrator_limit

    def reset(self):
        self._integral = 0.0
        self._last_error = None

    def step(self, t, measurement, dt):
        error = self.setpoint - measurement
        self._integral += error * dt
        # clamp integrator
        self._integral = max(min(self._integral, self.integrator_limit), -self.integrator_limit)
        d_error = 0.0
        if self._last_error is not None and dt > 0:
            d_error = (error - self._last_error) / dt
        self._last_error = error
        output = self.kp * error + self.ki * self._integral + self.kd * d_error
        return output
