import numpy as np
from .physics_base import Module

class EMOscillator(Module):
    """
    Simple harmonic oscillator model for an EM mode:
    state [E, dE/dt]
    """

    def __init__(self, E0=0.0, V0=0.0, omega=2.0, gamma=0.05):
        super().__init__()
        self.state = np.array([E0, V0], dtype=float)
        self.state_labels = ["E", "V"]
        self.omega = omega
        self.gamma = gamma

    def derivative(self, t, state, inputs):
        E, V = state
        driving = inputs.get("em_drive", 0.0) if inputs else 0.0
        dE_dt = V
        dV_dt = -self.gamma * V - (self.omega**2) * E + driving
        return np.array([dE_dt, dV_dt], dtype=float)

    def step(self, dt, inputs=None):
        try:
            from .integrator import rk4_step
            def f_wrap(t, y):
                return self.derivative(t, y, inputs)
            _, next_y = rk4_step(f_wrap, 0.0, self.state, dt)
            self.state = next_y
            return float(self.state[0])
        except Exception:
            deriv = self.derivative(0, self.state, inputs)
            self.state = self.state + deriv * dt
            return float(self.state[0])

# Backwards-compatibility alias
EMFieldOscillator = EMOscillator
