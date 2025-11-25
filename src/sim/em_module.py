import numpy as np
from .physics_base import Module

class EMOscillator(Module):
    """
    Simple harmonic oscillator model for an EM mode:
    state [E, dE/dt]
    d2E/dt2 + gamma dE/dt + omega^2 E = driving
    We'll store state as [E, V] where V = dE/dt
    """

    def __init__(self, E0=0.0, V0=0.0, omega=2.0, gamma=0.05):
        super().__init__()
        self.state = np.array([E0, V0], dtype=float)
        self.state_labels = ["E", "V"]
        self.omega = omega
        self.gamma = gamma

    def derivative(self, t, state, inputs):
        E, V = state
        driving = inputs.get("em_drive", 0.0)
        dE_dt = V
        dV_dt = -self.gamma * V - (self.omega**2) * E + driving
        return np.array([dE_dt, dV_dt], dtype=float)
