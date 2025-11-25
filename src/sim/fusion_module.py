import numpy as np

class Fusion0D:
    """
    Simple 0D fusion plasma model.
    Uses a cartoon energy balance:
        dT/dt = heating - losses
        P_fusion = n^2 * <σv> * E_fusion
    """

    def __init__(self):
        self.T = 1.0        # keV
        self.n = 1e19       # m^-3
        self.E_fusion = 17.6  # MeV (approx)
        self.sigma_v = 1e-22  # m^3/s (toy placeholder)

    def step(self, dt):
        """
        Perform one time-step update.
        Returns (T, n, P_fusion)
        """

        # toy heating & loss terms
        heating = 5.0
        losses  = 0.1 * self.T

        dTdt = heating - losses
        self.T += dTdt * dt

        # fusion power (toy)
        P_fusion = (self.n**2) * self.sigma_v * self.E_fusion * 1e-6

        return self.T, self.n, P_fusion
