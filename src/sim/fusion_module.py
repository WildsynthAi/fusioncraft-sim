import numpy as np
from .physics_base import Module

class Fusion0D(Module):
    """
    Toy 0D fusion model:
    state vector: [n, Ti, Te] (density [m^-3], ion temp [keV], electron temp [keV])
    Units are simplified / normalized for educational demo.
    """

    def __init__(self, n0=1e19, Ti0=2.0, Te0=2.0):
        super().__init__()
        # state: density (n), Ti, Te
        self.state = np.array([n0, Ti0, Te0], dtype=float)
        self.state_labels = ["n", "Ti", "Te"]
        # parameters (toy)
        self.loss_coeff = 1e-2    # radiation+transport coefficient
        self.heating_power = 1.0  # external heating (arbitrary units)
        self.sigma_v_prefactor = 1e-38  # toy cross-section scale

    def fusion_power(self, n, Ti):
        """
        Toy fusion power ~ n^2 * <sigma v> * E_fusion
        We approximate <sigma v> as prefactor * sqrt(Ti) (toy)
        """
        sigma_v = self.sigma_v_prefactor * np.sqrt(max(Ti, 1e-6))
        E_f = 17.6  # MeV per D-T reaction (toy constant)
        # convert to arbitrary units: keep E_f as scalar factor
        return n**2 * sigma_v * E_f

    def derivative(self, t, state, inputs):
        n, Ti, Te = state
        # simple heating: fraction goes to ions and electrons
        P_fusion = self.fusion_power(n, Ti)
        P_heating = self.heating_power + P_fusion  # input heating adds
        # losses proportional to temperature and loss_coeff
        loss_i = self.loss_coeff * Ti
        loss_e = self.loss_coeff * Te
        # densities change slowly in this toy model (fuel feed / pumping)
        dn_dt = inputs.get("fuel_feed", 0.0) - 0.0 * n
        # energy balance: dTi/dt = (alpha_i*P_heating - loss_i) / (n*C_i)
        # use normalized heat capacities C_i ~ 1 for simplicity
        alpha_i = 0.6
        alpha_e = 0.4
        dTi_dt = (alpha_i * P_heating - loss_i) / (1.0 + 0.1*n/1e20)
        dTe_dt = (alpha_e * P_heating - loss_e) / (1.0 + 0.1*n/1e20)
        return np.array([dn_dt, dTi_dt, dTe_dt], dtype=float)
