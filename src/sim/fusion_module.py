import numpy as np
from .physics_base import Module

class Fusion0D(Module):
    """
    0D fusion plasma model with more realistic physics:
    - Bremsstrahlung losses
    - Confinement time losses
    - EM heating
    - Fuel consumption
    State vector: [n, Ti, Te] (density [m^-3], ion temp [keV], electron temp [keV])
    """

    def __init__(self, n0=1e19, Ti0=2.0, Te0=2.0, B_field=5.0, fuel_inject_rate=1e18, em_coupling=0.1):
        super().__init__()
        self.state = np.array([n0, Ti0, Te0], dtype=float)
        self.state_labels = ["n", "Ti", "Te"]
        self.E_fusion = 17.6  # MeV per D-T reaction
        self.sigma_v_prefactor = 1e-38 # Toy cross-section scale

        # New physics parameters
        self.brems_coeff = 1.0e-32  # Toy Bremsstrahlung coefficient (units adjusted for simulation scale)
        self.confinement_factor = 0.5 # A factor for energy confinement time (e.g., related to plasma volume/surface)
        self.B_field = B_field # Magnetic field strength (Tesla)
        self.fuel_inject_rate = fuel_inject_rate # particles/m^3/s
        self.em_coupling = em_coupling # Efficiency of EM field coupling to plasma heating

        # Alpha heating fractions (simplified)
        self.alpha_heating_fraction_ion = 0.5
        self.alpha_heating_fraction_electron = 0.5

    def get_sigma_v(self, Ti):
        # A simplified approximation for <sigma v> (toy)
        # More realistic would be a D-T reaction rate curve
        return self.sigma_v_prefactor * np.sqrt(max(Ti, 1e-6)) # Avoid sqrt of negative

    def derivative(self, t, state, inputs):
        n, Ti, Te = state # n is ion density, assuming quasi-neutrality n_e = n_i = n

        # Extract EM field from inputs, default to 0 if not provided
        em_E_field = inputs.get("E_field", 0.0) if inputs else 0.0

        # --- Fusion Power (P_alpha) ---
        # Assuming D-T fusion, fusion power depends on n_i^2 and <sigma v>
        # E_fusion * 1.602e-13 converts MeV to Joules
        sigma_v = self.get_sigma_v(Ti)
        P_fusion_power_density = n**2 * sigma_v * (self.E_fusion * 1.602e-13) # J/m^3/s (power density)

        # --- Energy Losses ---
        # 1. Bremsstrahlung Losses (radiation): P_brems ~ n_e^2 * sqrt(Te)
        P_brems = self.brems_coeff * n**2 * np.sqrt(max(Te, 1e-6)) # keV / m^3 / s

        # 2. Confinement Losses: Energy loss due to finite energy confinement time (tau_E)
        # Simplified scaling for tau_E (energy confinement time)
        tau_E = self.confinement_factor * self.B_field # Toy scaling: Higher B, longer confinement
        tau_E = max(tau_E, 1e-6) # Ensure tau_E is not zero

        P_conf_loss_ion = n * Ti / tau_E # Loss rate for ions
        P_conf_loss_electron = n * Te / tau_E # Loss rate for electrons

        # --- Heating Terms ---
        # 1. Alpha particle heating from fusion
        P_alpha_ion = P_fusion_power_density * self.alpha_heating_fraction_ion
        P_alpha_electron = P_fusion_power_density * self.alpha_heating_fraction_electron

        # 2. EM Heating (from EM_E_field)
        # Assume P_em_heating ~ E_field^2 * n * em_coupling. Distribute based on temperature.
        P_em_heating_total = self.em_coupling * em_E_field**2 * n # units simplified for toy model
        P_em_heating_ion = P_em_heating_total * (Ti / (Ti + Te + 1e-6)) # Distribute based on temperature
        P_em_heating_electron = P_em_heating_total * (Te / (Ti + Te + 1e-6))

        # --- Fuel Consumption & Density Change ---
        # Fuel consumption (injection and losses)
        dn_dt_fusion_loss = -0.5 * n**2 * sigma_v # Particle loss from fusion reaction
        dn_dt_fuel_inject = self.fuel_inject_rate # External injection
        dn_dt_losses = -n / (100.0 * tau_E) # Some other particle losses, proportional to 1/tau_E
        dn_dt = dn_dt_fusion_loss + dn_dt_fuel_inject + dn_dt_losses

        # --- Temperature Changes ---
        # dTi/dt = (Heating_ion - Losses_ion) / (n * SpecificHeat_ion)
        # dTe/dt = (Heating_electron - Losses_electron) / (n * SpecificHeat_electron)
        # Specific heat capacity (simplified to unity, scaling factor n handles it implicitly)
        # (n + 1e6) in denominator to avoid division by zero for very low density

        dTi_dt = (P_alpha_ion + P_em_heating_ion - P_conf_loss_ion) / (n + 1e6)
        dTe_dt = (P_alpha_electron + P_em_heating_electron - P_brems - P_conf_loss_electron) / (n + 1e6)

        return np.array([dn_dt, dTi_dt, dTe_dt], dtype=float)

    def step(self, dt, inputs=None):
        """
        Advance state by dt using RK4.
        Returns (current Ti, current n, current P_fusion).
        """
        if inputs is None:
            inputs = {}

        # Import rk4_step from integrator.py dynamically
        try:
            from .integrator import rk4_step
        except ImportError:
            # Fallback to simple Euler if rk4_step not available
            n, Ti, Te = self.state
            dn_dt, dTi_dt, dTe_dt = self.derivative(0.0, self.state, inputs)
            self.state += np.array([dn_dt, dTi_dt, dTe_dt]) * dt
            # Re-calculate P_fusion for return
            current_n, current_Ti, current_Te = self.state
            sigma_v = self.get_sigma_v(current_Ti)
            P_fusion_out = (current_n**2) * sigma_v * self.E_fusion * 1e-6
            return float(current_Ti), float(current_n), float(P_fusion_out)

        # RK4 integration
        def _f_wrapped(t_val, y_vec):
            return self.derivative(t_val, y_vec, inputs)

        t_next, y_next = rk4_step(_f_wrapped, 0.0, self.state, dt)
        self.state = y_next

        # Recalculate fusion power for return value based on new state
        current_n, current_Ti, current_Te = self.state
        sigma_v = self.get_sigma_v(current_Ti)
        P_fusion_out = (current_n**2) * sigma_v * self.E_fusion * 1e-6

        # Return (Ti, n, P_fusion)
        return float(current_Ti), float(current_n), float(P_fusion_out)
