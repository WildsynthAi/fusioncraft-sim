"""
FusionCraft — 0D Deterministic Fusion Simulator (Toy Physics)

This is a minimal but real RK4-based simulation of:
- Plasma temperature (T)
- Plasma density (n)
- Fusion power generation
- Power losses
- Lawson triple product
- Confinement-time dependent stability
"""

import math


def rk4_step(f, t, y, dt):
    """Generic RK4 integrator."""
    k1 = dt * f(t, y)
    k2 = dt * f(t + dt/2, y + k1/2)
    k3 = dt * f(t + dt/2, y + k2/2)
    k4 = dt * f(t + dt, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6


# -----------------------------
# Fusion Physics Model
# -----------------------------

def sigma_v(T):
    """
    Toy fusion reactivity model.
    Real plasma physics is complicated.
    This is a simplified, smooth, monotonically increasing approximation.
    """
    return 1e-24 * T**2 * math.exp(-50/T)


def bremsstrahlung_loss(n, T):
    """Toy radiation loss: C * n * sqrt(T)."""
    return 5e-37 * n * math.sqrt(T)


def fusion_power(n, T):
    """Fusion power P = n^2 <σv> E."""
    E_fusion = 17.6e6 * 1.6e-19   # J (17.6 MeV)
    return (n**2) * sigma_v(T) * E_fusion


def confinement_time(T):
    """Toy confinement scaling: τ_E grows with temperature."""
    return 0.1 * (T**0.5)


def dT_dt(t, T):
    """Time evolution of temperature using power balance."""
    n = 5e19  # fixed density for baseline demo
    P_heat = 5e6                # 5 MW heater
    P_fus = fusion_power(n, T)
    P_loss = bremsstrahlung_loss(n, T) + T / confinement_time(T)
    return (P_heat + P_fus - P_loss) / 1e6   # simplified normalization


# -----------------------------
# Main Simulation
# -----------------------------

def run_sim():
    T = 1.0      # keV
    t = 0.0
    dt = 0.01
    history = []

    for step in range(2000):
        T = rk4_step(dT_dt, t, T, dt)
        t += dt

        n = 5e19
        tauE = confinement_time(T)
        triple_product = n * T * tauE

        history.append({
            "t": t,
            "T_keV": T,
            "fusion_power_MW": fusion_power(n, T) / 1e6,
            "triple_product": triple_product,
        })

    return history


if __name__ == "__main__":
    data = run_sim()

    print("Simulation complete.")
    print("Final T =", data[-1]["T_keV"], "keV")
    print("Final fusion power =", data[-1]["fusion_power_MW"], "MW")
