"""
FusionCraft – Deterministic Multi-Physics Simulation Driver
Runs:
- Fusion 0D model (toy energy balance)
- EM oscillator model
- PID controller
Uses RK4 integrator and returns simulation data.
"""

import numpy as np

from .fusion_module import Fusion0D
from .em_module import EMFieldOscillator
from .control import PID
from .integrator import rk4


def run_simulation(total_time=1.0, dt=0.001):
    steps = int(total_time / dt)

    # Initialize modules
    fusion = Fusion0D()
    em = EMFieldOscillator()
    pid = PID(kp=0.8, ki=0.2, kd=0.05)

    # Time series logs
    t_series = []
    T_series = []
    n_series = []
    pf_series = []
    E_series = []
    control_series = []

    # Simulation loop
    t = 0.0
    for _ in range(steps):
        # --- Fusion plasma integration ---
        T, n, pf = fusion.step(dt)

        # --- EM oscillator ---
        E = em.step(dt)

        # --- Control system ---
        control = pid.step(setpoint=5.0, measured=T, dt=dt)

        # --- Log data ---
        t_series.append(t)
        T_series.append(T)
        n_series.append(n)
        pf_series.append(pf)
        E_series.append(E)
        control_series.append(control)

        t += dt

    return {
        "time": np.array(t_series),
        "temperature": np.array(T_series),
        "density": np.array(n_series),
        "fusion_power": np.array(pf_series),
        "E_field": np.array(E_series),
        "control_signal": np.array(control_series)
    }


if __name__ == "__main__":
    results = run_simulation()
    print("Simulation complete.")
    print("Final Temperature:", results["temperature"][-1])
    print("Final Fusion Power:", results["fusion_power"][-1])
