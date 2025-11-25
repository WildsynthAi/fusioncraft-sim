"""
FusionCraft â€“ Deterministic Multi-Physics Simulation Driver
Runs:
- Fusion 0D model (toy energy balance)
- EM oscillator model
- PID controller
Uses RK4 integrator (if modules provide derivative interfaces) or module.step() methods and returns simulation data.
"""

import numpy as np
from typing import Dict, Any

from .fusion_module import Fusion0D
from .em_module import EMFieldOscillator
from .control import PID
# integrator.rk4 is available in the repo; not strictly required by this driver,
# but left available for future use.
from .integrator import rk4  # noqa: F401


def run_simulation(total_time: float = 1.0, dt: float = 0.001, progress: bool = False) -> Dict[str, Any]:
    """
    Run the deterministic multi-physics demo simulation.

    Args:
        total_time: total simulated time (seconds)
        dt: timestep (seconds)
        progress: if True print simple progress updates

    Returns:
        dict of numpy arrays: time, temperature, density, fusion_power, E_field, control_signal
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if total_time <= 0:
        raise ValueError("total_time must be > 0")

    # build explicit time grid (includes t=0 and t=total_time)
    time_grid = np.arange(0.0, total_time + dt * 0.5, dt)
    steps = len(time_grid)

    # Initialize modules
    fusion = Fusion0D()            # must implement .step(dt) -> (T, n, pf) or similar
    em = EMFieldOscillator()       # must implement .step(dt) -> E
    pid = PID(kp=0.8, ki=0.2, kd=0.05)

    # Time series logs (pre-allocate lists)
    t_series = []
    T_series = []
    n_series = []
    pf_series = []
    E_series = []
    control_series = []

    # Simulation loop
    for i, t in enumerate(time_grid):
        # --- Fusion plasma integration ---
        # try to call fusion.step(dt) and accept either tuple or single value
        fusion_out = fusion.step(dt)
        if isinstance(fusion_out, tuple) and len(fusion_out) >= 3:
            T, n, pf = fusion_out[0], fusion_out[1], fusion_out[2]
        else:
            # some toy modules may return (T,) or single value; be defensive
            try:
                T = float(fusion_out)
            except Exception:
                raise RuntimeError("fusion.step(dt) returned unexpected result; expected (T,n,pf) or numeric T")
            n = 0.0
            pf = 0.0

        # --- EM oscillator ---
        em_out = em.step(dt)
        try:
            E = float(em_out)
        except Exception:
            raise RuntimeError("em.step(dt) returned unexpected result; expected numeric E value")

        # --- Control system ---
        control = pid.step(setpoint=5.0, measured=T, dt=dt)

        # --- Log data ---
        t_series.append(t)
        T_series.append(T)
        n_series.append(n)
        pf_series.append(pf)
        E_series.append(E)
        control_series.append(control)

        # Simple progress printout (every 10%) if requested
        if progress and (i % max(1, steps // 10) == 0):
            pct = int((i / max(1, steps - 1)) * 100)
            print(f"[sim] {pct}%  t={t:.3f}s")

    # convert to numpy arrays and return
    return {
        "time": np.array(t_series),
        "temperature": np.array(T_series),
        "density": np.array(n_series),
        "fusion_power": np.array(pf_series),
        "E_field": np.array(E_series),
        "control_signal": np.array(control_series)
    }


if __name__ == "__main__":
    # quick smoke run when executed directly
    results = run_simulation(total_time=0.5, dt=0.001, progress=True)
    print("Simulation complete.")
    print("Final Temperature:", float(results["temperature"][-1]))
    print("Final Fusion Power:", float(results["fusion_power"][-1]))
