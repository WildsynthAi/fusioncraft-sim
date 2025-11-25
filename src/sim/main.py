"""
Deterministic multi-physics demo driver.

Usage:
  python src/sim/main.py

Produces: output.csv with time-series of selected states.
"""
import numpy as np
from integrator import integrate
from fusion_module import Fusion0D
from em_module import EMOscillator
from control import PID
import csv
import os

def build_coupled_state(fusion: Fusion0D, em: EMOscillator):
    """
    Combine states into a single vector and provide helper to unpack.
    Order: [fusion.n, fusion.Ti, fusion.Te, em.E, em.V]
    """
    def pack():
        return np.concatenate([fusion.get_state(), em.get_state()])
    def unpack(vec):
        fusion.set_state(vec[0:3])
        em.set_state(vec[3:5])
    return pack, unpack

def coupled_derivative(t, y, modules, controllers):
    # y is combined state vector
    # unpack into module states
    # modules: dict with 'fusion', 'em'
    fusion = modules['fusion']
    em = modules['em']
    # set local states (integrator keeps y but modules use state for internal convenience)
    fusion.set_state(y[0:3])
    em.set_state(y[3:5])

    # controller measures some variable (example: maintain Ti setpoint)
    pid = controllers['pid']
    Ti = fusion.get_state()[1]
    control_output = pid.step(t, Ti, dt=controllers['dt'])

    # prepare inputs for modules
    inputs_fusion = {
        "fuel_feed": 0.0,
        # use control_output to modulate heating power
        "heating_override": control_output
    }
    # The fusion module's heating_power will be supplemented by heating_override in a simple way
    # Build derivative from fusion: we intercept and add override.
    deriv_f = fusion.derivative(t, fusion.get_state(), inputs_fusion)
    # add override influence: we model as additive heating power that increases dTi/dt and dTe/dt
    deriv_f[1] += 0.1 * control_output
    deriv_f[2] += 0.05 * control_output

    # EM module driven by fusion energy output (toy coupling)
    # driving ~ fusion power normalized
    n, Ti, _ = fusion.get_state()
    P_fusion = fusion.fusion_power(n, Ti)
    inputs_em = {"em_drive": 1e-6 * P_fusion}  # small drive
    deriv_em = em.derivative(t, em.get_state(), inputs_em)

    # construct combined derivative vector
    deriv = np.concatenate([deriv_f, deriv_em])
    return deriv

def main():
    # simulation parameters
    t0 = 0.0
    t_final = 5.0  # seconds (toy)
    dt = 0.01

    # create modules
    fusion = Fusion0D(n0=1e19, Ti0=2.0, Te0=2.0)
    em = EMOscillator(E0=0.0, V0=0.0, omega=5.0, gamma=0.1)
    pid = PID(kp=0.5, ki=0.1, kd=0.02, setpoint=3.0)  # keep Ti ~ 3.0

    pack, unpack = build_coupled_state(fusion, em)
    y0 = pack()

    # wrapper function for integrator that matches (t, y) -> dy
    modules = {'fusion': fusion, 'em': em}
    controllers = {'pid': pid, 'dt': dt}

    def f_wrapper(t, y):
        return coupled_derivative(t, y, modules, controllers)

    out_file = os.path.join(os.path.dirname(__file__), "output.csv")
    with open(out_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        # header
        header = ["t"] + fusion.state_labels + em.state_labels + ["P_fusion", "control_out"]
        writer.writerow(header)

        # integrate deterministically
        for t, y in integrate(f_wrapper, t0, y0, t_final, dt):
            # update module states from y so we can compute derived quantities
            fusion.set_state(y[0:3])
            em.set_state(y[3:5])
            n, Ti, Te = fusion.get_state()
            P_f = fusion.fusion_power(n, Ti)
            control_out = pid._last_error if pid._last_error is not None else 0.0
            row = [float(t), float(n), float(Ti), float(Te), float(em.get_state()[0]), float(em.get_state()[1]), float(P_f), float(control_out)]
            writer.writerow(row)

    print(f"Simulation complete. CSV written to: {out_file}")

if __name__ == "__main__":
    main()
