import numpy as np
from typing import Callable, Tuple

def rk4_step(f: Callable[[float, np.ndarray], np.ndarray],
             t: float,
             y: np.ndarray,
             h: float) -> Tuple[float, np.ndarray]:
    """
    Single RK4 step.
    f: dy/dt = f(t, y)
    y: state vector (numpy)
    returns (t+h, y_next)
    """
    k1 = f(t, y)
    k2 = f(t + h/2.0, y + 0.5*h*k1)
    k3 = f(t + h/2.0, y + 0.5*h*k2)
    k4 = f(t + h, y + h*k3)
    y_next = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return t + h, y_next

def integrate(f: Callable[[float, np.ndarray], np.ndarray],
              t0: float,
              y0: np.ndarray,
              t_final: float,
              dt: float):
    """
    Integrate from t0 to t_final with fixed step dt.
    Yields (t, y) tuples.
    """
    t = t0
    y = y0.copy()
    yield t, y.copy()
    steps = int(np.ceil((t_final - t0)/dt))
    for _ in range(steps):
        t, y = rk4_step(f, t, y, dt)
        yield t, y.copy()
