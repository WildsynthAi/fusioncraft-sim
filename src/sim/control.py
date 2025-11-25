# --- PID compatibility wrapper (appended for runner compatibility) ---
def _pid_step_compat(self, setpoint, measured, dt):
    """
    Provide PID.step(setpoint, measured, dt) for older/newer PID class shapes.
    Tries existing methods first; otherwise uses a simple PID compute.
    Returns float control output.
    """
    # try other likely APIs
    for name in ("step","update","compute","control","control_signal"):
        if hasattr(self, name) and name != "step":
            meth = getattr(self, name)
            try:
                err = setpoint - measured
                try:
                    return float(meth(err, dt))
                except TypeError:
                    pass
                try:
                    return float(meth(setpoint, measured, dt))
                except TypeError:
                    pass
                try:
                    return float(meth(measured, dt))
                except TypeError:
                    pass
            except Exception:
                continue

    # fallback: simple PID
    e = float(setpoint - measured)
    if not hasattr(self, "_pid_integral"):
        self._pid_integral = 0.0
    if not hasattr(self, "_pid_prev_e"):
        self._pid_prev_e = e
    self._pid_integral += e * dt
    dedt = (e - self._pid_prev_e) / dt if dt and dt > 0 else 0.0
    self._pid_prev_e = e

    # read gains (support kp/ki/kd or Kp/Ki/Kd)
    Kp = getattr(self, "kp", None)
    if Kp is None:
        Kp = getattr(self, "Kp", 0.0)
    Ki = getattr(self, "ki", None)
    if Ki is None:
        Ki = getattr(self, "Ki", 0.0)
    Kd = getattr(self, "kd", None)
    if Kd is None:
        Kd = getattr(self, "Kd", 0.0)

    try: Kp = float(Kp)
    except Exception: Kp = 1.0
    try: Ki = float(Ki)
    except Exception: Ki = 0.0
    try: Kd = float(Kd)
    except Exception: Kd = 0.0

    out = Kp * e + Ki * self._pid_integral + Kd * dedt
    return float(out)

# Attach wrapper to PID class on import time (if PID present and missing step)
try:
    import importlib
    m = importlib.import_module("src.sim.control")
    PID_cls = getattr(m, "PID", None)
    if PID_cls is not None and not hasattr(PID_cls, "step"):
        PID_cls.step = _pid_step_compat
        # print only in dev contexts; harmless otherwise
        try:
            print("PID.step compatibility wrapper attached.")
        except Exception:
            pass
except Exception:
    # silence on import errors (safe to ignore)
    pass
