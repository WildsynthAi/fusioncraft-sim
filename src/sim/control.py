# -------------------------------------------------------------------
# PID compatibility shim (appended to ensure runner finds `PID`)
# If the repo already provides a PID class, this will not overwrite it.
# This shim will only define PID if it isn't already present.
# -------------------------------------------------------------------

if "PID" not in globals():
    class PID:
        """
        Minimal, safe PID class expected by the runner.
        - constructor: PID(kp=1.0, ki=0.0, kd=0.0)
        - method: step(setpoint, measured, dt) -> float
        Stores internal integral and previous-error state on the instance.
        """
        def __init__(self, kp=1.0, ki=0.0, kd=0.0):
            self.kp = float(kp)
            self.ki = float(ki)
            self.kd = float(kd)
            self._integral = 0.0
            self._prev_error = None

        def step(self, setpoint, measured, dt):
            # basic defensive checks
            try:
                dt = float(dt)
            except Exception:
                dt = 1.0
            err = float(setpoint - measured)
            if self._prev_error is None:
                dedt = 0.0
            else:
                dedt = (err - self._prev_error) / dt if dt > 0 else 0.0
            self._integral += err * dt
            self._prev_error = err
            out = self.kp * err + self.ki * self._integral + self.kd * dedt
            return float(out)

    # optional user-friendly message in dev contexts (no harm if suppressed)
    try:
        print("PID shim: default PID class defined (compatibility shim).")
    except Exception:
        pass
