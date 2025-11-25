# -------------------------------------------------------------------
# PID compatibility shim (ensures runner can import PID class)
# -------------------------------------------------------------------

if "PID" not in globals():
    class PID:
        """
        Minimal safe PID class:
        PID(kp, ki, kd).step(setpoint, measured, dt) -> float
        """

        def __init__(self, kp=1.0, ki=0.0, kd=0.0):
            self.kp = float(kp)
            self.ki = float(ki)
            self.kd = float(kd)
            self._integral = 0.0
            self._prev_error = None

        def step(self, setpoint, measured, dt):
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

            output = (
                self.kp * err +
                self.ki * self._integral +
                self.kd * dedt
            )

            return float(output)
