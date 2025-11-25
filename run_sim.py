#!/usr/bin/env python3
"""
run_sim.py
Light runner for FusionCraft simulation project.

Usage examples:
  python run_sim.py
  python run_sim.py --time 10 --dt 0.01 --out results.csv
"""

import argparse
import importlib
import os
import sys
from datetime import datetime

# default output
DEFAULT_OUT = "fusioncraft_run.csv"

def try_call_sim(mod):
    """
    Try common entrypoint names in the provided module.
    Expecting function signature like:
      result = fn(total_time, dt)
    Or fn() that returns a dict-like timeseries.
    """
    candidates = ["run_simulation", "run", "simulate", "main"]
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return name, fn
    return None, None

def save_timeseries_csv(ts, filename):
    """
    Accepts dict-like timeseries { 'time': [...], 'var1':[...], ... }
    Writes CSV where first column is time (if present) or index.
    """
    try:
        import csv
        # determine keys and length
        if hasattr(ts, "items"):
            keys = list(ts.keys())
            length = len(next(iter(ts.values())))
            with open(filename, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(keys)
                for i in range(length):
                    row = [ts[k][i] if i < len(ts[k]) else "" for k in keys]
                    writer.writerow(row)
        else:
            # fallback: try pandas
            import pandas as pd
            df = pd.DataFrame(ts)
            df.to_csv(filename, index=False)
        print(f"[OK] Saved timeseries to {filename}")
    except Exception as e:
        print("[ERR] Failed to save CSV:", e)

def simple_plot(ts, filename=None):
    try:
        import matplotlib.pyplot as plt
        if hasattr(ts, "items"):
            keys = list(ts.keys())
            time_key = None
            for k in ("time", "t", "timestamp"):
                if k in keys:
                    time_key = k
                    break
            if time_key:
                t = ts[time_key]
                others = [k for k in keys if k != time_key]
                for k in others:
                    plt.plot(t, ts[k], label=k)
                plt.xlabel(time_key)
            else:
                # plot all on index
                for k in keys:
                    plt.plot(ts[k], label=k)
            plt.legend()
            plt.tight_layout()
            out = filename or "fusioncraft_plot.png"
            plt.savefig(out)
            plt.close()
            print(f"[OK] Saved plot to {out}")
    except Exception as e:
        print("[WARN] Plot not created (matplotlib missing or plotting error):", e)

def main():
    parser = argparse.ArgumentParser(description="Run FusionCraft simulation (runner)")
    parser.add_argument("--time", "-T", type=float, default=5.0, help="Total sim time (s)")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step (s)")
    parser.add_argument("--out", "-o", default=DEFAULT_OUT, help="CSV output file")
    parser.add_argument("--module", "-m", default="src.sim.main", help="Module path to simulation main")
    args = parser.parse_args()

    # make sure repo root is on sys.path (so imports work)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    print(f"[RUNNER] Importing simulation module: {args.module}")
    try:
        sim_mod = importlib.import_module(args.module)
    except Exception as e:
        print(f"[ERR] Could not import module '{args.module}': {e}")
        print("Make sure the module path is correct and __init__.py exists in package directories.")
        sys.exit(1)

    name, fn = try_call_sim(sim_mod)
    if fn is None:
        print("[ERR] No suitable entrypoint found in module.")
        print("Looked for functions:", ["run_simulation","run","simulate","main"])
        print("Open src/sim/main.py and add a function `def run_simulation(total_time, dt):` that returns a dict of timeseries.")
        sys.exit(1)

    print(f"[RUNNER] Found function '{name}'. Calling it with total_time={args.time}, dt={args.dt} ...")
    try:
        # try call with (time, dt)
        res = fn(args.time, args.dt)
    except TypeError:
        try:
            res = fn()
        except Exception as e:
            print("[ERR] Calling function failed:", e)
            sys.exit(1)
    except Exception as e:
        print("[ERR] Simulation error:", e)
        sys.exit(1)

    # If function returned nothing, simulate may have written files already
    if res is None:
        print("[OK] Simulation executed (no return value). Check code for output files.")
        sys.exit(0)

    # If returned a path string
    if isinstance(res, str) and os.path.exists(res):
        print(f"[OK] Simulation produced file: {res}")
        sys.exit(0)

    # Assume returned timeseries-like dict
    if hasattr(res, "items"):
        save_timeseries_csv(res, args.out)
        # try plotting
        simple_plot(res, filename=os.path.splitext(args.out)[0] + ".png")
        print("[DONE] Runner finished successfully.")
    else:
        print("[WARN] Simulation returned unexpected type:", type(res))
        print("Returned value:", res)

if __name__ == "__main__":
    main()
