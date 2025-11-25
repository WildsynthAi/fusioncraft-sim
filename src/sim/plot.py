# plot.py
"""
Plot utilities for FusionCraft multi-physics simulation.
Generates simple time-series graphs for diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_fusion(results):
    t = results["time"]
    T = results["temperature"]
    n = results["density"]
    fusion_power = results["fusion_power"]

    plt.figure(figsize=(10, 5))
    plt.plot(t, T, label="Ion Temperature (keV)")
    plt.plot(t, fusion_power, label="Fusion Power (a.u.)")
    plt.title("Fusion Plasma Evolution")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_em(results):
    t = results["time"]
    E = results["E_field"]

    plt.figure(figsize=(10, 5))
    plt.plot(t, E, label="Electric Field", color="purple")
    plt.title("EM Field Oscillation")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_control(results):
    t = results["time"]
    control = results["control_signal"]

    plt.figure(figsize=(10, 5))
    plt.plot(t, control, label="PID Output", color="red")
    plt.title("Control Signal")
    plt.xlabel("Time")
    plt.grid(True)
    plt.legend()
    plt.show()
