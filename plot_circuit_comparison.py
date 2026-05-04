#!/usr/bin/env python3
"""
Plot reproduced circuit depth results vs paper values (partial or complete).
Saves to notebook_figures/circuit_depth_comparison.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

with open("data/circuit_depth.json") as f:
    paper_data = {(e["problem_size"]["n_particles"], e["problem_size"]["layers"]): e
                  for e in json.load(f)}

with open("data/circuit_depth_reproduced.json") as f:
    ours = json.load(f)

# Separate by layer count
def extract(data_list, layers, source="ours"):
    xs, ys = [], []
    for e in data_list:
        ps = e["problem_size"]
        if ps["layers"] != layers:
            continue
        if source == "ours":
            val = e["standard"]["hardware_torino"].get("two_qubit_gates")
        else:
            val = e["standard"]["hardware_torino"].get("two_qubit_gates")
        if val is not None:
            xs.append(ps["n_particles"])
            ys.append(val)
    return np.array(xs), np.array(ys)

def extract_paper(layers):
    xs, ys = [], []
    for (n, l), e in paper_data.items():
        if l != layers:
            continue
        val = e["standard"]["hardware_torino"].get("two_qubit_gates")
        if val:
            xs.append(n)
            ys.append(val)
    idx = np.argsort(xs)
    return np.array(xs)[idx], np.array(ys)[idx]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Circuit Complexity: Reproduced vs Paper (IBM Torino, 2-qubit gates)",
             fontsize=13, fontweight="bold")

for ax, layers, title in zip(axes, [3, 5], ["3 Layers", "5 Layers"]):
    # Paper (all sizes)
    px, py = extract_paper(layers)
    ax.loglog(px, py, "o--", color="gray", alpha=0.6, linewidth=1.5,
              markersize=7, label="Paper (pre-computed)")

    # Ours (partial)
    ox, oy = extract(ours, layers)
    ax.loglog(ox, oy, "s-", color="#d62728", linewidth=2,
              markersize=8, markeredgecolor="black", markeredgewidth=0.8,
              label="Reproduced (this run)")

    # Power-law fit to our data if enough points
    if len(ox) >= 3:
        log_x = np.log(ox)
        log_y = np.log(oy)
        b, log_a = np.polyfit(log_x, log_y, 1)
        x_fit = np.logspace(np.log10(ox.min()), np.log10(px.max()), 100)
        y_fit = np.exp(log_a) * x_fit ** b
        ax.loglog(x_fit, y_fit, ":", color="#d62728", alpha=0.5,
                  label=f"Fit: $N^{{{b:.2f}}}$")

    ax.set_xlabel("Number of particles (N)", fontsize=12)
    ax.set_ylabel("2-qubit gate count", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(which="both", direction="in", top=True, right=True)

plt.tight_layout()
out = Path("notebook_figures/circuit_depth_comparison.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
plt.show()
