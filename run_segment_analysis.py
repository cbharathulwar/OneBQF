#!/usr/bin/env python3
"""
Reproduce Figure 3 (segment efficiency analysis) from the OneBQF paper.
Generates angle data from scratch and saves plots to notebook_figures/.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from toy_model.state_event_model import *
from toy_model.state_event_generator import StateEventGenerator
from toy_model.velo_workflow import make_detector

# ── Config ─────────────────────────────────────────────────────────────
FIXED_RESOLUTION  = 0.005    # 5 µm
FIXED_SCATTERING  = 0.0001   # 0.1 mrad
FIXED_EPSILON     = 0.002    # 2 mrad (fixed threshold)
TRACKS_PER_EVENT  = 20
EVENT_COUNTS      = list(range(1, 11))   # 1–10 events → 20–200 tracks

# Output paths
BASE          = Path("data/segment_analysis")
RUNS_DIR      = BASE / "runs_fixed_epsilon"
PLOTS_DIR     = BASE / "plots"
ANGLE_CACHE   = RUNS_DIR / "fixed_epsilon_angle_data_v3.pkl"
FIG_PNG       = Path("notebook_figures/segment_efficiency.png")

BASE.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("FIXED EPSILON SEGMENT EFFICIENCY ANALYSIS")
print("=" * 60)
print(f"Resolution:   {FIXED_RESOLUTION*1000:.0f} µm")
print(f"Scattering:   {FIXED_SCATTERING*1000:.2f} mrad")
print(f"Epsilon:      {FIXED_EPSILON*1000:.1f} mrad")
print(f"Track counts: {[e*TRACKS_PER_EVENT for e in EVENT_COUNTS]}")
print("=" * 60)


# ── Step 1: Generate or load angle data ────────────────────────────────
def compute_segment_angles(event):
    """Return (true_angles, false_angles) for all adjacent-layer segment pairs."""
    true_angles, false_angles = [], []
    all_segs, seg_to_track = [], {}
    for track in event.tracks:
        for seg in track.segments:
            all_segs.append(seg)
            seg_to_track[id(seg)] = track.track_id

    for i in range(len(all_segs)):
        for j in range(i + 1, len(all_segs)):
            vi = np.array(all_segs[i].to_vect())
            vj = np.array(all_segs[j].to_vect())
            ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
            if ni == 0 or nj == 0:
                continue
            cos_a = np.clip(np.dot(vi / ni, vj / nj), -1.0, 1.0)
            angle = np.arccos(cos_a)
            if seg_to_track.get(id(all_segs[i])) == seg_to_track.get(id(all_segs[j])):
                true_angles.append(angle)
            else:
                false_angles.append(angle)
    return true_angles, false_angles


if ANGLE_CACHE.exists():
    print(f"\nLoading cached angle data from {ANGLE_CACHE} ...")
    with open(ANGLE_CACHE, "rb") as f:
        saved = pickle.load(f)
    fixed_eps_angle_results = saved["fixed_eps_angle_results"]
    print(f"Loaded {len(fixed_eps_angle_results)} configurations.")
else:
    print("\nGenerating angle data from scratch ...")
    Detector = make_detector()
    fixed_eps_angle_results = []
    t0 = time.time()

    for n_events in EVENT_COUNTS:
        total_tracks = n_events * TRACKS_PER_EVENT
        n_repeats = 50 if total_tracks <= 100 else 10
        print(f"\n  [{total_tracks} tracks] ({n_repeats} repeats) ...", end="", flush=True)

        per_repeat_data = []
        all_true, all_false = [], []
        n_particles_per_event = [TRACKS_PER_EVENT] * n_events

        for repeat in range(n_repeats):
            np.random.seed(repeat + n_events * 1000)

            seg = StateEventGenerator(
                Detector,
                phi_min=-0.2, phi_max=0.2,
                events=n_events,
                n_particles=n_particles_per_event,
                measurement_error=float(FIXED_RESOLUTION),
                collision_noise=float(FIXED_SCATTERING),
            )
            seg.generate_random_primary_vertices({"x": 1, "y": 1, "z": 1})
            seg.generate_particles([
                [{"type": "MIP", "mass": 0.511, "q": 1} for _ in range(total_tracks)]
                for _ in range(n_events)
            ])
            event = seg.generate_complete_events()

            ta, fa = compute_segment_angles(event)
            per_repeat_data.append({
                "n_true":       len(ta),
                "n_false":      len(fa),
                "true_accepted":  int(np.sum(np.array(ta) <= FIXED_EPSILON)),
                "false_accepted": int(np.sum(np.array(fa) <= FIXED_EPSILON)),
            })
            all_true.extend(ta)
            all_false.extend(fa)

        fixed_eps_angle_results.append({
            "n_events":       n_events,
            "total_tracks":   total_tracks,
            "true_angles":    np.array(all_true),
            "false_angles":   np.array(all_false),
            "n_true":         len(all_true),
            "n_false":        len(all_false),
            "per_repeat_data": per_repeat_data,
        })
        print(f" done ({len(all_true)} true, {len(all_false)} false pairs)")

    elapsed = time.time() - t0
    print(f"\nGenerated in {elapsed:.1f}s")

    with open(ANGLE_CACHE, "wb") as f:
        pickle.dump({
            "fixed_eps_angle_results": fixed_eps_angle_results,
            "FIXED_EPSILON":   FIXED_EPSILON,
            "FIXED_RESOLUTION": FIXED_RESOLUTION,
            "FIXED_SCATTERING": FIXED_SCATTERING,
            "TRACKS_PER_EVENT": TRACKS_PER_EVENT,
        }, f)
    print(f"Saved to {ANGLE_CACHE}")


# ── Step 2: Compute efficiency stats ───────────────────────────────────
print("\nComputing segment efficiency ...")
results = []
for r in fixed_eps_angle_results:
    ta, fa = r["true_angles"], r["false_angles"]
    true_acc  = int(np.sum(ta <= FIXED_EPSILON))
    false_acc = int(np.sum(fa <= FIXED_EPSILON))
    n_true, n_false = len(ta), len(fa)
    total_acc = true_acc + false_acc

    eff  = true_acc  / n_true  if n_true  > 0 else 0.0
    frate = false_acc / total_acc if total_acc > 0 else 0.0

    prd = r.get("per_repeat_data", [])
    def _stats(key):
        vals = np.array([p[key] for p in prd])
        n = len(vals)
        return vals.mean(), (vals.std() / np.sqrt(n) if n > 1 else 0)

    nt_m, nt_se   = _stats("n_true")
    nf_m, nf_se   = _stats("n_false")
    ta_m, ta_se   = _stats("true_accepted")
    fa_m, fa_se   = _stats("false_accepted")

    results.append({
        "total_tracks": r["total_tracks"],
        "seg_efficiency": eff * 100,
        "seg_false_rate": frate * 100,
        "n_true": n_true, "n_false": n_false,
        "true_accepted": true_acc, "false_accepted": false_acc,
        "total_accepted": total_acc,
        "n_true_mean": nt_m, "n_true_se": nt_se,
        "n_false_mean": nf_m, "n_false_se": nf_se,
        "true_acc_mean": ta_m, "true_acc_se": ta_se,
        "false_acc_mean": fa_m, "false_acc_se": fa_se,
        "n_repeats": len(prd),
    })
    print(f"  {r['total_tracks']:3d} tracks  eff={eff*100:.1f}%  false_rate={frate*100:.2f}%")


# ── Step 3: Plot ────────────────────────────────────────────────────────
print(f"\nPlotting → {FIG_PNG} ...")

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11, "axes.linewidth": 1.2,
    "lines.linewidth": 2, "lines.markersize": 8,
})

track_counts   = np.array([r["total_tracks"]    for r in results])
seg_eff        = np.array([r["seg_efficiency"]   for r in results])
false_rate     = np.array([r["seg_false_rate"]   for r in results])
n_true_mean    = np.array([r["n_true_mean"]      for r in results])
n_true_se      = np.array([r["n_true_se"]        for r in results])
n_false_mean   = np.array([r["n_false_mean"]     for r in results])
n_false_se     = np.array([r["n_false_se"]       for r in results])
true_acc_mean  = np.array([r["true_acc_mean"]    for r in results])
true_acc_se    = np.array([r["true_acc_se"]      for r in results])
false_acc_mean = np.array([r["false_acc_mean"]   for r in results])
false_acc_se   = np.array([r["false_acc_se"]     for r in results])
n_true_total   = np.array([r["n_true"]           for r in results])
total_acc      = np.array([r["total_accepted"]   for r in results])
false_acc_arr  = np.array([r["false_accepted"]   for r in results])

seg_eff_err   = np.sqrt(np.maximum(seg_eff * (100 - seg_eff) / n_true_total, 0))
false_rate_err = np.where(
    total_acc > 0,
    np.sqrt(false_rate * (100 - false_rate) / np.maximum(total_acc, 1)),
    0
)

c_true  = "#2166ac"
c_false = "#d6604d"
c_eff   = "#1b7837"
c_frate = "#c51b7d"

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

kw = dict(capsize=5, capthick=1.5, linewidth=2, markersize=8,
          markeredgecolor="black", markeredgewidth=0.8)

# (a) Segment efficiency
ax = axes[0, 0]
ax.errorbar(track_counts, seg_eff, yerr=seg_eff_err, fmt="o-", color=c_eff, **kw)
ax.axhline(100, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
ax.set(xlabel="Total Tracks", ylabel="Segment Efficiency (%)",
       title="(a) Segment Efficiency", ylim=(95, 101))
ax.grid(True, alpha=0.3); ax.minorticks_on()
ax.tick_params(which="both", direction="in", top=True, right=True)

# (b) False rate
ax = axes[0, 1]
ax.errorbar(track_counts, false_rate, yerr=false_rate_err, fmt="s-", color=c_frate, **kw)
ax.set(xlabel="Total Tracks", ylabel="Segment False Rate (%)",
       title="(b) Segment False Rate", ylim=(0, 10))
ax.grid(True, alpha=0.3); ax.minorticks_on()
ax.tick_params(which="both", direction="in", top=True, right=True)

# (c) Pair counts
ax = axes[1, 0]
ax.errorbar(track_counts, n_true_mean,  yerr=n_true_se,
            fmt="o-", color=c_true,  label="True pairs",  **kw)
ax.errorbar(track_counts, n_false_mean, yerr=n_false_se,
            fmt="s-", color=c_false, label="False pairs", **kw)
ax.set_yscale("log")
ax.set(xlabel="Total Tracks", ylabel="Number of Segment Pairs",
       title="(c) Segment Pair Counts")
ax.legend(loc="upper left", framealpha=0.95)
ax.grid(True, alpha=0.3, which="both"); ax.minorticks_on()
ax.tick_params(which="both", direction="in", top=True, right=True)

# (d) Accepted pairs
ax = axes[1, 1]
fa_plot   = np.where(false_acc_mean > 0, false_acc_mean, 0.5)
fa_se_plot = np.where(false_acc_mean > 0, false_acc_se,  0)
ax.errorbar(track_counts, true_acc_mean, yerr=true_acc_se,
            fmt="o-", color=c_true,  label="True accepted",  **kw)
ax.errorbar(track_counts, fa_plot,       yerr=fa_se_plot,
            fmt="s-", color=c_false, label="False accepted", **kw)
ax.set_yscale("log")
ax.set(xlabel="Total Tracks", ylabel="Number of Pairs Accepted",
       title="(d) Accepted Segment Pairs", ylim=(0.1, 1e4))
ax.legend(loc="lower right", framealpha=0.95)
ax.grid(True, alpha=0.3, which="both"); ax.minorticks_on()
ax.tick_params(which="both", direction="in", top=True, right=True)

plt.tight_layout()
fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(PLOTS_DIR / "fixed_epsilon_segment_efficiency.png",
            dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_PNG}")
print("Done.")
