#!/usr/bin/env python3
"""
Reproduce fidelity under noise analysis (Figure 5) from the OneBQF paper.
Runs OneBQF circuits through IBM fake backend noise models and measures:
  - Hellinger fidelity vs noiseless baseline
  - Signal Separation Index (SSI): mean active / mean inactive probability

Configs: 2T 3L, 2T 5L, 4T 3L, 4T 5L, 8T 3L  (tracks x layers)
Backends: FakeTorino, FakeFez, FakeMarrakesh (Pittsburgh unavailable in this version)
Saves to: data/fidelity_results_reproduced.json
"""

import sys, json, time, math
sys.path.insert(0, '.')
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeFez, FakeMarrakesh

from toy_model.state_event_generator import StateEventGenerator
from toy_model.simple_hamiltonian import SimpleHamiltonian
from toy_model import state_event_model
from quantum_algorithms.OneBQF import OneBQF as onebqf

# ── Config ────────────────────────────────────────────────────────────
CONFIGS = [
    (2, 3), (2, 5),
    (4, 3), (4, 5),
    (8, 3),
]
BACKENDS = {
    "Qiskit-Torino":    FakeTorino(),
    "Qiskit-Fez":       FakeFez(),
    "Qiskit-Marrakesh": FakeMarrakesh(),   # substitute for Pittsburgh
}
N_SHOTS   = 50_000    # per run
N_RUNS    = 10        # runs per backend/config
DZ        = 20.0


def build_hamiltonian(n, layers, seed=42):
    ids = list(range(1, layers+1))
    lx = ly = [33.0]*layers
    zs = [DZ*l for l in range(1, layers+1)]
    det = state_event_model.PlaneGeometry(module_id=ids, lx=lx, ly=ly, z=zs)
    np.random.seed(seed)
    seg = StateEventGenerator(det, phi_min=-0.2, phi_max=0.2, events=1,
        n_particles=[n], measurement_error=0.0, collision_noise=0.0)
    seg.generate_random_primary_vertices({"x":1,"y":1,"z":1})
    seg.generate_particles([[{"type":"MIP","mass":0.511,"q":1} for _ in range(n)]])
    event = seg.generate_complete_events()
    ham = SimpleHamiltonian(epsilon=1e-7, alpha=2.0, beta=1.0)
    ham.construct_hamiltonian(event=event, convolution=False)
    # Return A, b, and the set of true segment indices
    true_segs = set()
    for track in event.tracks:
        for seg_obj in track.segments:
            true_segs.add(seg_obj.segment_id)
    return ham.A.toarray(), ham.b, true_segs


def counts_to_probs(counts, n_system_qubits, n_shots):
    """Convert measurement counts (post-selected on ancilla=1) to probability vector."""
    size = 2**n_system_qubits
    probs = np.zeros(size)
    total_success = 0
    for bitstring, count in counts.items():
        parts = bitstring.split()
        # bitstring format: "system ancilla" or all together
        # ancilla is last bit
        full = ''.join(parts)
        ancilla = int(full[-1])
        if ancilla == 1:
            sys_bits = full[:n_system_qubits]
            idx = int(sys_bits, 2)
            if idx < size:
                probs[idx] += count
                total_success += count
    if total_success > 0:
        probs /= total_success
    return probs, total_success


def hellinger_fidelity(p, q):
    """Hellinger fidelity between two probability distributions."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Normalise to be safe
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    return float(np.sum(np.sqrt(p * q))**2)


def compute_ssi(probs, true_indices, n_system_qubits):
    """Signal Separation Index: mean(p_active) / mean(p_inactive)."""
    size = 2**n_system_qubits
    active   = [probs[i] for i in range(size) if i in true_indices]
    inactive = [probs[i] for i in range(size) if i not in true_indices]
    mean_act  = np.mean(active)  if active  else 0
    mean_inact = np.mean(inactive) if inactive else 1e-10
    return float(mean_act / mean_inact) if mean_inact > 0 else 0.0


results = {}
t_total = time.time()

for n, layers in CONFIGS:
    key = f"{layers}L_{n}T"
    msize = (n**2)*(layers-1)
    pad = 2**math.ceil(math.log2(msize)) if msize > 1 else 2
    n_sys = int(math.log2(pad))
    n_qubits = 1 + n_sys + 1

    print(f"\n{'='*60}")
    print(f"Config: {n} tracks, {layers} layers | matrix={msize} | qubits={n_qubits}")
    print(f"{'='*60}")

    A, b, true_segs = build_hamiltonian(n, layers)
    solver = onebqf(A, b, num_time_qubits=1, shots=N_SHOTS, debug=False)
    circ = solver.build_circuit()

    # ── Noiseless baseline ────────────────────────────────────────────
    print(f"  Running noiseless baseline ({N_SHOTS} shots × {N_RUNS} runs)...")
    noiseless_sim = AerSimulator()
    tc_noiseless = transpile(circ, noiseless_sim, seed_transpiler=0)
    baseline_probs_runs = []
    for run in range(N_RUNS):
        job = noiseless_sim.run(tc_noiseless, shots=N_SHOTS, seed_simulator=run)
        counts = job.result().get_counts()
        probs, _ = counts_to_probs(counts, n_sys, N_SHOTS)
        baseline_probs_runs.append(probs)
    baseline_mean = np.mean(baseline_probs_runs, axis=0)
    noiseless_ssi = compute_ssi(baseline_mean, true_segs, n_sys)
    print(f"    Noiseless SSI={noiseless_ssi:.2f}")

    results[key] = {"noiseless": {"ssi": noiseless_ssi}}

    # ── Noisy backends ────────────────────────────────────────────────
    for backend_name, fake_backend in BACKENDS.items():
        print(f"  [{backend_name}] building noise model...")
        t0 = time.time()
        try:
            # Use from_backend — correct way to get noise + coupling + basis gates
            noisy_sim = AerSimulator.from_backend(fake_backend)
            tc_noisy = transpile(circ, noisy_sim,
                                 optimization_level=1,
                                 seed_transpiler=42)

            fidelities, ssis = [], []
            for run in range(N_RUNS):
                job = noisy_sim.run(tc_noisy, shots=N_SHOTS, seed_simulator=run)
                counts = job.result().get_counts()
                probs, success = counts_to_probs(counts, n_sys, N_SHOTS)
                if success > 10:   # need enough counts to be meaningful
                    fid = hellinger_fidelity(baseline_mean, probs)
                    ssi = compute_ssi(probs, true_segs, n_sys)
                    fidelities.append(fid)
                    ssis.append(ssi)

            results[key][backend_name] = {
                "hellinger_fidelity": {
                    "mean": float(np.mean(fidelities)),
                    "std":  float(np.std(fidelities)),
                    "values": [float(x) for x in fidelities],
                },
                "ssi": {
                    "mean": float(np.mean(ssis)),
                    "std":  float(np.std(ssis)),
                    "values": [float(x) for x in ssis],
                },
            }
            print(f"    Fidelity={np.mean(fidelities):.3f}±{np.std(fidelities):.3f}  "
                  f"SSI={np.mean(ssis):.2f}±{np.std(ssis):.2f}  "
                  f"({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [WARN] {backend_name} failed: {e}")
            results[key][backend_name] = {"error": str(e)}

    # Save after each config
    with open("data/fidelity_results_reproduced.json", "w") as f:
        json.dump(results, f, indent=2)

print(f"\nAll done in {time.time()-t_total:.1f}s")
print("Saved → data/fidelity_results_reproduced.json")
