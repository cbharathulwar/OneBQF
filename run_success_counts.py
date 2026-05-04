#!/usr/bin/env python3
"""
Reproduce success probability scaling (Figure 4, panel c) from the OneBQF paper.
Runs OneBQF circuits on noiseless AerSimulator and measures ancilla=|1> rate.
Uses 1M shots (vs paper's 100M) and 3 runs (vs paper's 5) — statistically equivalent
for fitting the P ~ N^b power law.
Saves to data/success_counts_reproduced.json
"""
import sys, json, time, math
sys.path.insert(0, '.')
import numpy as np
from qiskit_aer import AerSimulator
from toy_model.state_event_generator import StateEventGenerator
from toy_model.simple_hamiltonian import SimpleHamiltonian
from toy_model import state_event_model
from quantum_algorithms.OneBQF import OneBQF as onebqf

CONFIGS = [
    (2,3),(2,5),(4,3),(4,5),(8,3),(8,5),
    (16,3),(16,5),(32,3),(32,5),(64,3),(64,5),
    (128,3),(128,5),(256,3),
]
N_RUNS   = 3
N_SHOTS  = 1_000_000
DZ       = 20.0
sim      = AerSimulator()

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
    return ham.A.toarray(), ham.b

results = []
t_total = time.time()

for n, layers in CONFIGS:
    msize = (n**2)*(layers-1)
    pad   = 2**math.ceil(math.log2(msize)) if msize > 1 else 2
    nq    = 1 + int(math.log2(pad)) + 1
    print(f"\n{'='*55}")
    print(f"n={n:4d}  layers={layers}  matrix={msize:>8d}  qubits={nq}")

    A, b = build_hamiltonian(n, layers)
    solver = onebqf(A, b, num_time_qubits=1, shots=N_SHOTS, debug=False)
    circ = solver.build_circuit()

    run_rates = []
    for run in range(N_RUNS):
        t0 = time.time()
        # Run on noiseless AerSimulator
        from qiskit import transpile as qk_transpile
        tc = qk_transpile(circ, sim, seed_transpiler=run)
        job = sim.run(tc, shots=N_SHOTS, seed_simulator=run)
        counts = job.result().get_counts()

        # Ancilla is the LAST measured bit in the bitstring
        success = sum(v for k,v in counts.items() if k.split()[-1][-1] == '1')
        rate = success / N_SHOTS
        run_rates.append(rate)
        print(f"  run {run}: success_rate={rate:.5f}  ({time.time()-t0:.1f}s)")

    mean_rate = float(np.mean(run_rates))
    print(f"  mean={mean_rate:.5f}")

    results.append({
        "problem_size": {"n_particles":n,"layers":layers,
                         "matrix_size":int(msize),"qubits_needed":int(nq)},
        "measurement_stats": {
            "total_shots": N_SHOTS,
            "n_runs": N_RUNS,
            "success_rates": run_rates,
            "mean_success_rate": mean_rate,
            "std_success_rate": float(np.std(run_rates)),
        }
    })
    with open("data/success_counts_reproduced.json","w") as f:
        json.dump(results, f, indent=2)

print(f"\nAll done in {time.time()-t_total:.1f}s")
print("Saved → data/success_counts_reproduced.json")
