#!/usr/bin/env python3
"""Run only the missing circuit depth configs and append to existing JSON."""
import sys, json, time, math
sys.path.insert(0, '.')
import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeTorino
from toy_model.state_event_generator import StateEventGenerator
from toy_model.simple_hamiltonian import SimpleHamiltonian
from toy_model import state_event_model
from quantum_algorithms.OneBQF import OneBQF as onebqf
from pytket.extensions.qiskit import qiskit_to_tk
from pytket.passes import FullPeepholeOptimise
from pytket import OpType

MISSING = [(256, 3)]
DZ = 20.0
backend = FakeTorino()

with open("data/circuit_depth_reproduced.json") as f:
    results = json.load(f)
done = {(e["problem_size"]["n_particles"], e["problem_size"]["layers"]) for e in results}

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

def gate_counts(circ):
    ops = circ.count_ops()
    two = sum(v for k,v in ops.items() if k in ("cx","cz","ecr","rzz"))
    single = sum(v for k,v in ops.items() if k not in ("cx","cz","ecr","rzz","measure","barrier"))
    return {"depth":circ.depth(),"two_qubit_gates":two,"single_qubit_gates":single,
            "total_gates":two+single,"num_qubits":circ.num_qubits,
            "gate_breakdown":{k:v for k,v in ops.items() if k!="barrier"}}

for n, layers in MISSING:
    if (n, layers) in done:
        print(f"Skipping {n}p {layers}L — already done"); continue
    msize = (n**2)*(layers-1)
    pad = 2**math.ceil(math.log2(msize)) if msize>1 else 2
    nq = 1 + int(math.log2(pad)) + 1
    print(f"\n{n}p {layers}L  matrix={msize}  qubits={nq}")

    t0=time.time(); A,b = build_hamiltonian(n, layers)
    print(f"  Hamiltonian: {time.time()-t0:.1f}s")

    t0=time.time()
    solver = onebqf(A, b, num_time_qubits=1, shots=100, debug=False)
    circ = solver.build_circuit()
    print(f"  Circuit built: {time.time()-t0:.1f}s")

    t0=time.time()
    decomp = circ
    for _ in range(5): decomp = decomp.decompose()
    qstats = gate_counts(decomp)
    print(f"  Qiskit: depth={qstats['depth']} 2Q={qstats['two_qubit_gates']} ({time.time()-t0:.1f}s)")

    t0=time.time()
    try:
        tr = transpile(decomp, backend=backend, optimization_level=3, seed_transpiler=42)
        tstats = gate_counts(tr); tstats["backend"]="fake_torino"
        print(f"  Torino: depth={tstats['depth']} 2Q={tstats['two_qubit_gates']} ({time.time()-t0:.1f}s)")
    except Exception as e:
        tstats = {"error": str(e)}; print(f"  Torino failed: {e}")

    t0=time.time()
    try:
        tk = qiskit_to_tk(decomp); FullPeepholeOptimise().apply(tk)
        tkstats = {"depth":tk.depth(),"two_qubit_gates":tk.n_gates_of_type(OpType.CX),
                   "total_gates":tk.n_gates,
                   "depth_reduction_percent":round((1-tk.depth()/qstats["depth"])*100,2)}
        print(f"  TKET: depth={tkstats['depth']} 2Q={tkstats['two_qubit_gates']} ({time.time()-t0:.1f}s)")
    except Exception as e:
        tkstats = {"error": str(e)}; print(f"  TKET failed: {e}")

    results.append({"problem_size":{"n_particles":n,"layers":layers,
        "matrix_size":int(msize),"qubits_needed":int(nq)},
        "standard":{"qiskit":qstats,"hardware_torino":tstats,"tket":tkstats}})
    with open("data/circuit_depth_reproduced.json","w") as f:
        json.dump(results, f, indent=2)

print("\nDone — data/circuit_depth_reproduced.json updated")
