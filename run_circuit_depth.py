#!/usr/bin/env python3
"""
Reproduce circuit depth scaling data (Figure 4, left panels) from the OneBQF paper.
Builds OneBQF circuits for 2-256 particles × 3 and 5 layers, transpiles for
IBM Torino (via FakeTorino), runs TKET FullPeepholeOptimise, and saves results
to data/circuit_depth_reproduced.json for comparison with data/circuit_depth.json.
"""

import sys, json, time
sys.path.insert(0, '.')

import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeTorino

from toy_model.state_event_generator import StateEventGenerator
from toy_model.simple_hamiltonian import SimpleHamiltonian
from toy_model import state_event_model
from quantum_algorithms.OneBQF import OneBQF as onebqf

try:
    from pytket.extensions.qiskit import qiskit_to_tk
    from pytket.passes import FullPeepholeOptimise
    from pytket import OpType
    TKET_AVAILABLE = True
except ImportError:
    TKET_AVAILABLE = False
    print("[WARN] pytket not available — skipping TKET optimisation")

# ── Problem sizes to test ──────────────────────────────────────────────
CONFIGS = [
    (2,   3), (2,   5),
    (4,   3), (4,   5),
    (8,   3), (8,   5),
    (16,  3), (16,  5),
    (32,  3), (32,  5),
    (64,  3), (64,  5),
    (128, 3), (128, 5),
    (256, 3),           # 256×5 would need >20 qubits; skip for speed
]

DZ = 20.0
backend = FakeTorino()


def build_event_and_hamiltonian(n_particles, n_layers, seed=42):
    """Generate a clean event and return the Hamiltonian matrix and vector."""
    module_ids = list(range(1, n_layers + 1))
    lx = ly = [33.0] * n_layers
    zs = [DZ * l for l in range(1, n_layers + 1)]
    det = state_event_model.PlaneGeometry(module_id=module_ids, lx=lx, ly=ly, z=zs)

    np.random.seed(seed)
    seg = StateEventGenerator(
        det, phi_min=-0.2, phi_max=0.2, events=1,
        n_particles=[n_particles],
        measurement_error=0.0, collision_noise=0.0,
    )
    seg.generate_random_primary_vertices({"x": 1, "y": 1, "z": 1})
    seg.generate_particles([[{"type": "MIP", "mass": 0.511, "q": 1}
                             for _ in range(n_particles)]])
    event = seg.generate_complete_events()

    ham = SimpleHamiltonian(epsilon=1e-7, alpha=2.0, beta=1.0)
    ham.construct_hamiltonian(event=event, convolution=False)
    A = ham.A.toarray()
    b = ham.b
    return A, b


def get_gate_counts(circuit):
    ops = circuit.count_ops()
    single = sum(v for k, v in ops.items()
                 if k not in ("cx", "cz", "ecr", "rzz", "measure", "barrier"))
    two = sum(v for k, v in ops.items()
              if k in ("cx", "cz", "ecr", "rzz"))
    return {
        "depth": circuit.depth(),
        "total_gates": sum(v for k, v in ops.items() if k != "barrier"),
        "single_qubit_gates": single,
        "two_qubit_gates": two,
        "num_qubits": circuit.num_qubits,
        "gate_breakdown": {k: v for k, v in ops.items() if k != "barrier"},
    }


results = []
t_total = time.time()

for n_particles, n_layers in CONFIGS:
    matrix_size = (n_particles ** 2) * (n_layers - 1)
    # OneBQF pads to next power of 2
    import math
    padded = 2 ** math.ceil(math.log2(matrix_size)) if matrix_size > 1 else 2
    n_sys_qubits = int(math.log2(padded))
    n_qubits = 1 + n_sys_qubits + 1   # time + system + ancilla

    print(f"\n{'='*60}")
    print(f"n_particles={n_particles}  layers={n_layers}  "
          f"matrix={matrix_size}  qubits={n_qubits}")
    print(f"{'='*60}")

    t0 = time.time()
    A, b = build_event_and_hamiltonian(n_particles, n_layers)
    print(f"  Hamiltonian built: {A.shape} in {time.time()-t0:.1f}s")

    # Build circuit
    t0 = time.time()
    solver = onebqf(A, b, num_time_qubits=1, shots=100, debug=False)
    circ = solver.build_circuit()
    print(f"  Circuit built ({circ.num_qubits} qubits) in {time.time()-t0:.1f}s")

    # Decompose fully
    t0 = time.time()
    decomp = circ
    for _ in range(5):
        decomp = decomp.decompose()
    qiskit_stats = get_gate_counts(decomp)
    print(f"  Qiskit decomposed: depth={qiskit_stats['depth']}  "
          f"2Q={qiskit_stats['two_qubit_gates']} in {time.time()-t0:.1f}s")

    # Transpile for IBM Torino
    t0 = time.time()
    try:
        transpiled = transpile(decomp, backend=backend, optimization_level=3,
                               seed_transpiler=42)
        torino_stats = get_gate_counts(transpiled)
        torino_stats["backend"] = "fake_torino"
        print(f"  Torino transpile: depth={torino_stats['depth']}  "
              f"2Q={torino_stats['two_qubit_gates']} in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  [WARN] Torino transpile failed: {e}")
        torino_stats = {"error": str(e)}

    # TKET optimisation
    tket_stats = {}
    if TKET_AVAILABLE:
        t0 = time.time()
        try:
            tk_circ = qiskit_to_tk(decomp)
            FullPeepholeOptimise().apply(tk_circ)
            ops = tk_circ.get_commands()
            from collections import Counter
            op_counts = Counter(str(cmd.op.type) for cmd in ops
                                if str(cmd.op.type) != "OpType.Measure")
            two_q = tk_circ.n_gates_of_type(OpType.CX)
            total = tk_circ.n_gates
            tket_stats = {
                "depth": tk_circ.depth(),
                "two_qubit_gates": two_q,
                "total_gates": total,
                "gate_breakdown": {k: v for k, v in Counter(
                    str(cmd.op.type) for cmd in tk_circ.get_commands()
                ).items()},
                "depth_reduction_percent": round(
                    (1 - tk_circ.depth() / qiskit_stats["depth"]) * 100, 2
                ) if qiskit_stats["depth"] > 0 else 0,
            }
            print(f"  TKET optimised:   depth={tket_stats['depth']}  "
                  f"2Q={tket_stats['two_qubit_gates']} in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  [WARN] TKET failed: {e}")
            tket_stats = {"error": str(e)}

    results.append({
        "problem_size": {
            "n_particles": n_particles,
            "layers": n_layers,
            "matrix_size": int(matrix_size),
            "qubits_needed": int(n_qubits),
        },
        "standard": {
            "qiskit": qiskit_stats,
            "hardware_torino": torino_stats,
            "tket": tket_stats,
        },
    })

    # Save after every config in case of interruption
    with open("data/circuit_depth_reproduced.json", "w") as f:
        json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"All done in {time.time()-t_total:.1f}s")
print(f"Results saved to data/circuit_depth_reproduced.json")
