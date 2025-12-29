# OneBQF
This is a repo for storing code of the paper on the 1-Bit Qunatum Filter

## Repository Structure

```
OneBQF/
├── quantum_algorithms/     # Quantum algorithm implementations
│   ├── HHL.py             # HHL (Harrow-Hassidim-Lloyd) algorithm implementation
│   └── OneBQF.py          # 1-Bit Quantum Filter implementation
│
├── toy_model/             # Toy model for simulations and testing
│   ├── hamiltonian.py     # Hamiltonian definitions
│   ├── simple_hamiltonian.py  # Simplified Hamiltonian models
│   ├── multi_scattering_generator.py  # Multi-scattering event generation
│   ├── state_event_generator.py       # State and event generation utilities
│   ├── state_event_model.py           # State event modeling
│   └── utils.py           # Utility functions for the toy model
│
├── data/                  # Experimental results and metrics
│   ├── circuit_depth.json     # Circuit depth measurements
│   ├── fidelity_results.json  # Fidelity analysis results
│   └── success_counts.json    # Success rate data
│
├── plotting_notebooks/    # Jupyter notebooks for visualization and analysis
│   ├── plot_fidelity_results.ipynb  # Fidelity visualization
│   ├── plotting_complexity.ipynb    # Complexity analysis plots
│   ├── combined_analysis_summary_fitted_exponents.json  # Fitted exponent data
│   └── Plots/             # Generated plot outputs
│
├── example.ipynb          # Example usage and demonstrations
├── LICENSE                # MIT License
└── README.md              # This file
```

### Getting Started

Run [example.ipynb](example.ipynb) to see demonstrations of the 1-Bit Quantum Filter implementation.

## Reproducing the Results

### Circuit Depth and Gate Count Analysis

The circuit complexity data in `data/circuit_depth.json` and `data/success_counts.json` was generated using an extension of the example.ipynb script that:


1. **Generates synthetic particle tracking events** using the LHCb VELO toy model with configurable:
   - Number of particles (2 to 1024)
   - Number of detector layers (3 or 5)
   - Detector geometry (layer spacing: 20mm, dimensions: 33×33mm)

2. **Constructs the Hamiltonian matrix** using GPU-accelerated C++/CUDA implementation with parameters:
   - epsilon = 1e-7 (regularization)
   - alpha = 2.0 (coupling strength)
   - beta = 1.0 (energy scale)

3. **Implements the 1-Bit Quantum Filter** via the HHL algorithm with:
   - 1 time qubit for phase estimation
   - Spectral folding for eigenvalue inversion
   - 100M shots per configuration for statistical reliability

4. **Performs multiple runs** (5 runs per configuration) to ensure reproducibility and statistical significance

5. **Analyzes circuit complexity** including:
   - Circuit depth (total and two-qubit only)
   - Gate counts (single-qubit and two-qubit gates)
   - Hardware transpilation for IBM quantum processors
   - TKET optimization analysis

6. **Computes success rates** by measuring the ancilla qubit and extracting solution vectors

### Fidelity and Noise Model Analysis

The fidelity results in `data/fidelity_results.json` were generated through a separate experimental pipeline:

#### IBM Quantum Hardware Noise Models
1. **Circuit preparation**: The OneBQF circuit is built and decomposed to native gates
2. **Noise model extraction**: Hardware noise models are obtained from IBM quantum backends:
   - `ibm_torino`
   - `ibm_marrakesh`
   - `ibm_pittsburgh`
   - `ibm_fez`
3. **Noisy simulation**: Each circuit configuration is executed 10 times per backend using the extracted noise models
4. **Solution extraction**: Normalized probability vectors are computed from measurement outcomes

#### Quantinuum Hardware (via Qnexus)
1. **TKET optimization**: Circuits are first optimized using `FullPeepholeOptimise` pass
2. **Qnexus compilation**: Optimized circuits are compiled through Qnexus with optimization level 3
3. **Emulator execution**: Circuits are run on the H2-Emulator (noisy emulator with realistic noise model)
4. **Noiseless baseline**: Circuits are also run on H2-1LE (noiseless emulator) to establish ground truth
5. **Multiple runs**: Each configuration is executed 10 times to gather statistics

#### Fidelity Metrics Computation
The analysis in `plotting_notebooks/plot_fidelity_results.ipynb` computes:
- **Hellinger fidelity**: Distribution similarity between noisy results and noiseless baseline
- **Signal Separation Index (SSI)**: Ratio of mean track probability to mean noise probability
- **Bootstrap error estimation**: 1000 bootstrap samples for confidence intervals

### Metrics Collected

- **Circuit Depth**: Total gate depth and two-qubit gate depth
- **Success Rate**: Fraction of shots with ancilla=|1⟩
- **Fidelity**: Agreement with classical solution (threshold T=0.45)
- **Hellinger Fidelity**: Distribution similarity to noiseless baseline
- **Signal Separation Index**: Track vs noise distinguishability
- **Success Probability**: Performance across problem sizes (2-1024 particles)

### Simulation Toy: segment_analysis.ipynb

The notebook `segment_analysis.ipynb` is responsible for the toy model analysis. It generates segment-level efficiency plots showing the Hamiltonian-based segment pairing performance.

#### Full Data Generation

To regenerate all data with full statistics run all cells in `plotting_notebooks/segment_analysis.ipynb`.

#### Configuration Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `FIXED_EPSILON` | 0.002 (2 mrad) | Segment angle threshold |
| `FIXED_RESOLUTION` | 0.005 (5 µm) | Measurement resolution |
| `FIXED_SCATTERING` | 0.0001 (0.1 mrad) | Multiple scattering |
| `TRACKS_PER_EVENT` | 20 | Particles per event |
| `EVENT_COUNTS` | [1, 2, ..., 10] | Events per configuration (20-200 tracks) |
| `N_REPEATS` | 100 | Statistical repeats per configuration |

#### Output Files

- `fixed_epsilon_segment_efficiency.png` (300 DPI raster)
- `fixed_epsilon_segment_efficiency.pdf` (600 DPI vector)
- `runs_fixed_epsilon/fixed_epsilon_angle_data_v3.pkl` (cached data)

### Individual Figures

| Figure | Description | How to Reproduce |
|--------|-------------|------------------|
| Fig. 1 | Event display | `example.ipynb`, Cell 6 |
| Fig. 2 | Circuit diagram | `example.ipynb`, Cell 13 |
| Fig. 2 | Circuit depth | `plotting_notebooks/segment_analysis.ipynb` |
| Fig. 4 | Circuit depth | `plotting_notebooks/plotting_complexity.ipynb` |
| Fig. 5 | Success probability | `plotting_notebooks/plot_fidelity_results.ipynb` |

The data files contain JSON-formatted results with complete metadata for each experimental run.