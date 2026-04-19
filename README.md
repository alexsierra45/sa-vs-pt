# Spin Glass Optimization: Simulated Annealing vs Parallel Tempering

A metaheuristic optimization project that benchmarks **Simulated Annealing (SA)** and **Parallel Tempering (PT)** for finding the ground state (minimum energy configuration) of a disordered Ising spin-glass system defined on a random 3-regular graph.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Project Structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Quick Start](#4-quick-start)
5. [Module Reference — `spin_glass_solvers.py`](#5-module-reference--spin_glass_solverspy)
6. [Algorithms](#6-algorithms)
7. [Experiments](#7-experiments)
8. [Data Format](#8-data-format)
9. [Design Decisions](#9-design-decisions)

---

## 1. Problem Definition

### The Ising Spin-Glass Model

An **Ising spin glass** is a statistical physics model of a magnetic system with *quenched disorder* — random, frozen-in interactions between spins. It is one of the canonical hard combinatorial optimization problems (related to MAX-CUT and graph partitioning) and serves as a rigorous benchmark for metaheuristics.

**Variables**

Each node `i` in the graph carries a binary spin:

```
s_i ∈ { -1, +1 }
```

**Energy function**

The energy (Hamiltonian) of a configuration `s = (s_0, …, s_{N-1})` is:

```
E(s) = − Σ_{<i,j>} J_ij · s_i · s_j  −  Σ_i H_i · s_i
```

| Symbol | Meaning |
|---|---|
| `<i,j>` | Sum over edges (each counted once) |
| `J_ij` | Coupling between spins i and j, drawn i.i.d. from N(0,1) |
| `H_i` | External magnetic field at node i, drawn i.i.d. from N(0,1) |

The goal is to **minimise E** — find the ground-state configuration. Because `J_ij` can be positive or negative, neighbouring spins simultaneously want to align and anti-align, creating *frustration* that makes the energy landscape highly non-convex with exponentially many local minima.

**Graph topology**

The system is defined on an undirected **3-regular graph** (every node has exactly 3 neighbours), generated randomly via `networkx.random_regular_graph`. The fixed degree-3 structure keeps the problem well-controlled: each `ΔE` computation touches exactly 3 neighbours, giving O(1) cost per spin flip regardless of N.

---

## 2. Project Structure

```
sa-vs-pt/
├── spin_glass_solvers.py        # Reusable module — all algorithms and helpers
├── create_notebook.py           # Run once to generate the notebook
├── experimentos_spin_glass.ipynb  # Jupyter notebook with the 3 experiments
├── resultados/                  # Auto-created; all saved results land here
│   ├── exp1_sa_*.pkl
│   ├── exp1_pt_*.pkl
│   ├── exp2_sa_30.pkl
│   ├── exp2_pt_30.pkl
│   ├── exp2_plots.png
│   ├── exp3_*.pkl
│   ├── exp3_escalabilidad.csv
│   └── exp3_escalabilidad.png
└── README.md
```

---

## 3. Installation

Python 3.9+ is required.

```bash
pip install networkx numpy matplotlib seaborn pandas scipy tqdm
```

All other dependencies (`json`, `pickle`, `time`, `pathlib`) are part of the standard library.

---

## 4. Quick Start

**Generate the notebook** (only needed if the `.ipynb` is missing):

```bash
python create_notebook.py
```

**Run the experiments:**

Open `experimentos_spin_glass.ipynb` in Jupyter and run all cells. The three experiments execute sequentially; everything is saved to `resultados/`.

**Use the module directly:**

```python
import spin_glass_solvers as sgs
import numpy as np

# Build a problem instance
G        = sgs.generar_grafo_regular(N=50, d=3, seed=7)
J, H     = sgs.generar_instancia(G, seed=7)
nbrs     = sgs.construir_lista_vecinos(G)

# Run Simulated Annealing
schedule = sgs.schedule_geometrico(T_ini=5.0, T_fin=0.1, n_pasos=100)
result   = sgs.simulated_annealing(50, J, H, nbrs,
               temp_schedule=schedule, iters_por_temp=2500, seed=0)
print(result['mejor_energia'])

# Run Parallel Tempering
temps  = np.geomspace(0.1, 5.0, 16)
result = sgs.parallel_tempering(50, J, H, nbrs,
             temp_schedule=temps, pasos_por_intercambio=10,
             iteraciones_totales=62500, seed=0)
print(result['mejor_energia'])

# Save and reload
sgs.guardar_resultado(result, 'resultados/mi_corrida.pkl')
data = sgs.cargar_resultado('resultados/mi_corrida.pkl')
```

---

## 5. Module Reference — `spin_glass_solvers.py`

### Problem generation

| Function | Description |
|---|---|
| `generar_grafo_regular(N, d=3, seed)` | Returns a random d-regular `nx.Graph` with N nodes. Wraps `networkx.random_regular_graph`. N·d must be even. |
| `generar_instancia(G, seed)` | Samples `J_ij ~ N(0,1)` for every edge and `H_i ~ N(0,1)` for every node. Returns `(J: dict, H: ndarray)`. |
| `construir_lista_vecinos(G)` | Returns `neighbors[i]` as a plain Python list of lists. Needed by the energy functions. |
| `problema_a_dict(G, J, H)` | Serialises the problem to a JSON-safe dict (`edges` as int lists, `J` keys as `"i,j"` strings). |
| `problema_desde_dict(d)` | Inverse of the above. Reconstructs `G, J, H, neighbors` from a saved dict. |

### Energy functions

| Function | Complexity | Notes |
|---|---|---|
| `energia(spins, J, H, neighbors)` | O(|E| + N) | Full energy — used for initialisation only. |
| `delta_energia(spins, i, J, H, neighbors)` | O(d) = O(3) | Energy change from flipping spin `i`. Used in every MC step. Formula: `ΔE = 2·s_i·(H_i + Σ_{j∈N(i)} J_ij·s_j)`. |

### Temperature schedules

All three return a `numpy.ndarray` of length `n_pasos` decreasing from `T_ini` to `T_fin`.

| Function | Shape | Notes |
|---|---|---|
| `schedule_geometrico(T_ini, T_fin, n_pasos)` | Exponential decay | `T_{k+1} = α·T_k`, constant ratio. Standard choice. |
| `schedule_lineal(T_ini, T_fin, n_pasos)` | Linear decay | Constant step `ΔT`. Cools too fast at low T. |
| `schedule_logaritmico(T_ini, T_fin, n_pasos)` | Slow at start, fast at end | `T_k ∝ 1/log(1+k)`, normalised to `[T_fin, T_ini]`. |

The dict `SCHEDULE_FNS` maps string names to these functions for convenient lookup.

### Simulated Annealing

```python
simulated_annealing(
    N, J, H, neighbors,
    temp_schedule,       # array of temperatures, high → low
    iters_por_temp,      # flip attempts per temperature level
    seed=None,
    registro_intervalo=1000,  # how often to append to trajectory
) -> dict
```

**Algorithm:**
1. Initialise spins randomly ∈ {−1, +1}.
2. For each temperature T in `temp_schedule`:
   - Repeat `iters_por_temp` times:
     - Pick a random spin index `i`.
     - Compute `ΔE` using `delta_energia` (O(3), not O(N)).
     - Accept the flip if `ΔE < 0` or with probability `exp(−ΔE/T)`.
     - Track the best configuration seen so far.
3. Every `registro_intervalo` evaluations, append the current energy to the trajectory list.

**Total evaluations:** `len(temp_schedule) × iters_por_temp`.

### Parallel Tempering

```python
parallel_tempering(
    N, J, H, neighbors,
    temp_schedule,           # one temperature per replica, low → high
    pasos_por_intercambio,   # MC steps per replica between swap attempts
    iteraciones_totales,     # total MC steps per replica
    seed=None,
    registro_intervalo=1000,
) -> dict
```

**Algorithm:**
1. Initialise R = `len(temp_schedule)` independent replicas, each with a random spin configuration.
2. Repeat for `iteraciones_totales // pasos_por_intercambio` cycles:
   a. **Local MC:** each replica runs `pasos_por_intercambio` standard Metropolis steps at its own temperature.
   b. **Replica swap:** attempt to exchange configurations between adjacent temperature pairs. Pairs alternate between even-offset (0↔1, 2↔3, …) and odd-offset (1↔2, 3↔4, …) to avoid correlations. Swap is accepted with probability:
      ```
      P(swap) = min(1, exp((β_i − β_{i+1})·(E_i − E_{i+1})))
      ```
      where `β = 1/T`. This preserves detailed balance.
3. Track the global minimum energy across all replicas.

**Total evaluations:** `R × iteraciones_totales` (plus R initial energy calculations).

**Why PT is stronger than SA:** the high-temperature replicas act as an annealing reservoir — they explore the landscape freely and occasionally donate good configurations to the cold replicas via swaps, allowing the cold replica to escape local minima that SA would get stuck in.

### Persistence

| Function | Description |
|---|---|
| `guardar_resultado(resultado, filepath, formato='pkl')` | Saves a result dict using `pickle` (default) or `json`. Creates parent directories automatically. |
| `cargar_resultado(filepath)` | Detects format from file extension (`.pkl` or `.json`) and deserialises. |

---

## 6. Algorithms

### Simulated Annealing (SA)

SA is a single-trajectory metaheuristic inspired by the physical process of slowly cooling a metal. The key parameter is the **cooling schedule**: cooling too fast traps the search in local minima; cooling too slowly wastes evaluations at high temperature where acceptance is near-uniform.

The acceptance criterion `exp(−ΔE/T)` degrades gracefully: at high T it is close to 1 (almost any move is accepted, exploratory behaviour), at low T it approaches 0 for uphill moves (exploitative behaviour).

**Three schedules are tested:**

- **Geometric** (`T_k = T_0 · α^k`): the most common choice. Constant percentage drop per step, so proportionally more time is spent at low temperatures. Recommended default.
- **Linear** (`T_k = T_0 − k·ΔT`): spends equal absolute time at each temperature. Tends to cool too aggressively at low T.
- **Logarithmic** (`T_k ∝ 1/log(1+k)`): slowest at the beginning, fastest at the end. Theoretically guaranteed to find the global optimum as `n_pasos → ∞`, but impractical for finite budgets.

### Parallel Tempering (PT)

PT (also called Replica Exchange Monte Carlo) runs R copies of the system simultaneously at different temperatures and periodically attempts to swap their configurations. This enables:

- **Ergodicity at low T:** cold replicas can tunnel through energy barriers by borrowing high-energy configurations from hot replicas.
- **Better exploration:** the hot replicas continuously generate diverse starting points that are then refined by cold replicas.

The number of replicas `R` and the temperature grid spacing are the main hyperparameters. Replicas must be close enough in temperature that swap acceptance rates are non-negligible (typically 20–40%).

**Comparison to SA:** PT is generally stronger on rugged landscapes but has higher memory usage (R × N spins) and more complex tuning. For small N (≤ 100) both methods are competitive; for large N the ergodic advantage of PT becomes more pronounced.

---

## 7. Experiments

All experiments are run in `experimentos_spin_glass.ipynb`. The fixed instance (N=100, seed=42) is shared across Experiments 1 and 2 to ensure a fair comparison.

### Experiment 1 — Hyperparameter Tuning

**Goal:** find the best temperature schedule for SA and the best number of replicas for PT, using ~1 000 000 energy evaluations per configuration.

**SA configurations tested** (`iters_por_temp = N² = 10 000`, `n_temps = 100`):

| `T_ini` | `T_fin` | Schedule |
|---|---|---|
| 5.0 | 0.1 | geometric |
| 10.0 | 0.01 | geometric |
| 3.0 | 0.05 | geometric |
| 5.0 | 0.1 | linear |
| 5.0 | 0.1 | logarithmic |

Each configuration is run 5 times; mean and std of best energy are reported. Results are sorted by mean energy; the best configuration is stored as `MEJOR_SA` for Experiments 2 and 3.

**PT configurations tested** (`T_min = 0.1`, `T_max = 5.0`, `pasos_por_intercambio = 10`):

| Replicas | `iters_totales` per replica | Total evals |
|---|---|---|
| 8 | 125 000 | 1 000 000 |
| 16 | 62 500 | 1 000 000 |
| 32 | 31 250 | 1 000 000 |

Results saved to `resultados/exp1_*.pkl`.

### Experiment 2 — 30 Independent Runs

**Goal:** characterise the distribution of final energies for both algorithms under equal evaluation budgets.

Using the best parameters from Experiment 1, both SA and PT are run 30 times (seeds 0–29) on the same N=100 instance. Each run records:

- Best energy found
- Total wall-clock time
- Number of energy evaluations
- Full energy trajectory (sampled every 5 000 evaluations)

**Statistics reported:** min, mean, median, mode (rounded to 2 decimal places), std of energy; mean and std of runtime.

**Plots generated (`exp2_plots.png`):**

1. **Boxplot** — side-by-side comparison of the 30 final energies for SA and PT.
2. **Histogram** — overlapping distribution of final energies.
3. **Trajectory plot** — all 30 energy curves superimposed, one per run, colour-coded by algorithm.

Results saved to `resultados/exp2_sa_30.pkl` and `resultados/exp2_pt_30.pkl`.

### Experiment 3 — Scalability

**Goal:** measure how solution quality and runtime scale with problem size.

**Setup:**

- `N ∈ {50, 100, 200, 400}`
- 3 independent instances per N (seeds 1000, 1001, 1002)
- 10 runs per instance × algorithm pair

**Parameter scaling:**

- SA: `iters_por_temp = N²` (budget grows quadratically with N, keeping N MC sweeps per temperature).
- PT: `num_replicas ∝ √N` (rounded, clamped to [4, 64]); `iteraciones_totales` scaled to ~`10·N²` total evaluations per replica set.

**Plots generated (`exp3_escalabilidad.png`):**

1. **Energy vs N** — mean best energy as a function of N (linear scale). Captures how solution quality degrades as the problem grows.
2. **Time vs N (log-log)** — empirical complexity. A straight line in log-log space indicates power-law scaling `T ~ N^α`; the exponent α is annotated on the plot.

Aggregate results saved to `resultados/exp3_escalabilidad.csv`; raw run data saved per instance to `resultados/exp3_*.pkl`.

---

## 8. Data Format

Every saved file (`.pkl` or `.json`) is a dict with the following structure. The `corridas` key wraps a list of individual run dicts; each run dict contains:

```python
{
    "problema": {
        "N":     int,                    # number of spins
        "edges": [[i, j], ...],          # list of edge pairs
        "J":     {"i,j": float, ...},    # couplings, keys as "i,j" strings
        "H":     [float, ...]            # external fields, length N
    },
    "config_inicial":      [int, ...],   # initial spin configuration (+1/-1)
    "algoritmo":           "SA" | "PT",
    "hiperparametros":     {dict},       # all hyperparameters used
    "trayectoria_energia": [float, ...], # energy sampled every registro_intervalo evals
    "mejor_energia":       float,        # best energy found in this run
    "mejor_configuracion": [int, ...],   # spin configuration achieving mejor_energia
    "tiempo_ejecucion":    float,        # wall-clock seconds
    "evaluaciones_energia": int          # total ΔE computations
}
```

**Reloading for further analysis:**

```python
import spin_glass_solvers as sgs

data   = sgs.cargar_resultado('resultados/exp2_sa_30.pkl')
corridas = data['corridas']

# Rebuild the problem
G, J, H, nbrs = sgs.problema_desde_dict(corridas[0]['problema'])

# Plot a saved trajectory
import matplotlib.pyplot as plt
traj = corridas[0]['trayectoria_energia']
plt.plot(traj)
plt.xlabel('Checkpoint index')
plt.ylabel('Energy')
plt.show()
```

---

## 9. Design Decisions

**ΔE instead of full energy recalculation.** On a 3-regular graph each spin flip affects only 3 edge terms. `delta_energia` therefore runs in O(3) = O(1), versus O(|E|) = O(3N/2) for a full recomputation. This is a ~50× speedup for N=100 and grows linearly with N.

**Alternating swap pairs in PT.** Adjacent swaps (0↔1, 1↔2, …) applied naively in sequence are correlated. Alternating between even-offset and odd-offset pairs on consecutive cycles removes this correlation and improves detailed balance in practice.

**`numpy.random.default_rng` over `numpy.random.seed`.** The new Generator API produces higher-quality random numbers (PCG64 algorithm) and is fully reproducible without affecting global state. Each run is seeded independently, making parallel execution safe.

**`time.perf_counter` for timing.** Higher resolution than `time.time` on Windows, where the latter has ~15 ms granularity. Timing is measured around the full algorithm body including initialisation.

**Pickle as default serialisation format.** Pickle preserves numpy arrays and Python tuples as native types without conversion, making reloading lossless and fast. JSON is offered as an alternative for human-readable inspection or interoperability with other languages — it converts array keys to strings and floats may lose precision at the last digit.
