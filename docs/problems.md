# Problems and data sets

The problem types that have an encoder and an experiment script, the extra
packages individual scripts need, and the benchmark data in the repository.

- [Supported problems](#supported-problems)
- [Graph partitioning parameters](#graph-partitioning-parameters)
- [Optional dependencies](#optional-dependencies)
- [Running the experiment scripts](#running-the-experiment-scripts)
- [Benchmark data](#benchmark-data)

## Supported problems

| Problem | Encoding | Script |
|---------|----------|--------|
| **3-SAT** | Clause $(a \vee b \vee c)$ → PBF penalty $(1-a)(1-b)(1-c)$; SATLIB benchmarks | `Skript_Solve_3SAT.py` |
| **Traveling Salesman (TSP)** | QUBO via binary position matrix $x_{ij}$ with constraint penalties | `Skript_Solve_Traveling_Salesman.py` |
| **Route planning on real roads** | TSP on a distance matrix from a routing engine (Valhalla via `routingpy`) | `Skript_Routing.py` |
| **Number Partitioning (NPP)** | QUBO formulation | `Skript_Solve_NumberPartitioning.py` |
| **Graph Partitioning** | Recursive DA bisection (`GraphPartitioning`) as divide-and-conquer for large TSP instances | `Skript_Solve_GraphPartitioning.py` |
| **2D Ising Model** | Direct QUBO encoding of spin interactions | `Skript_IsingModel_Simulator.py` |
| **Binary / Integer Programming** | General BIP/IP encoding as PBF | `Skript_Solve_Integer_Programm.py` |
| **Random PBF** | Randomly generated PBF landscapes for benchmarking | `Skript_Solve_Random_Generated_pbf.py` |
| **Database Join Ordering** | QUBO via cardinality/selectivity constraints; requires `QUBOGenerator.py` and `ProblemGenerator.py` from the [lfd/vldb24](https://github.com/lfd/vldb24) repository (GPL, not included here) | `Skript_Join_Ordering.py` |
| **Generalized Transportation** | QUBO formulation | `Skript_Solve_GTP.py` |

## Graph partitioning parameters

`GraphPartitioning(coords, parts, num_MC_GP=1, diff_border=1, steps=200)` in
`Funcs_Optimizers.py` splits a point set recursively into $2^{\text{parts}}$
groups by solving a balanced-bisection QUBO with DA at each level:

| Parameter | Meaning |
|---|---|
| `coords` | list of 2-D points |
| `parts` | number of bisection levels → $2^{\text{parts}}$ groups |
| `num_MC_GP` | DA trials per bisection; the best is kept |
| `diff_border` | tolerated size imbalance between the two halves; larger imbalances are repaired by moving the closest points across |
| `steps` | DA steps per bisection (schedule `["logarithmic", 2·N/(level+1), 0]`, offset rate 1000) |

## Optional dependencies

The solver itself needs only the packages in the
[README](../README.md#core-api). These are needed by individual scripts:

| Package | Needed by |
|---|---|
| `tensorflow` | `TensorAnnealing.py` (dense QUBOs on the GPU) |
| `routingpy` | `Skript_Routing.py` (route planning on real road networks) |
| `docplex`, `dimod`, `qiskit-optimization` | `Skript_Join_Ordering.py` (plus the external modules named in the table above) |
| `scipy` | `Skript_Solve_GTP.py` |
| `Pillow`, `imageio` | `Skript_IsingModel_Simulator.py`, `pythonVideoMaker.py` |
| `psutil` | `Test_Memory_Usage2.py` |
| `sympy` | `Test_symb_bib_stat_distrib.py` |

## Running the experiment scripts

All scripts are run from within the `Code/` directory, or from the repository
root for root-level scripts.

```bash
# From Code/
python Skript_Solve_3SAT.py
python Skript_Solve_NumberPartitioning.py
python Skript_Solve_Traveling_Salesman.py
python Skript_Solve_Integer_Programm.py
python Skript_Solve_Random_Generated_pbf.py
python Skript_IsingModel_Simulator.py
```

> **Note:** Several scripts contain hardcoded absolute file paths from the
> author's machine (`Funcs_Annealing2.py`,
> `Skript_IsingModel_Simulator.py`, `Skript_Join_Ordering.py`,
> `Skript_Solve_3SAT.py`, `Skript_Solve_GraphPartitioning.py`,
> `Skript_Solve_NumberPartitioning.py`, `Skript_Solve_Traveling_Salesman.py`).
> Adjust these to your local setup before running.

## Benchmark data

The benchmark data in `3SAT_DATA/` (SATLIB) and `TSP_data/` (TSPLIB) is
redistributed from the respective public benchmark collections and remains
subject to their terms; sources in
[references.md](references.md#benchmark-data).
