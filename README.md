# annealing-cop-approximator

The implementation part of my Master's thesis *Analyse und Implementierung der
Optimierungsheuristik Digital Annealing* (OTH Regensburg, 2025). Digital
Annealing is the quantum-inspired annealing heuristic behind Fujitsu's Digital
Annealer: like simulated annealing, but every step tests all possible moves at
once and picks among the accepted ones. Both are implemented here directly on
pseudo-Boolean functions of any degree and compared on 3-SAT, TSP, number
partitioning, Ising models and more ([problems](docs/problems.md),
[results](docs/performance.md)).

It is an experimenting library first. Every experiment of the thesis lives
here, the solver minimizes any pseudo-Boolean function, and an exact toolkit
computes the stationary distributions of both Markov chains
([toolkit](docs/stationary-distribution.md)) — which has since made the
repository the testbed for an ongoing comparison of the two chains
([findings](docs/findings.md)). It is not the fastest way to solve a QUBO; for
that, use the batched DA engine
[annealing-qubo-optimizer](https://github.com/LinoBugia/annealing-qubo-optimizer).

## Core API

There is no package to install: clone the repository, install the
dependencies, and import from `Code/`.

```bash
git clone https://github.com/LinoBugia/annealing-cop-approximator.git
cd annealing-cop-approximator/Code
pip install numpy pandas matplotlib plotly p_tqdm lk_heuristic
```

The main entry point for programmatic use is `pbf_min_solver` in
`Funcs_Optimizers.py`:

```python
import numpy as np
from Funcs_Annealing2 import createPolyDict
from Funcs_Optimizers import pbf_min_solver

# A QUBO as a PBF dictionary
pbf = {(0,): -1.0, (1,): -1.0, (0, 1): 2.0}
n = 2
pbf_var_dict = createPolyDict(pbf, n)

num_MC = 20                                # independent Monte-Carlo trials
Min_varAssignements, Mins, Trajectories, _ = pbf_min_solver(
    pbf, pbf_var_dict,
    type_alg="digitalAnnealing",           # algorithms: docs/configuration.md
    steps=10000,
    num_MC=num_MC,
    cooling_param=["logarithmic", 100, 0], # schedules: docs/configuration.md
    seed_rand=list(range(num_MC)),         # one RNG seed per trial
    seed_gen_initial_varAssignment=42,     # seed of the random start state
    offset_increase_rate=0,                # escape mechanism off
    save_addinfo=True,                     # needed to get Min_varAssignements back
    save_csv=False,
)
best = Min_varAssignements[int(np.argmin(Mins))]
```

All 20 trials reach the minimum $-1$, at `[0, 1]`. Every argument, the six
algorithms and all cooling schedules are in
[configuration.md](docs/configuration.md); the experiment scripts per problem in
[problems.md](docs/problems.md).

## Core concept

Every problem is encoded as a **pseudo-Boolean function (PBF)**:

$$P : \mathbb{F}_2^n \to \mathbb{R}, \quad P(x) = \sum_{\substack{S \subseteq \{1,\ldots,n\} \\ |S| \le d}} a_S \prod_{k \in S} x_k$$

where $d$ is the degree of the PBF. This general representation covers both
**QUBO** ($d \le 2$) and **PUBO** ($d > 2$, e.g. 3-SAT). The goal in all cases
is to find the binary assignment $x^* \in \mathbb{F}_2^n$ that minimizes $P(x)$.
In code, a PBF is a Python `dict` from monomial tuples to coefficients — see
[configuration.md](docs/configuration.md#input-the-pbf-dictionary).

| | Simulated Annealing | Digital Annealing |
|---|---|---|
| One step | draw one bit flip, apply the Metropolis test | test all $n$ bit flips, pick uniformly among the accepted |
| Stationary distribution | Boltzmann–Gibbs | differs — no detailed balance |
| At a local minimum | waits for a thermal escape | an energy offset grows until a flip is accepted |

Both are Markov chains over the modification-map framework of the thesis; the
definitions, the TSP move and the convergence results are in
[framework.md](docs/framework.md).

## Output

`pbf_min_solver` returns the best assignment, the minimum energy and the full
energy trajectory of every trial. With `save_csv=True` it also writes:

```
Code/Runs/Evaluation_<date>/
  Evaluation_<date>.csv       one summary line per call
  Trajectories/               energy per step, one column per trial
  AddInfo/                    time and seed per trial
```

Types, columns and a format caveat: [run-output.md](docs/run-output.md).

## Documentation

| Document | Contents |
|----------|----------|
| [docs/framework.md](docs/framework.md) | Modification maps, bit flip and city swap, the SA and DA chains, convergence, incremental update |
| [docs/configuration.md](docs/configuration.md) | PBF input objects, every `pbf_min_solver` argument, `type_alg`, cooling schedules, escape rate |
| [docs/run-output.md](docs/run-output.md) | Return values and the files written with `save_csv=True` |
| [docs/problems.md](docs/problems.md) | Supported problems, graph partitioning, optional dependencies, running the scripts, benchmark data |
| [docs/stationary-distribution.md](docs/stationary-distribution.md) | The exact toolkit for $\pi^{SA}$, $\pi^{DA}$ and their difference |
| [docs/findings.md](docs/findings.md) | Preliminary findings: high-temperature law, temperature scale, speed-up ramp, parameter choice |
| [docs/performance.md](docs/performance.md) | Benchmarks, step-time overhead, scalability, comparison with Fujitsu's hardware |
| [docs/references.md](docs/references.md) | Papers, Fujitsu sources, benchmark data, libraries |
| [Masterarbeitspräsentation.pdf](Masterarbeitspr%C3%A4sentation.pdf) | The thesis presentation (German) |

## Project structure

| Path | Purpose |
|------|---------|
| `Code/Funcs_Annealers.py` | SA, DA, DA_old, DA_parallel, DA_TSP and SCA annealing loops |
| `Code/Funcs_Annealing2.py` | Delta-energy evaluation, PBF helpers, cooling schedules |
| `Code/Funcs_Optimizers.py` | `pbf_min_solver` and `GraphPartitioning` |
| `Code/Funcs_pbfGenerators.py`, `Funcs_TSP.py`, `Funcs_lkh.py`, `tsp.py` | Problem encoders and TSP helpers |
| `Code/TensorAnnealing.py` | TensorFlow variant for dense QUBOs |
| `Code/Skript_Stationary_Distribution.py` | Stationary distribution toolkit |
| `Code/Skript_*.py` | Experiment scripts, one per problem |
| `Code/Visualize_Evaluation.py`, `MakeBoxplot.py`, `pythonVideoMaker.py`, `Test_*.py` | Plotting and tests |
| `Code/DATA Thesis/`, `Code/Data.csv`, `Code/Data_4.csv`, `Code/data_thesis_SAT.txt` | Raw results used in the thesis |
| `3SAT_DATA/`, `TSP_data/` | SATLIB and TSPLIB benchmark instances |
| `docs/` | Documentation; figures in `docs/img/` |
| `Boxplot.py`, `Timingplot.py`, `Aggregiert.csv` | Scripts for the two benchmark figures, raw step-time data |
| `Test_symb_bib_stat_distrib.py` | sympy toy example: DA stationary distribution for two variables |
| `Masterarbeitspräsentation.pdf` | Thesis presentation |

## License

MIT — see [LICENSE](LICENSE).
