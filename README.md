# annealing-cop-approximator

> Implementations of Simulated Annealing, Digital Annealing, and related algorithms for solving combinatorial optimization problems encoded as Pseudo-Boolean Functions — plus an exact toolkit for studying the stationary distributions of both Markov chains.

This repository contains the practical implementation component of the Master's thesis **"Analyse und Implementierung der Optimierungsheuristik Digital Annealing"** (OTH Regensburg, 2025). It benchmarks classical and quantum-inspired annealing heuristics on a range of NP-hard combinatorial optimization problems, and it has since become the testbed for an ongoing theoretical comparison of the two chains (see [Preliminary findings about DA](#preliminary-findings-about-da)).

---

## Table of Contents

- [Overview](#overview)
- [Mathematical Framework](#mathematical-framework)
- [Algorithms](#algorithms)
- [Cooling Schedules](#cooling-schedules)
- [Supported Problems](#supported-problems)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Stationary Distribution Toolkit](#stationary-distribution-toolkit)
- [Preliminary Findings about DA](#preliminary-findings-about-da)
- [Scalability & Hardware Context](#scalability--hardware-context)
- [References](#references)

---

## Overview

All optimization problems in this project are encoded as **Pseudo-Boolean Functions (PBFs)**:

$$P : \mathbb{F}_2^n \to \mathbb{R}, \quad P(x) = \sum_{\substack{S \subseteq \{1,\ldots,n\} \\ |S| \le d}} a_S \prod_{k \in S} x_k$$

where $d$ is the degree of the PBF. This general representation covers both **QUBO** ($d \le 2$) and **PUBO** ($d > 2$, e.g. 3-SAT). The goal in all cases is to find the binary assignment $x^* \in \mathbb{F}_2^n$ that minimizes $P(x)$. In code, a PBF is a Python `dict` from monomial tuples to coefficients — see [Core API](#core-api).

---

## Mathematical Framework

### The framework: modification maps

Every algorithm here is an iterative method on a finite state space $M$ with objective $f : M \to \mathbb{R}$. What it moves along is a **modification map**. This formulation is the author's own, introduced in the thesis so that bit-flip moves on PBFs and tour-based moves on the TSP can be treated in one framework:

> **Definition.** Let $(M, f)$ be an instance of a combinatorial optimization problem. A map
> $$\varphi : M \times K \to M, \qquad (x, k) \mapsto \varphi_k(x)$$
> with a finite **control set** $K$ is a *modification map* if every state can be reached from every other by a finite sequence of modifications: for all $x \ne y$ in $M$ there are $k_1, \dots, k_m \in K$ with $x = \varphi_{k_1}(\cdots\varphi_{k_m}(y))$.

The quantity that drives the search is the objective difference of one modification,

$$\Delta f_k(x) := f(\varphi_k(x)) - f(x),$$

and a modification is only useful if $\Delta f_k$ can be evaluated cheaply. Two properties of the chains below follow from the definition alone: they are **irreducible** (any two states are joined by a sequence of modifications, each of positive probability at $T > 0$), which makes the stationary distribution unique, and **aperiodic** (a state whose modifications are all rejected returns to itself in one step), which makes the chain converge to it.

### The modifications used here

**Bit flip on $\mathbb{F}_2^n$.** $K = \{1, \dots, n\}$ and

$$\theta_k(x) = (x_1, \dots, x_{k-1},\ x_k \oplus 1,\ x_{k+1}, \dots, x_n)^T, \qquad \oplus = \text{addition in } \mathbb{F}_2 .$$

Any state is reached in at most $n$ flips, so $\theta$ is a modification map. For a PBF the difference only involves the monomials that contain $k$:

$$\Delta E_k(x) := P(\theta_k(x)) - P(x) = (-1)^{x_k} \sum_{S \ni k} a_S \prod_{i \in S \setminus \{k\}} x_i ,$$

which costs $O(\lvert\text{support}(k)\rvert)$ per flip (`Eval_Delta_Energy` in `Funcs_Annealing2.py`). This is why the implementation works on the PBF dictionary and never needs a dense matrix.

**City swap on tours (TSP).** For a tour $t = (t_1, \dots, t_n)$, $K$ is the set of city pairs $(a, b)$ with $a \ne b$, and $\varphi_{(a,b)}(t)$ exchanges the positions of the two cities. For non-adjacent $a, b$ only eight distances change,

$$\Delta f_{(a,b)}(t) = -d_{a-1,a} - d_{a,a+1} - d_{b-1,b} - d_{b,b+1} + d_{a,b-1} + d_{a,b+1} + d_{a-1,b} + d_{a+1,b}$$

(indices are tour neighbours, cyclically), and $|K| = \binom{n}{2}$. In the permutation-matrix QUBO encoding a city swap changes exactly four bits; that is the move `digitalAnnealing_TSP` implements. The edge swap (2-opt, $\Delta f = -d_{a_1 a_2} - d_{b_1 b_2} + d_{a_1 b_2} + d_{a_2 b_1}$) is available in `Funcs_TSP.py`.

**Why the control set matters for DA.** SA evaluates one $\Delta f_k$ per step, DA evaluates all $|K|$ of them: a factor $n$ for bit flips, $\binom{n}{2}$ for city swaps. DA therefore needs encodings whose control set scales well and whose differences are cheap — the reason this codebase works on PBFs with bit flips, and the origin of the per-step overhead measured in [Results](#results).

### Simulated Annealing (SA)

SA is the Metropolis chain over a modification map: draw $k \in K$ uniformly at random and accept the move with

$$\alpha(\Delta f_k) = \min\left(1,\ e^{-\Delta f_k(x) / T}\right).$$

Throughout this repository $M = \mathbb{F}_2^n$, $f = P$ and $\varphi = \theta$, so $\Delta f_k = \Delta E_k$. At constant $T$ the chain is reversible and its stationary distribution is the **Boltzmann–Gibbs distribution**; under a sufficiently slow logarithmic cooling schedule it concentrates on the global minima (Hajek 1988):

$$\pi^G(x) = \frac{e^{-P(x)/T}}{Z}, \quad Z = \sum_{x'} e^{-P(x')/T}$$

### Digital Annealing (DA)

DA evaluates **all** $\Delta f_k$, $k \in K$, in one step, applies the Metropolis test to each of them independently, and then picks one of the accepted modifications uniformly at random. For bit flips the transition probability is

$$P^{DA}(x, \theta_k(x)) = \sum_{\substack{S \subseteq K : \\ k \in S}} \frac{1}{|S|} \prod_{i \in S} e^{-\Delta E_i(x)^+ / T} \prod_{i \notin S} \left(1 - e^{-\Delta E_i(x)^+ / T}\right), \qquad \Delta E_i^+ = \max(0, \Delta E_i),$$

and $P^{DA}(x, x) = \prod_i \big(1 - e^{-\Delta E_i(x)^+ / T}\big)$ when nothing is accepted. DA does **not** satisfy detailed balance, so its stationary distribution $\pi^{DA}$ differs from $\pi^G$ — by how much, and in which direction, is the subject of the [findings section](#preliminary-findings-about-da). The implementation additionally uses a **dynamic energy offset** (escape mechanism): a value $E_{\text{off}}$, initially 0, is *subtracted* from every $\Delta E_i$ before the acceptance test. Whenever no flip is accepted in a step, $E_{\text{off}}$ grows by `offset_increase_rate`; as soon as a flip is accepted it is reset to 0. This turns the exponentially long waiting time at a local minimum into a linear one.

**Convergence.** Without the escape mechanism the DA chain converges to the global minima under a logarithmic cooling schedule — the analogue of Hajek's theorem for SA; this was proved by Fukushima-Kimura et al. (2023) for the parallel-trial chain. With the escape mechanism the process is still a Markov chain, on the extended and still finite state space of pairs (state, number of consecutive idle steps), but Hajek's guarantee presumably does not survive: at a fixed offset rate the chain concentrates, as $T \to 0$, on $\arg\min\,[E + g_{\min}]$ rather than on $\arg\min E$ (see [findings, point 4](#4-what-this-suggests-for-choosing-annealing-parameters)). What replaces the guarantee is the subject of ongoing research.

### Incremental delta-energy update

After a flip of bit $k$ is accepted, the vector $(\Delta E_i)_{i \in K}$ is not recomputed: only the entries $i$ that share a monomial with $k$ change, and each changes by a local correction (see `Masterarbeitspräsentation.pdf`, p. 53–55). This incremental update is one of the core efficiency contributions of this codebase; it keeps DA's per-step cost at the measured 3–5× of SA instead of a factor $n$.

---

## Algorithms

The algorithm is selected with the `type_alg` string of `pbf_min_solver`:

| `type_alg` | Implementation | Description |
|---|---|---|
| `"simulatedAnnealing"` | `simulatedAnnealing` | Metropolis-Hastings with single bit-flip proposals. Converges to $\pi^G$. |
| `"digitalAnnealing"` | `DigitalAnnealing` | **Main benchmark algorithm**: parallel trial of all $n$ flips, incremental $\Delta E$ update, energy-offset escape mechanism. |
| `"digitalAnnealing_old"` | `DigitalAnnealing_old` | Earlier DA loop without the incremental update; kept for reference/benchmarks. |
| `"digitalAnnealing_parallel"` | `DigitalAnnealing_parallel` | DA with Python `multiprocessing` for the flip evaluation (8 worker processes). |
| `"digitalAnnealing_TSP"` | `digitalAnnealing_TSP` | DA on the TSP permutation encoding: moves are 4-bit swaps that keep the tour constraint satisfied. |
| `"scaAnnealing"` | `SCA_Annealing` | Stochastic Cellular Automata: simultaneous update of all bits per step, with a second schedule `Qs` for the pinning term. |

`TensorAnnealing.py` is a separate TensorFlow implementation for **dense QUBOs**: all delta energies of a step come out of one matrix-vector product $Qx$ on the GPU. It is not reached through `pbf_min_solver`.

---

## Cooling Schedules

`cooling_param` is a list whose first entry names the schedule; the remaining entries are its parameters. `Generate_Cooling_Schedule(cooling_param, steps)` in `Funcs_Annealing2.py` expands it to a list of temperatures $T_1, \dots, T_{\text{steps}}$. The formulas below are **what the code computes** ($t$ = step index, starting at 1):

| `cooling_param` | Formula | Notes |
|---|---|---|
| `["constant", C]` | $T_t = C$ | fixed temperature (the regime the stationary-distribution toolkit analyses) |
| `["linear", T_start, T_end]` | $T_t = T_{\text{start}} + t\,(T_{\text{end}}-T_{\text{start}})/\text{steps}$ | |
| `["logarithmic", c, 0]` | $T_t = c \,/\, \log\!\big(1 + t^{2.22}\big)$ | Hajek-type schedule with a fixed exponent $d = 2.22$; the third entry is not used by the schedule but is written to the CSV as `T_end`, so keep it when `save_csv=True` |
| `["logarithmic_step", c, k]` | as above, each value repeated $k$ times | plateaus of length $k$ |
| `["exponential", c]` | $T_t = e^{t/c} - 1$ | **rising** in $t$ — a heating schedule, not a cooling one |
| `["rising", T_start]` | $T_t = T_{\text{start}} \cdot t$ | used internally for the SCA pinning schedule `Qs` |
| `["auto_sa", steps_init, steps_avg, delta]` | adaptive | finite-time schedule after Aarts & Korst (*Simulated Annealing and Boltzmann Machines*): the initial temperature is calibrated on `steps_init` random flips; the solver appends `pbf`, `pbf_var_dict` and the start assignment to the list itself. Renamed from `"auto"` in this version. |
| `["da_gp", n_steps, None, lloyd_assign, E_lloyd, E_start, p_hot, p_cold, gamma]` | calibrated logarithmic | **experimental**, for DA warm-started from a graph-partition (Lloyd) solution: $c$ is calibrated from the median and lower quartile of the uphill $\Delta E$ at the start state, with the correction $T = \Delta E/\ln(n/p)$ for DA testing all $n$ flips per step. Slot 2 is filled by the solver. Defaults `p_hot=0.3, p_cold=0.01, gamma=0.05`. |
| `"auto_da"`, `"stepfunction"`, `"linear&logarithmic"` | — | reserved / not implemented — selecting them fails |

The escape mechanism is controlled separately by `offset_increase_rate`: the growth of $E_{\text{off}}$ per idle step as a number, `0` to switch the mechanism off, or `"auto_gp"` to set it to the median uphill $\Delta E$ at the start state divided by 25 (experimental, belongs to the `da_gp` work).

---

## Supported Problems

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

### Graph partitioning parameters

`GraphPartitioning(coords, parts, num_MC_GP=1, diff_border=1, steps=200)` in `Funcs_Optimizers.py` splits a point set recursively into $2^{\text{parts}}$ groups by solving a balanced-bisection QUBO with DA at each level:

| Parameter | Meaning |
|---|---|
| `coords` | list of 2-D points |
| `parts` | number of bisection levels → $2^{\text{parts}}$ groups |
| `num_MC_GP` | DA trials per bisection; the best is kept |
| `diff_border` | tolerated size imbalance between the two halves; larger imbalances are repaired by moving the closest points across |
| `steps` | DA steps per bisection (schedule `["logarithmic", 2·N/(level+1), 0]`, offset rate 1000) |

---

## Repository Structure

```
annealing-cop-approximator/
│
├── Code/                              # All Python source code
│   │
│   ├── Core algorithms
│   │   ├── Funcs_Annealers.py             # SA, DA, DA_old, DA_parallel, DA_TSP, SCA annealing loops
│   │   ├── Funcs_Annealing2.py            # Delta-E evaluation, PBF helpers, cooling schedules
│   │   ├── Funcs_Optimizers.py            # pbf_min_solver (Monte-Carlo orchestration, CSV export), GraphPartitioning
│   │   └── TensorAnnealing.py             # TensorFlow implementation for dense QUBOs
│   │
│   ├── Stationary distribution toolkit
│   │   └── Skript_Stationary_Distribution.py   # exact pi^SA, pi^DA, delta and its approximations (n <= 9)
│   │
│   ├── PBF / QUBO generators
│   │   ├── Funcs_pbfGenerators.py         # 3-SAT, Ising, NPP, clustering → PBF
│   │   ├── Funcs_TSP.py                   # TSP encoding, 2-opt, greedy, visualization
│   │   ├── Funcs_lkh.py                   # Lin-Kernighan-Helsgaun heuristic wrapper
│   │   └── tsp.py
│   │
│   ├── Experiment scripts
│   │   ├── Skript_Solve_3SAT.py
│   │   ├── Skript_Solve_Traveling_Salesman.py
│   │   ├── Skript_Routing.py
│   │   ├── Skript_Solve_NumberPartitioning.py
│   │   ├── Skript_Solve_GraphPartitioning.py
│   │   ├── Skript_Solve_Integer_Programm.py
│   │   ├── Skript_Solve_Random_Generated_pbf.py
│   │   ├── Skript_IsingModel_Simulator.py
│   │   ├── Skript_Solve_GTP.py
│   │   └── Skript_Join_Ordering.py
│   │
│   ├── Visualization & tests
│   │   ├── MakeBoxplot.py                 # Join ordering result boxplots (Plotly)
│   │   ├── Visualize_Evaluation.py        # Runtime vs. objective scatter plots
│   │   ├── pythonVideoMaker.py
│   │   └── Test_Memory_Usage*.py, Test_Permutationmatrix.py
│   │
│   ├── DATA Thesis/                       # raw result tables used in the thesis
│   ├── Runs/                              # created at runtime by save_csv=True (gitignored)
│   └── Data.csv, Data_4.csv, data_thesis_SAT.txt
│
├── 3SAT_DATA/                         # SATLIB benchmark instances (.cnf)
├── TSP_data/                          # TSPLIB instances
├── Boxplot.py                         # DA vs. SA boxplot (matplotlib)
├── Test_symb_bib_stat_distrib.py      # sympy toy example: DA stationary distribution for 2 variables
├── Aggregiert.csv
├── Masterarbeitspräsentation.pdf      # Master's thesis presentation (primary documentation)
├── Iterationbenchmarks.png
├── grouped_boxplot_da_sa.png
└── grouped_boxplot_da_sa_log.png
```

---

## Installation

Install dependencies manually:

```bash
pip install numpy pandas matplotlib plotly sympy p_tqdm Pillow
```

For GPU-accelerated tensor annealing:
```bash
pip install tensorflow
```

For the Lin-Kernighan TSP heuristic:
```bash
pip install lk_heuristic
```

For route planning on real road networks (`Skript_Routing.py`):
```bash
pip install routingpy
```

For join ordering (`Skript_Join_Ordering.py`; also needs the two generator modules from [lfd/vldb24](https://github.com/lfd/vldb24)):
```bash
pip install docplex dimod qiskit-optimization
```

> **Note:** Several scripts contain hardcoded absolute file paths from the author's machine (`Boxplot.py`, `Funcs_Annealing2.py`, `Skript_IsingModel_Simulator.py`, `Skript_Join_Ordering.py`, `Skript_Solve_3SAT.py`, `Skript_Solve_GraphPartitioning.py`, `Skript_Solve_NumberPartitioning.py`, `Skript_Solve_Traveling_Salesman.py`). Adjust these to your local setup before running.

---

## Usage

All scripts are run from within the `Code/` directory, or from the repository root for root-level scripts.

### Run an annealing experiment

```bash
# From Code/
python Skript_Solve_3SAT.py
python Skript_Solve_NumberPartitioning.py
python Skript_Solve_Traveling_Salesman.py
python Skript_Solve_Integer_Programm.py
python Skript_Solve_Random_Generated_pbf.py
python Skript_IsingModel_Simulator.py
```

### Core API

The main entry point for programmatic use is `pbf_min_solver` in `Funcs_Optimizers.py`:

#### Input: the PBF dictionary

A PBF is a Python `dict` mapping **monomial tuples** to real coefficients. Variables are integers `0 … n-1`; a tuple lists the variables of one monomial in ascending order:

```python
pbf = {
    (0,):      -1.5,   # linear term for variable 0
    (1,):      -2.0,   # linear term for variable 1
    (0, 1):     1.0,   # quadratic coupling between 0 and 1
    (0, 1, 2):  0.5,   # cubic term (PUBO)
}
```

Two derived objects accompany every PBF:

| Object | Type | Built by | Meaning |
|---|---|---|---|
| `pbf_var_dict` | `dict[int, tuple]` | `createPolyDict(pbf, n)` in `Funcs_Annealing2.py` | for each variable `k`, the monomials that contain `k` — this is what makes the delta-energy evaluation $O(\lvert\text{support}(k)\rvert)$ |
| `varAssignment` | `dict[int, bool]` | `getInitialVarAssignement(pbf, seed)` or passed in via `initial_varAssignement_pre=[...]` | the current binary state |

`Sort_pbf(pbf, degree)` normalises the key order; `createPoly(n, degree, density, seed)` and `createPoly_negative(...)` generate random instances with random coefficient signs. Problem-specific encoders live in `Funcs_pbfGenerators.py` and `Funcs_TSP.py`.

#### Example call

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
    type_alg="digitalAnnealing",           # see the Algorithms table
    steps=10000,
    num_MC=num_MC,
    cooling_param=["logarithmic", 100, 0], # see the Cooling Schedules table
    seed_rand=list(range(num_MC)),         # one RNG seed per trial
    seed_gen_initial_varAssignment=42,     # seed of the random start state
    offset_increase_rate=0,                # escape mechanism off
    save_addinfo=True,                     # needed to get Min_varAssignements back
    save_csv=False,
)
best = Min_varAssignements[int(np.argmin(Mins))]
```

Further keyword arguments: `initial_varAssignement_pre` (start from a given state), `Ising=True` (spin-trajectory output), `visual_inst=True` (Plotly figure of all trajectories), and for the TSP variant `inv_trans_dict`, `trans_dict`, `coords`, `tour`.

#### Return values

```python
Min_varAssignements, Mins, Trajectories, result_List = pbf_min_solver(...)
```

| Return value | Type | Content |
|---|---|---|
| `Min_varAssignements` | `list[list[bool]]` | best assignment found in each Monte-Carlo trial — **only filled if `save_addinfo=True`**, otherwise `[]` |
| `Mins` | `list[float]` | minimum energy reached in each trial |
| `Trajectories` | `list[list[float]]` | energy after every step, one list per trial |
| `result_List` | `list` | only used for `Ising=True` (spin trajectories); otherwise `[]` |

#### Run metadata written to disk (`save_csv=True`)

Every call appends one line to `Code/Runs/Evaluation_<date>/Evaluation_<date>.csv` and writes two per-run files. The run `ID` is the running number of files in `Trajectories/`.

| File | Columns |
|---|---|
| `Evaluation_<date>.csv` | `type_alg, ID, time` (sum of trial wall-clock), `timegen1, timegen2` (init times), `bestMin, seed_gen, variables, degree, monomials_pbf, tpye_cooling, T_start, T_end` (`cooling_param[1]`, `[2]`) |
| `Trajectories/Evaluation_<date>_<ID>_trajectories.csv` | one column `MC<i>` per trial, energy per step |
| `AddInfo/Evaluation_<date>_<ID>_addinfo.csv` | `Exec` (seconds per trial), `rand_seed` (per trial), `Min_var` (if `save_addinfo`) |

With `visual_inst=True` a Plotly figure of all trajectories is shown and, if `save_csv`, saved as HTML next to the CSV.

### Generate plots

```bash
# From repository root
python Boxplot.py
```

---

## Results

Benchmark experiments comparing SA and DA on the Number Partitioning Problem (300–1700 variables):

![Grouped boxplot of DA vs SA](grouped_boxplot_da_sa_log.png)

![Iteration Benchmarks](Iterationbenchmarks.png)

Key observations:
- DA is approximately **3–5× slower per step** than SA in this Python implementation: all $n$ acceptance tests per step plus the incremental update of the delta vector, against a single evaluation for SA.
- Despite the per-step overhead, DA shows competitive or superior solution quality on the tested instances.
- The findings below explain part of this: at the temperatures where annealing does its work, DA's Markov chain relaxes up to $n$ times faster than SA's.

---

## Stationary Distribution Toolkit

`Code/Skript_Stationary_Distribution.py` is a library plus a plot script that computes, **without simulating anything**, the stationary distributions of both chains at a constant temperature, by enumerating the state space ($2^n \le$ `max_states`, i.e. $n \le 9$ by default):

- $\pi^{SA}$ — the Boltzmann distribution, exactly (`boltzmann_distribution`);
- $\pi^{DA}$ — the left eigenvector of the closed-form DA transition matrix `da_transition_matrix`, whose entries are $p_k \cdot R(x,k)$ with $R(x,k) = \mathbb{E}[1/(|S|+1)]$ summed over all $2^{n-1}$ subsets (`R_neighbour`, formula (3.6) of Fukushima-Kimura et al.);
- $\delta = \pi^{DA} - \pi^{SA}$ and several analytic approximations of it, each with its $R^2$ against the exact value:
  - `delta_high_T` — the closed-form high-temperature law (below);
  - `delta_first_order_SA` / `delta_series_SA` — first and higher orders of the perturbation series around the SA chain, together with the spectral radius that tells where the series stops converging;
  - `delta_edge_flux` — a purely local approximation;
  - `delta_at_minimum` — the closed form at the global minimum for unfrustrated degree-2 instances;
- landscape quantities used by these formulas: `uphill_sum` ($G^+$), `phi_potential` ($\Phi = E + G^+$), `ratio_kernel` and `defect_source` (the bounded kernel of $\pi^{DA}/\pi^{SA}$), `frustrated_directions`;
- exact mean hitting times `exact_hit_time_matrix`.

`run_stationary_experiment(pbf, variables, temperatures)` returns the energy vector and a per-temperature result dictionary; `Plot_Stationary_Distributions` renders one column per temperature (row 1: the distributions, row 2: $\delta$ and its approximations, hover shows $E$, $\Phi$, $J$, hitting times). Parameters are set at the top of the `__main__` block: `variables`, `degree`, `max_states`, `seed_gen`, `temperatures`, `negative_couplings`.

---

## Scalability & Hardware Context

### This implementation

The implementation works directly on the **PBF dictionary** — no dense matrix is required. After a bit-flip at position $k$, only monomials containing $k$ are re-evaluated, and the delta-energy vector for all other variables is updated **incrementally**. For sparse problems (3-SAT, graph-structured QUBOs) this scales far better than dense matrix approaches and keeps memory and arithmetic close to the theoretical minimum.

The TensorFlow path (`TensorAnnealing.py`) maps the dense QUBO case — a matrix-vector product $Qx$ per step — onto GPU tensor cores.

### Fujitsu's Digital Annealer

The algorithm implemented here is the one Fujitsu ships: parallel trial of all bit-flips per step, uniform choice among the accepted ones, and the energy-offset escape mechanism (Aramon et al. 2019). How that algorithm is *executed* has changed over the product's generations, and with it the meaning of a comparison with this repository:

- **Dedicated CMOS hardware, up to 8,192 bits.** The first generation (2018) was a 1,024-bit chip; the second-generation Digital Annealer Unit (late 2018) an 8,192-bit fully connected CMOS ASIC that evaluates all flips in parallel with the coupling matrix on-chip. That is where the per-step speed advantage over any software implementation came from.
- **Beyond the chip: software on linked servers.** The problem sizes advertised since — 100,000 variables in the fourth generation (2022), and a demonstrated 1,014,000-bit scheduling problem in 2020 — are not reached by a larger chip. Fujitsu's own announcement describes "a technology to solve large-scale problems with multiple linked servers while ensuring consistency in overall solutions", and the service documentation notes that problem scaling and energy recalculation run on CPUs.

For the large-scale service the comparison is therefore algorithmic, not architectural: the same update rule and the same escape mechanism, hence comparable solution quality per step. What Fujitsu adds is an engineered software stack around it (vectorised evaluation, replica exchange, automatic parameter selection, distribution across servers); what this repository adds is the incremental delta-energy update for sparse PBFs. We have not benchmarked head-to-head against the service — the claim is algorithmic equivalence, not measured parity.

---

## References

### Papers

- Aramon, M., Rosenberg, G., Valiante, E., Miyazawa, T., Tamura, H., Katzgraber, H.G. (2019). *Physics-Inspired Optimization for Quadratic Unconstrained Problems Using a Digital Annealer.* Frontiers in Physics 7:48. [DOI: 10.3389/fphy.2019.00048](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2019.00048) — the Digital Annealer algorithm, parallel trial and escape mechanism.
- Fukushima-Kimura, B.H., Kawamoto, N., Noda, E., Saburi, K., Tasaki, H. (2023). *Mathematical aspects of the digital annealer's simulated annealing algorithm.* Journal of Statistical Physics. [DOI: 10.1007/s10955-023-03179-3](https://link.springer.com/article/10.1007/s10955-023-03179-3) — [arXiv:2303.08392](https://arxiv.org/abs/2303.08392) — closed-form DA transition probabilities (formula (3.6) used in the stationary distribution toolkit) and the proof that the parallel-trial DA chain without escape mechanism converges under logarithmic cooling (Hajek-type result).
- Kirkpatrick, S., Gelatt, C.D., Vecchi, M.P. (1983). *Optimization by Simulated Annealing.* Science.
- Hajek, B. (1988). *Cooling Schedules for Optimal Annealing.* Mathematics of Operations Research.
- Aarts, E., Korst, J. (1989). *Simulated Annealing and Boltzmann Machines.* Wiley — finite-time schedule behind `"auto_sa"`.

### Fujitsu Digital Annealer

- Fujitsu press release, 15 May 2018: [Fujitsu Quantum-Inspired Digital Annealer Cloud Service](https://info.archives.global.fujitsu/global/about/resources/news/press-releases/2018/0515-01.html) — first generation, 1,024 bits.
- Fujitsu press release, 21 Dec 2018: [Next Generation Quantum-Inspired Digital Annealer Service](https://info.archives.global.fujitsu/global/about/resources/news/press-releases/2018/1221-01.html) — second generation, 8,192 bits, CMOS ASIC.
- Fujitsu press release, 9 Nov 2020: [Megabit-class Performance for Large-Scale Combinatorial Optimization Problems](https://info.archives.global.fujitsu/global/about/resources/news/press-releases/2020/1109-01.html) — 1,014,000-bit problem solved with multiple linked servers.
- [Fujitsu Computing as a Service — Digital Annealer User's Guide](https://portal.aispf.global.fujitsu.com/apidoc/da/jp/da-guide-en.html) — current service; CPU-side processing for scaling and energy recalculation.
- [Fujitsu — overview at postquantum.com](https://postquantum.com/quantum-computing-companies/fujitsu/) — fourth generation (2022), 100,000 variables (secondary source).

### Benchmark Data

- SATLIB Benchmark Suite (3-SAT instances): https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html
- TSPLIB: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
- Dr. Manuel Schönberger, *Join Ordering via Quantum-Inspired Optimization.* VLDB 2024. https://github.com/lfd/vldb24

### Libraries & Tools Used

- `lk_heuristic` — Lin-Kernighan-Helsgaun TSP heuristic: https://github.com/pfayle/lk_heuristic
- `p_tqdm` — Parallel tqdm progress bars: https://github.com/swansonk14/p_tqdm
