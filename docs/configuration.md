# Configuration

Everything `pbf_min_solver` accepts: the PBF input objects, the remaining
keyword arguments, the algorithm switch `type_alg`, the cooling schedules and
the escape-mechanism rate.

- [Input: the PBF dictionary](#input-the-pbf-dictionary)
- [Further arguments](#further-arguments)
- [Algorithms](#algorithms)
- [Cooling schedules](#cooling-schedules)

## Input: the PBF dictionary

A PBF is a Python `dict` mapping **monomial tuples** to real coefficients.
Variables are integers `0 … n-1`; a tuple lists the variables of one monomial in
ascending order:

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
| `varAssignment` | `dict[int, int]` (values 0/1) | `getInitialVarAssignement(pbf, seed)` or passed in via `initial_varAssignement_pre=[...]` | the current binary state |

`Sort_pbf(pbf, degree)` normalises the key order;
`createPoly(n, degree, density, seed)` and `createPoly_negative(...)` generate
random instances with random coefficient signs. Problem-specific encoders live
in `Funcs_pbfGenerators.py` and `Funcs_TSP.py`.

## Further arguments

An example call with all common arguments is in the
[README](../README.md#core-api).

Further keyword arguments: `initial_varAssignement_pre` (start from a given
state), `Ising=True` (spin-trajectory output), `visual_inst=True` (Plotly figure
of all trajectories), and for the TSP variant `inv_trans_dict`, `trans_dict`,
`coords`, `tour`.

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

`TensorAnnealing.py` is a separate TensorFlow implementation for **dense
QUBOs**: all delta energies of a step come out of one matrix-vector product $Qx$
on the GPU. It is not reached through `pbf_min_solver`.

## Cooling schedules

`cooling_param` is a list whose first entry names the schedule; the remaining
entries are its parameters. `Generate_Cooling_Schedule(cooling_param, steps)` in
`Funcs_Annealing2.py` expands it to a list of temperatures
$T_1, \dots, T_{\text{steps}}$. The formulas below are **what the code
computes** ($t$ = step index, starting at 1):

| `cooling_param` | Formula | Notes |
|---|---|---|
| `["constant", C]` | $T_t = C$ | fixed temperature (the regime the stationary-distribution toolkit analyses) |
| `["linear", T_start, T_end]` | $T_t = T_{\text{start}} + t (T_{\text{end}}-T_{\text{start}})/\text{steps}$ | |
| `["logarithmic", c, 0]` | $T_t = c / \log\big(1 + t^{2.22}\big)$ | Hajek-type schedule with a fixed exponent $d = 2.22$; the third entry is not used by the schedule but is written to the CSV as `T_end`, so keep it when `save_csv=True` |
| `["logarithmic_step", c, k]` | as above, each value repeated $k$ times | plateaus of length $k$ |
| `["exponential", c]` | $T_t = e^{t/c} - 1$ | **rising** in $t$ — a heating schedule, not a cooling one |
| `["rising", T_start]` | $T_t = T_{\text{start}} \cdot t$ | used internally for the SCA pinning schedule `Qs` |
| `["auto_sa", steps_init, steps_avg, delta]` | adaptive | finite-time schedule after Aarts & Korst (*Simulated Annealing and Boltzmann Machines*): the initial temperature is calibrated on `steps_init` random flips; the solver appends `pbf`, `pbf_var_dict` and the start assignment to the list itself. Renamed from `"auto"` in this version. |
| `["da_gp", n_steps, None, lloyd_assign, E_lloyd, E_start, p_hot, p_cold, gamma]` | calibrated logarithmic | **experimental**, for DA warm-started from a graph-partition (Lloyd) solution: $c$ is calibrated from the median and lower quartile of the uphill $\Delta E$ at the start state, with the correction $T = \Delta E/\ln(n/p)$ for DA testing all $n$ flips per step. Slot 2 is filled by the solver. Defaults `p_hot=0.3, p_cold=0.01, gamma=0.05`. |
| `"auto_da"`, `"stepfunction"`, `"linear&logarithmic"` | — | reserved / not implemented — selecting them fails |

The escape mechanism is controlled separately by `offset_increase_rate`: the
growth of $E_{\text{off}}$ per idle step as a number, `0` to switch the
mechanism off, or `"auto_gp"` to set it to the median uphill $\Delta E$ at the
start state divided by 25 (experimental, belongs to the `da_gp` work).
