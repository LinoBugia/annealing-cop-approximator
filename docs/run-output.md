# Run output

What `pbf_min_solver` returns, and which files it writes with `save_csv=True`.

## Return values

```python
Min_varAssignements, Mins, Trajectories, result_List = pbf_min_solver(...)
```

| Return value | Type | Content |
|---|---|---|
| `Min_varAssignements` | `list[list[int]]` (values 0/1) | best assignment found in each Monte-Carlo trial — **only filled if `save_addinfo=True`**, otherwise `[]` |
| `Mins` | `list[float]` | minimum energy reached in each trial |
| `Trajectories` | `list[list[float]]` | energy of the start state and after every step (`steps + 1` entries), one list per trial |
| `result_List` | `list` | only used for `Ising=True` (spin trajectories); otherwise `[]` |

## Files written with `save_csv=True`

Every call appends one line to
`Code/Runs/Evaluation_<date>/Evaluation_<date>.csv` and writes two per-run
files. The run `ID` is the running number of files in `Trajectories/`.

| File | Columns |
|---|---|
| `Evaluation_<date>.csv` | `type_alg, ID, time` (sum of trial wall-clock), `timegen1, timegen2` (init times), `bestMin, seed_gen, variables, degree, monomials_pbf, tpye_cooling, T_start, T_end` (`cooling_param[1]`, `[2]`) |
| `Trajectories/Evaluation_<date>_<ID>_trajectories.csv` | one column `MC<i>` per trial, energy per step |
| `AddInfo/Evaluation_<date>_<ID>_addinfo.csv` | `Exec` (seconds per trial), `rand_seed` (per trial), `Min_var` (if `save_addinfo`) |

With `visual_inst=True` a Plotly figure of all trajectories is shown and, if
`save_csv`, saved as HTML next to the CSV.

> `Code/Visualize_Evaluation.py` does not read this format. It opens
> `Evaluation_2025-03-25.csv` (not in the repository) with `skiprows=5` and
> expects the columns `vars`, `degree`, `seed`, `Min` and `Type`; the summary file
> above names them `variables`, `degree`, `seed_gen`, `bestMin` and `type_alg`, and
> `skiprows=5` drops its header and first four rows. `Runs/` is gitignored.
