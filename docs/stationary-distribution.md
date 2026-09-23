# Stationary distribution toolkit

How `Code/Skript_Stationary_Distribution.py` computes both chains' stationary
distributions exactly, and which approximations of their difference it offers.

`Code/Skript_Stationary_Distribution.py` is a library plus a plot script that
computes, **without simulating anything**, the stationary distributions of both
chains at a constant temperature, by enumerating the state space ($2^n \le$
`max_states`, i.e. $n \le 9$ by default):

- $\pi^{SA}$ — the Boltzmann distribution, exactly (`boltzmann_distribution`);
- $\pi^{DA}$ — the left eigenvector of the closed-form DA transition matrix
  `da_transition_matrix`, whose entries are $p_k \cdot R(x,k)$ with
  $R(x,k) = \mathbb{E}[1/(|S|+1)]$ summed over all $2^{n-1}$ subsets
  (`R_neighbour`, formula (3.6) of
  [Fukushima-Kimura et al.](references.md#papers));
- $\delta = \pi^{DA} - \pi^{SA}$ and several analytic approximations of it, each
  with its $R^2$ against the exact value:
  - `delta_high_T` — a closed-form first-order law for high temperatures;
  - `delta_first_order_SA` / `delta_series_SA` — first and higher orders of the
    perturbation series around the SA chain, together with the spectral radius
    that tells where the series stops converging;
  - `delta_edge_flux` — a purely local approximation;
  - `delta_at_minimum` — the closed form at the global minimum for unfrustrated
    degree-2 instances;
- landscape quantities used by these formulas: `uphill_sum` ($G^+$),
  `phi_potential` ($\Phi = E + G^+$), `ratio_kernel` and `defect_source` (the
  bounded kernel of $\pi^{DA}/\pi^{SA}$), `frustrated_directions`;
- exact mean hitting times `exact_hit_time_matrix`.

`run_stationary_experiment(pbf, variables, temperatures)` returns the energy
vector and a per-temperature result dictionary; `Plot_Stationary_Distributions`
renders one column per temperature (row 1: the distributions, row 2: $\delta$
and its approximations, hover shows $E$, $\Phi$, $J$, hitting times). Parameters
are set at the top of the `__main__` block: `variables`, `degree`, `max_states`,
`seed_gen`, `temperatures`, `negative_couplings`.
