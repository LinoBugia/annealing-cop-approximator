# Framework

The mathematical model behind every algorithm in this repository: modification
maps, the two moves used here, the SA and DA Markov chains, their convergence,
and the incremental delta-energy update.

- [The framework: modification maps](#the-framework-modification-maps)
- [The modifications used here](#the-modifications-used-here)
- [Simulated Annealing (SA)](#simulated-annealing-sa)
- [Digital Annealing (DA)](#digital-annealing-da)
- [Incremental delta-energy update](#incremental-delta-energy-update)

## The framework: modification maps

Every algorithm here is an iterative method on a finite state space $M$ with
objective $f : M \to \mathbb{R}$. What it moves along is a **modification map**.
This formulation is the author's own, introduced in the thesis so that bit-flip
moves on PBFs and tour-based moves on the TSP can be treated in one framework:

> **Definition.** Let $(M, f)$ be an instance of a combinatorial optimization
> problem. A map
> $$\varphi : M \times K \to M, \qquad (x, k) \mapsto \varphi_k(x)$$
> with a finite **control set** $K$ is a *modification map* if every state can
> be reached from every other by a finite sequence of modifications: for all
> $x \ne y$ in $M$ there are $k_1, \dots, k_m \in K$ with
> $x = \varphi_{k_1}(\cdots\varphi_{k_m}(y))$.

The quantity that drives the search is the objective difference of one
modification,

$$\Delta f_k(x) := f(\varphi_k(x)) - f(x),$$

and a modification is only useful if $\Delta f_k$ can be evaluated cheaply. Two
properties of the chains below follow from the definition alone: they are
**irreducible** (any two states are joined by a sequence of modifications, each
of positive probability at $T > 0$), which makes the stationary distribution
unique, and **aperiodic** (a state whose modifications are all rejected returns
to itself in one step), which makes the chain converge to it.

## The modifications used here

**Bit flip on $\mathbb{F}_2^n$.** $K = \{1, \dots, n\}$ and

$$\theta_k(x) = (x_1, \dots, x_{k-1},\ x_k \oplus 1,\ x_{k+1}, \dots, x_n)^T, \qquad \oplus = \text{addition in } \mathbb{F}_2 .$$

Any state is reached in at most $n$ flips, so $\theta$ is a modification map.
For a PBF the difference only involves the monomials that contain $k$:

$$\Delta E_k(x) := P(\theta_k(x)) - P(x) = (-1)^{x_k} \sum_{S \ni k} a_S \prod_{i \in S \setminus \{k\}} x_i ,$$

which costs $O(\lvert\text{support}(k)\rvert)$ per flip (`Eval_Delta_Energy` in
`Funcs_Annealing2.py`). This is why the implementation works on the PBF
dictionary and never needs a dense matrix.

**City swap on tours (TSP).** For a tour $t = (t_1, \dots, t_n)$, $K$ is the set
of city pairs $(a, b)$ with $a \ne b$, and $\varphi_{(a,b)}(t)$ exchanges the
positions of the two cities. For non-adjacent $a, b$ only eight distances
change,

$$\Delta f_{(a,b)}(t) = -d_{a-1,a} - d_{a,a+1} - d_{b-1,b} - d_{b,b+1} + d_{a,b-1} + d_{a,b+1} + d_{a-1,b} + d_{a+1,b}$$

(indices are tour neighbours, cyclically), and $|K| = \binom{n}{2}$. In the
permutation-matrix QUBO encoding a city swap changes exactly four bits; that is
the move `digitalAnnealing_TSP` implements. The edge swap (2-opt,
$\Delta f = -d_{a_1 a_2} - d_{b_1 b_2} + d_{a_1 b_2} + d_{a_2 b_1}$) is
available in `Funcs_TSP.py`.

**Why the control set matters for DA.** SA evaluates one $\Delta f_k$ per step,
DA evaluates all $|K|$ of them: a factor $n$ for bit flips, $\binom{n}{2}$ for
city swaps. DA therefore needs encodings whose control set scales well and whose
differences are cheap — the reason this codebase works on PBFs with bit flips,
and the origin of the per-step overhead measured in
[performance.md](performance.md#results).

## Simulated Annealing (SA)

SA is the Metropolis chain over a modification map: draw $k \in K$ uniformly at
random and accept the move with

$$\alpha(\Delta f_k) = \min\left(1,\ e^{-\Delta f_k(x) / T}\right).$$

Throughout this repository $M = \mathbb{F}_2^n$, $f = P$ and $\varphi = \theta$,
so $\Delta f_k = \Delta E_k$. At constant $T$ the chain is reversible and its
stationary distribution is the **Boltzmann–Gibbs distribution**; under a
sufficiently slow logarithmic cooling schedule it concentrates on the global
minima (Hajek 1988):

$$\pi^G(x) = \frac{e^{-P(x)/T}}{Z}, \quad Z = \sum_{x'} e^{-P(x')/T}$$

## Digital Annealing (DA)

DA evaluates **all** $\Delta f_k$, $k \in K$, in one step, applies the
Metropolis test to each of them independently, and then picks one of the
accepted modifications uniformly at random. For bit flips the transition
probability is

$$P^{DA}(x, \theta_k(x)) = \sum_{\substack{S \subseteq K : \\ k \in S}} \frac{1}{|S|} \prod_{i \in S} e^{-\Delta E_i(x)^+ / T} \prod_{i \notin S} \left(1 - e^{-\Delta E_i(x)^+ / T}\right), \qquad \Delta E_i^+ = \max(0, \Delta E_i),$$

and $P^{DA}(x, x) = \prod_i \big(1 - e^{-\Delta E_i(x)^+ / T}\big)$ when nothing
is accepted. DA does **not** satisfy detailed balance, so its stationary
distribution $\pi^{DA}$ differs from $\pi^G$ — by how much, and in which
direction, is the subject of [findings.md](findings.md). The implementation
additionally uses a **dynamic energy offset** (escape mechanism): a value
$E_{\text{off}}$, initially 0, is *subtracted* from every $\Delta E_i$ before
the acceptance test. Whenever no flip is accepted in a step, $E_{\text{off}}$
grows by `offset_increase_rate`; as soon as a flip is accepted it is reset to 0.
This turns the exponentially long waiting time at a local minimum into a linear
one.

**Convergence.** Without the escape mechanism the DA chain converges to the
global minima under a logarithmic cooling schedule — the analogue of Hajek's
theorem for SA; this was proved by
[Fukushima-Kimura et al. (2023)](references.md#papers) for the parallel-trial
chain. With the escape mechanism the process is still a Markov chain, on the
extended and still finite state space of pairs (state, number of consecutive
idle steps), but Hajek's guarantee presumably does not survive: at a fixed
offset rate it no longer concentrates on $\arg\min E$ as $T \to 0$ (see
[findings.md, point 4](findings.md#4-what-this-suggests-for-choosing-annealing-parameters)).
What replaces the guarantee is the subject of ongoing research.

## Incremental delta-energy update

After a flip of bit $k$ is accepted, the vector $(\Delta E_i)_{i \in K}$ is not
recomputed: only the entries $i$ that share a monomial with $k$ change, and each
changes by a local correction (see
[`Masterarbeitspräsentation.pdf`](../Masterarbeitspr%C3%A4sentation.pdf), p.
53–55). This incremental update is one of the core efficiency contributions of
this codebase; it keeps DA's per-step cost within a small factor of SA's instead
of a factor $n$ (measurements in [performance.md](performance.md#results)).
