# Framework

The mathematical model behind every algorithm in this repository, built up in
order: modification maps, the two moves used here, the SA and DA Markov chains
with the incremental delta-energy update, their stationary distributions and
convergence, and an outlook on comparing the two chains.

- [The framework: modification maps](#the-framework-modification-maps)
- [The modifications used here](#the-modifications-used-here)
- [Simulated Annealing (SA)](#simulated-annealing-sa)
- [Digital Annealing (DA)](#digital-annealing-da)
- [Incremental delta-energy update](#incremental-delta-energy-update)
- [Stationary distributions and convergence](#stationary-distributions-and-convergence)
- [Outlook: comparing the chains at equal temperature](#outlook-comparing-the-chains-at-equal-temperature)

## The framework: modification maps

Every algorithm here is an iterative method on a finite state space $M$ with
objective $f : M \to \mathbb{R}$. What it moves along is a **modification map**.
This formulation is the author's own, introduced in the thesis so that bit-flip
moves on PBFs and tour-based moves on the TSP can be treated in one framework:

> **Definition.** Let $(M, f)$ be an instance of a combinatorial optimization
> problem. A map
> ```math
> \varphi : M \times K \to M, \qquad (x, k) \mapsto \varphi_k(x)
> ```
> with a finite **control set** $K$ is a *modification map* if every state can
> be reached from every other by a finite sequence of modifications: for all
> $x \ne y$ in $M$ there are $k_1, \dots, k_m \in K$ with
> $x = \varphi_{k_1}(\cdots\varphi_{k_m}(y))$.

The quantity that drives the search is the objective difference of one
modification,

```math
\Delta f_k(x) := f(\varphi_k(x)) - f(x),
```

and a modification is only useful if $\Delta f_k$ can be evaluated cheaply.

**Remark: self-inverse modifications.** The definition deliberately leaves open
whether a modification undoes itself, so that problems such as graph coloring,
whose natural recoloring moves do not, stay inside the framework. Every formula
in this repository, however, assumes *self-inverse* modifications,

```math
\varphi_k(\varphi_k(x)) = x \quad \text{for all } x \in M,\ k \in K,
```

so that the way back from $\varphi_k(x)$ to $x$ is again modification $k$, with
$\Delta f_k(\varphi_k(x)) = -\Delta f_k(x)$. This makes SA's proposal symmetric
and hence its chain reversible. Both moves used here, the bit flip and the city
swap, are self-inverse. The theory can be carried over to modifications that are
not, but the mathematics gets harder.

## The modifications used here

**Bit flip on $\mathbb{F}_2^n$.** $K = \lbrace 1, \dots, n\rbrace$ and

```math
\theta_k(x) = (x_1, \dots, x_{k-1},\ x_k \oplus 1,\ x_{k+1}, \dots, x_n)^T, \qquad \oplus = \text{addition in } \mathbb{F}_2 .
```

Any state is reached in at most $n$ flips, so $\theta$ is a modification map.
For a PBF the difference only involves the monomials that contain $k$:

```math
\Delta E_k(x) := P(\theta_k(x)) - P(x) = (-1)^{x_k} \sum_{S \ni k} a_S \prod_{i \in S \setminus \{k\}} x_i ,
```

which costs $O(\lvert\text{support}(k)\rvert)$ per flip (`Eval_Delta_Energy` in
`Funcs_Annealing2.py`). This is why the implementation works on the PBF
dictionary and never needs a dense matrix.

**City swap on tours (TSP).** For a tour $t = (t_1, \dots, t_n)$, $K$ is the set
of city pairs $(a, b)$ with $a \ne b$, and $\varphi_{(a,b)}(t)$ exchanges the
positions of the two cities. For non-adjacent $a, b$ only eight distances
change,

```math
\Delta f_{(a,b)}(t) = -d_{a-1,a} - d_{a,a+1} - d_{b-1,b} - d_{b,b+1} + d_{a,b-1} + d_{a,b+1} + d_{a-1,b} + d_{a+1,b}
```

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

```math
\alpha(\Delta f_k) = \min\left(1,\ e^{-\Delta f_k(x) / T}\right).
```

Throughout this repository $M = \mathbb{F}_2^n$, $f = P$ and $\varphi = \theta$,
so $\Delta f_k = \Delta E_k$. At constant $T$ the chain is reversible and its
stationary distribution is the **Boltzmann–Gibbs distribution**; under a
sufficiently slow logarithmic cooling schedule it concentrates on the global
minima (Hajek 1988):

```math
\pi^G(x) = \frac{e^{-P(x)/T}}{Z}, \quad Z = \sum_{x'} e^{-P(x')/T}
```

## Digital Annealing (DA)

DA evaluates **all** $\Delta f_k$, $k \in K$, in one step, applies the
Metropolis test to each of them independently, and then picks one of the
accepted modifications uniformly at random. For bit flips the transition
probability is

```math
P^{DA}(x, \theta_k(x)) = \sum_{\substack{S \subseteq K : \\ k \in S}} \frac{1}{|S|} \prod_{i \in S} e^{-\Delta E_i(x)^+ / T} \prod_{i \notin S} \left(1 - e^{-\Delta E_i(x)^+ / T}\right), \qquad \Delta E_i^+ = \max(0, \Delta E_i),
```

and $P^{DA}(x, x) = \prod_i \big(1 - e^{-\Delta E_i(x)^+ / T}\big)$ when nothing
is accepted. DA does **not** satisfy detailed balance with respect to $\pi^G$,
and in general $\pi^G$ is not its stationary distribution
([outlook](#outlook-comparing-the-chains-at-equal-temperature)). The
implementation additionally uses a **dynamic energy offset** (escape
mechanism): a value $E_{\text{off}}$, initially 0, is *subtracted* from every
$\Delta E_i$ before the acceptance test. Whenever no flip is accepted in a
step, $E_{\text{off}}$ grows by `offset_increase_rate`; as soon as a flip is
accepted it is reset to 0. This turns the exponentially long waiting time at a
local minimum into a linear one.

## Incremental delta-energy update

After a flip of bit $k$ is accepted, the vector $(\Delta E_i)_{i \in K}$ is not
recomputed: only the entries $i$ that share a monomial with $k$ change, and each
changes by a local correction (see
[`Masterarbeitspräsentation.pdf`](../Masterarbeitspr%C3%A4sentation.pdf), p.
53–55). This incremental update is one of the core efficiency contributions of
this codebase; it keeps DA's per-step cost within a small factor of SA's instead
of a factor $n$ (measurements in [performance.md](performance.md#results)).

## Stationary distributions and convergence

**Existence.** At constant $T$ with $0 \lt T \lt \infty$, and for DA without the
escape offset, both chains move from $x$ to each $\varphi_k(x)$ with
probability at least $e^{-\Delta f_k(x)^+ / T} / |K| \gt 0$. Chaining such
steps along the paths the definition guarantees gives $P^m(y, x) \gt 0$ for
all $x, y$: the chains are **irreducible**, so the stationary distribution is
unique. They are **aperiodic** as soon as one state $x$ has $P(x, x) \gt 0$:
every state can return to itself through $x$ in some $m$ steps, and in $m + 1$
by staying at $x$ once, so the gcd of its return times is 1. For SA such an $x$
exists whenever $f$ is not constant, for DA whenever $f$ has a strict local
minimum; otherwise the chain on $\mathbb{F}_2^n$ has period 2.

**Convergence under cooling.** Without the escape offset, the DA chain
converges to the global minima under a sufficiently slow logarithmic cooling
schedule, the analogue of Hajek's theorem for SA; this was proved by
[Fukushima-Kimura et al. (2023)](references.md#papers) for the parallel-trial
chain. With the offset, the process can still be read as a Markov chain on a
finite state space, namely on pairs $(x, j)$ of the current state and the
number $j$ of consecutive steps without an accepted modification. The offset is
determined by $j$, so the current pair carries all the information the next
step depends on, and earlier steps have no further influence; $j$ stays
bounded, because once the offset reaches the smallest uphill difference at $x$,
a modification is accepted with certainty. We state this reading with some
caution, since Fukushima-Kimura et al. assess the offset mechanism differently.
Whether a result in the spirit of Hajek's theorem holds for this chain is not
proven and remains open; we suspect that it does not.

## Outlook: comparing the chains at equal temperature

SA and DA are built from the same ingredients, the modifications $\varphi_k$
and the acceptance probabilities $e^{-\Delta f_k(x)^+ / T}$. SA runs the
acceptance test for one modification drawn at random, DA runs it for all of
them; in this sense SA's step is contained in DA's. Comparing the two chains at
the same temperature $T$ therefore isolates exactly this difference. The
quantity studied here is

```math
\delta(x) = \pi^{DA}(x) - \pi^{SA}(x).
```

Since $\pi^{SA} = \pi^G$ is known in closed form, $\delta$ determines
$\pi^{DA}$ directly, and how the two distributions relate is essential for
judging which algorithm is preferable at which temperature. This is ongoing
work; [stationary-distribution.md](stationary-distribution.md) computes
$\delta$ exactly for small instances.
