"""
Exact Stationary Distributions of SA and DA

Computes, without simulating anything, the stationary distributions of the two
annealing Markov chains at a CONSTANT temperature:

    pi^SA   the Metropolis chain is reversible w.r.t. the Gibbs measure, so its
            stationary distribution IS the Boltzmann distribution
            pi^B(x) = exp(-E(x)/T) / Z
    pi^DA   the digital annealing chain has a closed form transition matrix,
            P(x, theta_k(x)) = p_k * R(x, theta_k(x)) with p_k = exp(-beta dE_k^+)
            and R of formula (3.6); pi^DA is its left eigenvector

On top of that, the mean hitting times m(x, theta_k(x))_T are solved exactly,
one linear system per target state, and fed to approximate_pi_DA.

Everything rests on enumerating the state space, so the number of variables is
limited: 2^n <= max_states (512 by default, i.e. n <= 9). In exchange there is
no sampling noise, no burn-in and no seed — the numbers are the numbers.

All quantities are controlled by the parameters in the __main__ block.
"""

import time

import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Only the pbf helpers are imported, deliberately NOT the annealer
# implementations: nothing in this file simulates a chain.
from Funcs_Annealing2 import createPoly, createPoly_negative, Sort_pbf, evalPBF

pio.renderers.default = "browser"


# LANDSCAPE / REFERENCE DISTRIBUTION

def get_energy_landscape(pbf: dict[tuple:float], variables: int) -> np.ndarray:
    """
    Evaluates the pbf on the complete state space.

    A state is encoded as an integer s, where bit i of s holds x_i
    (i.e. s = sum_i x_i * 2^i). A monomial contributes its coefficient exactly
    when all of its variables are 1, which is a single bitmask test.

    Input:  pbf:        a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...}
            variables:  number of variables n

    returns: array E of length 2^n with E[s] = evalPBF(pbf, s)
    """
    states = np.arange(2 ** variables, dtype=np.int64)
    E = np.zeros(2 ** variables)
    for mon in pbf:
        if not mon:
            E += pbf[mon]            # constant term
            continue
        mask = 0
        for var in mon:
            mask |= (1 << var)
        E[(states & mask) == mask] += pbf[mon]
    return E


def state_to_varAssignement(s: int, variables: int) -> dict[int:bool]:
    """Converts a state index back to the varAssignement dict used by the repo."""
    return {i: (s >> i) & 1 for i in range(variables)}


def state_bits(variables: int) -> np.ndarray:
    """
    (2^n, n) matrix of the state vectors: row x is the bit vector of state x,
    column k the value of variable k across all states.
    """
    return ((np.arange(2 ** variables)[:, None] >> np.arange(variables)[None, :])
            & 1).astype(float)


def ising_coupling_term(pbf: dict[tuple:float], variables: int) -> np.ndarray:
    """
    Per state, the quadratic part of the pbf written in SPIN variables:

        J(x) = sum over the quadratic monomials (i,j) of
                   (2 x_i - 1) (2 x_j - 1) * J_ij

    with J_ij the pbf coefficient a_ij of the monomial (i,j). Substituting
    s_i = 2 x_i - 1 maps the 0/1 variables to the +-1 spins, so this is the
    coupling energy of the corresponding Ising model — same sign convention as
    the QUBO, only the variables are re-centred.

    Defined for QUBOs only: if the pbf carries a monomial of degree 3 or higher
    there is no J_ij to read off, and the function returns None rather than a
    number computed from part of the polynomial.

    returns: array of length 2^n, or None if the pbf is not of degree <= 2
    """
    if any(len(mon) > 2 for mon in pbf):
        return None

    spins = 2.0 * state_bits(variables) - 1.0          # (2^n, n), entries +-1
    J = np.zeros(2 ** variables)
    for mon in pbf:
        if len(mon) == 2:
            i, j = mon
            J += pbf[mon] * spins[:, i] * spins[:, j]
    return J


def verify_landscape(pbf: dict[tuple:float], E: np.ndarray, variables: int):
    """
    Sanity check: the vectorized landscape must agree with evalPBF on every state.
    Cheap for n <= 9 and protects against a wrong bit convention.
    """
    for s in range(2 ** variables):
        ref = evalPBF(pbf, state_to_varAssignement(s, variables))
        if abs(ref - E[s]) > 1e-9:
            raise AssertionError("landscape mismatch at state %d: %f != %f" % (s, E[s], ref))
    print("landscape verified against evalPBF on all " + str(2 ** variables) + " states")


def boltzmann_distribution(E: np.ndarray, T: float) -> np.ndarray:
    """
    Exact Boltzmann (Gibbs) distribution at temperature T.

    E is shifted by its minimum before exponentiating, which cancels in the
    normalization but avoids overflow for small T.
    """
    w = boltzmann_weights(E, T)
    return w / w.sum()


def boltzmann_weights(E: np.ndarray, T: float) -> np.ndarray:
    """
    Unnormalized Boltzmann weights w(x) = exp(-(E(x)-E_min)/T).

    Everything in this script uses the SAME E_min shift, so pi^B = w / Z holds
    exactly with Z = sum_x w(x). The shift only rescales Z by exp(E_min/T) and
    keeps the exponential from overflowing at small T.
    """
    return np.exp(-(E - E.min()) / T)


def estimate_Z(E: np.ndarray, T: float, pi_hat: np.ndarray) -> float:
    """
    Approximates the partition function Z from an empirical distribution.

    For a sampler with stationary distribution pi(x) = w(x)/Z the identity
        log Z = log w(x) - log pi_hat(x)
    holds for EVERY state x. The estimator averages the right hand side in log
    space over all visited states, weighted by pi_hat(x): well sampled states
    dominate, rarely visited ones (whose relative error is large) barely count.

    For SA this converges to the exact Z = sum_x w(x). For the DA, whose
    stationary distribution is not Boltzmann, the per state values disagree and
    the result is merely an effective Z — the gap to Z_exact is itself a measure
    of how far pi^DA is from a Gibbs distribution.
    """
    w = boltzmann_weights(E, T)
    visited = pi_hat > 0
    log_Z = np.log(w[visited]) - np.log(pi_hat[visited])
    return float(np.exp(np.average(log_Z, weights=pi_hat[visited])))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance 1/2 * sum |p-q|, in [0,1]."""
    return 0.5 * float(np.sum(np.abs(p - q)))


# SINGLE MOVE QUANTITIES
#
# Everything below is a closed form in the energy landscape: the acceptance
# probability of one flip, and R(x, theta_k(x)) — the probability that the DA
# picks direction k out of the set of directions it accepted in that step.

UNDEFINED = np.nan          # marker for "not defined" (e.g. an unreachable target)


def flip_probability(E: np.ndarray, x: int, k: int, T: float) -> float:
    """
    Acceptance probability of the single flip x -> theta_k(x):
        p_k = exp(-beta * deltaE_k^+),  deltaE_k^+ = max(E(theta_k(x)) - E(x), 0)

    This is the Metropolis acceptance of that flip: downhill flips give p_k = 1,
    uphill flips decay exponentially in deltaE/T.
    """
    deltaE = E[x ^ (1 << k)] - E[x]
    return float(np.exp(-max(deltaE, 0.0) / T))


def R_neighbour(E: np.ndarray, variables: int, x: int, k: int, T: float) -> float:
    """
    R(sigma, sigma^x) of formula (3.6), evaluated straight forward and exactly:

        R = sum over S subset of V\\{k} of
                1/(|S|+1) * prod_{y in S} e^{-beta E_y(sigma)^+}
                          * prod_{y in (V\\{k})\\S} (1 - e^{-beta E_y(sigma)^+})

    Careful with the names: the formula's x is the flipped SITE, which is k here,
    while x in this code is the STATE sigma. V = {0, ..., variables-1} are the
    sites, and E_y(sigma)^+ = max(E(theta_y(sigma)) - E(sigma), 0).

    Read probabilistically this is E[1/(|S|+1)], where S is the random set of the
    OTHER sites the DA accepts in this step: site y enters S independently with
    probability e^{-beta E_y^+}, and given S the DA picks site k out of S u {k}
    with probability 1/(|S|+1).

    The sum runs over all 2^(variables-1) subsets literally, as written. That is
    exponential but exact; for variables <= 9 it is 256 terms per entry.
    """
    others = [y for y in range(variables) if y != k]
    p = [flip_probability(E, x, y, T) for y in others]

    R = 0.0
    for mask in range(2 ** len(others)):
        term = 1.0
        size = 0
        for idx in range(len(others)):
            if (mask >> idx) & 1:               # y in S
                term *= p[idx]
                size += 1
            else:                               # y in (V\{k})\S
                term *= (1.0 - p[idx])
        R += term / (size + 1)
    return R


# EXACT SOLUTIONS
#
# For an enumerable state space the DA chain is just a 2^n x 2^n Markov chain
# whose transition matrix we can write down in closed form:
#
#     P(x, theta_k(x)) = p_k * R(x, theta_k(x))          p_k = exp(-beta dE_k^+)
#     P(x, x)          = prod_i (1 - p_i)                 (select came out empty)
#
# Mean hitting times and the stationary distribution then follow from linear
# algebra alone: no sampling, no burn-in, no censoring, every entry defined.

def da_transition_matrix(E: np.ndarray, variables: int, T: float) -> np.ndarray:
    """
    Full (2^n, 2^n) one step transition matrix of the DA chain at temperature T.

    Row x has at most n+1 nonzeros: the n neighbours theta_k(x) and the self
    loop. The self loop is the probability that the DA accepted nothing at all,
    prod_i (1 - p_i); it is nonzero only at strict local minima, because a single
    downhill direction already has p_i = 1 and kills the product.

    Assumes offset_increase_rate = 0 (pure DA dynamics). With a nonzero E_Offset
    the chain is no longer Markov on {0,1}^n, but it IS a time-homogeneous Markov
    chain on the extended space of pairs (x, j), j = number of consecutive idle
    steps, and that space is finite: j only advances at strict local minima and
    is capped at ceil(g_min(x)/rate), because a flip is accepted with certainty
    once its shifted gap drops to zero. See concepts/02_escape_offset.md and
    verify/konzept_02_offset.py in the latex-research-builder repo for the
    extended matrix.
    """
    n_states = len(E)
    P = np.zeros((n_states, n_states))
    for x in range(n_states):
        stay = 1.0
        for k in range(variables):
            p = flip_probability(E, x, k, T)
            P[x, x ^ (1 << k)] = p * R_neighbour(E, variables, x, k, T)
            stay *= (1.0 - p)
        P[x, x] = stay
    return P


def net_transition_flux(P: np.ndarray, variables: int) -> np.ndarray:
    """
    D(x) = sum over k of [ P(theta_k(x), x) - P(x, theta_k(x)) ]

    The net one step transition probability into x, summed over its n
    neighbours — the two directions of every incident edge, WITHOUT weighting
    them by pi. So this is a property of the kernel alone, not of the current
    distribution: it says nothing about probability actually flowing (the real
    flux pi(y)P(y,x) - pi(x)P(x,y) sums to zero at stationarity by definition),
    it says whether x is easier to enter than to leave.

    returns: array of length 2^n
    """
    n_states = P.shape[0]
    states = np.arange(n_states)
    D = np.zeros(n_states)
    for k in range(variables):
        neighbour = states ^ (1 << k)
        D += P[neighbour, states] - P[states, neighbour]
    return D


def exact_stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    The exact stationary distribution pi of a transition matrix P: pi P = pi,
    sum pi = 1.

    Solved as a least squares problem on the overdetermined system
    [P^T - I ; 1^T] pi = [0 ; 1], which is numerically better behaved than an
    eigenvector routine and enforces the normalization directly.

    Applied to the DA transition matrix this is pi^DA, exactly.
    """
    n_states = P.shape[0]
    A = np.vstack([P.T - np.eye(n_states), np.ones(n_states)])
    b = np.zeros(n_states + 1)
    b[-1] = 1.0
    pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return pi


def exact_hit_time_matrix(E: np.ndarray, variables: int, T: float,
                          P: np.ndarray = None, verbose: bool = True) -> np.ndarray:
    """
    Exact mean hitting times m(x, theta_k(x))_T for every state and every bit,
    in the (2^n, n) layout shared by all matrices here.

    For a fixed target t the mean hitting times solve the linear system
        h_t(t) = 0,   h_t(x) = 1 + sum_y P(x,y) h_t(y)   for x != t
    i.e. (I - Q) h = 1 with Q = P with row and column t deleted. One solve per
    target gives the n entries M[theta_k(t), k] = h_t(theta_k(t)).

    Cost is 2^n solves of size (2^n - 1); at n = 9 that is 512 solves of 511x511,
    a few seconds. Every entry is defined, including the transitions so rare that
    no finite simulation would ever observe them.

    An entry stays UNDEFINED only if its system is singular, which means the
    target is genuinely unreachable from that state.
    """
    n_states = len(E)
    if P is None:
        P = da_transition_matrix(E, variables, T)
    M = np.full((n_states, variables), UNDEFINED)
    all_states = np.arange(n_states)

    start_time_exact = time.time()
    singular = 0
    for t in range(n_states):
        keep = all_states != t
        A = np.eye(n_states - 1) - P[np.ix_(keep, keep)]
        try:
            h = np.linalg.solve(A, np.ones(n_states - 1))
        except np.linalg.LinAlgError:
            singular += 1
            continue
        h_full = np.zeros(n_states)
        h_full[keep] = h
        for k in range(variables):
            x = t ^ (1 << k)            # the states whose k-th neighbour is t
            M[x, k] = h_full[x]
    if verbose:
        print("exact hit times T = %.4g: %d of %d entries defined, %d singular targets, "
              "%.1f sec" % (T, np.count_nonzero(~np.isnan(M)), M.size, singular,
                            time.time() - start_time_exact))
    return M


def reverse_matrix(A: np.ndarray) -> np.ndarray:
    """
    Turns a matrix indexed by (x, k) into the one indexed by the REVERSE move:

        A_rev[x, k] = A[theta_k(x), k]

    So A[x, k] holds the quantity for the move x -> theta_k(x), and A_rev[x, k]
    holds it for the move theta_k(x) -> x.

    The COLUMN stays k, only the row changes: theta_k(theta_k(x)) = x, flipping
    bit k twice is the identity, so the way back from theta_k(x) to x is again
    the k-th flip. That is why the reverse value is a plain row lookup.

    Works for any of the (2^n, n) matrices:
        R_rev = reverse_matrix(R)  ->  R_rev[x, k] = R(theta_k(x), x)
        M_rev = reverse_matrix(M)  ->  M_rev[x, k] = m(theta_k(x), x)
    """
    n_states, variables = A.shape
    states = np.arange(n_states)
    A_rev = np.full_like(A, UNDEFINED)
    for k in range(variables):
        A_rev[:, k] = A[states ^ (1 << k), k]
    return A_rev


def sa_transition_matrix(E: np.ndarray, variables: int, T: float) -> np.ndarray:
    """
    pi^SA_k(x): the plain SA one step transition probability x -> theta_k(x).

        pi^SA_k(x) = (1/n) * exp(-beta * deltaE_k^+)

    The 1/n is SA picking one of the n variables uniformly, the exponential is
    the Metropolis acceptance of that flip — 1 for a downhill flip, decaying in
    deltaE/T for an uphill one. Note this is the probability of ACTUALLY moving
    to theta_k(x); the self loop (rejection) is what is left over.

    Same (2^n, n) layout and the same indexing as M and R, row = x, column = k,
    so reverse_matrix applies unchanged:
        reverse_matrix(P_SA)[x, k] = pi^SA_k(theta_k(x))

    Fully defined everywhere — it is a closed form, so it has no nan entries and
    is not affected by the probability threshold.
    """
    n_states = len(E)
    states = np.arange(n_states)
    P_SA = np.zeros((n_states, variables))
    for k in range(variables):
        deltaE = E[states ^ (1 << k)] - E[states]
        P_SA[:, k] = np.exp(-np.maximum(deltaE, 0.0) / T)
    return P_SA


def approximate_pi_DA(M: np.ndarray, R: np.ndarray, pbf: dict[tuple:float],
                      variables: int, T: float,pi_DA_exact) -> np.ndarray:
    """
    Approximation of pi^DA built from the R matrix:

        pi(x) = sum over k in K of [ R(theta_k(x), x) - R(x, theta_k(x)) ]

    K are the directions k for which the term is defined. Entries skipped by the
    probability threshold are nan in R and drop out of the sum. If NOT A SINGLE
    k is defined for a state, pi(x) is reported as UNDEFINED rather than as an
    empty sum of 0 — a state nothing is known about must not end up looking like
    a state of weight zero.

    The building blocks are all laid out in the same (2^n, n) indexing, so they
    can be combined entry by entry:
        R,        R_rev          R(x, theta_k(x)),    R(theta_k(x), x)
        M,        M_rev          m(x, theta_k(x))_T,  m(theta_k(x), x)_T
        P_SA,     P_SA_rev       pi^SA_k(x),          pi^SA_k(theta_k(x))
        Flow_SA,  Flow_SA_rev    pi^SA(x)*pi^SA_k(x), pi^SA(theta_k(x))*pi^SA_k(theta_k(x))
    plus pi_SA as a plain length-2^n vector (use pi_SA[:, None] to broadcast it
    over the k columns of any of the matrices).

    Input:  M:          (2^n, n) mean hit times m(x, theta_k(x))_T, nan where undefined
            R:          (2^n, n) R(x, theta_k(x)) of formula (3.6), nan where undefined
            pbf:        the polynomial the landscape came from
            variables:  number of variables n
            T:          the constant temperature the matrices were computed at

    returns: array of length 2^n, NOT normalized — whatever is returned is
             plotted as is. nan entries are gaps in the curve.
    """
    E = get_energy_landscape(pbf, variables)
    R_rev = reverse_matrix(R)               # R_rev[x, k] = R(theta_k(x), x)
    M_rev = reverse_matrix(M)               # M_rev[x, k] = m(theta_k(x), x)
    # plain SA transition, same (2^n, n) layout, no nan entries
    P_SA = sa_transition_matrix(E, variables, T)
    P_SA_rev = reverse_matrix(P_SA)         # P_SA_rev[x, k] = pi^SA_k(theta_k(x))
    # pi^SA itself: the stationary distribution of the SA chain IS the Boltzmann one
    pi_SA = boltzmann_distribution(E, T)                # length 2^n, per state
    Flow_SA =  P_SA*pi_SA[:,None]         # Flow_SA[x, k] = pi^SA(x) * pi^SA_k(x)
    # the state vectors themselves, same (2^n, n) layout and order as M and R:
    # row x is the bit vector (x_0, ..., x_n-1) of state x, column k is the value
    # of variable k across all states. X_inv is the complement, 1 - x_i.
    X = ((np.arange(2 ** variables)[:, None] >> np.arange(variables)[None, :])
         & 1).astype(float)
    X_inv = 1.0 - X
    
    # pi^SA evaluated at the NEIGHBOUR, again row = x, column = k:
    #     Pi_SA_theta[x, k] = pi^SA(theta_k(x))
    # Same layout as everything else, so it combines entry by entry with M, R,
    # P_SA, Flow_SA and X. Note pi_SA[:, None] is the same quantity at x itself.
    Pi_SA_theta = pi_SA[np.arange(2 ** variables)[:, None]
                        ^ (1 << np.arange(variables))[None, :]]
    
    C = np.nansum(
    P_SA * X_inv * (R_rev - R) * M_rev,
    axis=1
)
    contributions = (Pi_SA_theta*P_SA_rev *X_inv*R_rev + Flow_SA*X*R)*M_rev 

    pi_approx = pi_SA/(pi_SA-pi_SA*C+(np.nansum(contributions, axis=1)))
    n_defined = np.count_nonzero(~np.isnan(contributions), axis=1)
    pi_approx[n_defined == 0] = UNDEFINED   # nansum would have returned 0 here
    return pi_approx


# EXPERIMENT

def uphill_sum(E: np.ndarray, variables: int) -> np.ndarray:
    """
    G+(x) = sum over k of max(0, E(theta_k(x)) - E(x)).

    The total uphill barrier at x: every neighbour that lies above x contributes
    its gap, every neighbour at or below x contributes nothing. Purely local —
    it only looks at x and its n neighbours.
    """
    states = np.arange(len(E))
    neighbours = E[states[:, None] ^ (1 << np.arange(variables))[None, :]]
    return np.maximum(neighbours - E[:, None], 0.0).sum(axis=1)


def phi_potential(E: np.ndarray, variables: int) -> np.ndarray:
    """
    Phi(x) = E(x) + G+(x), the effective potential that governs the DA's
    deviation from Boltzmann at high temperature.

    For a QUBO of degree 2 it satisfies, exactly,

        -(Phi(x) - <Phi>) = J(x)/4 - (W(x) - <W>)/2,   W(x) = sum_k |deltaE_k(x)|

    with <.> the UNIFORM average over all states, not the Boltzmann one. So the
    spin coupling term J is exactly the |S| = 2 part of -Phi; the rest of Phi is
    the local field roughness W, which is what J alone cannot reproduce.
    """
    return E + uphill_sum(E, variables)


def delta_high_T(E: np.ndarray, variables: int, T: float) -> np.ndarray:
    """
    First order in 1/T of delta = pi^DA - pi^SA. No fitted parameters at all;
    the prefactor is analytic.

        delta(x) = -(Phi(x) - <Phi>) / (2^n (n-1) T)  +  O(1/T^2)

    Exact in the limit: 2^n * T * delta converges to -(Phi - <Phi>)/(n-1). The
    expansion parameter is deltaE/T, not T, so the useful range is set by the
    landscape: good from about T = 10 * std(deltaE), very good from 30 on. It
    says nothing at low temperature, where the stationary distribution is
    governed by the global barrier structure and no local formula can hold.
    """
    Phi = phi_potential(E, variables)
    return -(Phi - Phi.mean()) / (len(E) * (variables - 1) * T)


def sa_transition_full(E: np.ndarray, variables: int, T: float) -> np.ndarray:
    """
    The COMPLETE (2^n, 2^n) SA transition matrix, self loop included:

        P_SA(x, theta_k(x)) = (1/n) * exp(-beta * deltaE_k^+)
        P_SA(x, x)          = 1 - sum_k P_SA(x, theta_k(x))

    Note the 1/n. sa_transition_matrix above returns the (2^n, n) matrix of the
    ACCEPTANCE probabilities without that factor, so it cannot be scattered into
    a transition matrix as it stands.

    Unlike the DA chain this one is reversible, and its stationary distribution
    is exactly the Boltzmann distribution — at every temperature, not only
    asymptotically.
    """
    n_states = len(E)
    states = np.arange(n_states)
    neighbours = states[:, None] ^ (1 << np.arange(variables))[None, :]
    accept = np.exp(-np.maximum(E[neighbours] - E[:, None], 0.0) / T) / variables
    P = np.zeros((n_states, n_states))
    P[states[:, None], neighbours] = accept
    np.fill_diagonal(P, 1.0 - accept.sum(axis=1))
    return P


def delta_first_order_SA(E: np.ndarray, variables: int, T: float,
                         P_DA: np.ndarray = None) -> np.ndarray:
    """
    delta = pi^DA - pi^SA to first order in the perturbation around the SA chain.

        pi^DA = pi^SA + pi^SA * dP * Z + O(dP^2)
        dP = P_DA - P_SA        Z = (I - P_SA + 1 pi^SA)^-1   (fundamental matrix)

    Why SA rather than the uniform walk A: pi^SA is exactly Boltzmann at EVERY
    temperature, and the SA chain is irreducible for every finite T, so its
    eigenvalue 1 is simple and the perturbation is well posed throughout. The
    expansion around A only has that property in the limit T -> infinity.

    There is an exact resummation behind this,

        pi^DA  proportional to  pi^SA * (I - dP Z)^-1

    which holds at every temperature (it follows from Z(I-P_SA) = I - Pi and
    dP * 1 = 0). This function keeps only its first term.

    Range: dP is NOT small at low temperature — SA takes the single downhill
    flip with probability 1/n, the DA with probability 1, so ||dP|| grows to
    order 1. Measured usefully down to about T = 3 * std(deltaE) (R^2 ~ 0.94),
    against about T = 10 * std(deltaE) for delta_high_T.

    delta_series_SA is the same expansion carried to higher order, and it also
    computes the radius that says where the series stops converging. Use that
    instead of guessing; below the radius adding terms makes it worse, not
    better, and the radius is cheap.

    Cost: one (2^n, 2^n) inverse, so the same order as solving for pi^DA itself.
    It is a structural decomposition, not a cheaper route to the answer.
    """
    if P_DA is None:
        P_DA = da_transition_matrix(E, variables, T)
    n_states = len(E)
    P_SA = sa_transition_full(E, variables, T)
    pi_SA = boltzmann_distribution(E, T)
    Z = np.linalg.inv(np.eye(n_states) - P_SA
                      + np.outer(np.ones(n_states), pi_SA))
    return pi_SA @ (P_DA - P_SA) @ Z


def ratio_kernel(E: np.ndarray, variables: int, T: float,
                 R: np.ndarray = None) -> np.ndarray:
    """
    The kernel K of the ratio r = pi^DA / pi^SA, which satisfies r = K^T r:

        K(x, theta_k x) = p_k(theta_k x) * R(x, k)
        K(x, x)         = prod_i (1 - p_i(x))

    Every Boltzmann factor cancels out of K — the derivation is one line,
    pi^SA(x) P_DA(x,y) / pi^SA(y) = e^{-beta max(E(x)-E(y), 0)} R(x,k), using
    a + max(-a,0) = max(a,0). All entries land in [0,1], so K is computable in
    float64 down to T/std(dE) ~ 1e-4, where pi^DA itself needs mpmath at dps
    300 and still collapses.

    The SA analogue is K_SA(x, theta_k x) = p_k(theta_k x)/n, and because SA is
    reversible, K_SA(x,y) = P_SA(y,x), i.e. K_SA^T = P_SA exactly. That is what
    makes defect_source and delta_series_SA a Poisson problem for the SA chain.
    """
    n_states = len(E)
    states = np.arange(n_states)
    neighbours = states[:, None] ^ (1 << np.arange(variables))[None, :]
    p = np.exp(-np.maximum(E[neighbours] - E[:, None], 0.0) / T)
    if R is None:
        R = np.array([[R_neighbour(E, variables, x, k, T)
                       for k in range(variables)] for x in states])
    K = np.zeros((n_states, n_states))
    for x in range(n_states):
        for k in range(variables):
            K[x, x ^ (1 << k)] = p[x ^ (1 << k), k] * R[x, k]
        K[x, x] = np.prod(1.0 - p[x])
    return K


def defect_source(E: np.ndarray, variables: int, T: float,
                  R: np.ndarray = None) -> np.ndarray:
    """
    The source term S of the exact equation for g = pi^DA/pi^SA - 1:

        S(x) = sum_k p_k(x) [ R(theta_k x, k) - R(x, k) ]

    It is the column sum of K - K_SA, and the 1/n terms of K_SA cancel against
    the row sum identity sum_k p_k R(x,k) + prod_i(1-p_i) = 1, which is why S
    comes out purely local: it only ever looks at x and its n neighbours, no
    matrix inverse anywhere.

    S is the whole driving force of the deviation. It vanishes identically iff
    the DA chain is reversible, and sum_x pi^SA(x) S(x) = 0 always — that is
    the solvability condition of the Poisson equation in delta_series_SA, and
    it holds to 1e-19 in the tests.
    """
    n_states = len(E)
    states = np.arange(n_states)
    neighbours = states[:, None] ^ (1 << np.arange(variables))[None, :]
    p = np.exp(-np.maximum(E[neighbours] - E[:, None], 0.0) / T)
    if R is None:
        R = np.array([[R_neighbour(E, variables, x, k, T)
                       for k in range(variables)] for x in states])
    return (p * (reverse_matrix(R) - R)).sum(axis=1)


def delta_series_SA(E: np.ndarray, variables: int, T: float, order: int = 2,
                    R: np.ndarray = None) -> tuple:
    """
    delta to arbitrary order around the SA chain, plus the convergence radius.

    Write r = pi^DA/pi^SA = 1 + g. Then r = K^T r becomes, exactly,

        (I - P_SA) g - dK^T g = S,      dK = K - K_SA,  K_SA^T = P_SA

    a Poisson equation for the SA chain with a perturbation. This is the same
    structure as the high temperature law of document 1, where the equation was
    (I - A) (...) = source; here P_SA takes the place of A, and A is just its
    T -> infinity limit. Solving order by order,

        g_1 = Z S                                (Poisson, <g>_{pi^SA} = 0)
        g   = sum_{t >= 0} M^t g_1,              M = Z dK^T

    and delta = pi^SA * g. The first term is exactly delta_first_order_SA.

    returns: (delta, rho) with rho = spectral radius of M.

    rho < 1 is the convergence criterion, and it is sharp: at rho = 0.09 the
    error falls by a factor 10 per term (6.1e-3, 5.0e-4, 4.2e-5, ...), at
    rho > 1 it grows and every further term makes the answer worse. Measured
    crossing: T ~ 1.0 * std(dE) unfrustrated, T ~ 1.8 * std(dE) frustrated.

    R^2 against exact delta at T = 3 std(dE): 0.934 (order 1), 0.995 (2),
    0.9996 (3), 1.00000 (5). At T = 2 std(dE): 0.816, 0.960, 0.991, 0.9993.

    Cost beyond the first order: one matrix-vector product per term, so the
    higher orders are free once Z is there. rho needs one eigenvalue and can be
    skipped by passing order <= 1.
    """
    n_states = len(E)
    if R is None:
        states = np.arange(n_states)
        R = np.array([[R_neighbour(E, variables, x, k, T)
                       for k in range(variables)] for x in states])
    P_SA = sa_transition_full(E, variables, T)
    pi_SA = boltzmann_distribution(E, T)
    S = defect_source(E, variables, T, R=R)

    # Poisson solve (I - P_SA) g_1 = S under the constraint pi^SA . g_1 = 0,
    # which is the correct normalisation because sum_x delta(x) = 0.
    A = np.eye(n_states) - P_SA
    A = np.vstack([A[:-1], pi_SA])
    g = np.linalg.solve(A, np.concatenate([S[:-1], [0.0]]))

    if order <= 1:
        return pi_SA * g, np.nan

    Z = np.linalg.inv(np.eye(n_states) - P_SA
                      + np.outer(np.ones(n_states), pi_SA))
    M = Z @ (ratio_kernel(E, variables, T, R=R) - P_SA.T).T
    term = g.copy()
    for _ in range(order - 1):
        term = M @ term
        g = g + term
    rho = float(np.abs(np.linalg.eigvals(M)).max())
    return pi_SA * g, rho


def frustrated_directions(E: np.ndarray, variables: int) -> int:
    """
    Number of directions k whose local field L_k changes sign somewhere on the
    state space. Zero means the instance is UNFRUSTRATED.

    deltaE_k(x) = -2 s_k L_k(x), and L_k does not depend on s_k itself, so it is
    enough to look at the half of the space with x_k = 0 — there deltaE_k = 2 L_k.

    In Ising terms the condition is |h_k| > sum_j |J_kj|: the field of a site
    dominates its own couplings. A QUBO whose coefficients all carry the SAME
    sign always satisfies it, since h_k = a_k/2 + sum_j a_kj/4 while
    sum_j |J_kj| = sum_j |a_kj|/4 — so those instances are unfrustrated by
    construction, not by luck.

    Unfrustrated also implies the global minimum is the ONLY local minimum:
    every direction has a fixed descent orientation.
    """
    states = np.arange(len(E))
    count = 0
    for k in range(variables):
        lower = states[((states >> k) & 1) == 0]
        gaps = E[lower ^ (1 << k)] - E[lower]
        if (gaps > 0).any() and (gaps < 0).any():
            count += 1
    return count


def delta_at_minimum(pbf: dict[tuple:float], E: np.ndarray, variables: int,
                     T: float):
    """
    delta at the GLOBAL MINIMUM x*, in closed form, for an unfrustrated QUBO of
    degree 2:

        delta(x*) = (3/4) * J(x*) / (2^n (n-1) T)  +  O(1/T^2)

    Derivation, on top of the general first order result: unfrustrated means
    every |L_k| is linear, so <W> = 2 sum_k |h_k| and <Phi> = c_0 + sum_k |h_k|.
    At x* every spin sits against its own field, s_k* = -sign(L_k), hence

        G+(x*) = 2 sum_k |h_k| - 4 sum_{i<j} J_ij s_i* s_j*
        E(x*)  = c_0 - sum_k |h_k| +   sum_{i<j} J_ij s_i* s_j*

    and the fields cancel exactly:  Phi(x*) - <Phi> = -3 sum J_ij s_i* s_j*
    = -(3/4) J(x*), with J the spin coupling term in the QUBO's own convention.

    Only x* has this form — at any other state G+ runs over the uphill
    directions alone, and the cancellation fails.

    returns: (x_star, predicted delta, frustrated_count) with the prediction
             None if the pbf has degree > 2 or the instance is frustrated.
    """
    x_star = int(np.argmin(E))
    frustrated = frustrated_directions(E, variables)
    J = ising_coupling_term(pbf, variables)
    if J is None or frustrated > 0 or variables < 2:
        return x_star, None, frustrated
    value = 0.75 * float(J[x_star]) / (len(E) * (variables - 1) * T)
    return x_star, value, frustrated


def edge_weights(E: np.ndarray, variables: int, T: float) -> np.ndarray:
    """
    phi(x,k) = pi^SA(x) * p_k(x) = exp(-beta * max(E(x), E(theta_k(x)))) / Z

    The equilibrium probability flow of the SA chain across the edge
    {x, theta_k(x)}, up to the factor 1/n. Two properties make it the natural
    weight for everything below:

      - it is SYMMETRIC in the edge, phi(x,k) = phi(theta_k(x), k). That is
        exactly the detailed balance of the Metropolis chain, written out.
      - the exponent carries the SADDLE of the edge, the higher of the two
        endpoints, not the energy of x. An edge leaving a deep minimum is
        therefore damped by exp(-beta * barrier), not by the Boltzmann weight
        of that minimum. This is what makes the whole expression collapse at
        low temperature.

    In the electrical network picture of a reversible chain this is the edge
    conductance.

    Same (2^n, n) layout as M and R, so reverse_matrix applies unchanged.
    """
    states = np.arange(len(E))
    neighbours = states[:, None] ^ (1 << np.arange(variables))[None, :]
    saddle = np.maximum(E[:, None], E[neighbours])
    return np.exp(-(saddle - E.min()) / T) / np.exp(-(E - E.min()) / T).sum()


def delta_edge_flux(E: np.ndarray, variables: int, T: float,
                    R: np.ndarray = None) -> np.ndarray:
    """
    delta = pi^DA - pi^SA, as the LOCAL truncation of the expansion around SA:

        delta(x)  ~  sum_k phi(x,k) * [ R(theta_k(x), k) - R(x, k) ]

    This is pi^SA * dP written out — the first order term of the SA expansion
    with the fundamental matrix Z replaced by the identity. Multiplying it out,
    only x and its n neighbours contribute; detailed balance gives the neighbour
    term and the self loop term the SAME weight phi(x,k), and since dP has zero
    row sums the 1/n of the SA chain cancels. What is left is the difference of
    R across the edge, weighted by the edge's saddle Boltzmann factor.

    Everything here is a closed form: no mean hit times, no fundamental matrix,
    no linear system. Purely local, zero fitted parameters.

    Range: this is the one expression that does not blow up anywhere. Measured
    R^2 between 0.65 and 1.00 from T = 30*std(deltaE) down to 0.03*std(deltaE)
    — but it is never excellent either, and it does fail on some instances at
    very low T (one of four tested went to R^2 ~ 0). Use it where delta_high_T
    and delta_first_order_SA have already run away.
    """
    if R is None:
        R = np.array([[R_neighbour(E, variables, x, k, T) for k in range(variables)]
                      for x in range(len(E))])
    phi = edge_weights(E, variables, T)
    return (phi * (reverse_matrix(R) - R)).sum(axis=1)


def run_stationary_experiment(pbf: dict[tuple:float], variables: int, temperatures: list[float],
                              max_states: int = 512, verify: bool = True):
    """
    Computes, exactly and per temperature:

        pi^B        the Boltzmann distribution — at the same time the exact pi^SA,
                    since the stationary distribution of the Metropolis chain IS
                    the Gibbs distribution
        pi^DA       the exact stationary distribution of the DA chain, from the
                    left eigenvector of its transition matrix
        M           the exact mean hit times m(x, theta_k(x))_T, from one linear
                    system per target state
        R           R(x, theta_k(x)) of formula (3.6)
        DA_approx   whatever approximate_pi_DA builds out of M and R

    Nothing here is simulated: no chains are run, so there is no burn-in, no
    sampling noise and no seed. The only limit is the state space, which has to
    be enumerable — 2^n <= max_states.

    Input:  pbf:            the polynomial defining the landscape
            variables:      number of variables n, must satisfy 2^n <= max_states
            temperatures:   list of constant temperatures
            max_states:     guard against an unenumerable state space
            verify:         check the landscape against evalPBF

    returns: E, results  with results[T] = {"B", "Z_exact", "DA_exact",
                                            "Z_DA_exact", "M", "R", "D",
                                            "J_term", "Phi", "J_mod2",
                                            "delta_hiT", "DA_hiT",
                                            "delta_SA1", "DA_SA1",
                                            "delta_SA2", "DA_SA2", "rho_series",
                                            "delta_edge", "DA_edge", "DA_approx"}
    """
    if 2 ** variables > max_states:
        raise ValueError("state space 2^%d = %d exceeds max_states = %d"
                         % (variables, 2 ** variables, max_states))

    E = get_energy_landscape(pbf, variables)
    if verify:
        verify_landscape(pbf, E, variables)
    print("states = " + str(2 ** variables) + ", E_min = " + str(np.min(E))
          + ", E_max = " + str(np.max(E)))

    # temperature independent, so computed once: None unless the pbf is a QUBO
    J_term = ising_coupling_term(pbf, variables)
    if J_term is None:
        print("pbf has monomials of degree > 2, the spin coupling term is skipped")

    # also temperature independent: Phi = E + G+, centred on the UNIFORM mean.
    # Unlike J it is defined for every degree. gap_scale is the yardstick the
    # high temperature expansion is actually measured in — the expansion
    # parameter is deltaE/T, so T alone says nothing without it.
    Phi_centred = phi_potential(E, variables)
    Phi_centred = Phi_centred - Phi_centred.mean()
    _states = np.arange(len(E))
    gap_scale = float((E[_states[:, None] ^ (1 << np.arange(variables))[None, :]]
                       - E[:, None]).std())
    print("std(deltaE) = %.6g — the high T formula needs T well above this"
          % gap_scale)
    _frustrated = frustrated_directions(E, variables)
    print("frustrated directions: %d of %d — %s"
          % (_frustrated, variables,
             "UNFRUSTRATED, the closed form at x* applies" if _frustrated == 0
             else "frustrated, the closed form at x* does NOT apply"))

    results = dict()
    for T in temperatures:
        start_time_round = time.time()
        pi_B = boltzmann_distribution(E, T)
        Z_exact = float(boltzmann_weights(E, T).sum())

        # one closed form transition matrix, everything else is linear algebra
        P = da_transition_matrix(E, variables, T)
        pi_DA_exact = exact_stationary_distribution(P)
        M = exact_hit_time_matrix(E, variables, T, P=P)
        R = np.array([[R_neighbour(E, variables, x, k, T) for k in range(variables)]
                      for x in range(len(E))])

        D = net_transition_flux(P, variables)
        # J scaled by pi^SA, which brings it onto the scale of delta. J itself is
        # an energy in the hundreds, pi^SA(x) is the Boltzmann weight of that
        # state, so these two are the candidates to compare against
        # delta = pi^DA - pi^SA. Both are T dependent through pi^SA.
        if J_term is not None:
            J_mod2 = (J_term - D) * pi_B            # (J(x) - D(x)) * pi^SA_T(x)
        else:
            J_mod2 = None

        # the high temperature limit: delta to first order in 1/T, no free
        # parameters, and the distribution that follows from it
        delta_hiT = delta_high_T(E, variables, T)
        DA_hiT = pi_B + delta_hiT

        # first order around the SA chain instead of around the uniform walk:
        # the base point is exact at every temperature, which buys roughly a
        # factor 3 in the usable temperature range
        delta_SA1 = delta_first_order_SA(E, variables, T, P_DA=P)
        DA_SA1 = pi_B + delta_SA1

        # the same series carried to second order. The extra term costs one
        # matrix-vector product and is not a fitted scalar: rho reports where
        # the series stops converging, and below rho = 1 the second order is
        # worse than the first, not better.
        delta_SA2, rho_series = delta_series_SA(E, variables, T, order=2, R=R)
        DA_SA2 = pi_B + delta_SA2

        # the same expansion with Z replaced by the identity: local, closed form,
        # and the only one of the three that does not blow up at low T. R is
        # already computed above, so this costs nothing extra.
        delta_edge = delta_edge_flux(E, variables, T, R=R)
        DA_edge = pi_B + delta_edge

        # closed form at the global minimum, unfrustrated degree 2 QUBOs only
        x_star, delta_min_pred, n_frustrated = delta_at_minimum(pbf, E, variables, T)

        results[T] = {"B": pi_B, "Z_exact": Z_exact,
                      "DA_exact": pi_DA_exact,
                      "Z_DA_exact": estimate_Z(E, T, pi_DA_exact),
                      "M": M, "R": R,
                      "D": D,
                      "J_term": J_term, "Phi": Phi_centred, "J_mod2": J_mod2,
                      "delta_hiT": delta_hiT, "DA_hiT": DA_hiT,
                      "delta_SA1": delta_SA1, "DA_SA1": DA_SA1,
                      "delta_SA2": delta_SA2, "DA_SA2": DA_SA2,
                      "rho_series": rho_series,
                      "delta_edge": delta_edge, "DA_edge": DA_edge,
                      "x_star": x_star, "delta_min_pred": delta_min_pred,
                      "n_frustrated": n_frustrated,
                      "DA_approx": approximate_pi_DA(M, R, pbf, variables, T, pi_DA_exact)}
        delta_exact = pi_DA_exact - pi_B

        def _r2(pred):
            return 1.0 - (delta_exact - pred).var() / delta_exact.var()

        print("T = %-10.4g  TV(DA_exact, B) = %.4f  T/std(dE) = %8.2f   R^2:"
              "  high T %+8.4f | SA 1st %+8.4f | SA 2nd %+8.4f | edge %+8.4f"
              "   rho = %.3f%s   %.2f sec"
              % (T, total_variation(pi_DA_exact, pi_B), T / gap_scale,
                 _r2(delta_hiT), _r2(delta_SA1), _r2(delta_SA2),
                 _r2(delta_edge), rho_series,
                 "" if rho_series < 1.0 else "  (SERIE DIVERGENT)",
                 time.time() - start_time_round))
        if delta_min_pred is not None:
            print("    x* = %d:  delta exact = %+.6e   (3/4)J(x*)/(2^n (n-1) T)"
                  " = %+.6e   rel. err = %.2e"
                  % (x_star, delta_exact[x_star], delta_min_pred,
                     abs(delta_min_pred - delta_exact[x_star])
                     / max(abs(delta_exact[x_star]), 1e-300)))
    return E, results


# PLOTTING

def hamming_order(E: np.ndarray, variables: int):
    """
    State order by Hamming distance to the GLOBAL MINIMUM.

    Leftmost the minimizer x* itself (distance 0), then its n direct neighbours
    (distance 1), then the binom(n,2) states at distance 2, and so on out to the
    complement of x* at distance n. Within one shell the states are sorted by
    increasing energy, so the order is deterministic.

    returns: order, distance
             order:    indices in plotting order
             distance: Hamming distance to x* per state (NOT reordered)
    """
    x_star = int(np.argmin(E))
    distance = np.array([bin(s ^ x_star).count("1") for s in range(len(E))])
    # lexsort takes the LAST key as the primary one: shell first, energy within
    order = np.lexsort((E, distance))
    return order, distance


def Plot_Stationary_Distributions(E: np.ndarray, results: dict, sort_mode: str = "hamming",
                                  log_y: bool = False):
    """
    Plots the exact pi^DA, the exact pi^SA (= Boltzmann) and the approximation
    together in ONE plot.

    Two rows on top of each other:
        row 1: the distributions pi(x)
        row 2: their difference, delta(x) = pi^DA exact - pi^SA exact, with a
               zero line. It integrates to 0 over all states, so it shows where
               the DA moves weight to and where it takes it from. D(x) shares
               that same axis, so both cross the same zero line; it starts
               hidden — click it in the legend to switch it on.

    With more than one temperature the columns are ordered by DESCENDING
    temperature: the hottest run is the leftmost column, the coldest the
    rightmost one.

    sort_mode:
        "hamming"   — x-axis ordered by Hamming distance to the global minimum:
                      the minimizer first, then its n neighbours, then the
                      binom(n,2) states two flips away, and so on. The shells are
                      separated by thin vertical lines. This shows the structure
                      around the ground state rather than a monotone decay.
        "boltzmann" — states sorted by decreasing pi^B, i.e. by increasing energy,
                      which makes the Boltzmann reference fall off monotonically.
        "index"     — plain state index.
    All curves use the SAME order in every case, so they stay comparable point by
    point, and the order never depends on T.
    """
    temperatures = sorted(results.keys(), reverse=True)      # hot -> cold, left -> right
    variables = int(round(np.log2(len(E))))
    distance = None
    if sort_mode == "hamming":
        order, distance = hamming_order(E, variables)
        x_title = "state (by Hamming distance to the global minimum)"
    elif sort_mode == "boltzmann":
        # pi^B = exp(-E/T)/Z is strictly decreasing in E, so ascending energy IS
        # the descending-pi^B order — for every T and without the float ties that
        # sorting the near-uniform pi^B of a very hot run would produce.
        order = np.argsort(E, kind="stable")
        x_title = "state (ordered by pi^B)"
    else:
        order = np.arange(len(E))
        x_title = "state index"
    x = np.arange(len(E))

    # Hover info. Columns 0-3 are the same for every temperature, 4-6 are filled
    # per column further down. The positions are FIXED — a column that has no
    # value carries nan — so the template indices never shift with the sort mode
    # or with the kind of pbf:
    #   0 bit vector   1 energy   2 state index   3 Hamming distance
    #   4 D(x)         5 J(x)     6 shell sum of delta
    dist_plot = distance[order] if distance is not None else np.zeros(len(E), dtype=int)
    customdata_base = [[str(tuple((s >> i) & 1 for i in range(variables))), E[s], int(s),
                        int(dist_plot[pos])] for pos, s in enumerate(order)]

    has_J = results[temperatures[0]].get("J_term") is not None
    hover_lines = ["x = %{customdata[0]}",
                   "E = %{customdata[1]:.6g}",
                   "value = %{y:.6g}",
                   "state index = %{customdata[2]}"]
    if distance is not None:
        hover_lines.append("Hamming distance = %{customdata[3]}")
    hover_lines.append("D(x) = %{customdata[4]:+.4e}")
    if has_J:
        hover_lines.append("J(x) = %{customdata[5]:+.6g}")
    if distance is not None:
        hover_lines.append("sum delta in shell = %{customdata[6]:+.4e}")
    hover_lines.append("Phi(x) - &lt;Phi&gt; = %{customdata[7]:+.6g}")
    hovertemplate = "<br>".join(hover_lines) + "<extra>%{fullData.name}</extra>"

    # the yardstick of the high temperature expansion: its parameter is
    # deltaE/T, so an absolute T carries no information on its own
    _st = np.arange(len(E))
    gap_scale = float((E[_st[:, None] ^ (1 << np.arange(variables))[None, :]]
                       - E[:, None]).std())

    column_titles = []
    for T in temperatures:
        res = results[T]
        parts = ["T = %.4g" % T,
                 "<sub>Z_exact = %.5g | Z_DA_exact = %.5g" % (res["Z_exact"],
                                                              res["Z_DA_exact"]),
                 "TV(DA_exact, B) = %.3f" % total_variation(res["DA_exact"], res["B"])]
        if "delta_hiT" in res:
            d_ex = np.asarray(res["DA_exact"], float) - np.asarray(res["B"], float)
            d_hi = np.asarray(res["delta_hiT"], float)
            line = ("T/std(dE) = %.2f | R&#178;(high T) = %+.4f"
                    % (T / gap_scale, 1.0 - (d_ex - d_hi).var() / d_ex.var()))
            if "delta_SA1" in res:
                d_sa = np.asarray(res["delta_SA1"], float)
                line += (" | R&#178;(SA 1st) = %+.4f"
                         % (1.0 - (d_ex - d_sa).var() / d_ex.var()))
            if "delta_SA2" in res:
                d_s2 = np.asarray(res["delta_SA2"], float)
                line += (" | R&#178;(SA 2nd) = %+.4f"
                         % (1.0 - (d_ex - d_s2).var() / d_ex.var()))
            if "delta_edge" in res:
                d_ed = np.asarray(res["delta_edge"], float)
                line += (" | R&#178;(edge) = %+.4f"
                         % (1.0 - (d_ex - d_ed).var() / d_ex.var()))
            parts.append(line)
            # the convergence radius of the SA series: above 1 the second order
            # is worse than the first, and saying so beats letting the reader
            # wonder why the orange curve got wilder.
            rho = res.get("rho_series")
            if rho is not None and np.isfinite(rho):
                parts.append("rho(SA-Serie) = %.3f%s"
                             % (rho, "" if rho < 1.0 else "  &#8212; DIVERGENT"))
        if res.get("n_frustrated") is not None:
            if res["delta_min_pred"] is None:
                parts.append("<sub>frustrated in %d of %d directions — no closed"
                             " form at x*</sub>" % (res["n_frustrated"], variables))
            else:
                d_ex = np.asarray(res["DA_exact"], float) - np.asarray(res["B"], float)
                parts.append("<sub>unfrustrated | x* rel. err = %.2e</sub>"
                             % (abs(res["delta_min_pred"] - d_ex[res["x_star"]])
                                / max(abs(d_ex[res["x_star"]]), 1e-300)))
        column_titles.append("<br>".join(parts) + "</sub>")

    fig = make_subplots(rows=2, cols=len(temperatures),
                        column_titles=column_titles,
                        row_titles=["pi(x)",
                                    "delta = pi^DA exact - pi^SA exact"],
                        shared_yaxes=True, shared_xaxes=True,
                        horizontal_spacing=0.03, vertical_spacing=0.10)

    fig.update_layout(title_text="Exact stationary distributions of SA and DA at constant"
                                 " temperature, states = " + str(len(E)))
    fig.update_annotations(font_size=11)     # keeps the column titles from overlapping

    for col, T in enumerate(temperatures, start=1):
        res = results[T]
        # row 1: the distributions themselves
        series = [("pi^DA exact", "DA_exact", dict(color="darkred"), res["DA_exact"]),
                  # pi^B is at the same time the exact pi^SA
                  ("Boltzmann = pi^SA exact", "B", dict(color="black", dash="dash"),
                   res["B"])]
        if "DA_approx" in res:
            # plotted exactly as approximate_pi_DA returned it
            series.append(("pi^DA approx", "DA_approx", dict(color="green"),
                           np.asarray(res["DA_approx"], dtype=float)))
        if "DA_hiT" in res:
            # pi^SA + the first order term, no fitted parameters
            series.append(("pi^DA high T (1/T)", "DA_hiT", dict(color="royalblue"),
                           np.asarray(res["DA_hiT"], dtype=float)))
        if "DA_SA1" in res:
            # pi^SA + first order of the perturbation around the SA chain
            series.append(("pi^DA 1st order around SA", "DA_SA1",
                           dict(color="darkgoldenrod"),
                           np.asarray(res["DA_SA1"], dtype=float)))
        if "DA_SA2" in res:
            # one more term of the same series, no fitted parameter
            series.append(("pi^DA 2nd order around SA", "DA_SA2",
                           dict(color="darkorange", dash="dot"),
                           np.asarray(res["DA_SA2"], dtype=float)))
        if "DA_edge" in res:
            # the same with Z = I: local, closed form, stable at low T
            series.append(("pi^DA edge flux (local)", "DA_edge",
                           dict(color="darkslateblue"),
                           np.asarray(res["DA_edge"], dtype=float)))

        # column 4: D(x), the net one step transition probability into x summed
        # over its neighbours (see net_transition_flux)
        D = np.asarray(res["D"], dtype=float)

        # column 5: the spin form of the quadratic part (QUBOs only)
        J_term = res.get("J_term")
        J_col = (np.asarray(J_term, dtype=float) if J_term is not None
                 else np.full(len(E), np.nan))

        # column 6: per Hamming shell, the summed excess weight of the DA over
        # the SA, delta(x) = pi^DA(x) - pi^SA(x). Both are the exact
        # distributions, not the approximation. The value is attached to EVERY
        # state of a shell, so hovering any point of that block shows its shell
        # total. Over all shells the sums cancel to 0, since both distributions
        # are normalized — they show WHERE the DA puts weight that the SA does not.
        delta = np.asarray(res["DA_exact"], dtype=float) - np.asarray(res["B"], dtype=float)
        shell_sum = np.full(len(E), np.nan)
        if distance is not None:
            for d in range(variables + 1):
                in_shell = distance == d
                if in_shell.any():
                    shell_sum[in_shell] = delta[in_shell].sum()

        # column 7: the same with the linear coefficients h_i mixed in
        # column 7: Phi(x) - <Phi>, the potential the high temperature term is
        # built from. Defined for every degree, unlike J.
        Phi_c = res.get("Phi")
        Phi_col = (np.asarray(Phi_c, dtype=float) if Phi_c is not None
                   else np.full(len(E), np.nan))

        customdata = [row_data + [float(D[s]), float(J_col[s]), float(shell_sum[s]),
                                  float(Phi_col[s])]
                      for row_data, s in zip(customdata_base, order)]

        for name, group, line, y in series:
            fig.add_trace(go.Scatter(x=x, y=np.asarray(y, dtype=float)[order],
                                     name=name, legendgroup=group,
                                     showlegend=(col == 1), line=line,
                                     customdata=customdata,
                                     hovertemplate=hovertemplate),
                          row=1, col=col)

        # J(x) and the two pi^SA scaled variants, all on the SAME axis as pi and
        # all hidden until switched on in the legend. They are energies, so much
        # larger than pi — switch pi off in the legend to see them on their scale.
        for key, name, colour in (("J_term", "J(x)", "teal"),
                                  ("Phi", "Phi(x) - <Phi>", "darkcyan"),
                                  ("J_mod2", "(J(x) - D(x)) * pi^SA(x)", "magenta")):
            values = res.get(key)
            if values is not None:
                fig.add_trace(go.Scatter(x=x, y=np.asarray(values, dtype=float)[order],
                                         name=name + "  (click to show)", legendgroup=key,
                                         showlegend=(col == 1), visible="legendonly",
                                         line=dict(color=colour),
                                         customdata=customdata,
                                         hovertemplate=hovertemplate),
                              row=1, col=col)

        # row 2: the difference itself, pi^DA exact - pi^SA exact
        fig.add_trace(go.Scatter(x=x, y=delta[order],
                                 name="delta = pi^DA - pi^SA", legendgroup="delta",
                                 showlegend=(col == 1), line=dict(color="purple"),
                                 customdata=customdata, hovertemplate=hovertemplate),
                      row=2, col=col)
        # the high temperature prediction of delta, on the SAME axis as delta so
        # the comparison is direct. This is the one to look at: it should lie on
        # top of the purple curve whenever T is well above std(deltaE).
        if "delta_hiT" in res:
            fig.add_trace(go.Scatter(x=x,
                                     y=np.asarray(res["delta_hiT"], dtype=float)[order],
                                     name="delta high T = -(Phi - <Phi>)/(2^n (n-1) T)",
                                     legendgroup="delta_hiT",
                                     showlegend=(col == 1),
                                     line=dict(color="royalblue"),
                                     customdata=customdata,
                                     hovertemplate=hovertemplate),
                          row=2, col=col)
        # the same for the SA expansion — valid roughly three times further down
        # in temperature than the high T formula, so where the royalblue curve
        # starts running away this one should still track delta.
        if "delta_SA1" in res:
            fig.add_trace(go.Scatter(x=x,
                                     y=np.asarray(res["delta_SA1"], dtype=float)[order],
                                     name="delta 1st order around SA = pi^SA dP Z",
                                     legendgroup="delta_SA1",
                                     showlegend=(col == 1),
                                     line=dict(color="darkgoldenrod"),
                                     customdata=customdata,
                                     hovertemplate=hovertemplate),
                          row=2, col=col)
        # one more term of the same series. Where rho < 1 this tightens the
        # goldenrod curve onto delta; where rho > 1 it overshoots it, which is
        # the visual signature of the series having left its radius.
        if "delta_SA2" in res:
            fig.add_trace(go.Scatter(x=x,
                                     y=np.asarray(res["delta_SA2"], dtype=float)[order],
                                     name="delta 2nd order around SA = pi^SA (I + M) Z S",
                                     legendgroup="delta_SA2",
                                     showlegend=(col == 1),
                                     line=dict(color="darkorange", dash="dot"),
                                     customdata=customdata,
                                     hovertemplate=hovertemplate),
                          row=2, col=col)
        # the local edge flux — the only one of the three that stays finite all
        # the way down in temperature
        if "delta_edge" in res:
            fig.add_trace(go.Scatter(x=x,
                                     y=np.asarray(res["delta_edge"], dtype=float)[order],
                                     name="delta edge flux = sum_k phi(x,k)"
                                          " [R(th_k x,k) - R(x,k)]",
                                     legendgroup="delta_edge",
                                     showlegend=(col == 1),
                                     line=dict(color="darkslateblue"),
                                     customdata=customdata,
                                     hovertemplate=hovertemplate),
                          row=2, col=col)
        # the closed form at the global minimum. A single point, not a curve —
        # the cancellation that produces it only happens at x*, where every
        # direction is uphill. Drawn only when the instance is unfrustrated.
        if res.get("delta_min_pred") is not None:
            pos = int(np.flatnonzero(order == res["x_star"])[0])
            fig.add_trace(go.Scatter(x=[float(pos)], y=[res["delta_min_pred"]],
                                     mode="markers",
                                     marker=dict(color="crimson", size=13,
                                                 symbol="star",
                                                 line=dict(color="black", width=1)),
                                     name="delta(x*) = (3/4) J(x*) / (2^n (n-1) T)"
                                          "   [unfrustrated only]",
                                     legendgroup="delta_min",
                                     showlegend=(col == 1),
                                     hovertemplate="closed form at x*<br>"
                                                   "state index = %d<br>"
                                                   "predicted = %%{y:+.6e}"
                                                   "<extra></extra>" % res["x_star"]),
                          row=2, col=col)
        # D(x) on the SAME axis as delta, so both share one zero line and the
        # comparison at zero is honest. Hidden until switched on in the legend.
        fig.add_trace(go.Scatter(x=x, y=D[order],
                                 name="D(x)  (click to show)", legendgroup="D",
                                 showlegend=(col == 1), visible="legendonly",
                                 line=dict(color="orange"),
                                 customdata=customdata, hovertemplate=hovertemplate),
                      row=2, col=col)

        fig.add_hline(y=0.0, line_width=1, line_color="rgba(120,120,120,0.6)",
                      row=2, col=col)

        if distance is not None:
            # one thin line where the Hamming shell changes, so the blocks
            # (1, n, binom(n,2), ...) are visible as blocks
            for row in (1, 2):
                for boundary in np.flatnonzero(np.diff(dist_plot)) + 1:
                    fig.add_vline(x=float(boundary) - 0.5, line_width=1,
                                  line_color="rgba(120,120,120,0.35)",
                                  row=row, col=col)

        fig.update_xaxes(title_text=x_title, row=2, col=col)

    if log_y:
        fig.update_yaxes(type="log", row=1, col=1)   # row 2 changes sign, keep it linear
    fig.update_yaxes(title_text="pi(x)", row=1, col=1)
    fig.update_yaxes(title_text="delta(x),  D(x)", row=2, col=1)
    return fig


if __name__ == "__main__":
    # parameters
    variables = 8                   # 2^variables states, must stay <= max_states
    degree = 2
    max_states = 512
    seed_gen = 4132                # seed of the random instance, nothing else
    temperatures = [1000,10,300,50,8,3,1,2]     # one plot column per entry
    negative_couplings = True        # True -> createPoly_negative (mixed signs)

    print("variables = " + str(variables))
    print("temperatures = " + str(temperatures))

    start_time_gen = time.time()
    if negative_couplings:
        pbf = createPoly_negative(variables, degree, 1, seed=seed_gen)
    else:
        pbf = createPoly(variables, degree, 1, seed=seed_gen)
    pbf = Sort_pbf(pbf, degree)
    print("feature generation time:" + str(time.time() - start_time_gen))
    print("size pbf: " + str(len(pbf.keys())))

    E, results = run_stationary_experiment(pbf, variables, temperatures,
                                           max_states=max_states)

    fig = Plot_Stationary_Distributions(E, results, sort_mode="hamming", log_y=False)
    fig.show()
