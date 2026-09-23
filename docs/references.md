# References

Papers, Fujitsu sources, benchmark data and libraries cited anywhere in this
repository's documentation.

- [Papers](#papers)
- [Fujitsu Digital Annealer](#fujitsu-digital-annealer)
- [Benchmark data](#benchmark-data)
- [Libraries and tools](#libraries-and-tools)

## Papers

- Aramon, M., Rosenberg, G., Valiante, E., Miyazawa, T., Tamura, H., Katzgraber,
  H.G. (2019). *Physics-Inspired Optimization for Quadratic Unconstrained
  Problems Using a Digital Annealer.* Frontiers in Physics 7:48.
  [DOI: 10.3389/fphy.2019.00048](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2019.00048)
  — the Digital Annealer algorithm, parallel trial and escape mechanism.
- Fukushima-Kimura, B.H., Kawamoto, N., Noda, E., Saburi, K., Tasaki, H. (2023).
  *Mathematical aspects of the digital annealer's simulated annealing
  algorithm.* Journal of Statistical Physics.
  [DOI: 10.1007/s10955-023-03179-3](https://link.springer.com/article/10.1007/s10955-023-03179-3)
  — [arXiv:2303.08392](https://arxiv.org/abs/2303.08392) — closed-form DA
  transition probabilities (formula (3.6) used in the stationary distribution
  toolkit) and the proof that the parallel-trial DA chain without escape
  mechanism converges under logarithmic cooling (Hajek-type result).
- Kirkpatrick, S., Gelatt, C.D., Vecchi, M.P. (1983). *Optimization by Simulated
  Annealing.* Science.
- Hajek, B. (1988). *Cooling Schedules for Optimal Annealing.* Mathematics of
  Operations Research.
- Aarts, E., Korst, J. (1989). *Simulated Annealing and Boltzmann Machines.*
  Wiley — finite-time schedule behind `"auto_sa"`.

## Fujitsu Digital Annealer

- Fujitsu press release, 15 May 2018:
  [Fujitsu Quantum-Inspired Digital Annealer Cloud Service](https://info.archives.global.fujitsu/global/about/resources/news/press-releases/2018/0515-01.html)
  — first generation, 1,024 bits.
- Fujitsu press release, 21 Dec 2018:
  [Next Generation Quantum-Inspired Digital Annealer Service](https://info.archives.global.fujitsu/global/about/resources/news/press-releases/2018/1221-01.html)
  — second generation, 8,192 bits, CMOS ASIC.
- Fujitsu press release, 9 Nov 2020:
  [Megabit-class Performance for Large-Scale Combinatorial Optimization Problems](https://info.archives.global.fujitsu/global/about/resources/news/press-releases/2020/1109-01.html)
  — 1,014,000-bit problem solved with multiple linked servers.
- [Fujitsu Computing as a Service — Digital Annealer User's Guide](https://portal.aispf.global.fujitsu.com/apidoc/da/jp/da-guide-en.html)
  — current service; CPU-side processing for scaling and energy recalculation.
- [Fujitsu — overview at postquantum.com](https://postquantum.com/quantum-computing-companies/fujitsu/)
  — fourth generation (2022), 100,000 variables (secondary source).

## Benchmark data

- SATLIB Benchmark Suite (3-SAT instances):
  https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html
- TSPLIB: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
- Dr. Manuel Schönberger, *Join Ordering via Quantum-Inspired Optimization.*
  VLDB 2024. https://github.com/lfd/vldb24

## Libraries and tools

- `lk_heuristic` — Lin-Kernighan-Helsgaun TSP heuristic:
  https://github.com/pfayle/lk_heuristic
- `p_tqdm` — Parallel tqdm progress bars: https://github.com/swansonk14/p_tqdm
