# KAPS differential benchmark: two specialist environments

This is the first external differential comparator for the finite-family exact
policy-portfolio solver in
`crates/bridges/symthaea-complexity-lab/src/finite_family_portfolio.rs`.

## Pinned competitor

- Repository: https://github.com/SUMI-lab/KAPS
- Upstream commit: `4d35eef6c442521139d1786e31c6d6fbf322788f`
- Paper: Lutz, Vos, Spaan & Lukina, *Optimizing Minimax Regret in Uncertain
  MDPs with Small Sets of Policies*, arXiv:2608.02509.
- KAPS implements the authors' nested branch-and-bound search: an outer
  best-first partition search assigns environments to policy clusters; each
  cluster is solved by a resumable inner policy branch-and-bound search.
  Cluster solvers are memoized and reused across outer search nodes. Heuristic
  incumbent policies and bounds guide pruning.
- The paper/code policy class is deterministic and memoryless. The benchmark
  below is one-step, so it needs no state-clock augmentation and matches that
  policy class exactly.

KAPS's published project uses Python 3.13+, NumPy/Gymnasium/Rich and exposes
the `MultiPolicySolver` "ours" algorithm. The upstream project also declares
Gurobi among its dependencies for optional MILP/CREG/CEMR baselines; this
fixture imports only the MDP/family classes and the KAPS "ours" solver. It
does not invoke the Gurobi-backed solvers.

## Exact fixture

Two models share one state, two actions, initial probability one, and
`gamma=0`:

| Model | Action 0 reward | Action 1 reward | Optimal value |
|---|---:|---:|---:|
| M0 | 1 | 0 | 1 |
| M1 | 0 | 1 | 1 |

Each model transitions back to the sole state with probability 1. Since the
horizon is one and `gamma=0`, KAPS's discounted infinite-horizon value is
identical to the one-step finite-horizon reward. All input/output values
(0 and 1) are exactly representable in float64.

The exact regret oracle is:

| Portfolio size | Exact minimax regret | Interpretation |
|---:|---:|---|
| k=1 | 1 | Either single policy fails completely in the other model |
| k=2 | 0 | The portfolio contains both specialist policies |

The finite-family Rust reference exhaustively checks
`C(2,1)+C(2,2)=3` portfolios. It uses checked rational arithmetic and pins
model-family, policy-manifest, policy-value and decision-certificate SHA-256
known answers. KAPS returns a numeric regret, lower bound, partition, policy
vectors and runtime; its numeric path is floating-point, not a rational proof
certificate.

## Run the competitor

Clone the pinned upstream revision:

```sh
git clone https://github.com/SUMI-lab/KAPS.git
cd KAPS
git checkout 4d35eef6c442521139d1786e31c6d6fbf322788f
```

From this benchmark directory, set `KAPS_ROOT` to that checkout and run
the comparator in the environment containing KAPS's runtime imports:

```sh
KAPS_ROOT=/path/to/KAPS python3 run_kaps_specialist_fixture.py
```

The script requires NumPy, Gymnasium and Rich; it uses only KAPS's `ours`
nested branch-and-bound solver. It hard-fails if the k=1 or k=2 regret differs
from the oracle, or if the returned partition does not cover both models.
This repository does not currently claim that KAPS has been installed or
executed here.

## Fair extension to multi-step finite-horizon models

Our Rust reference enumerates deterministic time-dependent Markov policies
`pi_t(s)`; KAPS expects deterministic memoryless policies `pi(s)`. For a
finite-horizon comparison, clock-augment each state to `(t,s)`, using
`H*S` decision states plus one absorbing terminal state:

1. Put the original initial distribution on the `t=0` clock layer.
2. For each original transition row `P_t(s,a,s')`, route mass from
   `(t,s)` to `(t+1,s')`.
3. Encode the time/state/action reward as a transition reward constant over
   every destination of that row.
4. From the last decision layer, route every action to the absorbing terminal
   state; the terminal state self-loops with zero reward.
5. Use the same discount factor, reward timing, model family and k.

A stationary deterministic policy over `(t,s)` is then in bijection with a
deterministic time-dependent Markov policy over `s`. Compare all model
transition/reward tensors after this transformation before attributing any
difference to the algorithms.

## What to record

Record the KAPS commit, Python and dependency versions, hardware, seed,
model-family digest, policy class, gamma, clock expansion, k, returned
partition, each policy, regret, lower bound, solver runtime, total wall time,
search-node counts where instrumented, and whether the lower/upper gap closed.
Run KAPS's nested search and the Rust exact enumeration under the same policy
class and environment family.

## Research direction

KAPS already provides a strong exact-search architecture. The opportunity
for Symthaea is not to claim superiority over it based on this tiny fixture.
The proposed next solver should combine:

- KAPS-style outer partition branch-and-bound;
- exact rational inner cluster-policy search;
- proof-carrying lower/upper bounds and independently replayable pruning;
- resumable per-cluster search with content-addressed memoization;
- dominance/profile quotienting for policies that have identical exact values
  across the finite family;
- complete tie reporting, or a compact exact representation of tied
  portfolios when the tie set is too large to materialize.

The first target is 100% agreement with exhaustive exact enumeration on sealed
small families. Only after this should we compare scaling, search nodes and
runtime against KAPS on the published frozen-lake, maintenance, windy-routing
and disaster-rescue families.
