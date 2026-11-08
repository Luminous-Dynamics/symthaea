# IntervalMDP.jl exact-arithmetic comparator

This benchmark compares Symthaea's finite-horizon occupation-flow LP with
IntervalMDP.jl, a specialist interval Markov decision/process library.

## Pinned environment

- Julia: 1.11 or later
- IntervalMDP.jl: exactly 0.7.0 (compat pinned with =0.7.0; Julia package resolution is not run here, so no generated Manifest.toml is claimed)
- Arithmetic: Rational{BigInt}
- Algorithms: OMaximization() and VertexEnumeration()

The source repository is https://github.com/Zinoex/IntervalMDP.jl.
The package's own tests exercise interval Bellman/value iteration for
Float32, Float64, and Rational{BigInt}. Its API exposes finite-horizon
reward properties, robust value iteration, O-maximization and vertex
enumeration.

## Run

From this directory:

    julia --project=. -e 'using Pkg; Pkg.instantiate(); include("compare_occupation_flow.jl")'

The harness prints exact rational values, iteration count and residual for
each of six queries under both algorithms. It hard-fails if a value differs
from the preregistered rational oracle.

This repository/container has no Julia executable, so the harness is
authored but not executed. Do not treat the expected-value assertion as a
passed external benchmark until the script has been run in Julia 1.11+.

## Model mapping

The script uses the same eight-state time-expanded IMC as the Rust and PRISM
fixtures:

| Clock | State 1 | State 2 |
|---|---|---|
| t=0 | initial state 0 | state 1 |
| t=1 | state 0 | state 1 |
| t=2 | state 0 | state 1 |
| t=3 | terminal for state 0 | terminal for state 1 |

For each source state, the transition ambiguity set is a box intersected
with the probability simplex. Columns of IntervalMDP's lower/upper matrices
are source-state ambiguity sets; rows are destinations. This is the transpose
of the source/destination order used by the Symthaea Rust fixture.

The clock makes each stage/state pair a distinct source state. Therefore
time-varying stage transitions are represented as a stationary interval
chain over a larger state space, without introducing a shared transition
parameter across stages.

The horizon is three transitions. FiniteTimeReward(scores, 1, 3) evaluates
the reward at t=0,1,2,3; the t=3 terminal reward is zero. The canonical
Rust source-model SHA-256 for this exact initial distribution and transition
interval manifest is:

    b1fb89c3775613d7b9aeb20857e6eb1a15b4133419e6fa3a307ddec6c906c466

The same source digest is recorded in the companion PRISM expected-results
manifest.

## Exact query matrix

| Query | Mode | Exact expected value |
|---|---|---:|
| Action A minimum | Pessimistic | 9/25 |
| Action A maximum | Optimistic | 32/25 |
| Action B minimum | Pessimistic | 2/5 |
| Action B maximum | Optimistic | 4/5 |
| Action C minimum | Pessimistic | 29/50 |
| Action C maximum | Optimistic | 21/25 |
| Regret A vs B | Optimistic | 11/25 |
| Regret A vs C | Optimistic | 11/50 |
| Regret B vs A | Optimistic | 22/25 |
| Regret B vs C | Optimistic | 11/25 |
| Regret C vs A | Optimistic | 11/25 |
| Regret C vs B | Optimistic | 11/50 |

Action C is the pointwise midpoint of the A and B state-score vectors.
Its exact robust utility interval is [29/50, 21/25].

Regret is evaluated for every ordered pair using one shared transition
process: regret(a,b) = max_M [U_b(M) - U_a(M)]. The six values are:
A-vs-B 11/25, A-vs-C 11/50, B-vs-A 22/25, B-vs-C 11/25,
C-vs-A 11/25, C-vs-B 11/50. Therefore the complete minimax-regret
solution is a tie: actions A and C both attain 11/25; B attains 22/25.

IntervalMDP's finite-horizon reward support accepts signed reward vectors,
so no artificial path reward shift is needed. This is not a general MDP
regret problem where each action changes the transition kernel; the
comparison uses distinct state-score vectors under shared dynamics, matching
the current Symthaea occupation-flow API.

## Fair comparison

This is a particularly useful comparator because it checks both:
- O-maximization, a specialised exact interval-row Bellman method;
- vertex enumeration, an independent explicit local-polytope reference.

The algorithms solve the same finite-horizon interval Markov semantics as
the Symthaea fixture, but through dynamic programming rather than one global
occupation-flow LP.

Record:
- Julia and package versions;
- algorithm (OMaximization or VertexEnumeration);
- exact values, iterations, residual;
- cold and warm runtime separately;
- peak memory and allocations;
- state count, horizon, interval support sizes and rational bit widths.

The 8-state fixture is a correctness test, not a performance claim. The three-action extension tests complete pairwise coverage and tied minimax-regret decisions, not only a two-action special case. Scaling
experiments should vary state count, horizon, support density and rational
bit-width. Compare LP construction/solver/checker time against Bellman
iterations, and include the one-time compilation cost for Julia.

## Sources

- IntervalMDP.jl package repository:
  https://github.com/Zinoex/IntervalMDP.jl
- Package API and finite-horizon reward implementation:
  https://github.com/Zinoex/IntervalMDP.jl/blob/main/src/specification.jl
- O-max and vertex-enumeration algorithm declarations:
  https://github.com/Zinoex/IntervalMDP.jl/blob/main/src/algorithms.jl
- Exact rational Bellman/IMDP tests:
  https://github.com/Zinoex/IntervalMDP.jl/blob/main/test/base/bellman.jl
  https://github.com/Zinoex/IntervalMDP.jl/blob/main/test/base/imdp.jl
  https://github.com/Zinoex/IntervalMDP.jl/blob/main/test/base/vi.jl
