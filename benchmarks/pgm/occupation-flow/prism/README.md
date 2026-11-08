# PRISM 4.11 comparator: finite-horizon interval Markov decisions

This fixture is the external model-checking counterpart to Rust
occupation_flow.rs.

## Scope

The model is a two-state, three-reward-layer interval DTMC. Its clock variable
makes the two uncertain transition matrices stage-specific. A final
zero-reward terminal state turns the finite-horizon cumulative utility into
PRISM total reward over an absorbing model.

The model's interval rows are the same rational rows used by the Rust fixture:

| Stage | Source | Destination 0 | Destination 1 |
|---|---|---:|---:|
| 0 | 0 | [2/5, 4/5] | [1/5, 3/5] |
| 0 | 1 | [3/10, 7/10] | [3/10, 7/10] |
| 1 | 0 | [1/2, 9/10] | [1/10, 1/2] |
| 1 | 1 | [1/5, 3/5] | [2/5, 4/5] |

The initial state is state 0 at stage 0. Every row's interval box is
intersected with the probability simplex. Each stage/state row is distinct
because of the clock variable, so this fixture does not require a shared
transition parameter across time.

## Run

Use PRISM 4.11 or later:

    prism interval_markov_2state_3period.pm interval_markov_2state_3period.props

Record the PRISM version, engine, precision settings, runtime and reported
values. This repository does not currently include a PRISM executable or
runtime output for this fixture; the model and properties are authored
benchmark inputs, not a claimed PRISM pass.

## Expected exact rational results

The Rust occupation-flow model has:
- Action A utility interval: [9/25, 32/25].
- Action B utility interval: [2/5, 4/5].
- Worst-case regret of A: 11/25.
- Worst-case regret of B: 22/25.

The PRISM reward structures action_a and action_b encode the same
state-score functions. Compare Rmin and Rmax results to the exact
rational values above. PRISM's numeric result is an external differential
baseline; Symthaea's rational primal/dual certificate remains the proof
object for its compiled LP.

## Pairwise regret encoding

The raw state-score differences can be negative, while PRISM's total-reward
queries require nonnegative rewards (except for instantaneous reward
properties). The fixture uses path-constant shifts:

- Regret of selecting A: add 2 to the comparator-minus-selected state score
  at each complete path (one unit at t=1 and one at t=2). The resulting
  reward is nonnegative and its maximum is 2 + 11/25 = 61/25.
- Regret of selecting B: add 1 on every complete path at t=1. The resulting
  reward is nonnegative and its maximum is 1 + 22/25 = 47/25.

Subtract the corresponding path-constant after PRISM returns the maximum.
Because each run includes exactly one reward-bearing state at each of t=1
and t=2, the offsets are independent of the uncertain trajectory and do not
change which transition realization maximizes regret.

This gives a valid shared-dynamics comparison; it does not solve general MDP
regret when each action induces a different transition kernel.

## Semantic caveats

- This is an IDTMC-style interval model with one command per state, not an
  IMDP controller-synthesis problem with an additional strategic quantifier.
- The transition rows are stage-local by construction through the clock.
- The terminal clock state has zero reward and is absorbing.
- PRISM results may use floating-point numerical solving. Do not report them
  as exact rational certificates unless the specific engine/mode establishes
  exactness for this query.
- Model-checker value agreement is distinct from independent LP
  primal/dual certificate verification.

## References

- PRISM 4.11 changelog (released 9 October 2026):
  https://github.com/prismmodelchecker/prism/blob/master/CHANGELOG.txt
- PRISM uncertain-model syntax:
  https://www.prismmodelchecker.org/manual/ThePRISMLanguage/UncertainModels
- PRISM uncertain reward-property semantics:
  https://www.prismmodelchecker.org/manual/PropertySpecification/UncertainModels
- PRISM reward structure selection:
  https://www.prismmodelchecker.org/manual/PropertySpecification/Reward-basedProperties
