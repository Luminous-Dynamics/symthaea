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
- Action C utility interval: [29/50, 21/25].
- Pairwise worst-case regrets: A->B 11/25, A->C 11/50,
  B->A 22/25, B->C 11/25, C->A 11/25, C->B 11/50.
- Complete minimax-regret action set: {A, C}, with value 11/25.

The PRISM reward structures action_a, action_b and action_c encode the
same state-score functions. Compare all six Rmin/Rmax utility endpoints and
all six Rmax pairwise-regret results to the exact rational values above.
PRISM's numeric result is an external differential baseline; Symthaea's
rational primal/dual certificates remain the proof objects for its
compiled LPs.

## Pairwise regret encoding

Some raw score differences are negative, while this PRISM total-reward
fixture uses nonnegative reward structures. Each directed pair therefore
adds a path-constant offset and removes it after the maximum total reward:

| Selected -> comparator | Raw worst-case regret | Shift | PRISM shifted maximum |
|---|---:|---:|---:|
| A -> B | 11/25 | 2 | 61/25 |
| A -> C | 11/50 | 1 | 61/50 |
| B -> A | 22/25 | 1 | 47/25 |
| B -> C | 11/25 | 1/2 | 47/50 |
| C -> A | 11/25 | 1/2 | 47/50 |
| C -> B | 11/50 | 1 | 61/50 |

Each path visits exactly one state in each reward-bearing layer, so the
offset is independent of the uncertain transition realization. The complete
minimax-regret result is a tie between A and C at 11/25; B's worst-case
regret is 22/25.

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
