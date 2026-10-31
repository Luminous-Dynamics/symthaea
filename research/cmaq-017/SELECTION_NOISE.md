# CMAQ-017 denoised best-of-N selection and continuation-noise exploitability

**Status (2026-10-31):** executable selection-noise metric primitives and a proposed continuation protocol. The current CMAQ-017 planner/report is unchanged; this increment does not yet inject repeated continuation seeds into its existing episode runner or claim a benchmark result.

## Research trigger

The CMAQ-017 v0.1 environment is deterministic conditional on an episode seed and step: the true transition disturbance is derived from that fixed seed and step. The current episode runner is tested for exact repeatability. This is appropriate for deterministic model-error experiments, but it cannot yet isolate a planner's tendency to select a lucky candidate continuation.

A directly relevant new preprint, *How Much Can Best-of-N Help a World Action Model? Measuring Selectable Headroom and Realizing It in Closed Loop* (arXiv:2610.37196, 27 Oct 2026), identifies this confound. If each candidate is executed once and the largest observed outcome is selected, candidate quality is mixed with the randomness of what happens after the candidate. The paper proposes reproducible branching, repeated continuations, common random numbers and held-out continuations; it reports that one-shot best-of-N can overstate usable selectable headroom. This is a new external protocol to replicate, not a result that can be copied into CMAQ's scorecard. https://arxiv.org/abs/2610.37196

## Paradigm shift: continuation noise is an exploitability surface

CMAQ-017 currently measures how more planner search can exploit misspecified dynamics and unsupported regions. The same optimizer can exploit stochastic noise: as candidate count grows, it can select the candidate that received the luckiest continuation rather than the action with the best expected outcome.

Keep these causes separate:

- **Candidate main effect:** a candidate action is consistently better across continuations.
- **Continuation main effect:** one environment continuation/tape is globally easier or harder for every candidate.
- **Candidate × continuation interaction:** a candidate is unusually good or bad under a particular continuation. With one observation per matrix cell this also contains cell-level residual noise; it is not separately identified as a physical mechanism.
- **Selection optimism:** the same noisy outcomes are used both to choose a candidate and to estimate the benefit of that choice.

## Protocol

For each frozen observed history:

1. Freeze the candidate action/chunk bank before revealing continuation outcomes.
2. Branch all candidates from the same exact state/history.
3. For each candidate, run repeated continuations with a separate continuation identifier. A column represents the same continuation seed/tape across candidates only where the simulator permits valid common-random-number coupling.
4. Predeclare a selection partition and a scoring partition. The minimal executable diagnostic uses leave-one-continuation-out cross-fitting: select using all columns except j, then score the selected candidate on column j.
5. Compare the cross-fitted held-out value against the naive single-continuation best-of-N value. Their difference is the selection-optimism diagnostic.
6. Rotate held-out continuations or use a separate immutable held-out panel. Do not tune the candidate generator, planner, selection rule or thresholds on the final qualification panel.
7. Repeat the paired design across independent episode histories and held-out mechanism families. Bootstrap at the episode/history cluster; continuations are nested observations, not independent histories.
8. Charge every candidate model evaluation and every physical/simulator continuation to the resource ledger.

A starting continuation-count sweep is J = 1, 3, 5, 9 and 17, with the one-continuation arm retained as the intentionally optimistic reference. J=17 is motivated by the new paper's reported repeated-continuation protocol, not a universal minimum. Choose the actual qualification count using a blinded variance/sample-size pilot.

## Executable increment in this branch

The new src/selection_noise.rs provides a dependency-free outcome-matrix diagnostic:

- naive best-of-N mean (select and score on the same continuation);
- leave-one-continuation-out cross-fitted selected mean;
- a predeclared row-0 baseline mean and best empirical candidate mean;
- naive apparent gain, held-out gain and the optimism gap;
- balanced two-way candidate, continuation and candidate-by-continuation variance shares;
- selected candidate identity for each held-out fold;
- fail-closed matrix shape and finite-value validation.

The tests include an exact equal-mean counterexample where one-shot best-of-N reports +7.5 apparent gain while held-out selection scores -7.5, producing a 15.0 optimism gap. A stable +2.0 candidate advantage remains +2.0 when scored on held-out continuations. These are unit-test fixtures for the estimator, not CMAQ-017 simulator results.

## Integration into the next CMAQ-017 report version

The follow-on runner should preserve the existing deterministic model-error lanes and add a separate stochastic continuation lane:

- candidate counts K = 8, 32 and 64 (matching existing planner budgets);
- horizons 1, 4 and 8, with execution blocks kept distinct;
- continuation counts J = 1, 3, 5, 9 and 17 after a sample-size pilot;
- multiple independent episode histories and mechanism families;
- held-out continuation value, naive one-shot optimism, variance components and candidate-selection frequencies;
- risk/tail outcomes and termination false-safe rate by K/J/horizon;
- matched direct policy and true D1 cost baseline, with continuation simulator work included;
- shuffled-future, stale-model, random-selection and evaluator-only oracle controls.

The evaluator must keep world truth and unselected continuation outcomes hidden from the candidate. A planner must not be able to query held-out continuations or use their identifiers to infer outcomes.

## Qualification targets

Targets to freeze before sealed evaluation:

- Under equal-mean candidate null fixtures, the held-out gain should be statistically consistent with zero while naive best-of-N optimism grows with candidate count.
- Under a known stable candidate-effect fixture, the held-out estimator should preserve the true expected gain.
- Variance components should sum to one on non-degenerate balanced matrices and identify candidate-only, continuation-only and interaction-only fixtures.
- Report optimism gap against K and J; no single pooled scalar hides scenario dependence.
- Search expansion must not produce an apparent value gain that disappears on held-out continuations.
- The selected planner must still beat direct D0 and genuinely compute-matched D1 on realized held-out outcomes, after charging continuation cost.
- False-safe and lower-tail failures must not worsen as K, J or horizon increase without an explicit, preregistered risk trade-off.

## Boundaries

The current direct_search_control still matches candidate-sequence count only; this branch does not convert it into D1. No hardware FLOP/latency/energy matching is claimed. The metric module does not run a simulator, create real-world counterfactual ground truth or change the main CMAQ-017 report schema. It is a tested analytical substrate for a later simulator adapter.

WorldOracle's unresolved validation-data access restriction remains unchanged. No external dataset is downloaded or used by this branch.
