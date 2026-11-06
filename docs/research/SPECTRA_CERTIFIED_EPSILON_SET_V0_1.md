# SPECTRA Certified Epsilon-Optimal Sets v0.1

**Status:** formal research specification; no production API or set-level guarantee yet.  
**Issue:** #10688.  
**Parent selection contract:** robust scalar certificate in `ranking.rs`.

## 1. Why a set instead of a forced winner?

A deterministic tie-break is useful for reproducibility, but it can create an impression of uniqueness when the evidence supports multiple near-equivalent actions. Design selection, experimental protocols and operational planning frequently benefit from knowing which alternatives are already defensible, which remain unresolved, and which are excluded.

This is a distinct problem from merely returning all arms that have not been eliminated. A candidate can remain plausible without being certified epsilon-optimal.

Relevant comparator: Nonaga et al., *Risk-Averse Best Arm Set Identification with Fixed Budget and Fixed Confidence* (ACML 2025, PMLR 304). Their formulation uses a mean-variance reward-risk criterion. SPECTRA's finite-scenario Maximin and lower-tail CVaR objectives require a matched-objective comparison and separate proof.

## 2. Input and simultaneous coverage contract

Reuse a fully verified `ScoreBoundsReceipt` and its frozen action/scenario snapshots. The existing simultaneous utility intervals induce robust action intervals

`[L(a), U(a)]`

for every action in the declared finite candidate set and the selected robust objective (Maximin or lower-tail CVaR).

The set certificate assumes the same valid joint coverage event as the scalar ranking certificate. It must not treat the point-selection result as a substitute for simultaneous intervals, and must not introduce new confidence levels by reusing already-spent alpha.

Model-adequacy, operational-mapping, scenario-support and inference-risk qualifications remain independent gates. An action blocked by a mandatory gate must be classified as blocked/not-evaluable, not promoted into any evidence-supported recommendation set.

## 3. Three disjoint decision sets

Define

`U* = max_b U(b)`

`L* = max_b L(b)`

and for every action `a` the simultaneous upper bound on its robust regret:

`R_U(a) = max(0, U* - L(a))`.

Classify each otherwise eligible action as follows.

### Certified epsilon-optimal

`S_cert = { a : R_U(a) <= epsilon }`

Every action in this set individually satisfies the same robust epsilon-optimality guarantee as the scalar selected action. On the simultaneous coverage event,

`true_best - true_utility(a) <= U* - L(a) <= epsilon`.

This is a **safe recommendation set** under the declared objective and frozen scenario scope. It may contain one or multiple actions; it may be empty when no action yet meets the threshold.

### Certified excluded

`S_excluded = { a : U(a) < L* - epsilon }`

For every action in this set, the best true robust utility is at least `L*` and the action's true robust utility is at most `U(a)`. Thus it is more than epsilon below the best on the same simultaneous coverage event.

The inequality is strict. Equality remains unresolved.

### Unresolved

`S_unresolved = A_eligible \ (S_cert union S_excluded)`

These actions have not been proved epsilon-optimal and have not been proved epsilon-suboptimal. Do not label them winners, losers, safe alternatives or statistically equivalent.

The sets are disjoint because `L(a) <= U(a)` and `L* <= U*`. They partition the eligible action set.

## 4. Guarantee statement

Assume the simultaneous confidence intervals cover every frozen action's true robust utility with probability at least `1 - alpha`, under the registered score-sampling and scenario contract.

Then with probability at least `1 - alpha`:
1. Every action in `S_cert` is epsilon-optimal over the frozen candidate set and robust scenario objective.
2. No truly epsilon-optimal action is in `S_excluded`.
3. `S_unresolved` contains the remaining actions whose membership cannot be settled by the current intervals.

This does **not** prove that all actions in `S_unresolved` are epsilon-optimal. It does not prove scenario-bank coverage, model adequacy, operational realizability, or deployment validity.

A separate proof is required before these classifications are exposed as a production decision receipt.

## 5. Example

Suppose three eligible actions have robust intervals:

| Action | Lower | Upper |
|---|---:|---:|
| A | 0.90 | 0.96 |
| B | 0.88 | 0.94 |
| C | 0.50 | 0.70 |

With epsilon = 0.10:

- `U* = 0.96`
- `L* = 0.90`
- A has regret upper bound 0.06: certified epsilon-optimal.
- B has regret upper bound 0.08: also certified epsilon-optimal.
- C has upper bound 0.70, below `L* - epsilon = 0.80`: certified excluded.
- `S_cert = {A, B}`, `S_unresolved = {}`, `S_excluded = {C}`.

The scalar policy still selects A via its deterministic tie-break, but the set receipt communicates that B is also defensible at the registered tolerance.

For overlapping intervals where no action is independently certified and no action can be excluded, the entire eligible set is unresolved. That is not a failure of the set API; it is an honest representation of insufficient evidence.

## 6. Required typed receipt

A future `RobustEpsilonOptimalSetCertificate` should include:

- method ID and version;
- source score receipt digest / scope;
- action and scenario set digests;
- robust objective and epsilon;
- familywise confidence and score-bound method ID;
- selected scalar incumbent ID for compatibility;
- `U*` and `L*`;
- per-action `L(a)`, `U(a)`, and `R_U(a)`;
- sorted certified, unresolved and excluded action IDs;
- blocked / not-evaluable actions with explicit gate reasons, separate from the three statistical sets;
- replay method/version and deterministic ordering.

A verifier must reconstruct the source score receipt, re-run the scalar robust interval aggregation, recompute all regret upper bounds and membership decisions, and compare the complete receipt. Caller-provided digest strings remain opaque commitments unless cryptographic authentication is separately designed.

## 7. Tests and adversarial controls

- A unique winner with separated intervals produces one certified action and excludes clear losers.
- Exact ties with sufficiently narrow intervals certify multiple alternatives.
- Exact ties with wide intervals remain unresolved rather than being falsely certified.
- Epsilon boundary equality is classified as certified for `R_U(a) == epsilon` and unresolved for `U(a) == L* - epsilon`.
- Confirm no action can simultaneously be certified and excluded.
- Randomized interval corpora check disjointness and partition completeness.
- Maximin and lower-tail CVaR; fractional tail mass, scenario ties and nonuniform weights.
- Tampered interval, action-set, method, epsilon, class membership and digest fields fail replay.
- A blocked model-adequacy or operational-mapping action never enters `S_cert`.
- Compare with risk-averse best-arm set identification on a matched mean-variance sub-benchmark; report objective mismatch in all other comparisons.

## 8. Follow-on multiobjective Pareto set (separate version)

Do not hide utility, information, representativeness and participant burden inside an arbitrary weighted sum. A later extension should define per-objective direction, units, frozen range, robust aggregation, simultaneous intervals and epsilon-dominance, then distinguish certified dominated, certified nondominated, unresolved and blocked candidates.

This multiobjective Pareto problem requires separate semantics and error accounting; it is not implied by the scalar epsilon-optimal set certificate specified here.

## References

- Nonaga, S., Tabata, K., Mizuno, Y. & Komatsuzaki, T. (2025). *Risk-Averse Best Arm Set Identification with Fixed Budget and Fixed Confidence*. Proceedings of ACML 2025, PMLR 304:1182–1197. https://proceedings.mlr.press/v304/nonaga26a.html
- Yang, J., Tan, V. Y. F. & Jin, T. (2026). *Best Arm Identification with Minimal Regret*. Journal of Machine Learning Research 27(86):1–41. https://www.jmlr.org/beta/papers/v27/24-1612.html
- Jourdan, M., Delahaye-Duriez, A. & Réda, C. (2026). *An Anytime Algorithm for Good Arm Identification*. Journal of Machine Learning Research 27(19):1–90. https://www.jmlr.org/papers/v27/24-0680.html
