# Affective Emergence v0.2 — Functional Evaluation and Exploratory Promotion

Status: **normative design-only / blocked on Native Interoception v0.1 qualification**

This contract defines how the finite exploratory E00–E11 candidate set may be evaluated and reduced without selecting the candidate that merely looks most emotion-like.

The first exploratory study is allowed to produce no winner, several non-redundant candidates, or a result that simple baselines are sufficient.

## 1. Principle

Candidate computation and candidate evaluation are different evidence stages.

At cut point `t`:

1. compute/freeze the candidate payload using only the allowed prefix;
2. only after the candidate payload is immutable may the analysis derive later realized regulatory outcomes from the suffix;
3. use those outcomes to assess functional prediction/discrimination;
4. never feed suffix-derived evaluation targets back into the candidate definition or preprocessing inside the same lineage.

Retrospective evaluation targets are allowed as **outcomes**. They are forbidden as candidate inputs.

## 2. Why a functional target is required

Without an external selection criterion, exploratory reduction could become:

> choose the candidate whose plot looks most like intuitive valence.

That is forbidden.

The selection question is instead:

> Does any prefix-causal candidate contain reproducible, non-redundant information about later regulatory outcomes beyond simpler state/load/history/confidence baselines?

This is a functional-regulatory criterion, not an emotion label.

## 3. Frozen realized outcome vector

For each eligible candidate cut point, derive a retrospective `RealizedRegulatoryOutcomeVector` only after the candidate artifact is frozen.

Initial outcome components:

### Y0 — next-step realized viability change

Using W1 viability-only / A3 fixed-set burden:

`Y0(t) = H(t) - H(t+1)`

under the frozen sign convention where positive means realized improvement.

This is undefined when `t+1` is not available.

### Y1 — realized 16-step cumulative viability exposure

Over the next 16 realized post-cut-point states, compute the undiscounted sum of W1/A3 burden:

`Y1(t) = Σ_{k=1..16} H(t+k)`.

This target is `Unavailable(InsufficientFutureSupport)` when the full 16-step realized window is not present; do not shorten the horizon after seeing results.

### Y2 — realized first viability-breach latency

Record the first realized post-cut-point step in `1..=16` where any channel lies outside its viability range under the frozen boundary rule.

If no breach occurs in the 16-step window, emit typed `NoRealizedBreachWithinWindow` rather than zero/infinity.

### Y3 — realized terminal viability burden

`Y3(t) = H(t+16)` under W1/A3 semantics.

Unavailable when the terminal point is absent.

The outcome-vector schema/digest is evidence-critical. Adding, removing, or redefining Y components after exploratory outputs are inspected requires a new analysis/promotion identity.

## 4. Outcome isolation

The evaluator producing E00–E11 must not have access to Y0–Y3 or any suffix state used to construct them.

The analysis stage may join frozen candidate payloads with frozen outcome artifacts by opaque scenario/cut-point identity only after both upstream artifacts exist.

A candidate implementation whose source/dependency graph can read the outcome artifact fails the information firewall even if the final numeric value happens to match a legal implementation.

## 5. No single universal score

Do not collapse Y0–Y3 into one post-hoc weighted scalar.

Candidate evaluation reports a multi-target profile.

A candidate may be useful for one regulatory function and not another. For example:

- E07 breach latency may relate strongly to Y2 but weakly to Y1;
- E08 prospective cumulative exposure may relate to Y1/Y3 but not Y0;
- E04 residual may carry different information from E03 realized change.

Those differences are scientifically informative and must not be hidden by an arbitrary aggregate score.

## 6. Deterministic comparison metrics

Because the initial scenario program is deterministic, do not manufacture independent-sample p-values from scenario arms.

Use prospectively defined deterministic summaries such as:

- signed rank/order concordance where an ordering is meaningful;
- exact/equivalence classification under frozen tolerances;
- worst-case signed margin on registered contrasts;
- count/fraction of registered discriminator obligations satisfied;
- paired incremental margin beyond required simpler baselines;
- explicit failure-region/scenario-family report;
- target coverage vector across Y0–Y3.

The exact summary formulas/tolerances are frozen in the later `ExploratoryAnalysisPlan` before candidate outputs are inspected.

## 7. Incremental-information rule

A candidate is not promotion-eligible merely because it correlates with a future outcome.

For each candidate role, use its required simpler baseline obligations from the minimal candidate set.

Examples:

- E04 must add information beyond E03 actual change and E02 current drive;
- E05 must remain distinct from E04 one-step residual and E06 horizon turnover;
- E07 must remain distinct from E01 current burden and E02 drive;
- E08 must remain distinct from E01 current burden, E02 drive, and E07 urgency;
- E10 must justify any value beyond E01 rather than winning because precision rescales severity;
- E11 must add information beyond E01/E03 on X09-B while remaining explicitly external-history-derived.

If the complex candidate and simpler baseline are observationally equivalent under all registered discriminators and functional targets, the simpler explanation wins under parsimony.

## 8. Promotion states

The exploratory evaluator should classify each non-null candidate into one of a small set of states rather than producing an unrestricted leaderboard.

Suggested states:

- `IntegrityDisqualified` — candidate/evidence boundary failed;
- `NumericallyInvalid` — non-finite/undefined behavior outside the locked typed rules;
- `EquivalentToSimplerBaseline` — no registered incremental information;
- `InsufficientDiscrimination` — battery could not separate required alternatives;
- `PolicyFragile` — apparent value depends materially on one simple forecast policy under the locked X11 sensitivity rule;
- `SingleTargetIncremental` — adds reproducible information for at least one Y target beyond required baselines;
- `MultiTargetIncremental` — adds reproducible information for multiple independent Y targets beyond required baselines;
- `NoUniqueRepresentative` — remains in a non-trivial equivalence class with another candidate;
- `ExploratoryShortlistEligible` — passes all required integrity/discrimination gates and satisfies the prospectively frozen promotion thresholds.

These states do not imply affect or valence.

## 9. Shortlist rule

Do not require one winner.

The first exploration should produce a small set of non-redundant candidate classes.

Default shortlist policy:

1. remove integrity/numeric failures;
2. collapse candidates into observational equivalence classes under the frozen tolerance/fingerprint rules;
3. apply required simpler-baseline comparisons;
4. preserve target-specific specialists rather than force a universal score;
5. choose at most one representative per equivalence class using the prospective parsimony rule;
6. carry forward no more than **three** candidate representatives into confirmatory-design consideration.

If more than three classes remain eligible, the exploratory design is insufficiently selective; do not choose three by intuition. Add a prospectively justified new discrimination stage/new lineage.

If zero remain, preserve the null result.

## 10. Parsimony within an equivalence class

When two candidates are functionally/evidentially equivalent under the locked design, prefer the candidate requiring less information/authority.

Prospective preference order for otherwise equivalent candidates:

1. H0 current-state-only over H1 replay-history;
2. no forecast model over forecast-dependent candidate when functional coverage is equal;
3. no fitted preprocessing over fitted preprocessing;
4. simpler fixed channel/temporal support over a more elaborate support when the outputs are equivalent;
5. simpler baseline relation over a derived relation when all registered outputs/targets are equivalent.

This preference is used only to select a representative from an established equivalence class. It cannot erase a meaningful difference between candidates.

## 11. Forecast-policy robustness

Primary forecast-bearing candidates use `ObservedDrivePersistence`.

X11 evaluates the corresponding diagnostic outputs under `NativeZeroInputRecovery` and `KinematicVelocity`.

Do not require identical values across policies—the policies represent different assumptions.

Instead freeze a policy-fragility rule before exploratory output inspection. A candidate becomes `PolicyFragile` when its claimed incremental relation reverses or disappears across the prospectively required simple-policy diagnostics beyond the frozen tolerance/coverage criterion.

A policy-fragile candidate may remain scientifically interesting but cannot be promoted as a robust general regulatory observable without a new predictive-model justification.

## 12. Candidate-specific tautology guard

Do not count a relation as impressive merely because the candidate and evaluation target encode nearly the same quantity.

The analysis plan must label candidate-target pairs as:

- `PrimaryInformative`;
- `ExpectedNearTautologyDiagnostic`;
- `OrthogonalTarget`;
- `NotApplicable`.

Examples:

- E07 projected breach latency vs Y2 realized breach latency is a natural predictive diagnostic but should not alone justify broad promotion;
- E08 projected cumulative exposure vs Y1 realized cumulative exposure is similarly close to its stated functional purpose;
- evidence that E07/E08 add information on orthogonal Y targets beyond baselines is stronger than success only on their nearest target.

Promotion thresholds must require at least one non-tautological/orthogonal discrimination beyond any declared near-tautology diagnostic when making a broad multi-target claim.

## 13. Blinded selection

Exploratory candidate reduction should operate on neutral candidate IDs/factor coordinates and opaque scenario/arm identities.

Semantic condition labels, affect words, and human narrative descriptions are unavailable to the selection algorithm.

The semantic mapping can be consulted later for qualitative interpretation, but it cannot change candidate promotion states in the same exploratory lineage.

## 14. Exploratory-to-confirmatory boundary

`ExploratoryShortlistEligible` is not confirmatory evidence.

After exploration:

- freeze the candidate equivalence/discrimination report;
- freeze the functional-target evaluation report;
- choose at most three shortlist representatives under this contract;
- design a new confirmatory holdout specifically capable of challenging those representatives and their strongest baselines;
- freeze a new primary candidate/baseline set and analysis plan;
- only then generate confirmatory data.

A candidate that looks better after confirmatory unblinding cannot replace the frozen primary inside that confirmatory lineage.

## 15. Machine-readable artifacts

Future implementation should define:

- `RealizedRegulatoryOutcomeVector`;
- `CandidateFunctionalEvaluation`;
- `CandidateEquivalenceClassReport`;
- `ExploratoryPromotionReport`.

The promotion report binds:

- design-contract-registry digest;
- E00–E11 candidate-set manifest digest;
- X00–X11 scenario-battery manifest digest;
- malicious-fixture validation report digest;
- candidate payload/report digests;
- Y0–Y3 outcome artifact digests;
- target-classification/tautology map;
- exact comparison/tolerance rules;
- forecast-policy sensitivity report;
- equivalence classes;
- per-candidate promotion state;
- shortlist representatives (0–3);
- canonical SHA-256.

## 16. Claim boundary

This contract can support the statement that a prefix-causal regulatory candidate carried non-redundant functional information about later regulatory outcomes beyond specified simpler baselines under a locked deterministic design.

It still does not establish emotion, subjective valence, feeling, mood, suffering, sentience, consciousness, or phenomenal experience.