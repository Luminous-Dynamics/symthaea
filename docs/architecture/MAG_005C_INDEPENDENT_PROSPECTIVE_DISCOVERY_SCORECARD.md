# MAG-005C — Independent Prospective Discovery Scorecard Architecture

Status: architecture-only design subject

Date: 2026-09-26

Related issues:

- MAG-005 / prospective deferred DFT evaluation: #4485
- MAG-BENCH-002 / Fe-Co-X benchmark and contamination boundary: #5115
- MAT-MLIP-001 / calibrated surrogate evaluators: #4992
- MAT-CALC-001 / exact calculation execution and convergence: #5114
- MAT-RD-001 / materials research roadmap: #4997

## Purpose

Define the scoring boundary for Symthaea's first prospectively governed materials-discovery experiment without allowing the candidate generator, surrogate model, DFT executor, domain evidence normalizer, or scorecard to silently absorb one another's authority.

The intended first proving vertical is rare-earth-free `Fe-Co-X` discovery, but the scoring theorem should remain reusable for later prospective materials campaigns.

This document creates no scientific result and no production Rust type. It freezes architecture and adversarial requirements only.

## Core theorem

```text
candidate ranking committed before answers
+ control allocation committed before answers
+ high-fidelity execution performed after commitment
+ every preregistered evaluation slot reaches a terminal disposition
+ scorecard consumes immutable evidence identities
+ scoring implementation is independent of candidate generation/execution
-> prospectively scored computational discovery experiment
```

This does **not** imply:

```text
prospective computational enrichment
!= synthesis
!= physical property validation
!= useful permanent magnet
!= world-first novelty
!= patent novelty
!= manufacturability
!= economic value
```

## Required separation of authority

Preserve three independently committed objects:

```text
prospective question / candidate commitment
!= raw high-fidelity execution evidence
!= normalized scientific evidence
!= scorecard result
```

Recommended ownership:

- MAG-005A owns the prospective question/candidate commitment;
- MAT-CALC owns execution/convergence receipts;
- THERMO / PHONON / MAG property authorities own normalized scientific propositions;
- MAT-010 owns failed/null/OOD search memory where applicable;
- MAG-005C owns only scoring over immutable referenced evidence.

The scorecard must never mint or upgrade scientific evidence.

## Protocol classes

### `OneShotDeferredEvaluation`

All candidate ranking and evaluation strata are frozen before any high-fidelity answers exist.

Required pre-answer commitments include:

- exact candidate-universe identity;
- exact knowledge-cutoff / benchmark profile;
- exact generator/search/model identities;
- exact acquisition/ranking policy and parameters;
- exact ranked candidate order;
- exact top-K allocation;
- exact random-control allocation policy and seed/commitment;
- exact diversity/OOD-control allocation;
- optional conventional baseline allocation;
- exact evaluator profile identities;
- compute/resource budget;
- stopping policy;
- metric profile;
- denominator policy;
- exact SearchMemory / negative-history snapshot visible to ranking.

After the first high-fidelity outcome is revealed, the one-shot ranking may not change.

```text
observe DFT result
-> rerank remaining candidates
-> still call result one-shot top-K enrichment
```

is forbidden.

### `SequentialAdaptiveEvaluation`

Adaptive acquisition is a different estimand and requires an ordered decision/evidence log.

Each round must bind:

- exact evidence state available before selection;
- exact acquisition-policy implementation/profile;
- calibration artifacts consumed by the acquisition rule;
- candidate(s) committed for the next round;
- batch size;
- remaining budget;
- stop-rule state;
- deterministic/randomness commitments.

The admissible temporal order is:

```text
round-t evidence
-> acquisition decision t+1
-> commit candidate(s)
-> reveal fresh high-fidelity outcome(s)
```

Post-outcome reranking without a pre-outcome decision commitment receives no prospective credit.

A future MAG-005D may implement this protocol. MAG-005C must not silently score an adaptive trial as one-shot discovery.

## Scorecard identity

The scorecard subject should bind exact references to, at minimum:

- prospective campaign/question commitment;
- candidate universe;
- frozen ranking/acquisition output;
- control allocation receipt;
- high-fidelity evaluation bundle;
- normalized scientific-evidence bundle(s);
- terminal disposition census;
- metric profile;
- statistical/interval method profile;
- scorer implementation identity;
- scorer runtime/environment identity where executable evidence is claimed;
- scorecard schema/version;
- deterministic scorecard commitment.

Changing any authority-bearing input creates a different scorecard identity.

## Complete terminal census

Every preregistered evaluation slot must terminate exactly once.

Minimum disposition vocabulary should be able to distinguish:

```text
Completed
SolverNonConverged
ExecutionFailure
InvalidStructure
EvaluatorApplicabilityRefusal
DuplicateOrAliasDiscoveredAfterCommitment
ResourceBudgetExhausted
CancelledByPredeclaredStopRule
NotRunWithPredeclaredReason
```

Exact final names may reuse existing qualified owners where available.

A failed or nonconverged top-K candidate remains in the original denominator. It cannot be replaced by the next successful ranked candidate while preserving the original `K` claim.

Likewise:

```text
invalid after commitment
!= never selected
```

The historical selection still occurred and must remain visible.

## Metric vector — no universal discovery score

Report independent dimensions rather than one weighted scalar.

At minimum, where the frozen experiment supports them:

### Screening / hit metrics

- stable-candidate precision@K;
- near-hull precision@K under the exact threshold/profile;
- property-threshold precision@K where the property theorem is independently qualified;
- false-positive census;
- false-negative census where the evaluated design permits estimation.

### Control-relative metrics

- enrichment vs random control;
- enrichment vs diversity control;
- enrichment vs uncertainty/OOD control;
- optional enrichment vs conventional baseline model/policy.

### Calibration metrics

- property error under exact conditions;
- interval coverage where justified;
- calibration error/profile;
- error by chemistry-family slice;
- error by structure/prototype slice;
- error by applicability/OOD slice;
- error by evaluator/task capability.

### Diversity metrics

- distinct successful chemistry families;
- distinct successful structure/prototype families;
- duplicate/near-duplicate census;
- Pareto diversity where multiple property objectives are preregistered.

### Resource metrics

- high-fidelity compute consumed;
- wall-clock/provider resource use where evidence exists;
- evaluated candidates per compute budget;
- compute per validated hit;
- compute consumed by failures/nonconvergence;
- scorecard execution cost separately from scientific evaluation cost.

No metric may silently absorb scientific values that belong to another authority owner.

## Finite-sample discipline

A small prospective campaign may have very small K.

Therefore preserve:

- raw numerator/denominator counts;
- exact allocation sizes;
- uncertainty/confidence intervals or exact finite-sample statements appropriate to the frozen design;
- randomization procedure;
- all exclusions/refusals;
- sensitivity to preregistered thresholds where reported.

Do not convert a noisy small-sample difference into a categorical superiority claim.

```text
point estimate higher
!= established superiority
```

The first campaign may legitimately end `EvidenceInsufficientForControlDifference` while still being a valid prospective experiment.

## Calibration-gated acquisition

Preserve:

```text
raw model uncertainty
!= calibrated error detector
!= acquisition-policy eligibility
!= property uncertainty
```

An uncertainty signal may drive an ordinary acquisition arm only when an exact calibration artifact demonstrates useful error/risk detection under the relevant chemistry / structure / task / state profile.

An uncalibrated signal may still define an explicitly exploratory control arm, but the scorecard must preserve that status.

The calibration artifact belongs to MAT-MLIP / the relevant evaluator owner, not MAG-005C.

## Common-cause model ancestry

Multiple selectors/models may share:

- training data;
- foundation checkpoints;
- model family;
- feature pipeline;
- provider/codebase;
- fine-tuning data;
- structure canonicalization.

Therefore:

```text
N model outputs
!= N independent scientific sources
```

and:

```text
low disagreement among correlated models
!= calibrated low error
```

MAG-005C should consume existing evidence-dependency/common-cause representations rather than create a new independence ontology.

## Scorer independence

The reference scorecard implementation should be mechanically independent of the ranking/generator implementation wherever practical.

Minimum desired properties:

- no import of candidate-generation code;
- no mutable access to the ranking subject;
- score only exact immutable evidence references;
- recompute metric values from the frozen census/evidence rather than trusting stored summary values;
- fail closed on missing, duplicated, reordered, or mismatched candidate identities;
- fail closed if an evaluated candidate is outside the committed universe/stratum;
- fail closed if expected-answer artifacts are reachable from Phase-A candidate generation under a supposedly clean protocol.

An independent known-answer synthetic corpus should precede production scoring.

## Anti-hindsight rules

The following changes require a new scorecard/campaign lineage rather than rewriting the old result:

- threshold changed after seeing outcomes;
- K changed after seeing failures;
- random seed/control assignment changed;
- top-K candidate replaced after nonconvergence;
- candidate-universe membership changed;
- evaluator profile changed;
- result parser/convergence policy changed;
- OOD/refusal policy changed;
- metric weights/definitions changed;
- model retrained after target reveal;
- candidate duplicate/alias resolution changed material scientific identity.

Historical records remain addressable and are never rewritten to make the experiment look cleaner.

## Adversarial corpus requirements

The first synthetic/reference corpus should include at least these cases:

1. top-K candidate fails DFT convergence -> remains in denominator;
2. random-control candidate fails -> remains in control denominator;
3. candidate alias discovered after commitment -> disposition preserved, no silent replacement;
4. outcome threshold changed after reveal -> new scorecard lineage required;
5. same outcomes + changed metric profile -> different scorecard identity;
6. same score + different underlying failure census -> distinct scorecard evidence;
7. stored summary metric disagrees with recomputation -> reject;
8. scorer receives candidate outside committed universe -> reject;
9. one-shot campaign contains post-outcome reranking -> reject one-shot classification;
10. adaptive campaign omits pre-selection evidence snapshot -> reject prospective adaptive credit;
11. raw uncertainty arm represented as calibrated without calibration artifact -> reject;
12. calibration artifact from wrong task/domain -> reject calibrated acquisition claim;
13. several correlated uMLIPs agree -> no independent-replication claim;
14. random seed/control allocation changed after reveal -> new campaign lineage;
15. `NotRun` candidate silently omitted from census -> reject;
16. evaluator applicability refusal converted to ordinary property failure -> reject;
17. scorecard reads expected-answer artifact reachable from Phase A under clean benchmark -> contamination failure;
18. tiny-K point estimate higher than control -> preserve interval/evidence insufficiency;
19. changed scientific evidence identity with same numeric value -> different scorecard input;
20. scorecard PASS -> no synthesis/physical/manufacturing promotion.

## External design motivation

These are architectural references only; they are not local Symthaea evidence:

- Matbench Discovery separates discovery/stability-screening behavior from plain formation-energy error:
  https://www.nature.com/articles/s42256-025-01055-1
- flexible uncertainty calibration for MLIPs improves high-error-configuration identification:
  https://www.nature.com/articles/s41524-026-02080-3
- 2026 LLM active-learning work demonstrates acquisition efficiency while exposing stochastic/initialization sensitivity:
  https://www.nature.com/articles/s41524-026-02136-4
- OMat24 provides a large open DFT corpus/models useful for surrogate benchmarking, without becoming local calibration authority automatically:
  https://www.nature.com/articles/s43588-026-00996-w

## Dependency / implementation gate

Do not open production MAG-005C scoring code until usable qualified/current owner interfaces exist for the exact inputs it must consume.

Expected dependencies include qualified successors of:

- MAG-BENCH-002 cutoff/contamination/split semantics;
- MAG-005A prospective candidate commitment;
- MAT-CALC exact execution/convergence evidence;
- THERMO / PHONON / MAG property evidence used by the campaign;
- MAT-MLIP calibration/OOD profiles when surrogate acquisition is used;
- MAT-010 failure/negative-result memory where required;
- shared dependency/common-cause evidence owner.

The architecture document itself may be reviewed independently from those implementation gates.

## Proposed implementation train

```text
MAG-005C-000A
  architecture + ownership / no-authority-laundering contract

MAG-005C-001A
  frozen synthetic scorecard corpus

MAG-005C-001B
  independent known-answer reference scorer

MAG-005C-002A
  production scorecard contracts over qualified owner refs

MAG-005C-002B
  exact campaign scorer for first Fe-Co-X one-shot trial
```

Every executable semantic subject receives its own exact-head qualification. Queued/skipped/cancelled is never PASS.

## Claim ceiling

A qualified MAG-005C scorecard may establish that a frozen prospective computational materials campaign achieved the reported metric vector relative to its preregistered controls, evaluator profiles, terminal census, resource budget, and uncertainty statement.

It does not establish synthesis, experimental magnet performance, deployment readiness, world-first novelty, patentability, manufacturability, economic value, or universal superiority of Symthaea's acquisition strategy.