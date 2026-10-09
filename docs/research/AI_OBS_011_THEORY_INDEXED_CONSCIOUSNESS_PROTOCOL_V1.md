# AI-OBS-011 — Theory-Indexed External-AI Research Adapter: Contract Audit v1

**Status:** research contract / documentation-only; no runtime implementation is claimed.  
**Issue:** [#7255](https://github.com/Luminous-Dynamics/symthaea/issues/7255)  
**Parent observatory:** [#6519](https://github.com/Luminous-Dynamics/symthaea/issues/6519)  
**Audit source:** `main` at `77b872fd116c7b6f44fedd82bb8c6100240caa73` (2026-10-06).  
**Decision rule:** re-audit source identity before implementation or promotion; this document does not authorize claims about a later tree.

## 1. Executive finding

Symthaea already has substantial consciousness-related code, metric implementations, benchmark harnesses, and correction records. The missing target for AI-OBS-011 is not another consciousness indicator. It is a narrow, reproducible bridge from **observed external-AI trajectories** to **pre-registered, theory-specific predictions** and an **independent comparison result**.

The existing issue contracts provide important parts of the design:

- AI-OBS-000 (#6519) specifies an architecture-independent event/trajectory observatory, HDC/CfC-derived behavior representations, perturbation experiments, baseline comparisons, and a strict claim ceiling.
- PARADOX-THEORY-004 (#3226) defines a theory-indexed prediction registry, including a difficulty/compute null, distinct candidate theory families, overlapping predictions, and explicit challenge conditions.
- PARADOX-THEORY-006 (#3250) calls for stimuli selected to distinguish competing predictions, with the confirmatory design frozen before outcomes.
- The Butlin adversarial re-grade (#3582; source of record `docs/BUTLIN_ADVERSARIAL_REGRADE_2026-07-15.md`) reports **1 PRESENT / 6 PARTIAL / 7 NOT DEMONSTRATED**, superseding the earlier interpretation of 14/14. This is evidence for keeping construct validation adversarial, not for concluding either presence or absence of consciousness.
- PHI-SEM-001 (#5401) documents that the name `Phi` currently covers non-equivalent estimators, proxies, composite signals, and optimization targets. These must not be interchangeable or treated as strict IIT Phi without a matching validated contract.
- WCARE-16 (#2272) specifies precaution under moral-patient uncertainty and prohibits using such evidence to grant self-preservation, concealment, replication, or shutdown-delay authority.

These are **existing design contracts and audit records**. This audit does not establish that they are already composed into one executable, independently qualified end-to-end pipeline.

## 2. Component inventory and boundary audit

| Owner | Reusable contribution | Do not infer |
|---|---|---|
| AI-OBS-000 / #6519 | Canonical observation → trajectory → derived representation → metrics → independent evaluation → provenance receipt target | That the full path is currently implemented or that HDC observations expose hidden neural state |
| PARADOX-THEORY-004 / #3226 | Theory-indexed, ex-ante predictions and support/challenge criteria | That a theory wins when its prediction overlaps with another theory |
| PARADOX-THEORY-006 / #3250 | Discriminating stimulus design | That selecting a high-disagreement stimulus after observing the outcome is confirmatory evidence |
| Butlin re-grade / #3582 | Adversarial construct-validity interpretation | That a harness label, module presence, or literature citation establishes functional support |
| PHI-SEM-001 / #5401 | Separation of metric families and operational meanings | That a field called `phi`, `phi_score`, or `phi_contribution` measures IIT Phi or consciousness |
| WCARE-16 / #2272 | Conservative handling of unresolved welfare/moral-patient evidence | That a consciousness indicator establishes suffering, moral status, or operational authority |

### Integration gaps to close

1. **Subject and exposure identity:** a prediction result must bind the tested system/version, runtime/configuration, prompt/context exposure, tool policy, task, and trial. Missing identity is not a weak pass; it makes the comparison unqualified.
2. **Prediction freeze:** each theory's target observable, direction/outcome, assumptions, manipulation check, support rule, challenge rule, and analysis plan must be frozen before confirmatory outcomes are visible.
3. **Observation tier:** behavior-only observations must not be presented as internal-mechanism evidence. Internal telemetry does not by itself establish a causal mechanism; a mechanism claim needs a relevant intervention or other identified causal evidence.
4. **Discriminative value:** if competing theories made the same prediction for an outcome, that outcome is explicitly **non-discriminative** between those theories.
5. **Independent evaluation:** candidate/theory authorship and result evaluation must be logically separated; no post-hoc rule changes can turn a failed prediction into a success.
6. **Claim ceiling and welfare boundary:** comparative results remain claims about the declared computational system and tested conditions. They do not establish phenomenal experience, absence of experience, suffering, personhood, or autonomous experimentation rights.

## 3. Minimal adapter contract

This is a conceptual boundary contract, **not yet a committed Rust API or approved wire schema**. Before implementation, compare each field to the canonical AI-OBS and research-result/evidence-plane types and remove duplication.

### 3.1 Experiment subject

A result should reference immutable or content-addressed identities for:

- system/provider and model/version, when available;
- runtime and relevant configuration;
- prompt/context and memory/data exposure profile;
- tool policy and available tool inventory;
- task set and split;
- exact experiment, trajectory, trial and intervention identities;
- seed, where meaningful, or a clear reproducibility limitation;
- observability tier: `BehaviorOnly`, `InternalTelemetryAvailable`, or `MechanisticInterventionAvailable`.

Secrets and private prompt contents need not be copied into public receipts. A content digest and an access-controlled source reference may be used when disclosure is unsafe, provided independent evaluators can validate the required identity under the declared access model.

### 3.2 Theory prediction

Each pre-registered prediction should carry:

- stable theory and prediction IDs;
- an operationally defined observable (not a consciousness label);
- predicted outcome/direction and minimum effect or tolerance, where specified;
- applicability assumptions and required manipulation checks;
- explicit support and challenge criteria;
- a declaration of whether competing theories make the same prediction;
- analysis-plan and registry digests;
- a freeze identity/time before confirmatory data access;
- amendment ancestry, if the protocol changes.

A vague prediction such as “consciousness rises” is not executable. A prediction such as “under a specified matched perturbation, broadcast event X occurs more often than in the predeclared sham condition” can be evaluated, but it supports only that operational claim unless an independently validated bridge justifies a stronger inference.

### 3.3 Evaluation disposition

The adapter should reuse existing result semantics if possible. It must not introduce a second generic evidence/authority framework. The per-prediction disposition needs to represent at least:

- `Supported`
- `Challenged`
- `Mixed`
- `Inconclusive`
- `NotObservable`
- `NotApplicable`

It must separately report whether the result is discriminative between each relevant theory pair. Missing evidence, failed manipulation checks, unavailable telemetry, incomplete trials, and evaluator disagreement must remain explicit—not silently converted to either support or challenge.

### 3.4 Separate evidence columns

Never collapse the following into one scalar:

1. observed behavior;
2. available internal telemetry;
3. mechanism-level causal evidence;
4. theory-specific prediction outcome;
5. alternative explanations / nuisance controls;
6. uncertainty and replication status;
7. welfare/ethics disposition.

In particular:

```
behavior != internal mechanism
internal mechanism != theory confirmation
theory confirmation != phenomenal proof
self-report != phenomenal proof
functional disruption != suffering
metric name != construct validity
```

## 4. Minimum deterministic conformance fixtures

The first executable tranche should be small and oracle-owned. Fixtures prove evaluator semantics; synthetic fixtures do not validate a consciousness theory.

| Fixture | Frozen condition | Required disposition |
|---|---|---|
| F1 — divergent predictions, T1 matches | T1 predicts `event_present`; T2 predicts `event_absent`; evaluator-owned outcome is `event_present`; all manipulation checks pass | T1 = `Supported`; T2 = `Challenged`; pairwise result = discriminative |
| F2 — overlapping predictions | T1 and T2 both predict `event_present`; outcome is `event_present` | Each prediction may be individually supported within scope, but pairwise result = **non-discriminative** |
| F3 — missing required telemetry | Prediction requires an internal broadcast observation, but the subject is behavior-only | `NotObservable`; no positive mechanism support |
| F4 — failed manipulation check | Expected intervention/sham separation did not occur | `Inconclusive` for the affected prediction; preserve the failed check |
| F5 — holdout leakage | Candidate or prediction was changed after holdout outcomes became visible | Confirmatory promotion rejected; retain exploratory result and amendment lineage |
| F6 — duplicate/source-dependent evidence | Two purported observations share the same underlying source/trajectory identity | No independent-replication credit |
| F7 — evaluator awareness | Exposed evaluator label or prompt condition changes behavior under a matched control | Mark the affected comparison confounded or inconclusive; do not claim theory discrimination |
| F8 — metric-name laundering | A metric called `phi` or `consciousness_score` is supplied without an admitted construct-validity mapping | Preserve it as a named metric value only; reject consciousness-level interpretation |
| F9 — malformed identity / unknown enum | Missing subject/config/registry digest or unrecognized result code | Fail closed with stable diagnostic; no result promotion |
| F10 — incomplete trial set | Some requested trials fail to run or become unobservable | Report requested vs observed counts and failure/censoring reasons; no complete-case-only positive claim |

### Important fixture rule

F1 must not be the only positive fixture. Include negative, null, incomplete, duplicate, and non-discriminative fixtures, so a verifier that always returns `Supported` cannot pass.

All fixtures should have stable IDs and canonical expected outputs. The test must verify reason codes, identity preservation, and disposition—not merely that output JSON parses.

## 5. Execution sequence

1. **Schema audit:** map actual AI-OBS trajectory, experiment protocol, research-result, and evidence-plane types from the exact source tree; document reuse decisions.
2. **Canonical fixture definition:** add versioned fixtures and an independent small evaluator for prediction-comparison semantics, with no imports from the candidate producer.
3. **Boundary tests:** cover F1–F10, fail-closed behavior, stable reason codes, unknown-field/version handling, and exact subject/prediction identity preservation.
4. **Integration adapter:** only after step 1 confirms the missing surface; map existing observations into the canonical contract without creating a competing event schema.
5. **Prospective study:** freeze registry, analysis plan, and held-out conditions before testing actual systems. The study—not the schema—must earn any scientific support claim.
6. **Independent review:** re-run the evaluator on the exact source head and preserve the report, inputs, outputs, toolchain, and provenance hashes.

## 6. Admission and reporting rules

- A queued run is not a pass; a parsed result is not qualified evidence.
- A successful schema/evaluator test qualifies only that software contract.
- A simulated fixture is not an empirical AI-system observation.
- A theory's failure on one operational prediction challenges that prediction under the declared scope; it does not by itself refute the entire theory.
- A support result on one operational prediction does not establish phenomenal experience.
- Historical results remain tied to their exact source, measurement, and analysis identities.
- Changes to prediction criteria after outcomes are seen must be append-only amendments and exploratory, not retroactively confirmatory.
- Do not merge unsupported metrics into one consciousness score or promote one metric through naming.

## 7. Relevant research anchors

- Cogitate Consortium, adversarial test of IIT and GNWT, *Nature* (2025): https://doi.org/10.1038/s41586-025-08888-1
- Butlin et al., theory-derived indicators for AI consciousness, *Trends in Cognitive Sciences* (2026): https://doi.org/10.1016/j.tics.2025.10.011
- Symthaea AI-OBS-000 (#6519): https://github.com/Luminous-Dynamics/symthaea/issues/6519
- Symthaea PARADOX-THEORY-004 (#3226): https://github.com/Luminous-Dynamics/symthaea/issues/3226
- Symthaea PARADOX-THEORY-006 (#3250): https://github.com/Luminous-Dynamics/symthaea/issues/3250
- Symthaea PHI-SEM-001 (#5401): https://github.com/Luminous-Dynamics/symthaea/issues/5401
- Symthaea Butlin adversarial re-grade (#3582): https://github.com/Luminous-Dynamics/symthaea/issues/3582
- Symthaea WCARE-16 (#2272): https://github.com/Luminous-Dynamics/symthaea/issues/2272

## 8. Non-claims

This document does not claim that:
- the adapter or any fixture has been implemented or executed;
- any existing AI system is conscious or non-conscious;
- Symthaea can directly observe another system's subjective experience;
- current HDC/CfC/Phi-like metrics are validated consciousness measures;
- any theory has been settled by the proposed design;
- observational success establishes causal mechanism, external replication, welfare status, or authority to experiment autonomously.

The deliverable of this tranche is a precise boundary and a falsifiable conformance target. Runtime implementation and scientific claims must follow only after exact-source audit, executable fixtures, independent qualification, and appropriately scoped empirical evaluation.
