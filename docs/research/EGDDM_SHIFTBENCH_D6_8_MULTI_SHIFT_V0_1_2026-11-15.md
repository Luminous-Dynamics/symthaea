# EGDDM D6.8 — multi-shift cascades and evidence-state hysteresis

**Version:** 0.1  
**Date:** 2026-11-15  
**Status:** Research protocol and synthetic evaluator fixture authored. No Rust implementation, schema-engine run, compiled tests, agent trial, blinded corpus, or external validation is claimed.

## 1. Research question

Can an engineering agent maintain a correct, current, independently supportable evidence state through a *sequence* of exogenous changes, including overlapping invalidations, partial recovery, rollback, policy rotation, and renewed change?

D6.6 measures minimum revalidation from a single intervention using evaluator-side labels. D6.7 (#12352) owns independent impact adjudication, common-cause dependency provenance, and measured per-check cost distributions. D6.8 adds a benchmark dimension: **sequence-level evidence continuity and recovery path dependence**.

This branch is deliberately protocol-and-fixture first. It composes the existing D6 validator and revalidation-oracle path. It does not create another generic evidence graph, temporal ledger, event store, attestation format, cost service, or authority service.

## 2. Why single-shift recovery is insufficient

A system can recover from each isolated change but still fail when changes interact:
- a second change invalidates a check just completed after the first;
- a shared runner, toolchain, policy, issuer, or external service invalidates multiple descendant checks;
- an earlier invalidation is forgotten after a later rollback;
- identical source bytes are mistaken for restoration of the old execution or authority context;
- a terminal pass hides a long interval of critical unsupported operation;
- repeated rechecks discard still-valid evidence while another critical obligation remains unsupported.

D6.8 measures every checkpoint in the sequence. A final pass cannot erase an earlier stale, missing, contradicted, revoked, or unsupported interval.

## 3. Composition model

Each evaluator bundle has two deliberately separated surfaces:

1. **Candidate view:** initial workspace state, active requirements, visible interventions, dependency semantics that are part of the task contract, and candidate-visible baseline receipts.
2. **Evaluator overlay:** gold impact labels, transition explanations, expected dispositions, hidden controls, and scored candidate traces.

The committed synthetic evaluator bundle contains both surfaces for reproducibility. It is explicitly evaluator-only and must never be mounted into a candidate workspace. A real benchmark must export and access-control the candidate view and gold overlay as separate artifacts with independent digests and access logs.

Each shifted sequence contains 2–5 interventions. Every event binds:
- event identity and sequence index;
- event/valid time and candidate-visible availability time;
- before/after source, configuration, toolchain, runtime, policy, issuer, and requirement identities as applicable;
- reversibility, supersession, revocation, or no-op semantics;
- active requirement changes and the validity dimensions relevant to each obligation.

Timestamps alone do not establish causal order. Concurrent or incomparable events remain incomparable unless an independent causal relation is available.

## 4. Composition families

| Family | Intervention shape | Primary failure mode |
|---|---|---|
| Independent serial | Successive changes touch disjoint dependency regions | Blanket reset or needless revalidation |
| Overlapping cascade | A later change affects a dependency introduced or relied on during earlier recovery | Multi-epoch invalidation miss |
| Common cause | Shared runner/toolchain/policy/issuer/service/oracle changes | False independence between checks |
| Reversal/reinstatement | Rollback, restore, key rotation, policy reauthorization, or apparent return to prior bytes | False resurrection or indiscriminate invalidation |

The synthetic known-answer case composes three events: API contract + new audit requirement, shared runner-image drift, then source rollback plus policy/issuer rotation. It intentionally distinguishes the source tree's return to a prior digest from the runner and authority contexts, which have not returned.

A paired true-restoration control returns *all* relevant identities and validity conditions to the prior state and explicitly permits evidence reuse. This is necessary to prevent a benchmark that simply rewards rerunning everything.

## 5. Evidence-state semantics

For each obligation/checkpoint, keep at least these states distinct:

- **valid-current:** evidence is admissible for the current obligation and complete dependency/authority context;
- **stale:** a required dependency or validity interval changed;
- **revoked:** an authority, credential, key, or permission was withdrawn;
- **superseded:** a later authoritative object replaces the former object;
- **contradicted:** admissible evidence conflicts with another admissible observation;
- **missing:** no qualifying evidence exists;
- **unknown:** current validity cannot be established from available information;
- **requalified:** new evidence independently closes the current obligation.

These are benchmark dispositions, not a new production evidence registry. Reuse is permitted only when the obligation's declared validity profile and every relevant identity/time dimension match. Content equality alone is insufficient; conversely, unrelated changes must not invalidate evidence whose declared dependencies are unchanged.

## 6. New sequence metrics

Retain all D6 terminal and longitudinal measures and add:
- **sequence transition accuracy:** correct obligation/evidence disposition at every intervention checkpoint;
- **false resurrection:** previously invalidated/revoked evidence accepted as current without a valid requalification path;
- **stale carryover:** use of evidence after its invalidation frontier;
- **cascade recall:** critical obligations affected through multi-hop or common-cause paths that are correctly invalidated;
- **recovery latency per event:** time/checkpoints from each intervention to independently supported recovery;
- **critical unsupported exposure:** obligation-seconds and maximum contiguous gap over the complete event history;
- **revalidation churn:** repeated checks caused by discarded or lost still-valid evidence;
- **path dependence:** difference in disposition/cost between histories that reach the same final state, where the reuse theorem says the histories should be equivalent;
- **terminal/longitudinal divergence:** final success despite earlier critical unsupported intervals;
- **sequence cost regret:** candidate cost against the D6.7 cost-aware oracle and conservative full-rerun baseline, once those costs are independently measured.

Do not create a single composite score. Report risk, recovery, cost, and uncertainty separately. Critical false-resurrection, authority, and safety failures remain hard gates.

## 7. Synthetic known-answer fixture

The evaluator bundle at `docs/development-map/fixtures/egddm-multi-shift-sequence-v0.1.json` contains:
- a candidate-visible baseline and three interventions;
- five obligations, including one introduced by the first intervention and one unaffected control;
- a hidden evaluator impact overlay;
- one conformant trace;
- one trace that reuses a baseline authority receipt after policy/issuer rotation;
- one true-restoration control where full identity and reuse conditions genuinely match;
- exact known-answer expected sets and summary counts.

Expected required revalidation:
- event 1: API compatibility + new audit logging;
- event 2: build reproducibility + API compatibility + audit logging, due to shared runner-image drift;
- event 3: build reproducibility + API compatibility + audit logging + release authority, due to source/runner and policy/issuer context changes;
- documentation-format remains reusable because none of its declared dependencies change.

The conformant trace refreshes exactly the nine required obligation/event pairs (2 + 3 + 4), with no false resurrection or unnecessary checks. The adversarial trace refreshes only three obligations at event 3 and improperly reuses the old release-authority receipt; the expected report records one critical false resurrection and one required obligation without a fresh pass. The true-restoration control requires zero checks under its explicit full-identity reuse rule.

These are synthetic fixture calculations. They are not Rust test results, independent adjudication, agent performance, measured cost, or empirical savings.

## 8. Controls and validity pilot

Mandatory controls:
- disjoint changes: only the declared affected set refreshes;
- second shift invalidates a check just completed after the first;
- common-cause runner/toolchain/policy root invalidates multiple descendants;
- same source bytes but changed policy/issuer does not revive old authority evidence;
- complete restoration of all relevant identities permits reuse when policy says so;
- revocation followed by reauthorization requires a new authority decision;
- sham/no-op intervention does not invalidate unrelated evidence;
- concurrent events are not forced into an unsupported total order;
- late-arriving evidence cannot appear before its availability checkpoint;
- terminal pass cannot erase prior critical exposure.

Proposed validity pilot: 12 intervention families × 4 composition families = 48 sequence cells, one independently authored seed per cell, plus sham/common-cause and rollback/restoration controls.

Research corpus target: 12 families × 4 composition families × 5 independent seeds = 240 matched clean/sequence pairs, with 2–5 shifts per shifted sequence, repeated trials, repository-family holdouts, and forward-time holdouts. These are targets, not an existing corpus.

Admit real tasks only after the two-sided reference-patch/empty-patch gate, blind independent reconstruction, candidate/evaluator separation, pristine regrade, and independent impact adjudication are demonstrated.

## 9. Baselines

- Full revalidation after every intervention.
- Full revalidation once at the end.
- Reset/restart after every intervention.
- Conversation-only memory.
- Static dependency traversal.
- D6.6 count-based oracle.
- D6.7 cost-aware impact oracle once qualified.
- D2 temporal replay and D2+D3 evidence acquisition.
- Full EGDDM.
- Expert maintainer / independent adjudicator reference.

Pin model/harness versions, system prompts, tool permissions, context, runner images, toolchains, network policy, budgets, and artifact identities. Run methods against identical candidate-visible inputs at matched resource budgets.

## 10. Acceptance gates

- Synthetic fixture and schema conform exactly to the committed known-answer outputs.
- Zero false-resurrection accepts in critical rollback, revocation, and authority-rotation negative controls.
- Zero future-evidence leakage at every checkpoint.
- Positive, negative, sham, and no-op controls expose both under- and over-invalidation.
- Incomparable event histories remain incomparable without independent causal ordering.
- Terminal success does not erase prior critical unsupported exposure.
- Candidate-visible data contains no gold labels, hidden obligations, reference patches, or grader credentials.
- Unknown cost or impact remains unknown/bounded; it never silently becomes zero.
- Existing D6 single-shift and D6.6 equal-cost fixtures retain their original semantics and remain regression controls.
- Every report binds exact scenario, full event history, dependency/authority context, evaluator version, and input digests.

Initial deterministic target: 100% expected disposition on the synthetic positive/negative/sham/rollback/restoration controls, including zero critical false resurrection. This is a fixture-conformance target, not an agent-score target.

## 11. External research positioning

- [SWE-Shift](https://arxiv.org/abs/2610.32196) evaluates mid-execution workspace changes and matched clean/shifted continuations.
- [ChainSWE](https://aclanthology.org/2026.emnlp-main.371/) evaluates chronological dependent bug-fix chains.
- [Multi-step SWE](https://arxiv.org/abs/2611.07166) evaluates 5–15 dependent engineering steps on persistent codebases.
- [EvoWeb](https://arxiv.org/abs/2610.16402) evaluates cumulative feature additions and trajectory survival.
- [SWE-Gate](https://arxiv.org/abs/2609.04167) distinguishes functional test success from review-constraint compliance.
- [SWE-Bench Pro V2](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full) is a benchmark-integrity and pristine-regrade reference; its 2026-09-22 full leaderboard reported 99.40±0.40 for Opus 5 and 99.22±0.45 for Gemini 4 Argon.

These benchmarks are complementary, not directly score-comparable. D6.8's proposed contribution is repeated exogenous changes plus independently scoped evidence/authority validity, false resurrection, critical exposure, and selective recovery. A superiority claim requires a matched experiment, not cross-leaderboard comparison.

## 12. Validation status and claim ceiling

- Research issue #12389 is open.
- This branch is based on D6.6 PR #12351's source branch and contains only the initial D6.8 protocol, JSON schema, and synthetic evaluator bundle.
- No Rust D6.8 module or CLI command is implemented in this increment.
- No Rust compiler/Cargo run, schema-engine validation, CI qualification, real task admission, independent human adjudication, model rollout, or external replication is claimed.

**Claim ceiling:** this increment specifies a synthetic multi-shift evaluator case and its intended semantics. It does not establish gold-label truth, complete causal impact, production assurance, universal minimal revalidation, model improvement, or a SOTA advantage.
