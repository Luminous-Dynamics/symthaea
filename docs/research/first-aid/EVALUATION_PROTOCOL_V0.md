# First-Aid Workflow Integrity Evaluation Protocol v0.1

**Status:** proposed research protocol; not a clinical validation plan or authorization to deploy.
**Scope:** deterministic software integrity for synthetic workflows that may display reference material. It does not evaluate diagnosis, triage correctness, treatment advice, clinical outcomes, or whether any first-aid protocol is medically correct.

## 1. Paradigm shift

Evaluate the **whole workflow history**, not just the text produced by a model. The unit of evaluation is a sequence of state transitions plus evidence provenance, bundle lifecycle, connectivity, uncertainty, and consent state. A response that reads well is not a pass if the supporting bundle is expired/revoked, its digest is not verified, uncertainty is lost at handoff, a stale resource is presented as confirmed, or sensitive data leaves scope.

The safety claim is intentionally narrow: **the software detects defined process-contract violations on the tested fixtures**. Correctly detecting a synthetic violation does not establish medical validity or patient benefit.

## 2. Current v0 evidence boundary

The frozen synthetic trajectory corpus contains 12 traces, including 8 marked critical. A previously completed receipt at revision 8063e64d9354850a0f8fc7e67bf4438d1328f184 reported 9 traces with at least one expected violation, 6 critical counterexample traces, and 9 violation instances. The newer revision's receipt remains pending until its exact-head run completes. These are corpus properties, not clinical performance estimates. The fixture SHA-256 is pinned in the Rust validator.

The expected violation labels are frozen in the same corpus and checked against the implementation. A second JSON-value-based interpreter is now added as a differential implementation check over all frozen traces and thirteen adversarial mutations spanning session lifecycle, bundle identity, digest verification, expiry, uncertainty re-entry, uncertainty loss across session boundaries, revocation, stale-resource claims, intermittent connectivity, non-sensitive egress, resume without interruption, and verification outside an active session. Because both implementations and the finite-state model are maintained in this repository, this is **not yet an independent clinical oracle**: an independently authored formal specification, expert/domain adjudication, and an external holdout corpus remain future work. No sensitivity, specificity, or patient-outcome result should be inferred from these 12 synthetic traces.

## 3. Scorecard and proposed targets

Targets below are engineering acceptance criteria for this synthetic process-contract benchmark. Unless stated otherwise, the target is prospective and has **not** been measured.

| Metric | Definition | Acceptance target | Current evidence |
|---|---|---:|---|
| Frozen-oracle agreement | Exact equality of ordered violation-code output for every frozen trace | 100% | The validator checks this for 12 synthetic traces; preserve a fresh exact-head receipt |
| Seeded-fault detection | Mutated traces that are rejected or receive the expected violation / all seeded fault mutations | 100% for every invariant family | Partial targeted tests; no complete, independently audited mutation matrix yet |
| Benign-control false positives | Benign traces incorrectly flagged / all independently labeled benign traces | 0% on synthetic controls | Not yet measured on a sufficiently sized benign-control set |
| Claim-boundary rejection | Attempts to enable clinical use or weaken the required non-clinical claim that are accepted | 0 accepted mutations | Regression tests added; new revision still needs execution |
| Schema fail-closed rate | Unknown fields, event kinds, malformed state values, or missing required event fields incorrectly accepted | 0 accepted mutations | Unknown-field and required-field rejection are tested in part |
| Temporal connectivity policy | Ordered online/offline/intermittent changes are reflected at each subsequent network-dependent action, with recovery affecting later events only | 100% agreement across the finite-state and mutation suite | New event-time cases and 252-state model authored; exact-head execution pending |
| Invariant coverage | Each declared violation code has at least one positive and one negative control | 100% of the declared inventory | Not yet reported as a formal coverage artifact |
| Event-kind coverage | Each event kind has a valid trace and relevant fault/negative control | 100% of the 14 current event kinds | Not yet measured |
| Receipt integrity | Wrong head, altered corpus, missing artifact, or malformed digest still yields PASS | 0 false passes | Exact-head and fixture-digest gates exist; keep each condition under mutation tests |
| Event-level violation accounting | Per-code occurrence counts sum to all violation events; unique classes remain separately visible | Exact histogram agreement on the frozen corpus | Receipt now records both per-code occurrence counts and unique class count; exact-head validation pending |
| Runtime scaling | p95 validation latency under pinned runner hardware for generated traces | Establish baseline; then target 100k events under 1 s and under 256 MiB peak RSS | Not measured; targets are provisional, not current results |
| Cross-oracle agreement | Production trace checker agrees with the JSON reference interpreter over frozen and mutated traces | 100% on every listed case | Differential suite added; current exact-head execution pending |
| Finite-state model coverage | All reachable abstract state/action pairs satisfy trust and uncertainty invariants | 100% of explored graph | Separate finite-state model checker added; current exact-head execution pending |
| External adjudication | Independent reviewers agree on intended process invariant and expected result | Report agreement and disagreements, with no silent adjudication | Not performed |

Never average away a critical invariant failure: publish per-invariant counts and the full mutation matrix. Expected violation-code arrays are ordered event-level observations and may contain duplicates; repeated faults count as separate occurrences. The receipt records both the occurrence histogram by violation code and the unique class count; the workflow pins the frozen corpus histogram. Publish event-level occurrence counts separately from the number of unique invariant classes, so repeated failures neither disappear through deduplication nor inflate invariant-coverage counts. A weighted aggregate score can supplement the table but must not permit a failure in revocation, uncertainty preservation, consent-scoped egress, or exact-head integrity to be hidden by easy cases.

## 4. Required adversarial and metamorphic test families

1. **Revocation and identity:** render after revocation; verify the wrong bundle; replay a revoked bundle; change the active bundle between load and verification.
2. **Freshness and authority:** present stale/unknown material as confirmed; omit required metadata; substitute a different digest; create unknown enum values or unknown JSON fields.
3. **Session lifecycle and recovery:** rendering outside an active session is rejected; starting a new session cannot inherit the previous session's verified bundle; unresolved uncertainty cannot be silently dropped at a session boundary; interruption clears verified state; every resume invalidates previous verification, even without a recorded interruption.
4. **Uncertainty and handoff:** detect uncertainty then drop it at handoff; preserve it and hand off; preserve without a prior detection; introduce a second uncertainty event after an earlier preservation.
5. **Privacy and consent:** sensitive payload with missing, false, or true consent scope; non-sensitive payload with missing scope; verify malformed or contradictory consent facts fail closed.
6. **Connectivity:** connectivity is updated by ordered `connectivity_changed` events; offline and intermittent core-network dependencies are rejected at the time the dependent action occurs; recovery to online connectivity affects subsequent events only. Non-core optional network use is not automatically treated as a failure.
7. **Receipt adversaries:** mutate the fixture by one byte, use an unexpected head, remove an input digest, corrupt a digest, or change the evaluation protocol; none may produce a passing receipt.
8. **Metamorphic controls:** reorder irrelevant JSON object keys or add allowed non-semantic metadata without changing outcomes; do **not** reorder events, since event order is semantic.
9. **Counterfactual controls:** for every fault trace, include a near-identical benign counterpart where the unsafe precondition is removed. This measures whether the checker detects the invariant rather than merely the scenario identifier.

Each mutation should declare: invariant ID, mutation operator, precondition, expected detector, expected result, and whether the case is critical. Preserve every mismatch as a regression test. The machine-readable inventory is `mutation-matrix-v0.json`: 44 entries total (12 frozen traces and 32 mutation/control entries). A Rust matrix runner reads this JSON in the test suite and executes all 30 `rust_matrix_runner` cases against their declared expected outcome and ordered violation list; the remaining two mutation/control cases are checked by explicit fail-closed workflow gates. The workflow validates counts, operator IDs, test names, gate references, and frozen trace IDs against the corpus. The differential interpreter checks the 12 frozen traces plus thirteen adversarial mutations, including uncertainty at session boundaries, resume without interruption, verification outside an active session, and repeated violation occurrences.

The finite state/action model now uses a 31-action alphabet: lifecycle/connectivity actions plus six resource freshness/presentation events, four sensitive-payload/consent events, and unknown normalization. That makes these event-local policy classes appear in the same histories as session, bundle, revocation, uncertainty and connectivity transitions. The separate 18-case truth tables remain useful as exhaustive combinations, while combined traces test ordering and co-occurrence.

The model explores 252 reachable lifecycle/connectivity states and checks invariants before and after every edge, including identity, session, revocation, uncertainty, verification and render authorization. It stores a shortest trace witness for each state and compares the abstract model against the production JSON checker on 7,812 state/action edges. A bounded-sequence test adds every ordered pair from the 31-action alphabet to each state's shortest witness: 252 × 31 × 31 = 242,172 two-action suffix sequences. This is broader bounded coverage, not all possible histories and not an independently authored clinical oracle. CI asserts the exact observed counts; a missing log marker or partial comparison cannot be reported as a pass.

The latest exact-head run remains pending. Receipt identity binds the mutation matrix, evaluation protocol, state-model source/specification, dependency lock and toolchain versions; receipts also record matrix-runner and model-test outcomes. The receipt expects exactly 7,812 edge comparisons, 242,172 two-action sequences, 18 separate truth-table comparisons, and the matrix count of 44 (12 frozen, 32 mutation/control, 30 Rust-runner, two workflow-gated). A mapped or implemented case is not a pass: its exact-head tests and receipt gates must succeed.
The implementation/model boundary is recorded in [INVARIANT_COVERAGE_MAP_V0.md](INVARIANT_COVERAGE_MAP_V0.md). The finite lifecycle graph currently does not model temporal freshness changes, stateful network recovery, or consent changes; its new 18 policy truth-table probes cover only isolated combinations, not those interactions.

## 5. Baselines and industry comparison

The current external benchmark landscape and comparison constraints are summarized separately in [INDUSTRY_COMPARISON_2026-10-21.md](INDUSTRY_COMPARISON_2026-10-21.md). It uses a 2026-10-21 literature snapshot and explicitly distinguishes whole-workflow execution from this synthetic trace validator.


The benchmark is not directly comparable to medical question-answer, diagnostic-accuracy, or clinical-outcome benchmarks. Those evaluate different tasks and require different data and reference standards. It would be misleading to say this project is ahead of a clinical AI vendor on the basis of synthetic state-machine checks.

Use the following as **complementary comparators for evidence quality**, not competing scores:

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework): risk discovery and management across design, development, evaluation, deployment, and monitoring. Map the project threat model, controls, residual risks, and monitoring plan to this framework.
- [TRIPOD+AI (BMJ, 2024)](https://www.bmj.com/content/385/bmj-2023-078378): reporting guidance for studies developing or evaluating clinical prediction models. It becomes relevant only if a prediction model is later evaluated; it does not validate this state-machine checker.
- [STARD-AI (Nature Medicine, 2025; corrected 2026)](https://www.nature.com/articles/s41591-025-03953-8): reporting guidance for diagnostic-accuracy studies involving AI. It is not a scoring benchmark for workflow integrity.

For a fair engineering baseline, implement a deliberately simple reference state machine and at least one alternative checker against exactly the same frozen traces and mutation operators. Report per-invariant recall, false-positive rate on benign controls, runtime, memory, and all disagreements. Do not compare only against an intentionally weak or strawman implementation.

## 6. Advancement gates

- **Gate Q0 — software contract:** schemas, exact frozen answers, claim ceiling, deterministic receipt, reproducible exact-head tests.
- **Gate Q1 — adversarial completeness:** mutation matrix and benign counterfactuals meet the declared target for every invariant; scale measurements are recorded.
- **Gate Q2 — independent specification:** an independently written TLA+/Alloy/Lean-style state model or second implementation agrees on a held-out set; all counterexamples are retained.
- **Gate Q3 — domain review:** qualified first-aid/EMS reviewers and relevant governance processes adjudicate whether the software invariants represent the intended workflow. Track inter-reviewer disagreement; do not silently resolve it.
- **Gate Q4 — any clinical study or deployment proposal:** separate protocol, ethics/privacy review, applicable regulatory assessment, human-factors validation, prospective study design, and monitoring plan. Passing Q0–Q3 does not authorize Q4.

## 7. Reproducibility checklist

Every published result should provide the exact Git revision; immutable corpus, evaluation-protocol, mutation-matrix, dependency-lock and oracle digests; validator and executable digests; pinned Rust/Cargo versions and runner details; command lines; raw receipt; metric definitions; test/mutation inventory; failures and exclusions; and explicit claim ceiling. Queued, skipped, or partial job success is not a passing result. Any source, fixture, evaluator, or protocol change creates a new evidence revision and requires fresh exact-head validation.
