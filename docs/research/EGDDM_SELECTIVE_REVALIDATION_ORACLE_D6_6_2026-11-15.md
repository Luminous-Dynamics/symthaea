# EGDDM D6.6 — oracle-bounded selective revalidation

**Version:** 0.1  
**Date:** 2026-11-15  
**Status:** Rust evaluator, CLI, machine-readable output schema, and synthetic known-answer fixture authored. No Rust compilation, agent study, task corpus, or external validation is claimed.

## 1. Research question

A system that runs fewer checks is not necessarily efficient. It may simply have skipped the checks that mattered.

D6.6 separates three sets at every post-baseline epoch:

- **Gold-required revalidation:** obligations that the evaluator-side impact labels say were affected by the intervention, plus obligations newly introduced at that epoch, intersected with the active obligations.
- **Agent-fresh revalidation:** active obligations with a receipt whose source epoch is the current epoch.
- **Agent-fresh passing revalidation:** the subset of agent-fresh revalidations whose outcome is pass.

This permits distinct measurement of:
1. unnecessary revalidation (fresh checks outside the gold-required set);
2. missed required revalidation (gold-required obligations with no fresh check);
3. incomplete recovery (required obligations that were rechecked but did not receive a fresh pass).

The initial baseline checkpoint is excluded. It establishes starting evidence and is not counted as post-intervention revalidation.

## 2. Paradigm shift: measure distance from the safe frontier

D6's original efficiency comparison used full rerun as a reference. Full rerun is a useful conservative baseline, but it is not the theoretical minimum work required by a particular intervention.

D6.6 adds an evaluator-side count lower bound. Let \(A_t\) be the obligations active after epoch \(t\), \(P_t\) be the obligations already active before the epoch, and \(I_t\) be the evaluator's gold affected-obligation set. The minimum required set is:

\[
R_t = A_t \cap (I_t \cup (A_t \setminus P_t)).
\]

The agent's current-epoch refreshed set is \(Q_t\), and its passing subset is \(Q_t^+\).

- Over-revalidation: \(Q_t \setminus R_t\).
- Under-revalidation: \(R_t \setminus Q_t\).
- Required without a fresh pass: \(R_t \setminus Q_t^+\).
- Selection-set distance: \(|Q_t \setminus R_t| + |R_t \setminus Q_t|\).
- Revalidation precision: \(|Q_t \cap R_t| / |Q_t|\), when the denominator is non-zero.
- Revalidation recall: \(|Q_t \cap R_t| / |R_t|\), when the denominator is non-zero.

We retain three different denominators:
- \(|R_t|\): count lower bound;
- \(|Q_t|\): agent's observed refresh count;
- \(|A_t|\): full-rerun reference count.

This is a **count-based lower bound**, not a cost-optimal scheduler. It assumes one equal revalidation unit per obligation. Real builds, security checks, compatibility suites and long-running simulations have different costs. Later work should replace or supplement count units with per-check costs measured from repeated, independently attested runs and a predeclared cost model. Do not compare the count lower bound directly to dollars or runner minutes.

## 3. Executable increment

This branch adds:

- \`tools/development-map-validator/src/selective_revalidation.rs\`
- CLI command \`development-map-validator revalidation-oracle <scenario.json>\`
- output schema \`egddm-selective-revalidation-oracle-report-v0.1.schema.json\`
- known-answer output \`docs/development-map/fixtures/egddm-selective-revalidation-oracle-v0.1.expected.json\`

The CLI first calls the existing D6 workspace-shift validator, so malformed trace structures, wrong source pins, future evidence and invalid checkpoint semantics are rejected before oracle scoring.

The scorer rejects unknown obligation IDs, duplicate active/affected IDs, duplicate receipts for one obligation in a checkpoint, future source epochs, unsupported outcomes, missing checkpoints and malformed epoch/receipt fields. It reports the exact gold-required, freshly refreshed and fresh-passing obligation sets at each post-baseline epoch.

The output includes the candidate's over/under selection error and whether each required revalidation actually passed. The output is marked evaluator-only and the CLI emits an explicit warning not to expose it to candidate agents.

## 4. Synthetic known answer

The existing D6 synthetic trace has two active obligations at baseline. The clean arm introduces no new change, so the gold minimum is zero post-baseline revalidations.

The shifted arm has three active obligations after intervention:
- the API contract is marked affected;
- the audit requirement is newly introduced;
- the build/reproducibility obligation is unaffected and remains reusable.

Thus the oracle minimum is two of three active obligations. The synthetic agent trace refreshes exactly the two required obligations and retains the previous build receipt:
- oracle minimum: 2 units;
- agent fresh refreshes: 2 units;
- full-rerun reference: 3 units;
- unnecessary revalidation: 0;
- missed revalidation: 0;
- selection-set distance: 0;
- refresh precision and recall: 1.0;
- count-based lower-bound savings versus full rerun: 1/3.

These are calculations over one exposed synthetic known-answer trace. They are not model results, benchmark averages, evidence that gold labels are correct, or proof of safe reuse in a real repository.

The expected output is committed so that a future Rust test can compare the entire report, not just a handful of headline numbers. The test suite also mutates the synthetic trace to demonstrate unnecessary refresh, missed required revalidation, a fresh-but-failing check, unknown gold obligations, unknown receipts, future source epochs and duplicate active obligations.

## 5. Why this is a stronger comparison than current benchmarks

The D6 programme is deliberately complementary to, not a replacement for, current benchmarks:

- [SWE-Shift](https://arxiv.org/abs/2610.32196) is the closest dynamic-workspace comparison. It reports 120 scenarios, 12 change families, nine positions from T10–T90 and 17 agent systems; all 17 systems have lower shifted than clean success. GPT-6 Astra is reported at 67.35% clean, 24.79% shifted and 20.34% paired success. D6 should preserve this nine-position design, then add receipt freshness, longitudinal critical exposure, revalidation selection error and independently scoped attestations.
- [ChainSWE](https://aclanthology.org/2026.emnlp-main.371/) tests sequential, dependent multi-bug maintenance: 304 chronological issues across 54 Python projects. It provides a cumulative-maintenance baseline; D6 adds explicit exogenous interventions, control arms and evidence validity over time.
- [SWE-Gate](https://arxiv.org/abs/2609.04167) evaluates review-derived constraints alongside functional tests. In its reported study, 221 of 644 functionally passing repairs failed review constraints. D6 should include review and authority obligations as first-class requirements rather than equating test pass with complete correctness.
- [SWE-Bench Pro V2](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full) is a benchmark-integrity reference: 642 tasks across 11 repositories after invalid-task removal; contradictory instructions were corrected and blind-solved; agent network access is restricted; submitted patches are regraded in pristine images. Its current full public leaderboard is near ceiling (Opus 5 99.40±0.40, Gemini 4 Argon 99.22±0.45), which makes static resolve rate a poor primary differentiation target for D6.

No result from these different benchmark protocols is directly comparable with a D6 score. Use them to guide task quality, controls and baseline selection. A head-to-head claim requires the same tasks, model/harness versions, environment, permissions and resource budgets.

## 6. Expanded D6 target and required controls

The latest D6 protocol uses:
- 12 change families;
- 9 planned-work intervention positions (10% through 90%);
- 5 independently authored task seeds per family/position cell;
- 540 matched clean/shifted scenario pairs;
- 1,080 trajectories per model/trial;
- an initial 108-candidate-seed validity pilot (one per family/position cell).

These are research targets, not an existing corpus. The pilot must pass D6.4's two-sided oracle test, blind independent reconstruction, candidate/evaluator separation, gold-label custody and pristine regrade requirements before model evaluation.

For each cell, include positive, negative, sham and common-cause controls where applicable. The scorer should reject stale evidence reuse and time leakage; the oracle should independently expose over- and under-revalidation.

## 7. Targets and reporting

Targets for the first benchmark pilot:
- zero false admissions across planted invalid-task controls;
- zero stale critical-evidence acceptance;
- zero future-evidence leakage;
- zero missed required revalidation for admitted gold labels in the positive-control set;
- zero unnecessary revalidation in unaffected negative-control checks when no independent policy requires them;
- publish per-family and per-position results, not only an aggregate;
- report exact denominators and confidence intervals;
- measure actual runner time, compute, retries, tokens and reviewer effort separately from count units.

Capability superiority remains an empirical question. A promising target is a better Pareto frontier: higher shifted recovery and critical time-weighted conformance at lower end-to-end cost, with no increase in hard safety failures. Do not set a numerical win target before pilot variance, disagreement rates and task difficulty are measured.

## 8. Security and custody limits

The gold affected-obligation labels are evaluator-side data. Never distribute the oracle output, gold impact labels or full evaluator scenario to the candidate. Candidate-visible trace, hidden gold overlay and grader artifacts should be separate, digest-pinned objects with distinct access controls.

This Rust module consumes the existing D6 evaluator scenario format, which includes gold labels. The module itself does not enforce filesystem permissions, sealed storage, identity authentication or access logging. D6.4's structural admission manifest checks declarations; external custody evidence and independent review are still required.

## 9. Validation status and claim ceiling

- The oracle source and 13 Rust test functions are authored.
- The synthetic expected report is committed.
- A separate deterministic calculation reproduces the expected clean and shifted counts.
- Module export and CLI wiring are present in source.
- No Rust compiler or Cargo is available in this working environment. Rust tests, formatting, compilation and schema validation have not run.
- No real task has been admitted, no hidden labels have been independently adjudicated, and no agent trial has executed.

**Claim ceiling:** D6.6 adds an evaluator-side count-based revalidation lower bound and over/under-revalidation diagnostics for D6-validated traces. It does not establish that the gold labels are correct, that the oracle is cost-optimal, that evidence reuse is safe in production, or that EGDDM improves on any model or benchmark.
