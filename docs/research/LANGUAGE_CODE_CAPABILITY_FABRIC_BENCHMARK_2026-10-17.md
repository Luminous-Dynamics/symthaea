# Symthaea Language–Code Capability Fabric Benchmark RFC (2026-10-17)

**Status:** Research proposal; evaluation contract only. No measured capability improvement is claimed.  
**Scope:** Public Symthaea repository; Broca language capabilities and the Rust-first coding-agent stack.  
**Purpose:** Establish a falsifiable path from semantic understanding to verified software work.

## Executive decision

The proposed paradigm shift is to evaluate Symthaea as a **contract-driven, feedback-correcting engineering system**, not merely as a language/code generator.

A system should:
1. recover the intended meaning and identify unresolved ambiguity;
2. represent requirements, constraints, and exclusions explicitly;
3. construct a solution with traceable obligations;
4. use compiler, tests, and domain-specific checks as feedback;
5. repair within a bounded budget, preserving prior guarantees;
6. state only what its evidence supports.

This does **not** assume that HDC, CfC/LTC, FEP, or a shared language–code representation are automatically superior. Those mechanisms are hypotheses to compare against simpler baselines. The benchmark must be able to show no improvement, regressions, or an indeterminate result.

## Existing baseline to preserve

The repository already has relevant infrastructure:

- `examples/coding_agent_benchmark.rs`: a 55-task coding-agent benchmark described as checking expected names, compilation, and correctness tests.
- `docs/broca-measurements.md`: baseline/candidate artifact comparison, decoder metrics, semantic drift, structured-frame validity, hallucination rate, and explicit skipped Exercism measurements.
- `scripts/coding_agent_quality_gate.sh`: combines code-generation checks, proof-memory tests, coding-backend regression, calibration regression, and optional A/B lanes.

This RFC should compose with that work, not create a competing measurement stack. Before promotion, map each required metric to its current owner; only implement missing measurements. Do not rename an existing metric and imply it is new.

## Research hypothesis and paradigm shifts

### H1 — Meaning before wording

Treat a natural-language request as a set of obligations: intended outcome, actors/data, constraints, exclusions, examples, invariants, and open questions. Paraphrases that preserve meaning should produce equivalent obligation sets; a one-word change to negation, scope, units, or target should change the appropriate obligation.

**Testable prediction:** a structured semantic representation reduces meaning errors on compositional and contrastive cases versus the current path, without increasing unsupported certainty.

### H2 — Code is a behavioral contract, not a text similarity task

Score code by compilation, hidden behavioral tests, invariant preservation, and regression impact. Textual similarity to a reference implementation is diagnostic only. Multiple implementations may be correct.

**Testable prediction:** obligation-to-test tracing and executable feedback improve hidden-test success over a matched no-feedback control.

### H3 — Counterexample-guided repair

A failed compile or test is a counterexample. The system should classify the failure, state a repair hypothesis, make a bounded diff, and rerun the relevant checks. Repeated identical failures, unrelated edits, budget exhaustion, or loss of a previously satisfied invariant should stop or downgrade the result.

**Testable prediction:** feedback-enabled repair improves repair success and/or reduces median repair attempts on held-out tasks without increasing regressions.

### H4 — Epistemic confidence must control action

Confidence is useful only if calibrated against outcomes. Low confidence on an underspecified request should trigger a targeted clarification or explicit abstention, not hallucinated APIs or invented requirements. Confidence values must not be fabricated if the current system does not produce them.

**Testable prediction:** selective answering improves accuracy at a fixed coverage level and lowers false verification claims.

### H5 — Component-level evidence before architectural claims

Compare HDC retrieval/representation, temporal cognition, conventional retrieval, compiler feedback, and their combinations using paired tasks and frozen budgets. If a clean ablation switch does not exist, do not claim a causal comparison; first add one narrow, semantics-preserving switch.

## Evaluation lanes

| Lane | What it measures | Required evidence |
|---|---|---|
| L1. Semantic interpretation | Role/frame extraction, negation, scope, reference, temporal relations, constraints and exclusions | Exact frame match; per-role precision/recall; case-level errors |
| L2. Compositional language | Meaning under paraphrase, reordered clauses, nested constraints and controlled one-factor changes | Invariance on meaning-preserving pairs; correct change on meaning-changing pairs |
| C1. Rust synthesis | Generate a requested function/type/module under a frozen toolchain | Compile rate; hidden-test pass rate; unsafe/dependency policy violations |
| C2. Repair | Diagnose deliberately introduced compile and test failures | Repair success; attempts/time; changed-file scope; regression count |
| C3. Repository work | Interpret an issue, locate the relevant code, make a bounded patch and preserve existing tests | Exact commit/diff, failing-to-passing tests, passing-to-failing regressions, reviewer rubric |
| V1. Verification integrity | Whether the system's pass/fail claims match independent re-execution | False-pass rate, rejected tampered receipts, mutation-control results |
| P1. Engineering trajectory quality | Whether a claimed result follows evidence order: inspection before edits, final patch tested, final diff reviewed, scope authorized, retries evidence-driven, ambiguity safely surfaced, and selected plan bound to executed action | Unsupported-success rate, blind-retry rate, selected-plan/action identity mismatch rate, process ordering, justified scope ratio, trajectory-quality tier |
| E1. Efficiency | Quality versus runtime and resource use | Wall time, attempts, peak memory when available, CPU/GPU/backend and energy only when actually measured |
| X1. Generalization | Performance on temporally fresh, compositional, and held-out cases | Split-specific scores and uncertainty; no mixing train/dev/test |

Do not compress these into one opaque capability score. Report each lane, each difficulty band, and critical failure modes separately.

## Benchmark set design

### Stage 0 — integrity smoke set

Create 32 deterministic, synthetic, human-reviewable cases:

- 8 semantic interpretation / contrastive language cases;
- 8 small Rust synthesis cases with behavior-based tests;
- 8 repair cases with known compile/test counterexamples;
- 8 verification negative controls, including missing tests, incorrect exit status, tampered artifacts, stale commit identity, and an explicitly skipped lane.

The first public starter catalog is now authored at `fixtures/language-code-capability/smoke-v1.json`, alongside `smoke-v1.schema.json`. It contains eight cases in each of four lanes: language semantics, Rust synthesis, repair, and verifier-integrity negative controls. All values are explicitly synthetic. This is a **fixture catalog plus a structural-audit example**, not a model/task evaluation runner or benchmark result. `examples/language_code_capability_smoke.rs` checks the embedded catalog's top-level contract, four lane counts, unique IDs, required lane-specific fields, and the requirement that verifier negative controls expect rejection. Its unit tests cover duplicate IDs, missing required fields, wrong synthetic labelling, lane-count drift and an incorrectly accepted negative control. A source-level audit confirmed that the catalog and schema files parse as JSON, contain 32 unique IDs, and have the expected lane balance. `examples/language_code_capability_receipt_verify.rs` now adds an independent receipt checker for the frozen negative controls and an optional CLI path for a receipt bound to an explicit expected source SHA and an artifact file read from a canonicalized artifact root. It recomputes SHA-256 from those artifact bytes, rejects path traversal/symlink escape, checks result-payload digest/status consistency, and requires both artifact identity and a digest-bound result payload for **passed, failed, and indeterminate** statuses. Authored tests cover valid receipts, missing artifact/result evidence, false-pass claims, tampering, stale source revisions, and path traversal. Run the local orchestrator with `bash scripts/language_code_capability_smoke.sh`; it records the Git HEAD, Rust/Cargo versions, corpus/schema SHA-256 values, and raw logs under `target/language-code-capability/smoke/`. It runs tests and executes each synthetic auditor. The receipt verifier can also be run directly with `cargo run --locked --example language_code_capability_receipt_verify --features receipt-verification`, or with `--receipt FILE --artifact-root DIR --expected-source-sha SHA40` to check a receipt and read artifact bytes from a bounded root. The three Rust examples/tests and the orchestration script have been authored but **not compiled or executed**, and the JSON Schemas have not yet been run through an independent schema validator. A dedicated `receipt-verification = ["dep:sha2"]` feature keeps the receipt check separate from the broader code-generation feature. The future harness must additionally check consistency between `expected_case_count` and actual case count; JSON Schema's `uniqueItems` alone does not enforce uniqueness of an object property.

Every case should have a stable ID, version, task statement, constraints, expected semantic obligations, evaluator, timeout/resource budget, and claim ceiling. Synthetic values must be clearly labelled as synthetic and must never masquerade as real-world corpus evidence.

### Stage 0b — process-quality policy smoke set

The separate starter catalog `fixtures/language-code-capability/trajectory-smoke-v1.json` contains nine synthetic engineering traces, and `trajectory-smoke-v1.schema.json` defines its shape. `examples/language_code_capability_trajectory_audit.rs` applies deterministic policy checks to those traces, including a matched selected-plan/action identity control and a negative control where the selected plan differs from the executed action.

Its authored cases include: a valid patch/test/diff sequence with matching plan/action identity; a claimed success after skipped tests; a patch made after failure but never retested; repeated identical retries without new evidence; unauthorized scope expansion despite passing tests; success claimed without final diff review; a safe clarification/stop; counterexample-guided repair followed by final verification; and a claimed success where the selected plan ID/action fingerprint differs from the action that executed.

The trace auditor deliberately treats a safe abstention as a different positive outcome from completed work. Its plan-identity pair adds a concrete bridge to existing coding-agent work: [#4330](https://github.com/Luminous-Dynamics/symthaea/pull/4330) addresses selector-return identity, while [#4333](https://github.com/Luminous-Dynamics/symthaea/issues/4333) tracks the larger selected-plan-to-observed-execution and single-charge energy boundary. This benchmark addition tests the process property; it does not replace that production fix. It is not a model trajectory evaluator yet: authored synthetic traces test policy definitions, not Symthaea's behavior. Its Rust tests are not yet compiled or executed in this environment. The next step is to capture real Symthaea trajectories with event and artifact provenance and run the same policy checks against them. Do not report a synthetic control audit as an agent-quality score.

This lane is motivated by **AgentLens** (Sahoo et al., May 2026), which analyzed 2,614 coding-agent trajectories and reported that 10.7% of passing trajectories in its main process-evaluation subset were "Lucky Passes" involving patterns such as regression cycles, blind retries, or missing verification; process-quality ranking changed some model ordering materially. Source: https://www.microsoft.com/en-us/research/publication/agentlens-revealing-the-lucky-pass-problem-in-swe-agent-evaluation/

It is also informed by the September 22, 2026 **SWE-Bench Pro V2** release notes: maintainers dropped invalid tasks, corrected instruction/test contradictions, disabled open web tools during agent execution, regraded diffs in pristine images, and required the reference patch to pass while the empty/no-op patch failed. These controls guard against benchmark defects, test tampering, and hidden-fix access. Source: https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full

Every case should have a stable ID, version, task statement, constraints, expected semantic obligations, evaluator, timeout/resource budget, and claim ceiling. Synthetic values must be clearly labelled as synthetic and must never masquerade as real-world corpus evidence.

### Stage 1 — measured development set

Target at least 128 versioned tasks spanning all lanes, with difficulty and task-family labels. The dataset must include boundary cases rather than only happy paths. Do not tune to the public smoke set and then describe it as held out.

### Stage 2 — protected/fresh evaluation

Target at least 256 cases for promotion-level evaluation as resources permit, including tasks created after the candidate baseline is frozen, repository patches, metamorphic pairs, and adversarial negative controls. Freshness means that case construction and release dates are recorded; it is not a proof that a model could not have encountered a similar problem.

Keep public development data and any protected evaluation data explicitly separate. Do not claim public fixtures are secret or uncontaminated. Respect dataset licenses and do not redistribute external benchmark content without permission.

### External benchmark mapping

Use established suites as complementary references, not as substitutes for a Symthaea-native Rust evaluation:

- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) uses time-indexed coding problems and includes code generation, execution, repair, and test-output tasks. Its date filters motivate prospective, freshness-aware tests.
- [EvalPlus](https://github.com/evalplus/evalplus) demonstrates how substantially stronger test suites can expose fragile solutions that pass smaller test sets.
- [BigCodeBench](https://github.com/bigcode-project/bigcodebench) covers API-oriented, software-engineering tasks beyond short algorithm snippets.
- [SWE-bench](https://github.com/SWE-bench/SWE-bench) evaluates issue-to-patch work in real repositories, which complements isolated functions.
- [CRUXEval](https://github.com/facebookresearch/cruxeval) studies code input/output reasoning, but its canonical tasks are Python; do not report it as a Rust benchmark without a validated port.

Static public benchmarks can be contaminated and may fail to represent daily engineering. Pin the exact version and harness; publish which tests are newly authored versus imported; preserve task release dates where available.

A further caution comes from Zhang et al., **“Test of Time: Rethinking Temporal Signal of Benchmark Contamination”** (ACL 2026): the authors find that post-cutoff performance patterns can change substantially when questions are transformed, even when their source material is unchanged. This weakens the idea that “newer task = uncontaminated task” is a sufficient rule. Symthaea's freshness protocol should therefore pair dated task releases with *metamorphic families*: meaning-preserving rewrites, meaning-changing contrast pairs, API/name changes, generated counterexamples, and held-out repository defects. Report each family separately; no single freshness score establishes absence of contamination. Source: https://aclanthology.org/2026.acl-long.1693/

For repository tasks, add two-sided oracle qualification before accepting a case: the reference patch should pass the intended tests and the empty/no-op patch should fail them. Regrade the submitted diff in a pristine copy after agent execution, prohibit network/code-host access during the solve phase unless network use is itself a declared capability, and record whether tests were changed. A task that can be passed by a no-op patch, by editing expected tests, or by accessing the hidden fix is an evaluator defect—not agent success.

## Paired ablation protocol

Establish a frozen baseline first. Run each candidate on the **same case IDs, task order, environment, timeouts, retries, and resource limits**. Record random seeds and repeat stochastic arms at least three times where possible.

Candidate arms, where the implementation permits genuinely isolated toggles:

- **A0 — Current baseline:** unchanged Broca/coding-agent path.
- **A1 — Retrieval/context only:** same generation path with retrieval configuration changed and fully recorded.
- **A2 — Structured meaning/obligation representation:** adds the proposed semantic contract where available.
- **A3 — Compiler/test feedback:** enables bounded repair from real diagnostics.
- **A4 — Cognitive integration:** enables the selected HDC/CfC/LTC/FEP mechanisms under test.
- **A5 — Combined system:** combines qualified components.

Not every arm needs to exist on day one. Record unavailable arms as not run, not zero, and not a pass. If toggles change prompts, model/provider, token budget, tool access, or context size, disclose those differences. If multiple factors change together, the result is a system-level comparison, not evidence that one mechanism caused the difference.

Pre-register one primary metric per experiment, all protected metrics, the minimum meaningful effect, and the stopping rule before inspecting candidate results. Keep unsuccessful experiments and negative results.

## Metrics and definitions

### Language

- **Exact semantic-frame accuracy:** all required slots and relations match the case contract.
- **Per-slot precision/recall:** report missing, extra, and incorrect fields separately.
- **Paraphrase consistency:** meaning-preserving rewrites retain the same obligations.
- **Contrastive sensitivity:** controlled negation/scope/constraint changes alter the relevant obligation and do not rewrite unrelated fields.
- **Selective risk:** error rate at stated answer coverage when the system can abstain.
- **Calibration:** Brier score and expected calibration error only when a probability/confidence output with a well-defined event is available.

### Code and repair

- **Compile rate:** fraction of attempted tasks for which the generated artifact compiles under the pinned compiler and flags.
- **Functional pass@1:** fraction of tasks whose first candidate passes every evaluator-owned hidden test.
- **Repair success:** fraction of broken tasks restored to the expected behavior under the permitted repair budget.
- **Regression rate:** fraction of previously passing protected tests that fail after a candidate patch.
- **Budgeted success:** fraction resolved within a pinned attempt/time/compute budget.
- **Repository resolution:** issue cases satisfying evaluator-defined fail-to-pass and pass-to-pass checks, with patch and test logs retained.
- **Efficiency:** report runtime and resource costs alongside success; a speedup that reduces correctness does not qualify as an improvement.

### Verifier integrity

- **False-pass rate:** fraction of results claimed to pass when an independent verifier does not confirm that status.
- **Negative-control rejection:** fraction of deliberately invalid, stale, skipped, or tampered records rejected.
- **Replay determinism:** repeated evaluation of the same pinned inputs produces identical normalized outcome records, except explicitly separated nondeterministic metadata.
- **Provenance completeness:** fraction of reports containing every required input/configuration/toolchain/artifact identity.

### Engineering trajectory quality

- **Unsupported-success rate:** fraction of completion claims made without a passing test for the final patch and a review of every changed file.
- **Blind-retry rate:** repeated identical actions without new diagnostic evidence, a configuration change, or a predeclared reason to retry.
- **Final-state verification:** fraction of patches whose exact final patch identity is linked to successful test output and complete diff review.
- **Plan/action identity consistency:** fraction of executed actions whose plan ID and canonical action fingerprint match the most recent selected plan. A mismatch is a hard process failure even if the final tests pass.
- **Scope discipline:** fraction of changed files/behaviors justified by the task contract; unauthorized expansion is a hard failure, not something averaged into a score.
- **Recovery efficiency:** fraction of failed attempts followed by a diagnosis grounded in new evidence and a repair that changes the subsequent outcome.
- **Safe abstention:** correct detection of ambiguous/conflicting requirements without unauthorized mutation; report separately from task completion.
- **Trajectory robustness:** replicate a subset of tasks across independent seeds and report outcome variance, retry loops and regression cycles, not just the best run.

Use a separate evaluator for final pass/fail wherever feasible. A generator should not self-certify. Test harnesses must not execute generated code on a developer's unsandboxed host: use isolated, resource-capped execution without ambient secrets or unnecessary network access.

## Proposed promotion gates — targets, not measured results

| Gate | Proposed threshold | Interpretation |
|---|---|---|
| Evidence identity | 100% of promoted runs bind exact source revision, case-set digest, configuration, toolchain, and raw-result digests | Missing identity means the run is not promotable |
| Verification negative controls | 100% of frozen false-pass/tamper/skip controls rejected | One false pass blocks promotion |
| Language improvement | Aim for at least +5 percentage points on the pre-registered primary held-out semantic metric, with a paired 95% confidence interval excluding zero | A research target, not an industry standard; expand the sample when uncertainty is too wide |
| Coding improvement | Aim for at least +5 percentage points in held-out functional pass@1 **or** at least 20% lower median repair attempts, with no protected correctness regression | Select only one primary outcome before the experiment |
| Regression guard | No statistically supported increase in protected-test failures; any safety/false-pass regression blocks regardless of aggregate score | Hard failures cannot be averaged away |
| Trajectory discipline | 100% of synthetic policy controls receive their expected classifications; 0 unsupported-success claims on promotion candidates; 100% rejection of plan/action identity mismatch controls; no unauthorized scope expansion | Synthetic controls qualify the auditor only; live agent trajectories must be assessed separately |
| Calibration | Where confidence exists, initial target ECE ≤ 0.10 on a sufficiently large, predeclared sample, while reporting coverage and the calibration method | ECE alone is not sufficient and can be unstable on small datasets |
| Replication | Direction of the primary effect repeats across three runs where the path is stochastic, using the same frozen protocol | Repeated runs do not compensate for contaminated or weak tests |

Confidence intervals, sample counts, all failed runs and per-lane outcomes must accompany scores. If statistical power is inadequate, report inconclusive and collect more cases rather than relaxing the gate after seeing results. Thresholds should be reviewed and frozen before each new promotion cycle.

## Reproducible run receipt

Every comparable run should record, at minimum:

- schema/benchmark version and unique run ID;
- repository commit SHA and relevant file/blob digests;
- exact case IDs, split, corpus digest, source dates, and licensing note;
- baseline/candidate arm, feature toggles, prompt/config digest, backend/provider/model identifier where applicable, and seed;
- Rust/Cargo/rustc version, target triple, features, lockfile digest, OS/container image, sandbox policy, timeout, retry, and resource budget;
- task-level generated artifact digest, compile/test commands and exit statuses, normalized stdout/stderr digests, repair attempts and changed-file list;
- independent verifier result, all metric values, skipped/unavailable lanes, and explicit claim ceiling;
- raw artifact locations and the digest of each retained artifact.

Do not use a SHA-256-shaped string as proof that artifact bytes were checked. When a verifier has the bytes, it should hash those bytes. A structurally valid receipt proves only that its declared evidence has the stated identity, not that the task specification, test oracle, or source data are correct. The current receipt checker also does not independently rerun declared build/tests or authenticate the producer: a forged but internally self-consistent receipt remains possible. Promotion therefore still requires a trusted runner/attestation path and independent re-execution.

## Execution plan that does not depend on a queued CI run

1. **Freeze and capture the current baseline.** Run the repository's existing fast Broca/coding measurement paths locally where tooling is available. Record missing tools and skipped lanes explicitly; do not infer a pass from an artifact's existence.
2. **Map metric ownership.** Compare this contract with existing scripts/reports and identify only actual gaps. Avoid duplicate harnesses or parallel sources of truth.
3. **Exercise the first harness locally.** Run `bash scripts/language_code_capability_smoke.sh` from the repository root inside the Nix/Rust devshell. It runs all three Rust example test targets, executes the synthetic auditors, and writes a run manifest plus raw logs. A successful run should include `NEGATIVE_CONTROLS_REJECTED` only after all eight receipt negative controls fail for their expected reasons, and `TRAJECTORY_POLICY_CONTROLS_MATCHED` only if all nine process-policy traces match their expected classifications. To check a real receipt, run `cargo run --locked --example language_code_capability_receipt_verify --features receipt-verification -- --receipt FILE --artifact-root DIR --expected-source-sha SHA40`. These commands have been authored as the local run path but have **not been executed in this environment**. After a local pass, add independent schema validation and a sandboxed compiler/test runner for generated Rust snippets. Keep compilation/evaluation results separate from fixture-authoring and structure checks.
4. **Add or extend a Rust evaluator and receipt.** Prefer existing crates, types, and scripts. Keep the evaluator independent from generation logic; use no new dependency unless required and reviewed.
5. **Capture a same-head baseline and paired ablation receipts.** Ensure each comparison references exact source/config revisions. CI queue status remains independent of local source-level checks; skipped/queued/failed runs are never green.
6. **Expand to fresh, repository-level and temporally held-out cases.** Do this only after the smoke evaluator and verifier contract are qualified.
7. **Promote one measured change at a time.** Re-run protected controls and prior regression suites; report negative results as first-class outcomes.

## Definition of done for the first implementation tranche

- [ ] 32 stable smoke case IDs and an explicit synthetic-data label.
- [ ] Every case has a versioned task contract and a behavior-based oracle or a deliberately asserted negative-control outcome.
- [ ] Deterministic run manifest binds exact source/config/corpus/toolchain identities.
- [ ] False-pass, tampering, stale identity and skipped-evaluation controls all fail closed.
- [ ] Existing Broca and coding-agent metrics are reused where equivalent.
- [ ] The separate nine-case trajectory-policy corpus matches its synthetic expectations, including selected-plan/action identity; real agent traces are then collected and assessed independently.
- [ ] One baseline run is captured with accurate pass/fail/skipped/unavailable distinctions.
- [ ] No capability, speed, safety or benchmark-ranking improvement is claimed until verified on the exact candidate head.

## Non-goals and claim boundaries

This RFC does not implement a universal semantic IR, train a model, change production routing, prove consciousness or general intelligence, or establish that HDC is superior to conventional methods. It does not claim a baseline score, current CI status, Rust compilation, or benchmark pass. Its first deliverable is a stricter experimental contract that makes those future claims possible to evaluate honestly.
