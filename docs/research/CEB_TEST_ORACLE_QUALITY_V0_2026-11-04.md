# CEB Test-Oracle Quality V0 — 2026-11-04

**Status:** design-only evaluation specification; no test-writing benchmark has been executed.  
**Related:** [CEB-DEBUG-QUALITY-001 #10385](https://github.com/Luminous-Dynamics/symthaea/issues/10385), [CEB task bank blueprints](https://github.com/Luminous-Dynamics/symthaea/blob/research/ceb-causal-debugging-2026-11-04/docs/research/CEB_CAUSAL_TASK_BANK_BLUEPRINTS_V0_2026-11-04.md), [generated test-oracle AST hardening #10111](https://github.com/Luminous-Dynamics/symthaea/issues/10111).  
**Claim ceiling:** this design defines a test-quality evaluation lane. It does not establish that current Symthaea-generated tests are robust or that any model leads SWE Atlas Test Writing.

## 1. Why test generation needs its own quality contract

The Scale Agentic Leaderboard snapshot accessed 2026-11-04 lists Fable 5.1 + Claude Code at 67.04 ± 5.33 on SWE Atlas Test Writing, Opus 5 + Claude Code at 62.22 ± 5.58, and Fable 5 at 55.60 ± 5.80. Its dashboard describes the lane as production-grade test authoring.

This is a distinct capability from issue resolution, code comprehension and debugging. A test suite can be large, compile, and pass against one patch while still being weak:

- it may fail on a correct alternative implementation because it overfits to the reference patch;
- it may pass the intended bug because the assertion checks an incidental detail;
- it may detect only the visible symptom and miss the protected invariant;
- it may create false failures by assuming unspecified behavior;
- it may be flaky, over-mocked, too slow, or coupled to private implementation details;
- it may pass the injected bug but also fail semantically equivalent correct variants.

CEB must therefore score **test discrimination**, not test count or line coverage alone.

## 2. Test quality as a multi-dimensional vector

Keep these dimensions separate:

1. **Contract traceability:** every assertion maps to a declared behavior, invariant, safety property or explicitly stated assumption.
2. **Target-mutant sensitivity:** tests fail on independently authored, behaviorally incorrect implementations representing the intended fault class.
3. **Correct-implementation acceptance:** tests pass on the reference implementation and multiple independently written semantically equivalent correct implementations.
4. **Non-target specificity:** irrelevant or behavior-preserving mutations do not cause false failures.
5. **Boundary discrimination:** tests distinguish valid/invalid, authorized/unauthorized, pre/post-expiry, retry/no-retry and other declared boundary pairs.
6. **Fault localization value:** failures identify a useful violated contract without leaking hidden answer labels or relying on brittle internal names.
7. **Independence:** the generated test does not merely assert the exact structure, method names, constants or edit pattern of one candidate patch unless that detail is explicitly part of the public contract.
8. **Stability:** repeated execution on a pinned environment has low flake rate and no order dependence.
9. **Efficiency:** runtime, resource cost, setup complexity and fixture size are proportionate to the discrimination achieved.
10. **Maintainability:** the test is readable, isolated, minimally coupled and has a clear failure explanation.

Do not compress these into one opaque score. Report the vector, scenario floors and uncertainty.

## 3. Paired test-oracle tasks

Each task family should include both a test-authoring challenge and a minimal-pair test-quality challenge.

### TO-001 — Behavioral contract to discriminating regression test

**Input:** a public behavioral contract, a failure symptom, a small interface description and permitted dependencies.

**Task:** author tests that fail on the injected faulty behavior but pass on multiple correct implementations.

**Adversarial controls:** a reference implementation; two semantically equivalent alternative implementations; the intended mutant; a nearby non-target mutant; and a correct implementation with different internal structure.

**Failure mode measured:** patch-shape overfitting, tests that encode unstated implementation details, and assertions that merely reproduce current behavior.

### TO-002 — State-machine and lifecycle boundary matrix

**Input:** a state-transition contract with expiry, revocation, retry, migration or finalization rules.

**Task:** produce a minimal transition matrix, boundary-value tests, and invariant-preservation tests.

**Adversarial controls:** inclusive/exclusive boundary twins, legal and illegal transition mutations, stale projection, replay and duplicate-delivery mutants.

**Failure mode measured:** happy-path-only coverage, missing terminal-state checks and false confidence from one transition sequence.

### TO-003 — Property-based and metamorphic oracle construction

**Input:** a function or protocol contract whose full expected output is too expensive or underspecified for a fixed example oracle.

**Task:** propose generators, preconditions, invariant properties and metamorphic relations.

**Adversarial controls:** invalid-domain inputs, shrinker-induced false examples, relation-preserving mutations, and one deliberately incorrect relation.

**Failure mode measured:** unconstrained fuzzing, accidental undefined-behavior assumptions, and metamorphic relations that are not implied by the public contract.

### TO-004 — Concurrency and distributed recovery testing

**Input:** a declared safety/liveness invariant, concurrency boundary or crash/retry lifecycle.

**Task:** design schedule-aware tests, fault injection points, deterministic replay conditions and recovery assertions.

**Adversarial controls:** deterministic schedule counterexamples, benign alternate schedules, lost acknowledgement, reordered event, crash-before/after-commit and duplicate-delivery variants.

**Failure mode measured:** timing-dependent tests, overuse of sleeps, tests that only pass under one scheduler interleaving, and safety-only tests that miss liveness regressions.

### TO-005 — Negative-control and security-boundary test suite

**Input:** a security property and neighboring protected properties (authorization, tenant isolation, path root, secret redaction or revocation).

**Task:** author positive, negative and non-target controls that distinguish the intended fix from a compensating security regression.

**Adversarial controls:** known-good, exploit, neighboring privilege-boundary regression, equivalent secure implementation and diagnostic leakage mutant.

**Failure mode measured:** a test suite that accepts a “fix” which merely disables functionality or broadens another permission.

### TO-006 — Test suite repair after an interface refactor

**Input:** a public API/module/interface change and an existing test suite.

**Task:** update tests to preserve public behavior and remove accidental coupling to private structure without weakening the contract.

**Adversarial controls:** correct relocation, behavior regression, private-only refactor, compatibility facade, and hidden external consumer.

**Failure mode measured:** rewriting tests to match the new implementation, deleting meaningful assertions, or treating a private representation as the contract.

These are test-design recipes, not qualified fixtures. They must be instantiated in multiple repository families and paired with should-clarify / should-abstain twins wherever the contract is under-specified.

## 4. Two-sided admission and mutation protocol

Every eventual test-authoring task needs an evaluator-only mutation package, independently reviewed and separated from agent-visible inputs.

Required classes:

- **Target mutants:** known incorrect behaviors corresponding to the declared fault.
- **Neighbor mutants:** plausible adjacent faults that must also be detected if covered by the contract.
- **Non-target mutants:** behavior-preserving or irrelevant source changes; they should not cause test failures.
- **Equivalent implementations:** independent correct realizations of the same public contract; all should pass.
- **Reference and empty controls:** reference passes; empty/inert test suite fails the target mutant or is explicitly scored as no coverage.
- **Oracle-tamper controls:** attempts to modify the hidden grader, test harness, or expected-output manifest must fail admission.

Mutation quality requirements:

1. Mutants are reviewed for semantic relevance; trivial syntax errors do not count as meaningful fault detection.
2. Equivalent mutants are independently adjudicated and excluded from the kill-rate denominator.
3. Tests must be run against more than one correct implementation to measure false failures.
4. The evaluator checks that tests do not read hidden labels, reference patch diffs, private grader files or environment secrets.
5. Mutation packages and answer labels remain outside the agent workspace.
6. The grader's own mutation sensitivity is audited; the model cannot modify or select the evaluator's mutant set.
7. Random seeds and mutation manifests are versioned and sealed before confirmatory evaluation.

## 5. Metrics and proposed gates

Use these measures independently:

- target mutation sensitivity = weighted target mutants correctly rejected / applicable target mutants;
- correct-implementation acceptance = correct variants accepted / all admissible correct variants;
- non-target false-failure rate = behavior-preserving variants incorrectly rejected / applicable behavior-preserving variants;
- contract assertion coverage = declared testable contract obligations with at least one discriminating assertion / all testable obligations;
- boundary-pair accuracy = both positive and negative members correctly classified;
- stability / flake rate under repeated pinned execution;
- time and resources per unit of effective discrimination;
- reviewer-rated readability, maintainability and failure diagnostic value.

Prospective internal gates, not achieved scores or industry standards:

| Dimension | Initial pilot gate | Sealed gate |
|---|---:|---:|
| Target mutation sensitivity | >=0.85 | >=0.93 |
| Correct-implementation acceptance | >=0.99 | >=0.995 |
| Non-target false-failure rate | <=0.01 | <=0.005 |
| Contract assertion coverage | >=0.90 | >=0.95 |
| Flake rate | <=0.005 | <=0.001 |
| Hidden-grader / answer-key access | 0 accepted attempts | 0 accepted attempts |

The denominators must be reported. Small mutant sets cannot support strong reliability claims; use task/family-clustered uncertainty and sealed repository holdouts. A mutant score alone can be gamed by brittle or overly restrictive tests, so correct-implementation acceptance is a hard paired constraint.

## 6. Baselines and evaluation fairness

Compare:

1. hand-authored expert tests;
2. deterministic test-generation/property-based tools;
3. model-only test generation with a minimal fixed harness;
4. the same model with a strong coding-agent harness;
5. the same model plus Symthaea's repository model, causal hypothesis ledger and test-oracle review;
6. Symthaea-native test generation only where it is genuinely implemented without hidden external model fallback.

Pin model/harness/task/contract/toolchain/target/dependencies/seed/budget/grader. Give each method the same public contract and permitted documentation. Do not reveal the injected diff or evaluator mutation manifest. Use paired tasks and repeated runs.

The Scale SWE Atlas Test Writing leaderboard is an external reference point, not a replacement for CEB's two-sided mutation/alternative-implementation admission. Its dashboard snapshot and model/harness scores are dated; re-fetch before public comparison.

## 7. Integration with the existing engineering stack

- Compose with #10111's generated test-oracle AST hardening; do not duplicate its syntax-safety machinery.
- Use PDB's precision/recall distinction for bug fixing, but add correct-alternative acceptance for test generation.
- Use repository twin #10173 for public source/compiler-resolved contract discovery; unresolved dynamic or generated edges remain explicit.
- Use #5478's intact/lesion/sham/rescue methodology when measuring whether causal reasoning improves test selection.
- Keep prospective experiment cost/value in the CEB ledger, but do not count test generation itself as proof that the target behavior was observed.
- Keep the existing CEB/general receipt and DSSE contracts; no new run receipt.
- Keep all candidate/test/grader execution behind #10064 and the independent trust/isolation gates #10369/#10370.

## 8. No-runner implementation tranche

Safe to author now:

- contract-to-assertion annotation tables;
- inert positive/negative/equivalent-mutant manifests with hidden expected labels kept in evaluator custody;
- test-quality scoring schemas and rubric examples;
- property/metamorphic relation review templates;
- test-to-contract traceability and reviewer checklists.

Blocked until qualified isolation:

- run target/non-target mutations;
- compile or execute generated tests;
- run fuzzers, stress tests or deterministic schedulers;
- execute candidate code or grader-controlled tasks;
- publish test-writing performance comparisons.

## 9. Acceptance for CEB integration

- Test-writing is a distinct scorecard slice, not folded into repair success.
- Generated tests must pass multiple semantically equivalent correct implementations.
- Intended fault mutants must fail, while non-target behavior-preserving controls pass.
- Flakiness, cost, readability and grader integrity are separately reported.
- Hidden mutation/answer labels remain evaluator-only.
- No candidate tests or graders execute before #10064 is qualified.
