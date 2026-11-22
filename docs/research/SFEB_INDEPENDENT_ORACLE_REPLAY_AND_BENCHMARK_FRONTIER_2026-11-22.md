# SFEB independent oracle replay and benchmark frontier

**Date:** 2026-11-22  
**Status:** oracle-attestation contract and comparator ledger authored; replay engine pending

## 1. Why a second attestation is required

The execution worker and the semantic evaluator must not share the authority to declare a run successful. A signed execution trace authenticates the origin and identity of the captured evidence under a trust policy; it does not establish that the expected behavior was satisfied.

The independent verifier therefore emits a separate in-toto Statement, using `sfeb-oracle-verification-predicate-v1.schema.json`. It binds the exact execution statement, trace, evidence artifacts, task/fixture/source/environment identities, verifier binary, oracle policy and derived per-oracle outcomes.

The existing qualification-report v1 still consumes runner-supplied `observed_oracle_ids`. That format remains a useful local artifact-integrity/adjudication prototype, but it must not be represented as independently oracle-adjudicated. A v2 report must consume a separately signed verifier statement.

## 2. Independent verifier contract

The verifier must open the signed execution statement and raw artifact bundle itself. It then:

1. checks execution attestation identity and the one-time run challenge;
2. verifies raw trace and event-chain hashes;
3. re-hashes every screenshot, accessibility, layout, network and console artifact;
4. binds every observation to the exact case, task, fixture, source revision and environment;
5. evaluates the frozen oracle rules from raw events/artifacts;
6. emits one result per required oracle, each referencing evidence event/artifact digests and the exact rule digest;
7. produces accepted/rejected/inconclusive using fail-closed semantics;
8. signs a second in-toto Statement under a distinct trusted verifier workflow identity.

A result is inconclusive—not accepted—when an oracle cannot be evaluated, the policy is incomplete, evidence is absent, or trace/artifact identities mismatch.

## 3. External benchmark position — 2026-11-22

The machine-readable record is `benchmarks/frontend/external-comparator-ledger-v1.json`. Important distinctions:

- Vision2Web is the closest broad website-development comparator: 193 tasks, 16 categories, 918 prototype images and 1,255 test cases across static, interactive and full-stack levels. Its S4-2026 historical row lists 82.3 overall, with 87.4 static, 83.2 interactive and 76.3 full-stack averages. The public page selected for the 2026-11-22 check displayed no results for that selected season, so that row is historical, not a current-season SOTA claim.
- Design2Code evaluates 484 screenshot-conditioned real-world webpage reconstructions, with automatic rendering metrics complemented by human evaluation. It is a visual reconstruction comparator, not an end-to-end repair/security benchmark.
- WebdriverIO's page exposes two model contexts and seven browser tools over a fixed 100-task sample. Its indexed October 9 snapshot lists 2,511 total agent runs and an aggregate WebdriverIO session row of 62% (103/165; 55–69% interval). A separate DeepSeek V4.1 Flash run reports 58/100 for WebdriverIO session. A direct page render on November 22 showed an empty run-log state. Therefore 62% is a historical, model-context-dependent aggregate—not a single-run score or a universal browser-tool lead. The benchmark warns about roughly ±10-point intervals at 100 tasks and WebJudge/human disagreement.
- OpenAI's July 8, 2026 SWE-Bench Pro audit flagged 200/731 tasks automatically and 249/731 in human annotation. This is evidence for benchmark QA, not a claim about the defect rate of SFEB.

Do not pool raw results. Reproduce each benchmark's original task set, model, prompt, harness, judge and scoring scripts before presenting an external comparison.

## 4. What we should aim to beat

### Matched external reproduction
- Reproduce Vision2Web's official task distribution and scoring pipeline before tuning; submit an authorized official result when the runner is ready.
- On the matched protocol, target at least a three-point improvement over a freshly reproduced reference in each level, with paired confidence intervals excluding zero, and no hidden hard-gate regressions. This is a stretch target, not a current SFEB score.
- Reproduce Design2Code's native automatic and human evaluation separately; target improved element recall and layout correctness, not an invented composite.
- For browser-tool comparisons, match one model and harness at a time. Use the published 100-task sample for development only, then expand to at least 300 distinct tasks and repeated trials for a more stable estimate. Do not treat repeated trials on the same task as independent tasks.

### SFEB's own differentiated claim
A credible primary claim remains A4 versus A0 on paired tasks with the same model/tools/budgets:
- superiority only when the paired task-level 95% interval is positive and all independent hard gates pass; or
- non-inferiority within two percentage points plus at least 30% lower cost per accepted task.

### Causal repair lane
Proposed internal qualification scale:
- 150 seeded defects across visual/responsive, functional/state, accessibility, security and runtime/maintenance families;
- at least 30% hidden or controller-only;
- Top-1 root-cause localization at least 80%, Top-3 at least 95%;
- repair success at least 85%;
- unrelated regression below 5%;
- placebo false-credit below 1%;
- zero critical security/evidence-integrity false acceptance;
- calibration ECE below 0.10;
- cost per accepted repair at most 70% of matched baseline.

These are targets, not measured results. They require independent oracle qualification first.

## 5. Paradigm shift

The field should move from code-output competition to verified change competition. Each task should create an evidence graph:

`requirement -> contract -> implementation -> intervention -> browser observations -> independent oracle -> repair -> invariant preservation -> signed verifier result`

The resulting research contribution is not that an agent can write a page. It is that an engineering system can establish, independently and repeatedly, that a specific change satisfies a specified requirement and preserves the guarantees that were not meant to change.

## 6. Execution boundary

- The Rust trace-integrity verifier source and adversarial tests are now committed in PR #13162.
- The verifier has not been compiled or executed; the current environment has no Rust toolchain.
- The independent oracle-result schema and comparator ledger are committed as contracts/data, not as a running evaluator.
- The current report v1 still trusts runner-supplied oracle IDs and remains a legacy local adjudicator.
- #13228 owns execution attestation, trace verification and isolated runner.
- #13247 owns the independent oracle engine, verifier statement and qualification report v2.

## Sources

- Vision2Web: https://vision2web-bench.github.io/ and https://arxiv.org/abs/2603.26648
- Design2Code: https://aclanthology.org/2025.naacl-long.199/
- WebdriverIO benchmark and methodology: https://benchmark.webdriver.io/ and https://github.com/webdriverio/benchmark
- WebdriverIO October 9 run: https://github.com/webdriverio/benchmark/blob/main/results/2026-10-09-run-37883072636/report.md
- OpenAI SWE-Bench Pro quality audit: https://openai.com/index/separating-signal-from-noise-coding-evaluations/
