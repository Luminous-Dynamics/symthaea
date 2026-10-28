# Verified Legacy Engineering Benchmark (VLEB): Research Protocol

**Status:** draft research specification; no benchmark has been run and no performance advantage is claimed.  
**Date:** 2026-10-24  
**Owner:** Symthaea / Luminous Dynamics

## Executive thesis

Coding-agent evaluation should measure verified engineering outcomes—not only whether one patch passes a test script. This matters most for legacy and financial systems, where silent data corruption, changed decimal semantics, missing side effects, broken restart behavior, or an incomplete migration can pass shallow tests.

This protocol proposes an internal pilot, the **Verified Legacy Engineering Benchmark (VLEB)**. It complements, rather than replaces, public benchmarks such as Factory's Legacy-Bench and SWE-bench Multilingual.

The paradigm shift is to make the evaluated unit a reproducible **engineering episode** containing:

1. a task contract and locked source/environment identity;
2. a valid, discriminating oracle with independent negative controls;
3. an attempted code change or explicit abstention;
4. independently reproduced test and runtime evidence;
5. behavioral-equivalence, data-integrity, and recovery checks where applicable;
6. uncertainty, cost, latency, and failure classification.

A syntactically valid or compiling patch is not automatically a successful repair. A successful repair does not automatically establish a safe migration. A valid benchmark score does not automatically establish out-of-distribution generalization.

## Why a specialized evaluation is needed

### What existing public benchmarks contribute

- **Factory Legacy-Bench** evaluates legacy engineering across COBOL, Java 7, BASIC, C89, Fortran, and Assembly. As of the Specialized Intelligence Index listing updated **October 22, 2026**, Claude Opus 5.5 and Muse Spark 1.3 are tied at **62.0% avg@3**. The listing reports Opus at **$0.49/task and 434 seconds/task**, Muse Spark at **$1.98 and 744 seconds/task**, DeepSeek V4.1 Flash at **57.7%, $0.14 and 691 seconds**, and GPT-6 Astra at **55.7%, $2.33 and 629 seconds**. This positions Opus as a cost-efficiency comparator (about 75% lower listed task cost than Muse Spark) as well as a score leader. These are three-attempt average results, not single-attempt pass@1, and costs/agent setups must be checked before making an apples-to-apples claim. The published task analysis reports that 31 of 44 tasks no evaluated model solved were COBOL tasks. Ten representative samples are public; the full benchmark is available by request. Use public samples for development only, not as a full-benchmark score. Sources: [Legacy-Bench methodology](https://docs.factory.com/benchmarks/legacy-bench), [results current as of October 22](https://fireworks.ai/specialized-intelligence-index/benchmarks/code-generation/legacy-bench), [public sample tasks](https://github.com/factory-ai/legacy-bench).
- **SWE-bench Multilingual** contains 300 issue/PR repair tasks across 42 repositories and nine languages (C, C++, Go, Java, JavaScript, TypeScript, PHP, Ruby, Rust). Its authors describe a median gold patch of 10 changed lines and note that the benchmark does not measure broad, large-scale transformations. Source: [benchmark and limitations](https://www.swebench.com/multilingual).
- **SWE-bench Pro V2** demonstrates why task eligibility and clean regrading should be first-class: its September 2026 release dropped 89 tasks it judged invalid from a 731-task version, corrected 69 task instructions that contradicted grading tests, disabled web tools during the agent phase, and re-graded diffs on pristine images. Source: [Scale AI's methodology/result page](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full).
- **Mutation-based benchmark work** warns that passing tests do not necessarily discriminate correct from incorrect implementations. SWE-Mutation's 2026 paper evaluates 2,636 mutated variants from 800 original tasks; its reported strongest model result includes 10.20% verification and 36.15% detection, and more realistic agent-generated mutants were harder to detect than conventional mutants (39.81% versus 71.04% average detection). SWE-ABS (ICML 2026) separately reports that roughly one in five patches classified as solved among top-30 agents were semantically incorrect after adversarial test strengthening; its approach combines coverage-guided and mutation-driven checks. This supports measuring test/Oracle strength independently from repair success, with both conventional and agent-generated mutants. Sources: [SWE-Mutation, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1976/), [SWE-ABS, PMLR 306](https://proceedings.mlr.press/v306/yu26b.html).
- **Mainframe modernization** is a system-level change. IBM cautions that mere code translation often fails to address the complexity of modernization. Source: [IBM, mainframe modernization](https://www.ibm.com/think/topics/mainframe-modernization).

These sources imply that VLEB should avoid optimizing one leaderboard number. It needs rigorous task construction, hidden negative controls, business-behavior checks, and a separate migration protocol.

## Pass receipts are evidence contracts, not truth claims

The receipt schema now makes a `pass` structurally impossible when the record reports only a runtime-self-reported trust boundary, lacks hashed command/test logs, marks required tests skipped/failed, leaves mutations unresolved, reports a differential mismatch, or omits complete timing/cost accounting. It also requires an auditor-output digest and either an immutable runtime image identity or host attestation.

JSON Schema cannot by itself prove that the receipt corresponds to real artifacts or that its `required_test_coverage = 1` claim is true. The independent auditor must recompute artifact hashes; reconcile required test IDs against unique observed results; compare the task, protocol, candidate, and environment identities; validate the actual test process; and verify every preregistered mutation. A well-formed receipt is necessary evidence, not sufficient proof.

## Next-wave research agenda (2026-10-28)

### Fresh external comparison

The Legacy-Bench index was updated **October 22, 2026**. Its leader is a tie at **62.0% avg@3** (Claude Opus 5.5 and Muse Spark 1.3). The same listing reports $0.49/task for Opus, $1.98/task for Muse Spark, $0.14/task for DeepSeek V4.1 Flash at 57.7%, and $2.33/task for GPT-6 Astra at 55.7%. Use these as the current legacy-engineering reference, not as pass@1; cost comparison must also disclose harness, model and runtime differences. Source: [Legacy-Bench, updated Oct 22](https://fireworks.ai/specialized-intelligence-index/benchmarks/code-generation/legacy-bench).

SWE-bench Pro V2 is a distinct evaluation with a different task set and turn budget. Its September 22 refresh reduced the public split from 731 to 642 tasks after removing 89 invalid tasks, corrected 69 instruction/test contradictions, disabled browser/web retrieval during agent execution, and regraded submitted diffs on pristine images. The leaderboard currently lists results up to 99.40 under its own protocol; do not compare that percentage directly with Legacy-Bench. Source: [SWE-bench Pro V2 methodology](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full).

SWE-ABS reports that adversarial strengthening rejected **19.78% of previously passing patches** on 500 SWE-bench Verified instances, lowering the top result in its study from 78.80% to 62.20%. This supports using an adversarially strengthened score as a second metric, not replacing the standard public result. Source: [ICML 2026 / PMLR](https://proceedings.mlr.press/v306/yu26b.html).

Security needs a separate axis. Endor Labs' AI Coding Agent Security Benchmark lists a best functional score of 87.2% but a best security-correctness score of 37.4% among the systems it evaluated. The numbers are vendor-benchmark-specific and not interchangeable with VLEB; the disparity nevertheless supports measuring secure behavior independently from functional success. Source: [Agent Security League](https://www.endorlabs.com/research/ai-code-security-benchmark).

### Operational run manifests

The `tools/vleb/run-manifest.schema.json` contract separates the evaluator's trusted run identity from the agent-produced receipt. `tools/vleb/verify_run_manifest.py` requires out-of-band SHA-256 pins for the manifest, manifest-verifier source/schema, permitted receipt-auditor source and receipt-verifier binary; it recomputes referenced artifact digests and rejects unsafe paths. For confirmatory runs it verifies the attestation of the complete evidence bundle and checks the immutable repository ID, workflow identity, source-repository digest/ref, run ID/attempt and subject digest before extracting the ZIP. The extracted members—not adjacent files—are then hash-checked and passed to the pinned independent receipt auditor. In the follow-up claim gate, confirmatory eligibility also requires a non-empty evaluator-pinned claim inventory, independently pinned claim auditor/schema, witness-receipt schema/producer, matching source/span and witness artifact digests, and out-of-band pins for each witness-verifier kind. Each witness receipt binds exact task/claim/source, repository/workflow/run ID and attempt, candidate checkout, runtime identity, verifier identity, invocation, input/output hashes and process status; the output reports supported, contradicted, unresolved and failed claim counts separately.

The expected manifest digest must be supplied from a protected evaluator record or other independently verified trust root; deriving the pin from the same untrusted manifest would be circular. Candidate checkout revision, source-repository digest/ref and workflow identity SHA/ref are separate identities; a pull-request workflow's source SHA may be a merge commit, while the candidate checkout is the pull request's head commit. Each must match its own manifest and certificate field rather than being coerced into one SHA. A development-smoke manifest may validate its integrity, but it always reports `score_eligible=false`. A confirmatory run becomes eligible only after the manifest hash, artifact hashes, attestation, two-sided task controls and independent receipt audit all pass. See [the run procedure](../../tools/vleb/CONFIRMATORY_RUNS.md).

This is a provenance and evidence-integrity gate, not a complete semantics oracle: GitHub attestation ties bytes to a workflow identity but does not prove the workflow's claims are true or the patch is correct. The trusted recorder captures a verifier invocation and the manifest verifier checks the receipt against the attested run, but the trusted evaluator still must control result classification, command selection, and network isolation. The task-level behavior contract and independent receipt auditor remain separate requirements; see [VLEB issue #8702](https://github.com/Luminous-Dynamics/symthaea/issues/8702) for remaining classifier review and anti-replay controls.

### Paradigm shift: three clocks, three claims

1. **Patch clock:** did the agent produce a change that satisfies the locked contract?
2. **Proof clock:** did an independent verifier reproduce hashes, observe all required tests, and kill the registered semantic mutants?
3. **Provenance clock:** can a separate trusted evaluator establish that the logs came from the claimed checkout, image and workflow?

A hash-only receipt covers artifact integrity, not execution provenance. A malicious or faulty runner can fabricate logs and hash them consistently. For confirmatory results, package the receipt, patch, test logs, mutation evidence and differential details into one evidence bundle. Keep the run manifest as a separately pinned sidecar: it contains the evidence-bundle digest, so including the manifest inside the bundle would create a self-referential hash. Verify the bundle's signed provenance from a dedicated trusted workflow. GitHub documents `actions/attest@v4` and `gh attestation verify` for signed artifact provenance; that attestation links artifacts to their workflow identity but does not prove semantic correctness. Sources: [actions/attest](https://github.com/actions/attest), [GitHub attestation verification](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations), [gh attestation verify](https://cli.github.com/manual/gh_attestation_verify). Avoid issuing attestations for every routine development CI build; reserve signed evidence bundles for pre-registered, confirmatory benchmark runs.

### Add a secure-legacy lane

Keep semantic repair, migration equivalence and security correctness as separate lanes. The secure-legacy lane should test whether a proposed repair introduces injection, unsafe deserialization, authorization bypass, dependency confusion, secret leakage, insecure temp-file use, unchecked integer boundaries, or supply-chain/cache tampering. Every issue should have a severity, exploitable precondition, reproducible proof, and false-positive adjudication path. Report functional pass rate, security-correctness rate, critical/high escape rate, and security regressions among functionally passing patches separately.

Two 2026 evaluations strengthen the case for **joint functional-and-security scoring**. DualGauge-Bench evaluates 307 specification-only coding tasks with paired functional and security tests; its paper reports that the strongest evaluated model remained below 15% joint success in every language it tested (Python, C++ and JavaScript). The Last Secure Code Benchmark (preprint dated October 10, 2026) includes 450 tasks derived from 150 real vulnerabilities across seven languages and 24 CWE classes; it reports 73.6–91.3% functional success across tested agents while the strongest joint functional-and-secure result was 25.1%. These are different datasets/harnesses and early benchmark reports, not a direct VLEB comparison. They suggest the secure-legacy pilot should count a task as jointly successful only when both the behavioral contract and security tests pass. Sources: [DualGauge, EMNLP 2026 Findings](https://aclanthology.org/2026.findings-emnlp.1335/), [Last Secure Code Benchmark](https://arxiv.org/abs/2610.14766).

For financial and mainframe migration tasks, add state-transition traces as first-class observables: commit/rollback, restart/replay, duplicate delivery, batch return codes, control totals, field encodings and ordering. Byte-identical output is appropriate when the contract requires it; otherwise use independently specified business-state invariants and exact decimal semantics. Never use approximate floating-point agreement to excuse a mismatch in financial minor units.

### Proposed target additions (not achieved)

- Keep the Legacy-Bench target at **65%+ avg@3** under a matched full-benchmark protocol.
- Add a cost–quality Pareto target: **at least 60% avg@3 at no more than $0.25/task** on the full authorized Legacy-Bench under matched budgets. This is a proposed target relative to the October 22 leader listing of 62% at $0.49/task, not an achieved result. Source: [Legacy-Bench index](https://fireworks.ai/specialized-intelligence-index/benchmarks/code-generation/legacy-bench).
- Require **100% of scored tasks** to pass two-sided oracle admission and independent review.
- Maintain **>=90% mutation kill rate** for reviewed non-equivalent compilable mutants, reported by mutation family.
- Track the gap between ordinary resolution and **Assurance-Adjusted Semantic Success Rate (AASSR)**; the goal is to reduce that gap without sacrificing raw resolution.
- On the secure-legacy lane, pre-register a maximum critical/high vulnerability escape rate before the confirmatory run; do not choose the threshold after seeing results.
- Report cost per independently verified fix and p50/p90 duration, alongside abstention and unresolved rates.

## Measurement ledger and claim discipline

The current evidence/target split is recorded in [`tools/vleb/scorecard.json`](../../tools/vleb/scorecard.json) and guarded by `tools/vleb/test_scorecard.py`. External model scores, proposed targets and Symthaea measurements are separate objects. The current 4/4 source-mutant result is recorded only as a synthetic ledger smoke test; Legacy-Bench score, AASSR, security correctness and cost per verified fix remain `not_run` or `not_measured`.

This distinction should be preserved in papers and progress reports: a passing implementation test, an oracle-smoke mutation result, a model benchmark score and a signed provenance result are four different evidence claims.

## Research questions

1. Can Symthaea improve verified repair success over a matched baseline agent on previously unseen legacy tasks?
2. Can Symthaea detect when the specification, build environment, or available evidence is insufficient to certify a patch?
3. Does its proposed verifier actually reject subtle incorrect implementations, not just syntax errors and empty patches?
4. Can a candidate migration preserve observable behavior, numeric semantics, data layout, transaction effects, and recovery behavior?
5. Can improvements lower total cost per independently verified outcome without reducing correctness?
6. Which gains come from language/toolchain adapters, Symthaea planning/memory, stronger oracles, and additional inference budget? Measure these separately.

## Task model and lanes

A task is a locked engineering episode with a repository snapshot, task instructions, a declared language/dialect, build/runtime image, budgets, an oracle contract, and evidence requirements. Task families are distinct strata; do not pool them into a single headline without publishing per-family results.

| Lane | Purpose | Representative work | Required oracle |
|---|---|---|---|
| A. Read and diagnose | Measure understanding without patching | Explain data flow, identify call sites, locate numeric conversion risk, enumerate dependencies | Expert-checked answer key plus exact source/line references or graph assertions |
| B. Legacy repair | Repair a defect in an existing program | COBOL batch output defect, Java 7 behavior bug, C89 parser issue, HLASM branch/register bug | Hidden behavior tests, regression controls, and mutation suite |
| C. Specification implementation | Implement a bounded behavior in an existing codebase | Add a new record rule, parser branch, or business rule | Hidden tests plus properties and independent reviewer audit |
| D. Cross-language migration | Port one bounded component | COBOL calculation to Rust/Java; Fortran numerical routine to C++; assembly helper to a higher-level implementation | Differential reference execution, properties, numeric tolerances where justified, and output/state invariant checks |
| E. Data / transaction integrity | Preserve state-changing behavior | Batch posting, settlement reconciliation, duplicate handling, rollback, restart and replay | Independent state model, control totals, transaction assertions, failure-injection and idempotency checks |
| F. Test and oracle construction | See whether agents can verify code, not only write it | Build property tests, edge cases, discriminating tests, or mutation suites from a spec | Reference implementation, validated mutants, test-validity and mutation-score reports |
| G. Runtime / operational repair | Keep build and job orchestration correct | JCL dependency change, failure return code, restart condition, deployment/build config | Pinned environment, execution evidence, expected side effects and recovery behavior |

### Minimum pilot scope

Start with **60 tasks** as an internal feasibility pilot, 10 per lane across lanes B–G. Lane A can be a separate diagnostic set because it has a different response format. This is a design target, not an assertion that these tasks already exist.

Ensure the six implementation/operational lanes have a meaningful mix of COBOL batch and fixed-record processing; SQL or file-backed state behavior; JCL/job-control or equivalent workflow behavior; HLASM or another explicitly named ISA/dialect; Java 7/C89/Fortran/other languages from the public legacy task mix; and one or more cross-language migrations.

Do not fabricate six-lane parity by padding tasks with trivial syntax edits. Each task should require observable engineering reasoning and carry an independently reviewed difficulty/skill label. Publish per-language and per-task-family results even when the aggregate is useful.

The public Legacy-Bench sample tasks may be used as open development smoke tests, clearly labelled **public / exposure-expected**. For external comparison, obtain permitted access to the full evaluation set and follow its official submission conditions. Do not report a score on ten public samples as a full Legacy-Bench result.

## Task admission and benchmark-validity gate

No task enters a headline score unless all admission checks pass before the agent rollout.

### Eligibility and environment

- Pin task ID, source revision, candidate row, container image digest/immutable image ID, toolchain/compiler version, operating system and relevant dependencies.
- Require a pristine checkout and record the repository HEAD and worktree state before agent access.
- Disable container network access during task execution unless a narrowly scoped exception is preregistered. Disable agent browser/search/retrieval separately; container network isolation alone does not prove the model endpoint or all tools are offline.
- Sanitize image, refs, reflogs, hooks, caches and other solution-bearing state. Inspect retained ignored files and dependency caches as a possible leakage surface.
- Preserve raw commands, logs, output hashes, exit statuses and verifier inputs. Runtime-reported evidence is telemetry—not cryptographic hardware attestation.
- Make the task solvable with the declared tools and constraints; missing proprietary toolchains must not be confused with agent incapability.

### Two-sided task oracle

For every repair task:

1. The empty/baseline patch must fail at least one explicit required fail-to-pass test.
2. The reference patch must pass all declared required tests in a pristine environment.
3. Baseline-passing regression tests must remain passing after the reference patch.
4. Repeated reference runs must be deterministic, or variance/flakiness must be modelled and disclosed.
5. Test/patch order must be checked for hidden state leakage.
6. A reviewer who did not author the task must assess whether the test checks required behavior rather than a specific implementation detail.
7. The task's eligibility decision and all oracle hashes must be locked before the agent run.

For migrations, a "reference patch passes" gate is necessary but insufficient. Also lock a **behavioral contract**: input domains, output structure/encoding, arithmetic rules, persistent-state effects, ordering constraints, restart rules and explicitly allowed behavior changes.

### Adversarial oracle audit

Create mutants that are semantically wrong but likely to compile. Examples include:

- change a decimal rounding rule or truncate a monetary field;
- invert a branch only at a boundary;
- drop duplicate detection or process a record twice on restart;
- omit a record, alter its byte length or change field alignment;
- change EBCDIC/ASCII conversion assumptions or ordering/collation;
- move a commit boundary or suppress an error/return code;
- ignore one file/table/child record;
- change register/condition-code behavior in assembly;
- weaken a regression assertion while leaving the happy path passing;
- return success without performing the required side effect.

Mutants must be executed and independently classified as non-equivalent or equivalent/uncertain. A mutant that fails to compile is not evidence that the behavioral oracle detects semantic errors. Equivalent mutants must not be counted as killed; unresolved mutant runs must remain unresolved. Report results per mutation class, language, and test family.

## Mainframe and financial-behavior protocol

The migration lane should capture system behavior rather than trust a line-by-line translation.

### Required input/output semantics

For each mainframe task, record where relevant:

- source and compiler dialect/version; fixed/free format; compiler flags;
- copybooks and replacement rules; linkage and called modules;
- record layouts, field widths, signedness, COMP-3/packed decimal and zoned-decimal interpretation;
- code pages, EBCDIC conversion, byte ordering, sorting/collation rules and whitespace/record terminators;
- numeric precision, intermediate precision, rounding mode, overflow/truncation behavior and date conventions;
- JCL steps, datasets, return-code conditions, restart/re-run behavior, job dependencies and spool/output contracts;
- embedded Db2 SQL, SQLCA/error handling, isolation/locking, commit/rollback, and observable transaction effects;
- CICS/IMS/PL/I/HLASM/external routines when present.

Not every task uses every subsystem. Each task must explicitly list applicable concerns and the checks that cover them.

### Differential execution

For each migration task, run baseline and candidate implementations against the same authorized, de-identified inputs. Compare, as relevant:

- exact output bytes and record lengths;
- parsed business fields and financial totals;
- database state before/after, committed operations and rollback behavior;
- return codes and declared external side effects;
- ordering, duplicate treatment, idempotency and restart recovery;
- numeric values under an explicit exact-decimal or tolerance contract.

Default to exact comparison for financial amounts and fixed-format records unless a domain owner specifies a justified tolerance. Floating-point numerical workloads need a predeclared error envelope tied to the algorithm and domain, not a tolerance invented after observing candidate errors.

Use property-based and boundary-value tests in addition to saved fixtures. Where feasible, include input minimization so a mismatch is reproducible as a small counterexample. Record the first semantic divergence and all subsequent state/output differences.

GnuCOBOL may be a practical development toolchain for an open pilot, but successful execution under GnuCOBOL is not proof of equivalence to a declared IBM Enterprise COBOL and z/OS environment. For every trial, label the exact runtime it measured. Never test unreviewed changes against live banking systems or customer data.

## Evidence contract

Each task's evaluator receipt should carry:

- protocol/version ID and preregistered task-manifest digest;
- task ID, source revision, candidate-row hash and per-task assignment;
- source/patch/test-oracle hashes, exact patch-application result and checkout HEAD;
- toolchain/container immutable identity, platform, network mode and sanitation result;
- every command, exit code, timeout, raw-log hash and test ID/status;
- baseline, gold/reference, adversarial-control and candidate evidence;
- differential-comparison summary and unresolved mismatch details;
- mutation inventory, validity review, equivalent/uncertain classifications;
- agent model/version/configuration, tool permissions, budget and attempt count;
- outcome (pass, fail, unresolved, or abstained) and reason;
- total elapsed time, model/tool/compute cost, retries and human intervention.

The evaluator must independently recompute aggregates from raw evidence. It must reject evidence whose hashes, IDs, required test inventories, immutable image IDs, checkout identities or exit statuses disagree. Do not trust a task-supplied summary JSON as the score oracle.

The machine-readable receipt contract is `tools/vleb/evidence-receipt.schema.json`. The pre-registered planned negative controls are catalogued in `tools/vleb/mutation-catalog.json`; catalog entries are hypotheses until their mutants are actually built and evaluated. It binds the task and protocol digests, patch identity, model/scaffold/tool policy, environment and checkout identity, per-command log hashes and exits, required test inventory, mutation results, differential checks, verifier build identity, outcome and cost. The `trust_boundary` field distinguishes runtime-reported telemetry from independent inspection and external attestation; schema validation alone never certifies a pass. For a pass receipt, the independent auditor must recompute required-test coverage, check environment identity and confirm that no unresolved reason remains.

One attempt per task is the primary pass@1 condition. Retries, repair loops, multi-sample selection, network-enabled runs and hidden-test-aware runs are separate conditions, explicitly preregistered and reported.

### Four outcomes, not one green check

- **Pass:** eligible task, resolved execution, required behavior satisfied, no prohibited regressions, required evidence valid.
- **Fail:** eligible task with valid execution showing required behavior was not satisfied or a regression occurred.
- **Unresolved:** toolchain/evidence failure, missing test result, unexplained exit, invalid environment, or verifier disagreement.
- **Abstained:** the agent declines to certify or to make a change because it reports insufficient context, capability or evidence. This is not a task pass; report it separately to measure calibration and safe behavior.

Unresolved tasks remain in the assigned denominator. Publish completion rate and unresolved rate separately. Never silently drop a difficult task or convert a tool failure into an agent fail/pass without retaining the distinction.

## Metrics and scorecard

No single weighted aggregate should conceal safety-critical failures. Publish a scorecard with raw denominators and per-lane results.

### Assurance-Adjusted Semantic Success Rate (AASSR)

Use a transparent denominator rather than a composite weighted score:

`AASSR = independently verified passes / all tasks assigned to the locked confirmatory cohort`.

A task is an independently verified pass only when the semantic behavior contract is satisfied, every required test passes, every declared observable is covered by passing differential evidence, the mutation campaign meets its pre-registered task-level gate, evidence hashes and environment identities validate, and the trusted run/receipt audit succeeds. A model's unsupported claim of success does not count.

Keep these companion metrics separate: (1) raw candidate-resolution rate before assurance checks; (2) AASSR; (3) raw-minus-AASSR assurance gap; (4) unresolved, failed, and abstained rates; (5) false acceptance on negative controls; (6) cost per verified pass. Do not re-weight them into a single number. Unresolved tasks and abstentions remain non-passes in the AASSR denominator and are reported separately with reasons. Eligibility/exclusion decisions must be frozen before arm assignment and before scores are examined; post-hoc exclusions are not allowed except documented data corruption adjudicated blind to arm.

For a model comparison, report paired task-level differences and confidence intervals stratified by task family/language/repository; use repository-clustered bootstrap or a preregistered hierarchical model when tasks share codebase lineage. Repeat trials to estimate nondeterminism, but keep primary one-attempt pass@1, repeated-trial reliability, and avg@3 as distinct metrics.

| Metric | Definition / rationale | Proposed initial gate |
|---|---|---|
| Verified task resolution | Pass / all assigned eligible tasks, with unresolved shown separately; headline pass@1 only if protocol conditions are met | Establish matched baseline first; no target claim before baseline exists |
| Task-oracle validity | Fraction of tasks with valid environment, baseline/empty-patch fail control, reference pass control and reviewed behavior contract | 100% of tasks admitted to headline evaluation |
| False acceptance on negative controls | Incorrect/mutated candidates incorrectly certified as passing | 0 observed false accepts in the locked pilot adversarial set; publish binomial confidence bounds and do not infer zero population risk |
| Semantic mutant kill rate | Non-equivalent, compilable mutants rejected by the independent verifier divided by resolved non-equivalent compilable mutants; equivalence/unresolved separately reported | Initial internal target >=90% for reviewed non-equivalent compilable mutants, separately per mutation family |
| Migration equivalence | Required output/state invariants preserved on the locked validation corpus | 100% on required fixtures and invariants for a task to pass; no silent exceptions |
| Regression rate | Previously passing required behavior broken by candidate patch | Zero required P2P regressions per passing task |
| Evidence completeness | Required artifacts present and consistent with independently recomputed receipts | 100% for a resolved pass |
| Abstention / calibration | Coverage versus error risk; Brier score or expected calibration error where confidence is meaningful | Establish baseline/calibration curve before selecting a target |
| Unresolved rate | Unresolved assigned tasks / all assigned tasks, with reasons split by agent, harness, image and verifier | Proposed pause-and-investigate if >2% or arm difference >1 percentage point |
| Cost per verified outcome | Total inference/tool/runtime cost divided by independently verified passes; report median, tails and failures | Compare matched baseline; proposed alternative target: >=20% lower cost per verified fix with correctness non-inferiority margin fixed before trial |
| End-to-end time | Wall clock including environment, tools, retries, verification and human intervention | Report p50/p90 and budget-capped results |
| Generalization | Performance on private repositories/task families held out from prompt examples and tuning | Use a new locked cohort; no tuning on its test results |

These are proposed internal gates, not industry standards or achieved measurements. A zero-error pilot only bounds observed risk; report its confidence interval and expand the negative-control set before making strong reliability claims.

## Experimental design: isolate Symthaea's contribution

Do not compare different models, tools and budgets and attribute the entire gap to Symthaea. Run a matched, randomized experiment with the same base model, repository snapshot, environment, tool permissions, maximum turns/tokens, timeout and inference settings.

Required arms:

- **A. Baseline agent:** fixed reference scaffold and no Symthaea-specific planning/memory;
- **B. Language adapter:** A plus pinned language-aware parsing, build/test diagnostics and edit interface;
- **C. Symthaea assistance:** B plus the selected Symthaea planning, HDC retrieval/memory and repair loop;
- **D. Verification assistance:** the same agent condition, scored by the full independent oracle/evidence protocol (and, in a separate verifier-quality experiment, compared with the normal grading protocol).

Arm D is principally a measurement comparison; it must not be presented as a treatment effect on code generation if it changes only the scorer. If the verifier changes the feedback available to the agent, that is a separate preregistered treatment.

Stratify by repository, language, task family and estimated difficulty; randomize task order; freeze manifests before trials; cluster uncertainty by repository/task family rather than pretending every task is statistically independent. Use an untouched private holdout for the confirmatory comparison.

### Power and sample size

Start with the 60-task feasibility pilot to estimate rates, clustering, variance, harness cost and failure modes. Do not assert that 60 tasks are enough to establish a small improvement. Use those pilot estimates to calculate sample size and power for a confirmatory held-out study. A 250–500 task private cohort is a reasonable planning range, not a substitute for a formal power analysis. For reliability/variance, use repeated runs on a stratified subset but keep them separate from one-attempt pass@1.

Pre-register a superiority hypothesis such as **+5 percentage points in verified resolution** and require the repository-clustered 95% confidence interval for the treatment effect to exclude zero. The 5-point target is a desired effect threshold, not an assumed or already observed gain. Alternatively, test a preregistered cost-efficiency hypothesis with a 1.5-point correctness non-inferiority margin; define the margin and sample size before observing results.

## Comparison to industry and claims policy

- Run the open Legacy-Bench samples for toolchain/dev smoke tests, but label them public and never use their result to imply full-benchmark ranking.
- Seek authorized full Legacy-Bench evaluation if an apples-to-apples comparison is a priority; pin the exact version, agent/model, budget and harness.
- Use SWE-bench Multilingual as a broad repair comparison, not the mainframe migration metric.
- Use a separately curated VLEB private cohort for migration equivalence and verifier integrity. Publish task construction and verifier code when safe, while keeping held-out task inputs/solutions unavailable until results are locked.
- For a direct Legacy-Bench comparison, obtain authorized access to the complete benchmark and match the published avg@3 protocol. The October 22, 2026 listing's 62.0% avg@3 is the reference point captured by this revision; a proposed stretch target is **at least 65% avg@3**, or a preregistered cost-efficiency result that remains within a fixed correctness non-inferiority margin. Re-check the reference leaderboard before each experiment because it can change.
- Publish per-task-family results, confidence intervals, unresolved rates, costs, tool versions, and all protocol deviations.
- Label a claim "better than SOTA" only after comparing the same benchmark version, same eligible tasks, same attempt protocol, equivalent resource budget and verified scoring rules. Otherwise report the precise measured improvement without a ranking claim.

## Execution plan

### Phase 0 — now, no runner required

1. Freeze this spec and the evaluation vocabulary.
2. Build the task registry schema, support manifest, evidence fields and task-admission checklist.
3. Download/inspect only licensed public Legacy-Bench sample tasks for static inventory and environment review; record the exact repository commit and per-task requirements, not a score.
4. Draft task specifications, test contracts, mutation families and source/runtime requirements before implementing agents.
5. Build a deterministic synthetic financial-record fixture generator using integer minor units and explicit field layouts, plus a differential-comparison schema. Use synthetic/de-identified data only.

### Phase 1 — harness conformance

1. Create baseline-empty-gold task controls; prove invalid task admission is rejected.
2. Add mutations for numeric rounding/truncation, duplicate processing, record widths, ordering, return codes and restart behavior.
3. Verify mutated solutions cannot reach a pass without the independent verifier reporting why.
4. Test hash tampering, missing logs, mismatched image IDs, missing predictions and unexplained exits.
5. Keep all CI/static evidence separate from executed compiler/runtime results.

### Phase 2 — 60-task pilot

1. Curate six task families with independent review.
2. Execute baselines and reference solutions on pinned local/open toolchains.
3. Run the A/B/C ablation and a verifier-quality experiment.
4. Publish all task-level outcomes and confidence intervals, including null/negative results.

### Phase 3 — confirmatory evaluation

1. Select 250–500 held-out tasks from independent repositories and task families.
2. Lock the dataset, protocol, agent settings, and analysis before runs.
3. Compare against authorized industry baselines where the protocols permit.
4. Repeat on a stratified subset for reliability/variance; do not pool retries into pass@1.
5. Release the evaluation artifacts, reproducible commands, task inclusion/exclusion record, and limitations.

## Threats to validity and guardrails

- **Contamination:** public samples are for iteration only; hold out repositories/task families; record exposure and freeze tasks before evaluation.
- **Reward hacking:** monitor for grader/test access, harness modifications, hardcoded fixtures, suspicious cache use, output-only wrappers, skipped work or forged evidence; adjudicate traces.
- **Weak oracles:** require gold/empty controls plus nontrivial semantic mutations and human review.
- **Environment mismatch:** distinguish missing/incompatible toolchain from model failure, and make target runtime compatibility explicit.
- **False confidence:** include abstention, unresolved outcomes and calibration, and inspect the large class of failures in which the agent thinks it succeeded.
- **Migration scope:** report only the system boundary actually tested. One converted module is not evidence that a full bank application is migrated safely.
- **Safety and privacy:** synthetic or authorized de-identified data only; no unreviewed agents on live banking infrastructure; human owners approve release and cutover.

## Decision

Proceed with the evidence/task schema and oracle-validation harness before expanding language coverage. Start the legacy lane with **read-only COBOL/JCL/copybook analysis plus a small synthetic financial-record differential harness**. This creates concrete, locally verifiable work that does not depend on an external runner or access to a production mainframe. It also tests whether the new language adapters and Symthaea reasoning improve verified outcomes instead of only improving a leaderboard score.
