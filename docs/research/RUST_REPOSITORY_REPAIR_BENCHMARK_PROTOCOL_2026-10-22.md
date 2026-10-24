# Rust Repository-Repair Benchmark Protocol (2026-10-22)

**Protocol status:** proposal / preregistration required before scored trials  
**Protocol ID:** `symthaea-rust-repair`  
**Version:** `0.1.0`

This protocol is an execution contract and research plan, not a report of measured performance. No repository-repair SOTA result is claimed. The current eight-task GCS pilot remains a small, public-fixture mechanism study.

## 1. Questions and falsifiable claims

Separate three claims that must not be conflated:

1. **Harness validity:** did the intended task, toolchain, model route and test oracle actually run, and can an independent auditor reproduce the scoring from preserved evidence?
2. **Mechanism effect:** with everything else fixed, do Geodesic Code Synthesis (GCS) advisories improve correct patch rate, reduce cost per verified fix, or improve another preregistered endpoint?
3. **Repository generalization:** does an observed effect transfer from short functions to held-out issues across repositories, task families and realistic build/test conditions?

A result at one level does not establish the next. Receipt consistency is not proof of execution; application-reported model-tier telemetry is not provider attestation.

## 2. Two complementary evaluation lanes

### Lane A — external Rust comparison

Use the official Rust slice of [SWE-bench-Live MultiLang](https://github.com/microsoft/SWE-bench-Live) as the first external starting point. Its project announcement dated October 1, 2026 identifies C/C++, C#, Java, TypeScript/JavaScript, Go and Rust support. The public leaderboard's 71.0% entry dated October 8 is over 300 valid instances and is **not** a Rust-only score. Do not compare Symthaea's internal pilot percentage to it.

As inspected on October 22, 2026, the upstream *code repository* was at commit `62d02936a0726fbf4597b05bd07b09fb99eddae6` (README update, September 25, 2026). Separately, the Hugging Face dataset repository is pinned here to snapshot `3638632e8153a10ca422c1022bed79023084b5c9`. Its `rust` split metadata reports 171 candidate rows; the file `data/rust-00000-of-00001.parquet` has published SHA-256 `02ac78e2c51a84eb174ac393bb07b77478f1f96cf260af18835a711dd8074ebc`. See the [pinned dataset metadata](https://huggingface.co/datasets/SWE-bench-Live/MultiLang/blob/3638632e8153a10ca422c1022bed79023084b5c9/README.md), [pinned Rust file metadata](https://huggingface.co/datasets/SWE-bench-Live/MultiLang/blob/3638632e8153a10ca422c1022bed79023084b5c9/data/rust-00000-of-00001.parquet), and [official evaluation guidance](https://github.com/microsoft/SWE-bench-Live/blob/62d02936a0726fbf4597b05bd07b09fb99eddae6/evaluation/README.md). Before scored execution, retrieve and verify the exact bytes; use the upstream procedure to run gold patches three times and determine the actually-valid subset. The raw row count is not the final denominator. Also freeze:
- immutable dataset revision or release identifier;
- exact split and Rust filter definition;
- ordered task IDs plus SHA-256 digest;
- per-task base-repository commit/container image where supplied;
- official evaluation harness commit and configuration.

If the upstream task data cannot be resolved to an immutable revision and reproducible task list, stop and report the blocker. Do not silently substitute the latest moving split or describe a partial sample as the official benchmark. Respect upstream license, redistribution and leaderboard-submission rules.

#### Blind rollout and contamination controls

For the strict-comparability lane, expose only the task's `problem_statement` and the prebuilt sandbox to the agent. Do not expose `patch`, `test_patch`, `hint`, `hints_text`, `all_hints_text`, `FAIL_TO_PASS`, `PASS_TO_PASS`, commit URLs, evaluator logs, or task metadata that is not part of the official agent interface. Disable web/search access and task-specific retrieval during the rollout; fetch dependencies before the run or from a recorded allowlist/cache. Preserve full input/output tool trajectories so protocol compliance can be independently reviewed.

Use one rollout per task for the primary pass@1 result; the official SWE-bench-Live protocol asks for a single rollout based on the problem statement in the task container. Do not rerun a failed task until one attempt passes and then report it as pass@1. Retry, pass@k, best@k, browser-enabled or retrieval-enabled variants may be valuable separate experiments, but must have distinct preregistrations, budgets and result labels and must never be pooled with strict pass@1.

### Lane B — Symthaea-held-out Rust repairs

Curate 50–100 reviewed development tasks, then a held-out confirmatory set targeting 250–500 tasks distributed over many repositories and task families. The final count must be supported by a preregistered power/precision analysis; task count cannot compensate for too few independent repository clusters.

Include multi-file fixes, regression-preserving refactors, dependency and compiler errors, concurrency, parser/security boundaries, performance regressions, stateful behavior and tasks requiring understanding of repository conventions. Do not let all tasks be synthetic one-function changes. Keep public development tasks distinct from private confirmatory tasks.

Every task must have a prompt whose requested behavior can reasonably be inferred, an immutable base commit, a reference patch that satisfies the stated behavior, hidden fail-to-pass tests and pass-to-pass regression tests. Record explicit exclusions and rationale before observing candidate system results.

## 3. Task-quality contract

A task is eligible only after all of these controls pass:

1. **Reference control:** the reference patch applies cleanly, builds and passes all required fail-to-pass and pass-to-pass tests.
2. **Base control:** the unpatched base exhibits the expected fail-to-pass failure; pre-existing pass-to-pass tests pass.
3. **Oracle-discrimination controls:** plausible semantic mutants compile and are rejected by behavioral assertions. A compiler error is not a behavioral kill. Categorize equivalent mutants and adjudicate them before calculating mutation adequacy.
4. **False-negative review:** a valid alternative implementation or independently written solution is not rejected merely for differing from the gold patch's implementation details.
5. **Prompt/test consistency:** reviewers inspect the prompt, expected behavior and hidden assertions separately from the gold patch. Flag underspecification, prompt/test contradictions, over-specific tests, low-coverage tests and tests that reward implementation-specific details.
6. **Reproducibility:** the task can be built and evaluated in the pinned clean image; flaky, unavailable-dependency or unresolvable cases are excluded before the test set is locked, with a public exclusion ledger that does not reveal hidden assertions.

Use at least two independent reviewers per task for initial curation and a third adjudicator for disagreements. Reviewers should decide from the task statement, interface/context and behavior before seeing the gold diff where practical. Re-run quality checks when a task, verifier, container or dependency changes.

This review burden is deliberate. OpenAI's July 8, 2026 audit of a public SWE-Bench Pro split found 27.4% flagged by its automated pipeline and 34.1% labelled broken in a human annotation campaign, and it withdrew its earlier recommendation to adopt the benchmark. The exact rates apply to the audited split, not every later revision; the method is the lesson: task quality must be measured, not inferred from popularity.


### 3.1. Audit the upstream transition labels rather than trusting them

The pinned upstream implementation at [`evaluation/validation.py`](https://github.com/microsoft/SWE-bench-Live/blob/62d02936a0726fbf4597b05bd07b09fb99eddae6/evaluation/validation.py) derives `FAIL_TO_PASS` as `post_pass - pre_pass` and `PASS_TO_PASS` as `pre_pass & post_pass`. That is not a sufficient task-validity contract:

- A test absent from the parsed baseline but passing after the gold patch is included in `post_pass - pre_pass`, despite never being observed to fail before the patch.
- A test that passed at baseline and fails after the gold patch is omitted from `pre_pass & post_pass`; if that derived list is carried forward, the regression test disappears from the benchmark's pass-to-pass expectations.

The Rust audit tool added alongside this protocol recalculates **explicit fail → pass** transitions only, preserves every baseline-passing test in the expected pass-to-pass set, requires identical test-ID sets, rejects skip/missing statuses, requires all gold-aggregated tests to pass, and checks the three per-task gold-run markers. It records hashes for the baseline log, three-run gold log, status JSON, prompt, gold patch and test patch without putting prompts, gold patches, test patches, or test names into the manifest.

This is an independent *eligibility filter*, not a patch to the upstream evaluator. It can reject a task or fail closed when evidence is missing; it cannot compensate for a compromised runner or prove the raw log truthful. Review the [task-audit design and commands](SWE_BENCH_LIVE_RUST_TASK_AUDIT_2026-10-23.md) before locking the valid ID list.

## 4. Paired experiment for GCS

**Primary causal contrast:** the same agent and model with GCS advisories enabled versus disabled.

Hold constant model/provider and revision, system/developer prompt, tool set, sampling settings, context/token/time budgets, network access, repository base commit, image digest, compiler, dependencies, test oracle and stopping rules. Randomize the order of on/off trials per task and repeat. Store the randomization seed before scoring.

For each pair:
- classify both outcomes as `pass`, `fail` or `unresolved`;
- define `pass` only when the patch applies, real compilation completes, all hidden fail-to-pass tests pass, all required pass-to-pass tests pass, and the evaluator reports no invalid/missing test execution;
- count only pairs with complete oracles in both arms for the paired treatment-effect estimate;
- preserve all unresolved runs and all assigned tasks in the published denominator. Never convert unresolved into a behavioral failure or a tie.

Report sensitivity bounds assigning unresolved outcomes against each arm in turn. If unresolved rates are materially different across arms or task families, the result is not decision-grade until the cause is understood. Provisional release gate: unresolved outcomes below 2% overall and an arm difference below 1 percentage point; these are proposed internal quality gates, not an industry standard.

The pinned RepoLaunch Linux runtime fails closed unless it sanitizes Git refs, stash/reflog history, hook configuration/files, pseudo-refs such as `FETCH_HEAD`/`ORIG_HEAD`, merge/rebase/bisect replay state, the rerere cache, and unreachable objects before handing a container to an agent or grader. It preserves only the checked-out branch ref and retains ignored dependency caches for reproducibility; those caches remain an explicitly documented residual leakage risk requiring image review.

The evaluator captures the actual repository `HEAD` and a clean-worktree probe before any test or solution patch is applied. Baseline, every gold control, and every scored run must report a clean worktree and a `HEAD` equal to the locked task `base_commit`. These fields are included in the signed-off task identity/digest record. This blocks common dirty-checkout and wrong-revision artifacts, but is not a substitute for image sanitization against ignored files, extra refs, hooks or caches.

The strict protocol is Linux-only. The pinned RepoLaunch overlay modifies its Linux runtime; each receipt therefore includes the runtime object's platform label, and the task/scored auditors require exactly `linux`. A Windows, Android, missing, or otherwise different runtime platform is unresolved and cannot enter headline pass@1. This platform label is runtime-reported telemetry rather than a hardware attestation, so the pinned runtime source and Docker evidence remain part of the audit package.

The pinned Linux RepoLaunch task containers use `network_mode="none"`; the inspected network mode is included in task-validation and scored-run evidence and is a required manifest gate. This does not replace disabling network-backed tools in the outer agent harness. Task-validation provenance and scored-agent-run provenance are separate evidence streams. The pinned upstream overlay now writes one sidecar for the baseline/gold controls and a different `evaluation_evidence.json` for each scored prediction patch. A valid task-manifest audit does not establish that the scored rollout ran correctly; the scored run needs its own patch, log, process-status and container-image checks. Observed generation tiers should be recorded, but labelled as application-reported. For any provider-backed trial, preserve the provider request/response identifiers and usage records where available. Requested model name plus observed tier is not proof of which model the provider served.

## 5. Primary and secondary outcomes

**Primary endpoint:** absolute paired pass@1 difference (GCS-on minus GCS-off) on eligible confirmatory task pairs, reported in percentage points. Pre-register whether the target population is task-weighted or repository-weighted; task-weighted is the default, with repository-weighted as a sensitivity analysis.

Secondary endpoints:
- repeated-run pass frequency and pass@k, reported separately;
- compile success, oracle-completion and unresolved rates;
- paired wins, losses and ties;
- pass-to-pass regression rate, including security-relevant regression categories;
- patch application rate and test execution completeness;
- cost per verified fix, input/output tokens, retries, wall-clock time and p50/p95 latency;
- energy/compute only when the measurement method and hardware attribution are recorded;
- results by task family, repository and task difficulty.

A task that is not attempted, times out, refuses or fails to produce a patch remains in the assigned-task denominator. Distinguish agent failure, infrastructure failure and invalid oracle in run statuses; apply exclusions only by preregistered rules and report sensitivity results including excluded/unresolved tasks.

## 6. Statistics and superiority gates

For GCS on/off comparisons, use paired binary-outcome analysis and confidence intervals for the paired absolute effect. Cluster resampling by repository is required for repository-level claims; within a repository, tasks are not assumed independent. Report the number of repositories and task families alongside the interval. Pre-register random seed, resample count and analysis code revision. Ten thousand bootstrap resamples may be used for reporting, but do not imply that resampling creates information absent from a small sample.

Target for a confirmatory superiority claim: at least +5 absolute percentage points in held-out pass@1 versus the same-model, same-budget baseline and a repository-clustered 95% confidence interval excluding zero. Alternative practical-success target: at least 20% lower cost per verified fix while remaining within a 1.5-percentage-point correctness non-inferiority margin. These are goals, not evidence of current performance; finalize sample size and non-inferiority rationale before data collection.

Also report negative and inconclusive results. Require independent replication before claiming state-of-the-art or general superiority. Cross-model results should be stratified by exact model and harness configuration; never attribute a model+harness result solely to the model or Symthaea.

The standalone Rust binary `rust-repair-ablation-audit` implements the paired analysis contract in [the ablation-audit specification](RUST_REPOSITORY_REPAIR_ABLATION_AUDIT_2026-10-24.md). It verifies each task outcome against a hashed `rust-repair-score-audit` report, checks preregistered configuration equality and hash-derived randomized order, keeps missing arms unresolved, reports full-assignment verified-pass rates plus resolved-pair effects and unresolved sensitivity bounds, and withholds repository-clustered 95% intervals below ten resolved repository clusters. Its superiority and cost/non-inferiority gates are implementation of the proposed targets above, not evidence that any target has been achieved.

## 7. Run receipt and provenance minimums

The locked preregistration and per-run receipt must distinguish preregistered identity from observed telemetry and include, at minimum:

- protocol version, benchmark/dataset revision, split, ordered task IDs and externally pinned task-manifest digest;
- exact Symthaea source commit, agent/harness revision, GCS configuration, prompt/configuration digest and randomization seed;
- requested backend/provider/model/revision separately from observed generation tiers, provider usage/request IDs where available, and an explicit indication when identity is only self-reported;
- OS/architecture, CPU/GPU, Rust toolchain, dependency lockfile, container image digest, network policy and resource limits;
- task/base commit, patch digest, build/test commands, discovered and expected test counts, raw exit statuses, fail-to-pass/pass-to-pass counts, stdout/stderr/log references, timing and usage/cost records;
- status for every assigned run, error taxonomy, auditor version, and a signature or separately controlled immutable artifact reference where available.

Hashing supports byte-integrity checks. A digest by itself does not authenticate its producer or prove a process ran. Preserve raw evidence and audit in a separately managed environment.

## 8. Versioning and change control

Treat prompt/task changes, dataset/split changes, tool/model/budget changes, and container/resource changes as a new evaluation release requiring new scored runs. Verifier-only changes require regrading archived patches when the environment supports it and must be called out. Pure metadata/documentation fixes may reuse results only after an explicit review. Do not pool scores from incompatible versions as one series.

The release artifact should contain the locked preregistration, task manifest digest, raw receipts, independent audit output, exclusions, analysis script, confidence intervals and a machine-readable result summary. Keep task contents private where necessary, while publishing enough version and validity metadata to make the claim auditable.

## 9. Industry benchmark snapshot and strategy (2026-10-24)

This is a dated comparability note, not a claim that Symthaea has been scored against these leaderboards.

| Reference | Public evidence observed | Correct interpretation for this project |
|---|---|---|
| [SWE-bench-Live](https://swe-bench-live.github.io/) and [official source repository](https://github.com/microsoft/SWE-bench-Live) | The official leaderboard supports a Rust tab. Its indexed mixed-language table reports 71.0% over 300 valid instances dated 2026-10-08; the current page render exposes an error while loading its table, so that row is not an auditable Rust-specific target here. The upstream repository states that the MultiLang dataset grew to 1,077 instances across 431 repositories and eight languages in its 2026-08 update. | **First external target:** resolve the current official Rust-filtered leaderboard rows and exact matching split before claiming a rank or percent-point gap. Submit trajectories according to the benchmark's protocol. Never compare Symthaea's Rust score to the 71% mixed-language row. |
| [SWE-Bench Pro V2 methodology and leaderboard](https://labs.scale.com/blog/swe-bench-pro-v2) | The 2026-09-22 refresh reduced the public set from 731 to 642 tasks after dropping 89 invalid instances; 69 contradictory prompts were corrected, 211 task environments gained dependency support, evaluation became network-locked, and patches are regraded on pristine images. The published public V2 leaderboard lists results near 99% for its strongest setups. The V2 task mix is not a Rust-specific comparison. | Copy the **evaluation mechanics**, not the headline number: two-sided oracle gating, blind agent phase, locked networking, trajectory review, fresh-image regrading and explicit residual limitations. Do not compare scores across benchmark/task/harness versions. |
| [Terminal-Bench 4.0 release](https://www.tbench.ai/news/terminal-bench-4-0) and [Epoch AI's version-specific review](https://epoch.ai/benchmarks/terminal-bench-4/review) | The release notes describe resource calibration and task fixes; Epoch AI's 2026-09-04 review judged version 4.0.0 flawed and reported scoring defects affecting 30 of 66 tasks (45.5%) in that reviewed version, including reward hacks, answer leakage and false negatives. | Treat evaluator security, oracle adequacy and task validity as measurable outcomes. The review concerns that exact version; it is not evidence that all versions or all benchmarks have the same defect rate. |

### What would count as doing better than the field

There are two independent axes. **Capability** means resolving genuinely valid unseen tasks at a better verified rate for a fixed model, harness, time/token budget and task split. **Evaluation integrity** means being harder to game, easier to reproduce, more transparent about unresolved tasks, and better at identifying false-positive/false-negative scoring. A system can improve one axis without improving the other. Never turn stronger auditing infrastructure into a claim of stronger coding capability.

1. **Comparable public result.** Resolve the current SWE-bench-Live Rust rows, pinned task/data revisions, denominator, protocol and required rollout trajectory policy. Run a one-attempt primary pass@1 submission with the exact upstream evaluator; preserve all inputs/outputs and publish excluded-task reasons. First establish the same-model baseline, then GCS-on and other mechanisms as separately labelled experiments.
2. **Held-out generalization.** Use a pre-registered 250–500-task private Rust cohort spanning many repositories and task families, with repository-cluster power/precision analysis, reference-patch and empty-patch controls, hidden tests, mutation checks and human adjudication. A large count dominated by one repository is not sufficient.
3. **Causal mechanism claim.** The first GCS study asks whether the same model/harness with GCS on beats its randomized off control. Lock model identity/revision, prompt, scaffold, budget, image digest, task order and scoring contract before exposure. Require the preregistered +5 percentage-point target, repository-clustered 95% interval above zero, conservative unresolved-sensitivity lower bound above zero, at least 10 resolved repository clusters and the unresolved-rate gates; alternatively require the 20% cost-per-verified-fix reduction and correctness non-inferiority gates documented above.
4. **Red-team the grader, not only the agent.** Add controlled negative cases for test-result forgery, grader-file tampering, cache/ignored-file contamination, network or code-host retrieval, patch-time verifier mutation, incorrect task/base commit, flaky tests, changed task manifest and corrupted receipt. Test both false passes and false failures. A compile error is not a semantic mutant kill; a pass from a tampered grader is an immediate integrity failure.
5. **Regrade independently.** Treat the agent workspace and final grading workspace as separate trust zones. Preserve the emitted patch by digest, then apply it in a fresh pinned task image with evaluator-side tests and ground truth not mounted in the agent environment. Compare the primary-run verdict with the fresh regrade. Any discrepancy becomes an explicit invalid/unresolved category and an investigated incident, never silently resolved in favour of the better score.
6. **Economics and reliability.** Report pass@1, repeated-run pass frequency, correctness regressions, unresolved/infrastructure failure rates, p50/p95 wall time, token usage, total cost and cost per independently verified fix. Report Pareto frontiers; do not collapse score, cost and safety into a single opaque weighted score. Run repeated trials on a stratified subset and disclose confidence intervals.
7. **Public release discipline.** Freeze a versioned preregistration and exact task IDs before the run; version the task set, harness and grader separately; archive immutable input/output trajectory bundles and auditor outputs; publish score and unresolved sensitivity results. Independent replication and matching conditions are required before an SOTA claim.

### Immediate blockers (as of 2026-10-24)

- The pinned SWE-bench-Live Rust dataset candidate has 171 reported candidate rows, but its exact bytes have not yet been downloaded and rehashed in this authoring session. The three-run gold-control validity pass and final eligible task-manifest digest have not been produced.
- No valid external Rust leaderboard score, private held-out result, GCS treatment effect or cost-per-fix comparison has been measured under this protocol.
- The new Rust tooling adds testable contracts. Until the fresh runner workflows complete on the current commits and their results are independently reviewed, queued/pending checks remain unverified. A passing CI job validates the tested code path, not the security of the runner itself.

## 10. Execution checklist

Before running a scored task, require a locked preregistration validated against `schemas/rust-repair-preregistration.schema.json`. Start with dataset provenance and task quality; then validate the environment; then run paired GCS trials; only after that run confirmatory held-out tasks. Do not wait on a queued CI run to do independent source review, dataset investigation or protocol development—but do not represent unexecuted tests as passing.

**Current status:** the candidate Hugging Face data snapshot is pinned to `3638632e8153a10ca422c1022bed79023084b5c9`; its Rust split reports 171 candidate rows and the Parquet metadata reports SHA-256 `02ac78e2c51a84eb174ac393bb07b77478f1f96cf260af18835a711dd8074ebc`. The actual file bytes have not been downloaded and independently hashed in this authoring session, the gold-patch three-run validity filter has not been executed, and the final valid task-ID/manifest digests are not yet frozen. No external or held-out score has been measured under this protocol.
