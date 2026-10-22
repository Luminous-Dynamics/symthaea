# Coding-Agent Benchmark Comparison and Scale-Up Roadmap (2026-10-22)

## Executive assessment

Symthaea's GCS ablation is an internal mechanism study, not a frontier coding-agent benchmark result. The current pilot has eight short Rust function tasks, a paired GCS-advisory on/off intervention, reference/empty-patch controls and ten semantic mutants. It has not yet been compiled or executed in the authoring environment, and its fixtures are public. There is no basis to claim SOTA from the current artifact.

The path to a credible comparison is: validate the measurement harness; expand task realism; hold out tests and tasks; compare identical model/backend/budget conditions; report paired effect size, confidence intervals, regressions and cost per verified fix. The companion [Rust repository-repair protocol](RUST_REPOSITORY_REPAIR_BENCHMARK_PROTOCOL_2026-10-22.md) defines the proposed task, scoring, provenance and release gates.

## Industry comparison: what the published numbers measure

| Evaluation | Recent public result / protocol | Relevance and limitation for Symthaea |
|---|---|---|
| Terminal-Bench 4.0, official board | The roadmap's checked September 2026 snapshot lists Claude Opus 5.5 + Claude Code at 64.8% ± 3.1, Sonnet 5.5 + Claude Code at 61.8% ± 2.9, and GPT-6 Astra + Codex at 58.2% ± 2.8. Source: [Terminal-Bench](https://www.tbench.ai/?efforts=all&version=4.0). | Useful end-to-end terminal-work context, but task distribution, harness, timeout, tools and model budget differ from this eight-function pilot. Do not compare the percentages directly. |
| Terminal-Bench 4.0, alternative harness | Vals AI's October 19, 2026 snapshot reports Claude Opus 5.5 + Mini-SWE-agent at 65.15%, Sonnet 5.5 + Mini-SWE-agent at 64.14%, and GPT-6 Astra + Mini-SWE-agent at 59.60%. Source: [Vals AI](https://www.vals-ai.com/benchmarks/terminal-bench-4). | The differing values illustrate that a result belongs to a model-plus-harness-plus-protocol, not a model in isolation. |
| SWE-bench-Live MultiLang | The public board lists a 71.0% result on 300 valid instances dated October 8, 2026; the MultiLang project supports Rust. Sources: [leaderboard](https://swe-bench-live.github.io/) and [project](https://github.com/microsoft/SWE-bench-Live). The observed project-code revision on 2026-10-22 is `62d02936a0726fbf4597b05bd07b09fb99eddae6`. | Best available Rust-native external starting point found in this pass. The 71.0% headline is **not a Rust-only score**. The project-code revision does not pin the task data. Select the official Rust slice, pin the immutable dataset revision and task-ID digest, and grade under a documented equivalent harness before reporting any Symthaea comparison. |
| SWE-Bench Pro / versions and splits | Public descriptions have changed across dataset snapshots. The public dataset card has described a 642-task, 11-repository mix in Go, Python, JavaScript and TypeScript. Separately, OpenAI's July 8, 2026 audit of a 731-task public split reported 27.4% flagged as broken by an automated pipeline and 34.1% labelled broken by human annotation, and explicitly retracted its earlier recommendation to adopt SWE-Bench Pro. Source: [OpenAI evaluation audit](https://openai.com/index/separating-signal-from-noise-coding-evaluations/). | Do not treat the benchmark name or a newer-looking version number as a quality guarantee, and do not project the audit rates onto an unexamined later snapshot. Pin the exact split/version; conduct independent task-quality review. Rust is not in the cited 642-task language mix. |
| Benchmark maintenance | [Terminal-Bench 4.0 notes](https://www.tbench.ai/news/terminal-bench-4-0) document resource calibration, task fixes and removal of saturated tasks; the published review of the exact [Terminal-Bench 4.0.0 set](https://epoch.ai/benchmarks/terminal-bench/4/review) reported scoring flaws in 30 of 66 reviewed tasks. These reports concern specific revisions. | Task fixes, verifier audits, resource changes and release versioning are part of evaluation science. A leaderboard score without its exact protocol revision is incomplete evidence. |

## Paradigm changes to prioritize

### 1. Tests become an experimentally validated oracle

A reference implementation passing and an empty patch failing are necessary, not sufficient. The current pilot requires each curated mutant to compile and fail at least one assertion under a complete, non-simulated test run. CodeExecutor may record a nonzero test-binary exit in `runtime_error` when an ordinary assertion fails; exact full test counts therefore indicate a completed behavioral failure, while missing counts/simulated runs remain unresolved. For a larger task bank, record mutation categories and report killed, surviving, invalid/compile-failing and unresolved mutants separately. A mutant that cannot compile is not evidence that tests discriminate behavior.

The fixed eight-task pilot declares expected test counts explicitly and binds them into each task digest (schema v2), removing producer/auditor text scanning. Schema v3 additionally records requested backend/model identity and observed per-run generation tiers, corrects the local model label to the actual `gemma4:e2b` constructor default, and fails closed on cloud mode without a non-empty API key. These tiers are app-reported telemetry, not cryptographic provider attestation. The real preflight and each run must account for exactly the declared count. Before arbitrary-repository ingestion, generalize this into a machine-readable harness contract with independent test-discovery evidence.

### 2. Unknown is not failure—and not a tie

Report behavioral failures separately from incomplete compiler/test execution. Paired GCS effect statistics should include only pairs with complete oracles in both arms; publish unresolved-pair count and rate next to the effect. Never remove unresolved outcomes silently. If unresolved rate differs by arm or task class, treat it as a validity issue. Also publish conservative sensitivity bounds in which unresolved outcomes are assigned against each arm in turn.

### 3. Test intervention effects, not just the final agent

Use the same model revision, system prompt, token/time budget, tools, compiler, sandbox, task, hidden tests and stopping rules in each arm; change only GCS advisories. Randomize treatment order per task/repeat rather than relying only on a fixed alternating pattern. Record requested model identity and per-run observed generation tiers. Report both within-task paired effects and the full agent+model+harness configuration. A real comparison needs an external baseline on the same task sample—not imported leaderboard percentages.

### 4. Make held-out tasks harder than memorized function synthesis

Move from toy functions to repository-level fixes: multi-file changes, regression-preserving refactors, compiler/dependency failures, concurrency, security-sensitive parsing, performance regressions and changes where tests alone do not reveal intended behavior. Freeze a public development set and maintain a private/held-out evaluation set. Grade patches in pristine checkouts with fail-to-pass plus pass-to-pass tests, and keep hidden tests, gold patches and evaluator metadata out of the agent runtime.

The next external validation lane should use a pinned Rust-only slice of [SWE-bench-Live MultiLang](https://github.com/microsoft/SWE-bench-Live). The independent Symthaea lane should add reviewed tasks across multiple repositories and task families. Neither lane substitutes for the other: an external suite provides comparability; a private set permits controlled mechanism studies and contamination resistance.

### 5. Measure the engineering frontier, not one score

For each configuration report:
- pass@1, per-task paired wins/losses/ties, repeated-run success frequency, and the probability of succeeding at least once after (k) tries as separate quantities;
- valid-oracle, mutation-adequacy and unresolved-execution rates;
- pass-to-pass regression rate and security-regression rate;
- cost per verified fix, wall time, p50/p95 latency, tokens, retries and measured/estimated compute or energy—with the measurement method labelled;
- results by task family and repository, plus confidence intervals clustered at the independent unit.

Use a correctness/cost Pareto frontier. A system can be more useful through cheaper verified fixes without claiming a higher absolute benchmark score. Do not combine unrelated benchmark scores into one headline number.

## Proposed acceptance gates (targets, not measured results)

### Gate A — make the pilot trustworthy
- Exact-head `cargo fmt --check`, focused tests and example compilation pass in a pinned Rust/Nix environment.
- Reference controls pass, the empty patch is rejected by a real Rust compiler diagnostic, and all ten semantic mutants are behaviorally killed after successful compilation.
- The receipt auditor accepts a valid producer receipt and rejects fixture edits, missing/duplicated runs, incomplete oracle data, tampered summaries, malformed digests and source/manifest pin mismatches.
- No improvement claim until both arms execute with a named model/backend; report unresolved pairs rather than counting them as failures or ties.

### Gate B — demonstrate a mechanism effect
- Expand to 50–100 carefully reviewed Rust tasks with category and difficulty stratification.
- Start with at least three independent repeats per task/arm; use pilot variance and a predeclared precision/power analysis to set the confirmatory repetition count.
- Require high and balanced oracle coverage; publish the task-level reference/mutation-control outcomes.
- Report paired confidence intervals and a paired binary-outcome test. The eight-task pilot cannot justify a significance or field-wide claim.

### Gate C — credible frontier comparison
- Target 250–500 held-out Rust repository tasks across many repositories and task families, with hidden tests and pristine grading. The number of tasks alone is not a sufficiency guarantee; set the final sample size by power/precision analysis and ensure repository clusters are numerous enough for the proposed interval.
- Pre-register model ID/revision, agent/harness commit, token/time/cost budget, task-manifest digest, exclusions, primary endpoint and analysis before the first scored run.
- Candidate performance target: at least +5 absolute percentage points in held-out pass@1 versus the same-model, same-budget baseline, with a repository-clustered 95% confidence interval excluding zero; alternatively, reduce cost per verified fix by 20% while remaining within a predeclared 1.5-point correctness non-inferiority margin.
- Treat these as decision thresholds, not guaranteed or currently achieved results. Require independent replication before claiming superiority.

## Immediate sequence

1. Validate PR #8402 at its exact head in a pinned environment: formatting, focused producer/auditor tests and compilation. The authoring environment cannot run these checks, so no passing execution is claimed here.
2. Run the reference, empty-patch and ten mutation controls; repair weak fixtures instead of weakening the gate.
3. Capture a real schema-v3 receipt on every available backend; independently audit it against source and manifest digests pinned outside the receipt.
4. Create a locked preregistration using [the JSON template](RUST_REPOSITORY_REPAIR_PREREGISTRATION_TEMPLATE_2026-10-22.json) and validate it against [the schema](schemas/rust-repair-preregistration.schema.json). Do not launch the scored evaluation until the data revision, task IDs, model, harness, image digest and analysis are locked.
5. Run the exact official Rust slice with baseline and GCS arms, followed by the private multi-repository task bank. Publish negative/inconclusive results and unresolved denominators alongside positive results.

## Epistemic boundary

A digest proves byte identity relative to the bytes being hashed; it is not a signature or proof that execution occurred. The receipt auditor checks internal consistency, not the trustworthiness of the machine running the compiler. A generation-tier label is application-reported telemetry. For an external claim, preserve runner logs and raw artifacts, run the auditor in an independently managed environment, keep the holdout private, and sign the final audit record with a separately controlled key. Do not interpret receipt consistency as independent execution attestation or as an SOTA result.
