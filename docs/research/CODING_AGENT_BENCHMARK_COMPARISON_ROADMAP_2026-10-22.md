# Coding-Agent Benchmark Comparison and Scale-Up Roadmap (2026-10-22)

## Executive assessment

Symthaea's GCS ablation is an internal mechanism study, not a frontier coding-agent benchmark result. The current pilot has eight short Rust function tasks, a paired GCS-advisory on/off intervention, reference/empty-patch controls and ten semantic mutants. It has not yet been compiled or executed in the authoring environment, and its fixtures are public. There is no basis to claim SOTA from the current artifact.

The path to a credible comparison is: validate the measurement harness; expand task realism; hold out tests and tasks; compare identical model/backend/budget conditions; report paired effect size, confidence intervals, regressions and cost per verified fix.

## Industry comparison: what the published numbers measure

| Evaluation | Recent public result / protocol | Relevance and limitation for Symthaea |
|---|---|---|
| Terminal-Bench 4.0, official board | The board's September 2026 snapshot lists Claude Opus 5.5 + Claude Code at 64.8% ± 3.1, Sonnet 5.5 + Claude Code at 61.8% ± 2.9, and GPT-6 Astra + Codex at 58.2% ± 2.8. Source: [Terminal-Bench](https://www.tbench.ai/?efforts=all&version=4.0). | Best high-level comparator for end-to-end terminal work, but its task distribution, harness, timeout, tools and model budget are not equivalent to this eight-function pilot. |
| Terminal-Bench 4.0, alternative harness | Vals AI's October 19, 2026 snapshot reports Claude Opus 5.5 + Mini-SWE-agent at 65.15%, Sonnet 5.5 + Mini-SWE-agent at 64.14%, and GPT-6 Astra + Mini-SWE-agent at 59.60%. Source: [Vals AI](https://www.vals-ai.com/benchmarks/terminal-bench-4). | The differing numbers demonstrate why a score belongs to a model-plus-harness-plus-protocol, not a model in isolation. Do not compare these percentages to Symthaea's pilot. |
| SWE-bench-Live MultiLang | The current public board lists a 71.0% result on 300 valid instances dated October 8, 2026; the project supports Rust in its MultiLang task set. Source: [leaderboard](https://swe-bench-live.github.io/) and [project](https://github.com/microsoft/SWE-bench-Live). | Use its Rust filter / a pinned Rust-only snapshot for an industry-aligned baseline; the 71.0% headline is not a Rust-only score and must not be compared to this pilot. |
| SWE-Bench Pro V2 | The refreshed public set lists 642 repository tasks across 11 repositories (Go, Python, JavaScript and TypeScript); it uses hidden fail-to-pass and pass-to-pass tests and grades diffs in pristine containers. Source: [official dataset card](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro). | A better design target for repository-level Rust repair, but Rust is not currently represented in this public task mix. Build a Rust-native held-out set rather than claim direct leaderboard comparability. |
| Benchmark validity | The V2 description says 89 of the prior public tasks were removed as invalid, 69 instruction/test contradictions were corrected, and clean-image grading caught benchmark-hacking behaviors. Source: [SWE-Bench Pro V2 notes](https://model-eval-leaderboard.vercel.app/leaderboard/swe_bench_pro_public_v2?tab=full). A separate earlier audit estimated roughly 30% of SWE-Bench Pro tasks had issues in the audited version: [OpenAI evaluation audit](https://openai.com/index/separating-signal-from-noise-coding-evaluations/). | Task validity, hidden tests, pristine grading and anti-cheating controls are part of the benchmark itself—not optional polish. These audit figures refer to earlier versions and should not be projected to V2 unchanged. |

Terminal-Bench itself describes version 4 as a protocol change that calibrates time/CPU/memory, fixes tasks and removes saturated tasks. Therefore, even adjacent versions of a benchmark may not be directly comparable: [Terminal-Bench 4.0 release notes](https://www.tbench.ai/news/terminal-bench-4-0). A separate September 4, 2026 review of the exact Terminal-Bench 4.0.0 task set classified it as flawed, reporting scoring defects in 30 of 66 reviewed tasks—including verifier reward hacks, answer leakage and false negatives: [Epoch AI review](https://epoch.ai/benchmarks/terminal-bench-4/review). Treat that as an audit of that benchmark revision, not a claim about later revisions. The lesson for Symthaea is to maintain adversarial verifier tests and regrade checks as first-class benchmark releases.

## Paradigm changes to prioritize

### 1. Tests become an experimentally validated oracle

A reference implementation passing and an empty patch failing are necessary, not sufficient. The current pilot now requires every curated mutant to compile and fail at least one assertion under a complete, non-simulated test run. CodeExecutor may record a nonzero test-binary exit in `runtime_error` when an ordinary assertion fails; exact full test counts therefore indicate a completed behavioral failure, while missing counts/simulated runs remain unresolved. For a larger task bank, record the mutation catalogue and report killed, surviving, invalid/compile-failing and unresolved mutants separately. A mutant that cannot compile is not evidence that tests discriminate behavior.

The fixed eight-task pilot declares expected test counts explicitly and binds them into each task digest (schema v2), removing producer/auditor text scanning. Schema v3 corrected requested model identity and failed closed on missing cloud credentials. Schema v4 separates selected tier decisions from actual backend attempts (including failed calls and fallback paths), and the auditor recomputes selected/actual route summaries per treatment arm. This avoids falsely treating a selected CloudLLM tier as evidence that cloud supplied the output. These are application-reported telemetry, not cryptographic provider attestation. The real preflight and each run must account for exactly the declared count. Before ingesting arbitrary repositories, generalize this into a machine-readable harness contract with independent test-discovery evidence.

### 2. Unknown is not failure—and not a tie

Report assessed behavioral failures separately from incomplete compiler/test execution. Paired GCS effect statistics should include only pairs with complete oracles in both arms; publish the unresolved-pair count and rate next to the effect. Never remove unresolved outcomes silently: if their rate differs by arm or task class, treat that as a validity issue and investigate it.

### 3. Test intervention effects, not just the final agent

Use the same model revision, prompt budget, tools, compiler, sandbox, task, hidden tests and stopping budget in each arm; change only GCS advisories. Record requested model identity, selected tier decisions, and each actual backend invocation in order (including failures/fallback). Compare treatment arms not only for pass rate but also for changes in backend attempts, simulated fallback frequency, tail latency, cost, and unresolved-oracle rates. Alternate/randomize treatment order and repeat tasks to estimate run variance. Include a no-GCS baseline and, later, ablations for advisory type, confidence threshold and repair feedback. Report the agent+model+harness as the unit of comparison.

### 4. Make the held-out tasks harder than memorized function synthesis

Move from toy functions to repository-level fixes: multi-file changes, regression-preserving refactors, dependency/compiler errors, concurrency, security-sensitive parsing, performance regressions and changes where tests alone do not reveal the intended behavior. Freeze a public development set, maintain a private/held-out evaluation set, and grade patches in pristine checkouts with hidden fail-to-pass plus pass-to-pass tests. Keep task source, model prompts and reference patches out of the agent's accessible runtime. A practical Rust-native starting point is [SWE-bench-Live MultiLang](https://github.com/microsoft/SWE-bench-Live), which supports Rust and reported 1,077 task instances across 431 repositories and eight languages in its August 2026 snapshot. Pin an exact dataset revision and report the Rust slice, not the aggregate leaderboard result.

### 5. Measure the engineering frontier, not one score

For each configuration report:
- pass@1, repeated pass rate and paired win/loss/tie counts, plus a deterministic task-cluster bootstrap interval for the equal-task paired effect;
- valid-oracle, mutation-adequacy and unresolved-execution rates;
- pass-to-pass regression rate and security-regression rate;
- cost per verified fix, wall time, nearest-rank p50/p95 agent and verification latency, tokens, retries, observed generation-tier counts, and energy/compute estimates;
- result confidence intervals clustered by repository/task family, plus per-category results.

Use a cost/correctness Pareto frontier. A system can beat a more expensive agent on verified fixes per dollar without claiming a higher absolute benchmark score. Do not combine unrelated benchmark scores into one headline number.

## Proposed acceptance gates (targets, not measured results)

### Gate A — make the pilot trustworthy
- Exact-head `cargo fmt --check`, focused tests and example compilation pass in the pinned Rust/Nix environment.
- All reference controls pass, the empty source is rejected by a real Rust compiler diagnostic, and all ten semantic mutants are behaviorally killed after successful compilation.
- Receipt auditor accepts a valid producer receipt and rejects fixture edits, missing/duplicated runs, incomplete oracle data, tampered summaries, malformed digests and source/manifest pin mismatches.
- No claim of improvement until both arms execute with a named model/backend; report unresolved pairs rather than counting them as failures or ties.

### Gate B — demonstrate a mechanism effect
- Expand to 50–100 carefully reviewed Rust tasks with category and difficulty stratification.
- Use at least three independent repeats per task/arm initially; select the final repetition count using observed variance and a predeclared power/precision target.
- Require complete oracle coverage to be high and balanced across arms; publish every task's reference/mutation control outcomes.
- Report paired confidence intervals and a paired binary-outcome test; avoid claiming statistical significance on the eight-task pilot.

### Gate C — credible frontier comparison
- Run 250–500 held-out Rust repository tasks across multiple repositories and task families, with hidden tests and pristine grading.
- Pre-register model ID/revision, agent/harness commit, token/time/cost budget, task-manifest digest, exclusions, primary endpoint and analysis before the first scored run.
- Candidate performance target: at least +5 absolute percentage points in held-out pass@1 versus the same-model, same-budget baseline, with a repository-clustered 95% confidence interval excluding zero; alternatively, reduce cost per verified fix by 20% while staying within a predeclared 1.5-point correctness non-inferiority margin.
- Treat those thresholds as goals for deciding whether to continue, not guaranteed or currently achieved performance. Include an independent replication before claiming superiority.

## Immediate sequence

1. Before spending on an evaluation, use `--print-manifest` to capture and independently pin the source/task-manifest digest; then run exact-head formatting, focused producer/auditor unit tests and compile both targets in a clean pinned environment. The auditor now prints a deterministic 10,000-resample task-cluster bootstrap interval, but the current toy-function pilot's interval is descriptive only; larger studies need repository and task-family identifiers and should resample at the independent unit relevant to the claim.
2. Inspect the 10 mutation controls; if any fail compilation or survive tests, improve the task fixture rather than weakening the mutation gate.
3. Capture a real receipt from each available backend (native dispatcher, Ollama and/or cloud) and audit it using externally pinned source and task-manifest digests.
4. Only then scale task count and invest in a hidden Rust repository-repair set. Fix oracle quality before increasing leaderboard ambition.

## Epistemic boundary

A digest proves byte identity relative to the bytes being hashed; it is not a signature or proof that execution occurred. The receipt auditor checks internal consistency, not the trustworthiness of the machine running the compiler. For a serious external claim, preserve runner output and environment identity, independently rerun the verifier, keep the holdout private, and sign the final raw artifact in a separately controlled pipeline.
