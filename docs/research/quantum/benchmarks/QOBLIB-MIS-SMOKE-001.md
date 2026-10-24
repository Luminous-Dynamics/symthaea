# QOBLIB-MIS-SMOKE-001: deterministic local smoke runner

Status: **runner authored; Rust execution not yet confirmed**

## Run locally

From the Symthaea workspace root, after Rust dependencies/toolchain are available:

    cargo run -p symthaea-quantum-comp --bin qoblib_mis_smoke -- --seed-base 42 --target-objective 10 /path/to/QOBLIB/07-independentset/instances/farm.gph > farm-smoke.json

The command reads one DIMACS edge-format graph and writes exactly one JSON record to stdout. On successful runs the receipt has return_status=completed; on input/read/verification failures it emits a return_status=failed receipt with an error kind, escaped error text, elapsed time, and source byte/hash identity when the input was readable, then exits nonzero. It also writes a diagnostic to stderr. This preserves failures as data instead of silently dropping them. Keep stdout as the raw inspectable receipt. Receipt schema v5 also contains a bounded deterministic 1-for-2 local-improvement result for graphs of at most 250 vertices; pair-search is capped at 2,000,000 checks, and a budget-exhausted result is explicitly not labelled locally optimal. The pinned source cohort is tracked in QOBLIB-MIS-COHORT-001.json.

## Re-verify a published solution file

The runner also supports the independent QOBLIB solution check:

    cargo run -p symthaea-quantum-comp --bin qoblib_mis_smoke -- --verify-solution /path/to/instance.gph /path/to/instance_solution.sol 88 > solution-audit.json

The third argument (88 above) is optional; supply it when comparing against the objective declared by a submission. This value is passed to the independent graph verifier and must equal the unique selected-vertex count. The solution parser itself ignores objective comments and treats the vertex list as untrusted data. On a parseable solution, stdout contains a single JSON audit receipt with graph and solution checksums, counts, computed/reported objective, feasibility and every detected graph/objective violation. Invalid graph/solution input and file I/O errors emit a separate failed JSON receipt with the escaped error and available path/objective context; infeasible or malformed candidates produce a nonzero exit status. This checks feasibility, not optimality, reproduces no submitted algorithm and does not validate its runtime claims. The reported checksums are FNV-1a, not cryptographic hashes; source provenance should continue to use pinned upstream commit and Git object IDs.

## Fail-closed parser limits

The runner reads file metadata before reading graph contents and refuses an input larger than 64 MiB. The parser also defaults to at most 100,000 vertices and 5,000,000 declared/observed edges; declarations are checked before graph allocation, and the observed edge budget is enforced during parsing. These defaults cover the pinned QOBLIB cohort, including its documented 45 MB C4000-5 file. The library API exposes MisDimacsLimits and parse_dimacs_mis_with_limits for deliberately provisioned custom workloads. Raising those limits does not add a runtime or memory watchdog, which remain explicitly unconfigured in the smoke receipt.

## Output contract

The receipt records:
- schema (currently v5) and explicit correctness-smoke track label;
- supplied source path, byte count and FNV-1a 64-bit checksum of the exact input bytes;
- parsed vertex/edge counts;
- the deterministic minimum-residual-degree greedy solution, its one-based vertex list, independently recomputed objective and feasibility verdict;
- a deterministic repeated feasible 1-for-2 improvement that replaces one selected vertex with two mutually nonadjacent compatible vertices; the result has an explicit locally-optimal, pair-check-budget-exhausted, or skipped-over-limit status;
- the tiny exact reference result and feasibility verdict when the graph has at most 24 vertices, or an explicit skipped-over-limit status otherwise;
- parser, greedy, portfolio, local-improvement, exact-oracle and end-to-end elapsed nanoseconds;
- QOBLIB-compatible incumbent time series in seconds: `qoblib_objective_time_series.deterministic_greedy` contains one run, `multi_start_greedy_per_trial` contains one trace per seeded restart, and `local_improvement_stage` contains its own stage-relative trace when run;
- Optional fixed-target analysis via `--target-objective <usize>`: records per-arm trial counts, target-hit counts, empirical hit rate and time-to-target seconds for each run. Unreached targets are `null`, never dropped; targets above the graph's vertex count fail closed. With no target supplied, `target_analysis` is `null`;
- explicit seed-base and restart-count provenance for the randomized portfolio, with `--seed-base <u64>` available to replay or run a distinct deterministic campaign. The 64-bit `seed_base` and `best_seed` are JSON decimal strings, not JSON numbers, so consumers preserve the full integer exactly;
- a stable non-cryptographic checksum over normalized input, policy/configuration, seeds and outcomes, excluding variable runtime fields;
- package version, target OS/architecture, compile-time Rust version when provided, explicit determinism/replay wording, and null time/memory limits when none were configured;
- failure receipts for source-read errors, DIMACS parse errors, local verification failures and exact-oracle failures;
- limitations that prevent treating smoke timings as an SOTA result.

The greedy control selects a minimum residual-degree vertex with the lowest-index tie break and is deterministic. The separate 16-run portfolio uses the recorded base seed plus the zero-based restart index for seeded tie breaking. Identical input bytes, source/toolchain and seed base should reproduce algorithmic outcomes; elapsed fields are intentionally different and are excluded from the normalized-result checksum. The local improvement is deterministic, only runs on graphs up to 250 vertices, and returns a feasible solution even if its pair-check budget is exhausted. A locally-optimal label means only that no improving one-selected-for-two-unselected exchange remains in the searched neighborhood; it is not a global-optimality claim. It is a baseline/control, not a competitive replacement for dedicated MIS methods. The exact reference uses the deliberately bounded branch-and-bound oracle and must not be run on large benchmark instances.

## Classical baseline arms emitted

Each receipt now preserves three distinct solution tracks: (1) deterministic minimum-residual-degree greedy with lowest-index tie-breaking; (2) a reproducible 16-start minimum-residual-degree greedy portfolio with seeded tie-breaking, recording each run's objective, distinct returned solutions, winning seed and best feasible set; and (3) deterministic one-for-two local improvement applied to the first greedy result, with explicit pair-check-budget status. The tiny exact oracle is run only up to the declared 24-vertex limit.

The residual-degree greedy implementation uses a lazy min-heap and updates degrees only across edges incident to removed vertices, avoiding full adjacency rescans at every selection. The multi-start portfolio is intentionally a modest classical control, not a replacement for strong methods such as the published QOBLIB ParallelILS record. All 16 starts count toward its reported runtime. The smoke runner is not a portfolio router and does not claim that selecting the best of these heuristics beats industry solvers.


## Incumbent time series and time-to-solution

Each trace is encoded as QOBLIB's outer array of runs and inner array of `{"Time": seconds, "Incumbent": objective}` observations. Times start at zero relative to each traced run or local-search stage. The greedy traces begin at objective zero and append the feasible independent-set cardinality after each chosen vertex; the local-improvement trace begins at the greedy input's cardinality and appends values only after accepted 1-for-2 improvements. Every trace ends with a terminal observation at or after the last update, with the final incumbent repeated if it did not change. For MIS maximization the incumbent objective is nondecreasing.

A fixed target can be supplied using `--target-objective <usize>`. Choose it before the run from an authoritative optimum/best-known objective or the application's declared quality threshold; do not pick it after inspecting outputs. The report returns each trial's first hit time in seconds, the number of hits and empirical hit rate. This is an observed cohort summary, not a calibrated future-success probability. A target above the graph's vertex count is rejected, and an unreached target remains `null` in the per-trial list. Local-improvement target time remains relative to that stage, so it is not directly comparable to the complete-run timing without including prior stages.

The 16 stochastic greedy restarts each have an independent relative-time trace and retain their individual final objective. This supports target-hit / empirical time-to-solution analysis across trials without treating all run steps as independent samples. The local-improvement trace measures only that stage, not the full end-to-end pipeline; add preceding-stage costs or use the complete-run outcome when comparing it with a full solver. The top-level end-to-end time remains the operational whole-run measure. Time fields are local diagnostics, not cross-machine SOTA timings.

## Example correctness anchor

The pinned QOBLIB farm instance has 17 vertices and 39 edges. Independent subset enumeration on the pinned source graph finds maximum independent-set cardinality 10, matching the published ParallelILS best-known objective. The Rust runner is designed to parse the same source, independently verify the greedy set, and compare with the exact reference. This repository contribution has not yet been compiled/executed in the current authoring environment, so no Rust runner output is represented as already produced.

## Interpreting costs and claims

Failure receipts retain a source checksum for parse/verification failures and use null when the source could not be read. FNV-1a is not cryptographic; the immutable QOBLIB source commit and file-level Git object IDs remain the source provenance. The rustc_version field is labelled not-recorded-at-build unless a build environment explicitly sets RUSTC_VERSION. Runtime figures are local diagnostics and must not be compared across machines without a controlled protocol. Parse time plus solver time should not be added to end-to-end time; stage metrics overlap with the total. The output does not claim quantum execution, a state-of-the-art result, or quantum advantage.
