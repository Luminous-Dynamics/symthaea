# QOBLIB-MIS-SMOKE-001: deterministic local smoke runner

Status: **runner authored; Rust execution not yet confirmed**

## Run locally

From the Symthaea workspace root, after Rust dependencies/toolchain are available:

    cargo run -p symthaea-quantum-comp --bin qoblib_mis_smoke -- /path/to/QOBLIB/07-independentset/instances/farm.gph > farm-smoke.json

The command reads one DIMACS edge-format graph and writes exactly one JSON record to stdout. On successful runs the receipt has return_status=completed; on input/read/verification failures it emits a return_status=failed receipt with an error kind, escaped error text, elapsed time, and source byte/hash identity when the input was readable, then exits nonzero. It also writes a diagnostic to stderr. This preserves failures as data instead of silently dropping them. Keep stdout as the raw inspectable receipt. Receipt schema v2 also contains a bounded deterministic 1-for-2 local-improvement result for graphs of at most 250 vertices; pair-search is capped at 2,000,000 checks, and a budget-exhausted result is explicitly not labelled locally optimal. The pinned source cohort is tracked in QOBLIB-MIS-COHORT-001.json.

## Fail-closed parser limits

The runner reads file metadata before reading graph contents and refuses an input larger than 64 MiB. The parser also defaults to at most 100,000 vertices and 5,000,000 declared/observed edges; declarations are checked before graph allocation, and the observed edge budget is enforced during parsing. These defaults cover the pinned QOBLIB cohort, including its documented 45 MB C4000-5 file. The library API exposes MisDimacsLimits and parse_dimacs_mis_with_limits for deliberately provisioned custom workloads. Raising those limits does not add a runtime or memory watchdog, which remain explicitly unconfigured in the smoke receipt.

## Output contract

The receipt records:
- schema and explicit correctness-smoke track label;
- supplied source path, byte count and FNV-1a 64-bit checksum of the exact input bytes;
- parsed vertex/edge counts;
- the deterministic minimum-residual-degree greedy solution, its one-based vertex list, independently recomputed objective and feasibility verdict;
- a deterministic repeated feasible 1-for-2 improvement that replaces one selected vertex with two mutually nonadjacent compatible vertices; the result has an explicit locally-optimal, pair-check-budget-exhausted, or skipped-over-limit status;
- the tiny exact reference result and feasibility verdict when the graph has at most 24 vertices, or an explicit skipped-over-limit status otherwise;
- parser, greedy, exact-oracle and end-to-end elapsed nanoseconds;
- a stable non-cryptographic checksum over the normalized result fields, excluding variable runtime fields;
- package version, target OS/architecture, compile-time Rust version when provided, deterministic/no-seed declaration, and explicit null time/memory limits when none were configured;
- failure receipts for source-read errors, DIMACS parse errors, local verification failures and exact-oracle failures;
- limitations that prevent treating smoke timings as an SOTA result.

The greedy policy recomputes each active vertex's residual degree at every selection and breaks ties by lower vertex index. It is deterministic and requires no random seed. The local improvement is deterministic, only runs on graphs up to 250 vertices, and returns a feasible solution even if its pair-check budget is exhausted. A locally-optimal label means only that no improving one-selected-for-two-unselected exchange remains in the searched neighborhood; it is not a global-optimality claim. It is a baseline/control, not a competitive replacement for dedicated MIS methods. The exact reference uses the deliberately bounded branch-and-bound oracle and must not be run on large benchmark instances.

## Classical baseline arms emitted

Each receipt now preserves three distinct solution tracks: (1) deterministic minimum-residual-degree greedy with lowest-index tie-breaking; (2) a reproducible 16-start minimum-residual-degree greedy portfolio with seeded tie-breaking, recording each run's objective, distinct returned solutions, winning seed and best feasible set; and (3) deterministic one-for-two local improvement applied to the first greedy result, with explicit pair-check-budget status. The tiny exact oracle is run only up to the declared 24-vertex limit.

The residual-degree greedy implementation uses a lazy min-heap and updates degrees only across edges incident to removed vertices, avoiding full adjacency rescans at every selection. The multi-start portfolio is intentionally a modest classical control, not a replacement for strong methods such as the published QOBLIB ParallelILS record. All 16 starts count toward its reported runtime. The smoke runner is not a portfolio router and does not claim that selecting the best of these heuristics beats industry solvers.

## Example correctness anchor

The pinned QOBLIB farm instance has 17 vertices and 39 edges. Independent subset enumeration on the pinned source graph finds maximum independent-set cardinality 10, matching the published ParallelILS best-known objective. The Rust runner is designed to parse the same source, independently verify the greedy set, and compare with the exact reference. This repository contribution has not yet been compiled/executed in the current authoring environment, so no Rust runner output is represented as already produced.

## Interpreting costs and claims

Failure receipts retain a source checksum for parse/verification failures and use null when the source could not be read. FNV-1a is not cryptographic; the immutable QOBLIB source commit and file-level Git object IDs remain the source provenance. The rustc_version field is labelled not-recorded-at-build unless a build environment explicitly sets RUSTC_VERSION. Runtime figures are local diagnostics and must not be compared across machines without a controlled protocol. Parse time plus solver time should not be added to end-to-end time; stage metrics overlap with the total. The output does not claim quantum execution, a state-of-the-art result, or quantum advantage.
