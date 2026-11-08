# symthaea-nano-policy

A small research-only Rust policy kernel for deterministic candidate selection from a frozen feature pool. It does not predict energy, spectra, stability, toxicity, synthesizability, or any other physical property.

## Contract

One JSON request per input line; one JSON response per output line. The CLI uses a bounded line reader: it retains at most 16 MiB per request, hashes oversized records incrementally without retaining them, drains the record to its line ending, and rejects it as `request_too_large`. Invalid UTF-8 is rejected instead of terminating the whole JSONL stream. CRLF and LF line endings are normalized by excluding line terminators from the request digest. Blank or whitespace-only lines are ignored. Example request:

    {"schema_version":"nano-policy-request-v1","policy":"farthest_first","budget":2,"seed":42,"feature_contract_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","candidates":[{"candidate_id":"c-a","family_id":"family-a","normalized_features":[0.0,0.0]},{"candidate_id":"c-b","family_id":"family-a","normalized_features":[0.2,0.2]},{"candidate_id":"c-c","family_id":"family-b","normalized_features":[1.0,1.0]}]}

The repeated `a` characters in this example are a placeholder only. Replace them with the lowercase SHA-256 of the actual frozen feature/preprocessing contract; the kernel checks the digest format but cannot independently fetch or verify that external contract.

Policies:
- random_seeded: deterministic pseudo-random sampling without replacement.
- farthest_first: greedy max-min distance over normalized features, starting from the lexicographically smallest candidate ID.
- group_balanced_farthest_first: seeded initial candidate, then prefers the least represented family before max-min distance.
- cost_aware_farthest_first: starts with the cheapest candidate, then selects maximum minimum descriptor distance divided by declared evaluation cost. It requires a positive cost on every candidate, in one frozen unit.

## Verify a receipt independently

The CLI has a structural-verification mode. Each input line is a JSON object holding the two exact JSONL line contents as strings:

    {"request_line":"<exact original request JSON>","response_line":"<exact original response JSON>"}

Run:

    cargo run -p symthaea-nano-policy -- --verify-receipt < receipt-pairs.jsonl > audits.jsonl

The output includes request and response SHA-256 digests and checks for budget/count agreement, selected-ID uniqueness and membership, family/cost lineage, and independently recomputed nearest-selected descriptor distances. The verifier does not rerun the selector and cannot prove a scientific result. Malformed or extra envelope fields produce a separate `rejected_input` JSON response.

## Schemas and benchmark criteria

- Request schema: schema/request.schema.json
- Response/error receipt schema: schema/response.schema.json
- Structural audit schema: schema/receipt-audit.schema.json
- Structural verifier API: `symthaea_nano_policy::receipt::verify_receipt(request_line, response_line)`
- Benchmark and go/no-go protocol: BENCHMARK.md

## Receipt verification

Call `symthaea_nano_policy::receipt::verify_receipt(request_line, response_line)` with the exact JSONL line contents, excluding line terminators. A `passed` audit means the request and response are structurally consistent: it checks the request SHA-256, feature-contract digest, policy/seed/budget/count, candidate membership and uniqueness, family/cost lineage, bounds and the reported nearest-selected distances recomputed from the frozen descriptors.

This checker does **not** rerun the selector or prove that the chosen candidates are optimal, scientifically useful, safe, novel, or physically valid. Use it as a structural receipt gate; a separate benchmark checker must independently recompute task-specific metrics from held-out labels/observations.

## Independent post-outcome metrics

The crate exposes `symthaea_nano_policy::metrics::evaluate_json_ledger` and a separate CLI that evaluates a completed outcome ledger:

    cargo run --locked -p symthaea-nano-policy --bin nano-policy-metrics -- --input crates/domains/symthaea-nano-policy/fixtures/synthetic-outcome-ledger.json --bootstrap-replicates 200 --seed 17 > metrics-report.json

The evaluator is deliberately downstream of selection. **Never feed the outcome ledger or metrics report to the selector.** It hashes the exact input bytes itself, rejects unknown fields, verifies identical target IDs and declared group IDs across arms, validates query-rank and status/loss semantics, and preserves all attempted-call costs including failures.

It computes success-at-budget, normalized success AUC (mean success fraction across budget points 1..K), target coverage, oracle failure/valid rate, repeated candidate attempts, best loss among covered targets, mean cost per target, cost to first success, and paired group-bootstrap comparisons against the declared baseline. Metrics-report v2 additionally emits each group’s success curve, success-at-budget, normalized AUC, target coverage and mean cost, plus explicit worst-group success/AUC/coverage values so a pooled win cannot conceal a failing chemistry/surface/temperature subgroup. Comparisons include final-budget success-rate difference, normalized success-AUC difference, best-loss reduction (baseline best loss minus candidate best loss; positive favors the candidate), and separate worst-group success/AUC deltas defined as the minimum candidate-minus-baseline group mean. The continuous-loss comparison requires a valid best loss for every paired target in both arms; if any target is uncovered, the report says `insufficient_target_coverage` and emits no confidence interval. Pooled comparison intervals are withheld unless at least five groups are declared. Worst-group intervals additionally require at least two paired targets in every group; otherwise the point estimate is reported with `insufficient_within_group_replication`. For worst-group intervals, paired target deltas are bootstrapped within each predeclared group and Bonferroni-adjusted percentile bounds are combined as the minimum across groups. This is stricter than pooled inference and still cannot establish transfer to unseen groups. Five groups and two targets per group are minimum guards, not a power guarantee or proof that group IDs are truly independent. Worst-group intervals are adjusted across groups but remain unadjusted across policy arms and multiple pooled/worst-group metrics; preregister the primary comparison or apply an additional declared multiplicity adjustment.

Schema versions:
- Post-outcome ledger remains `nano-outcome-ledger-v1`: `schema/outcome-ledger.schema.json`
- Metrics report is now `nano-benchmark-metrics-v2` with per-group and worst-group outputs: `schema/benchmark-metrics-report.schema.json`

Use the same frozen `group_id` values as the shift manifest in the NANO-010 audit contract and populate optional `shift_manifest_sha256` in the outcome ledger. Metrics report v2 carries that digest forward so detached per-group/worst-group curves remain bound to the exact shift-manifest bytes. The standalone metrics evaluator summarizes those groups but does not independently establish that split membership or physical shift labels are valid. The Rust artifact auditor in [PR #7717](https://github.com/Luminous-Dynamics/symthaea/pull/7717) checks shift-manifest hashes and rejects sample/independence keys that cross data splits.

Each observation's `cost` is the all-in cost for selecting and evaluating that attempt in the ledger's common unit. For end-to-end efficiency claims, include selector/runtime plus oracle costs; if only oracle cost is logged, label the metric accordingly rather than implying total pipeline cost.

The checked-in ledger is a synthetic unit/smoke fixture, not scientific data. Real scored comparisons still require the source/rights, holdout, oracle, and independent structural-receipt gates described in `BENCHMARK.md`.

## Fail-closed contract

- Unknown fields are rejected. Target/label fields, optical target spectra, test membership and oracle observations must not be included.
- Candidate IDs must be unique; family IDs non-empty; vector dimension fixed; all feature values finite and within [0, 1].
- Optional `estimated_cost` must be finite and positive when supplied; cost-aware policy rejects any missing cost. All costs must share one declared unit and be frozen before evaluation.
- feature_contract_sha256 identifies the externally frozen feature and normalization definition.
- Request-line SHA-256 is returned with the result. It is a digest of exact input text, not proof that source features are true or legally usable.
- The selector performs no feature scaling or fitting. Fit transforms only on permitted training data and freeze them outside this program.
- No dataset, model checkpoint, external solver, network access, synthesis instruction, or production Python dependency is included.

Limits: 10,000 candidates, 256 features per candidate, 1,000 selected candidates, and 16 MiB per input line.

## Run and test

    cargo test -p symthaea-nano-policy
    cargo run -p symthaea-nano-policy < request.jsonl > response.jsonl

This is a baseline/adapter foundation. A Symthaea HDC representation or active-inference policy should be added as a separately measured arm after the feature contract and baselines are frozen. Descriptor-space diversity is neither physical novelty nor calibrated uncertainty.
