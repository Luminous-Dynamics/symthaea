# Unguided Threat-Hunting Benchmark: Initial Synthetic Store

Status: implementation seed; **UNSCORED**. Parent: Symthaea #7742. Related programme: #2105, #2113, #2217. Praxis learning contract: Mycelix #5124 and draft PR #5125.

## Research question

Can Symthaea find and explain planted malicious event sequences in a noisy evidence store without the evaluator giving away the attack family, event IDs, or query plan—and can it preserve benign alternatives, data-quality uncertainty, and action-authority boundaries?

The Cyber Defense Benchmark paper evaluates iterative queries against 75k–135k Windows-event records and reports 3.8% mean malicious-event recall for its best evaluated model in that study. CyberSOCEval separately targets malware analysis and threat-intelligence reasoning. These motivate task-specific evaluation; neither result is a prediction for Symthaea.

- https://arxiv.org/abs/2604.19533
- https://arxiv.org/abs/2509.20166
- https://www.nist.gov/itl/applied-cybersecurity/nice/nice-framework-resource-center/nice-framework-current-versions

## Artifact and trust boundary

The generator emits three distinct roots/artifacts:

- `solver/events.jsonl`: event observations only; strict event v1 schema.
- `solver/manifest.json`: exact event digest/bytes/count, seed, source/version, and non-claims. It contains no oracle path, digest, record count, or answer labels.
- `evaluator/oracle.json`: per-event truth, planted sequences, benign decoys, and data-quality labels.
- Root `manifest.json`: operator-side combined manifest that hashes both sides; never mount this parent directory into the solver process.

This is a data-layout boundary, not a substitute for OS isolation. In a real range, mount only `solver/` into the solver's filesystem namespace. The typed query loader resolves only files inside that directory and never opens the root manifest or evaluator path.

## Typed investigation interface

The Rust query interface accepts only the versioned `cyber-hunt-query-v1` schema: exact-text equality, bounded event-/ingestion-time ranges, minimum bytes transferred, explicit projected fields, and a hard row limit. It has no free-form SQL, shell, mutation, network, or effect API.

Each query receipt binds the exact normalized request SHA-256, solver-manifest SHA-256, event-stream SHA-256, source metadata, rows examined/matched/returned, serialized result bytes, and cumulative budget usage. The session has a bounded query count, total rows examined, total returned rows, returned-byte budget, maximum limit, and maximum filters/projection width. Failed queries do not consume usage, and the runner consumes an ordered JSONL batch in one process so budgets are shared across the investigation.

The query output is an observation projection, not a security conclusion. The query system cannot write events or use campaign truth. Duplicate delivery and source-ID/payload conflict remain data-quality observations, not automatic malicious labels.

## Prediction scoring contract

The benchmark has a versioned prediction contract and a separate scoring command. A solver prediction set binds to the exact source commit, solver-only manifest hash, and event-store hash. Suspected event IDs require existing evidence references and a rationale; abstained IDs must be known and cannot also be marked as suspects.

The evaluator runs separately with access to the oracle, recomputes the dataset's structural qualification first, then scores only the supplied synthetic truth. It reports TP/FP/FN/TN, precision and recall in basis points, false positives per 10,000 benign rows (milli-scaled), abstention counts, false positives on approved-change decoys, and recall by campaign phase and campaign. Zero-alert baselines are supported but do not count as model performance. A scored result is explicitly `SCORED_UNQUALIFIED_SYNTHETIC_ONLY`, never a security qualification or grant of operational authority.

The source tree now includes `schemas/predictions-v1.schema.json` and `schemas/evaluation-receipt-v1.schema.json`; JSON Schema documents receive draft meta-validation in the Rust test suite. The CI baseline uses an empty suspect list to verify scorer plumbing and the expected zero-recall disposition; a real model comparison still requires independently generated predictions and held-out campaigns.

## Prediction scoring contract

The benchmark has versioned prediction and scoring-receipt contracts. A solver prediction set binds to the exact source commit, solver-only manifest hash, and event-store hash. Suspected event IDs require existing evidence references and a rationale; abstained IDs must be known and cannot also be marked as suspects.

A separate evaluator re-verifies the complete synthetic dataset before reading ground truth. It reports TP/FP/FN/TN, precision and recall in basis points, false positives per 10,000 benign rows (milli-scaled), abstention counts, false positives on approved-change decoys, and recall by campaign phase and campaign. Empty-alert baselines are supported but do not count as model performance. A scored result is explicitly `SCORED_UNQUALIFIED_SYNTHETIC_ONLY`, never a security qualification or grant of operational authority.

The source includes `schemas/predictions-v1.schema.json` and `schemas/evaluation-receipt-v1.schema.json`. JSON Schema documents receive Draft 2020-12 meta-validation in the Rust test suite using the offline configuration (remote/file reference resolution disabled). Cross-document reference resolution and full instance conformance remain distinct from meta-schema validity and are not promoted by a passing meta-validation test.

## Scale and measurement plan

- S0: seed=42, 1k and 10k records; verify determinism, bytes, solver/oracle parity, and read-only query receipt.
- S1: 100k records; report query latency, records/bytes examined, query cost, precision/recall, and result truncation.
- S2: at least 250k records and varied seeds/time shifts after generator review; add independent benign-only and changed-campaign controls.
- S3: separately controlled held-out campaigns and independent evaluator; split by campaign/environment/causal pattern, never by random event rows.

Report event-level precision/recall, recall by campaign phase/domain, false positives per 10k benign rows, exact ID/timestamp attribution, query count/latency/bytes, abstention, benign-change false alarms, duplicate/conflicting telemetry handling, and complete failed/unscored counts. Compare deterministic/rule-based baselines under the same evidence access and query budget. Do not reduce results to a single score.

## Proposed hard gates (not measured results)

- Same seed + exact generator/version/source yields byte-identical event/oracle artifacts.
- Every solver record has exactly one oracle record; all campaign links resolve.
- Solver view contains no evaluator labels and can be copied/mounted separately without an oracle file.
- 100% rejection of invalid hashes/counts, unknown IDs, invalid query fields, unrestricted operations, and safety negative controls.
- Query budgets are shared for the complete investigation; no query-induced state mutation, effect dispatch, live scanning, or credential access.
- No capability claim until semantic scoring runs on independent holdout with confidence intervals, baseline comparison and complete failure counts.

## Next engineering steps

1. Compile, test, and repair generator + query engine at the exact PR head.
2. Keep the solver/evaluator directory separation and verify the solver runner mounts only the solver view.
3. Add deterministic read-only event-indexing only if baseline query latency proves full scan is too costly; benchmark any index against the same truth corpus to avoid hidden coverage gaps.
4. Implement an independent semantic evaluator and baseline runners.
5. Compose the benchmark into the existing #2217 crucible, then render Praxis teaching scenarios without publishing hidden oracle records.

## Claim ceiling

Only deterministic synthetic fixture generation, internal structural consistency, and bounded read-only query behavior can be established by the first tranche. It does not establish detector effectiveness, real incident truth, SOC readiness, learner efficacy, or production security.