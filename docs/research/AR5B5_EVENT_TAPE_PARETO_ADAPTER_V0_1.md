# AR-5B5 EventTape-to-Pareto adapter v0.1

**Status:** adapter implementation committed; test cases authored but not compiled or executed. The adapter requires a strictly qualified EventTape and creates a digest-bound candidate-level Pareto report.

## Purpose

The Pareto objective evaluator accepts a frozen vector of candidate metrics. The EventTape adapter derives those metrics from the canonical event history and resource receipts, instead of requiring a researcher to re-enter them manually.

File:
\`crates/domains/symthaea-architecture-engineering/src/event_tape_pareto.rs\`

Public entry point:
\`event_tape_to_pareto(tape, config)\`

The adapter calls \`EventTape::validate()\` first. It therefore rejects empty, incomplete, over-budget or incompletely metered tapes before deriving candidate objectives.

## Candidate aggregation

Candidate evaluations are grouped by candidate content SHA-256. Each retry and failed attempt remains in that candidate's resource total. Verified scores are aggregated separately from failure outcomes.

Supported objective mappings:

| Metric | Meaning | Required direction |
|---|---|---|
| MeanVerifiedScore | Mean of verified evaluation scores | Maximize |
| MinimumVerifiedScore | Worst observed verified score | Maximize |
| LowerConfidenceBoundVerifiedScore | Mean minus z × sample standard-error proxy; requires at least two verified repeats | Maximize |
| VerifiedAttemptRate | Verified evaluation count / all candidate attempts, including retries | Maximize |
| MeanReportedUncertainty | Mean evaluator-reported uncertainty across all verified repeats | Minimize |
| MaximumReportedUncertainty | Worst evaluator-reported uncertainty | Minimize |
| EvaluationAttemptCount | Candidate evaluation attempts, including failed retries | Minimize |
| TotalHostCpuMillis | Sum of host CPU time over candidate evaluation attempts | Minimize |
| SumActionWallMillis | Sum of action wall durations (worker-time, not elapsed campaign latency) | Minimize |
| TotalAcceleratorMillis | Sum of accelerator time | Minimize |
| MaxActionPeakMemoryBytes | Maximum per-action host memory peak; not concurrent campaign memory | Minimize |
| MaxActionAcceleratorPeakMemoryBytes | Maximum per-action accelerator memory peak | Minimize |
| TotalEnergyMillijoules | Sum of measured action energy | Minimize |
| TotalInputTokens / TotalOutputTokens | Token totals across candidate evaluation attempts | Minimize |
| TotalToolCalls / TotalEvaluatorCalls / TotalVerifierCalls | Measured call totals | Minimize |
| TotalChargedBudgetMicrounits | Sum of normalized event charges | Minimize |

The objective IDs and their order must match between the mapping list and the frozen objective-spec list. A metric/direction mismatch is rejected.

Candidates below the configured minimum verified-evaluation count remain in the Pareto plan as infeasible, with explicit violation reasons. Their resource expenditure is retained; their score metric uses the explicitly configured unverified fallback. Infeasible candidates do not contribute to the Pareto front or hypervolume.

For mean/worst reported uncertainty objectives, every verified evaluation must carry an uncertainty value. Missing uncertainty is never converted to zero. When no verified result exists, the candidate is infeasible and uses the objective's upper scale anchor as a conservative placeholder for the uncertainty coordinate.

The lower-confidence-bound score is a predeclared optimization statistic, not a calibrated confidence interval. A statistically valid uncertainty-aware front across independent runs remains future work.

## Resource accounting and overhead

Three levels are reported:

1. \`candidate_evaluation_resources\`: all candidate-evaluation actions, including rejected, inconclusive, timed-out, crashed and retried work.
2. \`direction_proposal_overhead\`: all direction-proposal/verifier actions, not assigned repeatedly to individual candidates.
3. \`campaign_resources\`: all actions, including both categories.

Additive meters are summed with checked arithmetic. Memory is summarized as the maximum per-action peak, not summed. The adapter verifies that candidate-evaluation totals plus proposal overhead exactly reconcile to campaign totals. None of these max-per-action memory fields represents campaign-wide concurrent memory.

This separation prevents two common accounting errors: excluding proposal-generation cost from system totals, and charging the same proposal overhead to every candidate as if each candidate caused a separate proposal.

## Evidence and receipts

Each candidate receives a \`CandidateEvidenceManifest\` with ordered event sequence, event hash, event ID, direction ID, retry reference, disposition, score, reported uncertainty and outcome evidence digest. The manifest SHA-256 becomes the Pareto candidate's evidence reference.

The report binds:
- serialized source EventTape digest;
- adapter configuration digest;
- Pareto plan and analysis receipt;
- all per-candidate evidence manifests and summaries;
- campaign, candidate-evaluation and proposal-overhead resource totals.

The adapter does not retrieve artifact bytes from the evidence digests. It proves which event and artifact identifiers were recorded, not that an external artifact store currently returns those bytes. A separate artifact resolver/verifier is still required for that claim.

## Tests authored

- failed retry costs remain charged to the original candidate;
- proposal overhead remains distinct but reconciles into campaign totals;
- per-candidate memory uses max-per-action semantics;
- candidate with no verified result remains in the report as infeasible;
- strict validation rejects unknown resource meters;
- metric/objective direction mismatch is rejected;
- lower-confidence-bound metric rejects a single verified sample;
- uncertainty objective rejects missing uncertainty on a verified repeat.

These tests are authored but not run. Acceptance commands:

\`\`\`bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering event_tape_pareto -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_objectives -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
\`\`\`

## Current limits and next steps

1. The current adapter reports policy-level campaign totals and candidate-level evaluation costs. Proposal overhead is not allocated to candidate rows because any such allocation requires a declared cost-attribution policy.
2. It operates on one policy tape. Cross-policy matched Pareto comparisons require aligning common candidate identities and campaign/seed blocks across reports.
3. Resource metrics come from EventTape receipts, but external hardware telemetry, concurrent process memory and independent energy-meter calibration remain responsibilities of the execution harness.
4. Artifact digest resolution and independent verifier re-execution remain outstanding.
5. Next: semantic alias and verifier corruption challenge fixtures, multi-run uncertainty-aware Pareto fronts, ZDT/DTLZ/WFG and constrained benchmark expansion, plus pinned algorithm adapters.

No SOTA result is claimed by this adapter.
