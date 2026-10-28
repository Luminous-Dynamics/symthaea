# Broca Risk-Aware Memory Escalation (V1)

**Status:** executed synthetic research lane; not a Broca integration, not an official benchmark run, and not evidence of industry/SOTA superiority.

## Research question

Can a memory-using agent decide among three actions—answer from current evidence, spend more budget to escalate retrieval and re-evaluate, or abstain—while improving task outcomes at controlled harmful-answer risk?

This extends binary “retrieve or not” gating to a sequential decision problem. It separates initial retrieval, escalation to a more expensive second retrieval, and abstention when evidence is weak, absent, contradictory, stale, revoked or out of scope.

The evaluator records counterfactual outcomes for all three actions per generated case. Policies receive only query-time feature dictionaries, never case objects, latent state, answerability labels or counterfactual outcomes. The oracle is evaluator-only.

## Reproduce

Requires Python 3.11+ and only the Python standard library. No model endpoint, internet, NumPy or pandas is required for the synthetic run.

    bash research/broca-risk-aware-escalation-v1/run.sh /tmp/broca-risk-run

The runner refuses to overwrite non-empty output directories. It generates an outcome JSONL, configuration, summary, report and receipt; then runs an independent streaming verifier and tests. The full 150,000-row outcome artifact is deliberately not checked into Git due to its size. Re-run the script to reproduce it.

## Policy arms

- `always_answer`, `always_abstain`, `always_escalate`: structural controls.
- `similarity_threshold`, `evidence_threshold`: simple selectors tuned on calibration data.
- `sequential_rule`: interpretable three-action policy.
- `learned_value`: multi-output ridge regression predicts utility for each action.
- `risk_budgeted`: point-estimated harm constraint; comparator, not a guarantee.
- `risk_control_*permille`: thresholds chosen on a separate calibration split to satisfy a Wilson-upper-bound criterion on observed calibration harm.
- `oracle_diagnostic`: evaluator-only upper bound; not deployable.

Development data is split by complete groups into policy-training and calibration partitions. Thresholds are frozen before five synthetic test regimes are evaluated: IID, high-similarity distractors, stale revisions, scope shift and compound shift.

## Risk-control caveat

The 95% Wilson upper endpoint is an empirical calibration screen. It is not distribution-free deployment-risk control, does not guarantee performance under distribution shift and does not certify the risk model itself. The benchmark intentionally reports each shifted regime separately.

## Evidence boundary

All features, classes and outcomes are constructed by the generator. Token costs and utility are declared simulation parameters—not observed token usage, dollar costs, energy or wall-clock latency. This experiment does not execute Symthaea, an LLM endpoint or the official LongMemEval-V2 harness. It makes no SOTA claim.

See `RESULTS_SUMMARY.md`, `BENCHMARK_TARGETS.md` and the report generated into the run directory.
