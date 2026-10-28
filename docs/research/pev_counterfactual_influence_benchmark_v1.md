# PEV counterfactual influence benchmark v1

**Status:** reference benchmark and synthetic execution only. It is not production Rust verification and is not a task-utility/SOTA result.

## Paradigm shift: from provenance to causal influence

A well-formed, hash-chained trace does not prove that prediction-error (PE) evidence actually affected the intended learner—or that invalid PE had no effect. The next gate is a paired counterfactual experiment. Re-run the same decision from the same initial state with the same observation/action inputs and externally supplied gate/outcome calibration; intervene only on PE availability and compatibility payload.

The auditor requires, per trial:

1. **No-PE baseline:** no PE evidence admitted and no PE-driven transition.
2. **Invalid-payload variants:** status is invalid, evidence is not admitted, and consumer state digests match the no-PE baseline exactly. V1 generates seven variants: null, 0, 1, -1, NaN, +Infinity, -Infinity (the non-finite tokens are represented as strings in JSON; a production encoder must separately test actual IEEE values).
3. **Valid-PE positive control:** the predeclared target consumer admits the exact evidence ID exactly once and has an observable state change in this deliberately PE-sensitive control case.
4. **Protected channels:** PE manipulation must not change the designated non-target consumer digests or the independent gate/outcome calibration digest.
5. **Paired-input integrity:** input, initial-state and externally held-outcome digests must match across branches. Otherwise the comparison is confounded and fails closed.

The positive control is a designed sensitivity case, not a claim that every valid PE observation must change learner state. Real task utility is measured later on separate matched task panels; correctness and utility may not be collapsed into one score.

## Reproduce

From the repository root:

    python3 docs/research/pev_counterfactual_influence_benchmark_v1.py --self-test --cases 100000 --seed 20261028 --output /tmp/pev_counterfactual_influence_benchmark_v1_results.json

The local run accepted 100,000 generated paired trials (900,000 branch summaries: one baseline, seven invalid variants, one valid-PE positive control per trial) and detected all seven injected faults: invalid-payload state divergence, valid-PE no-op learner, missing positive-control admission, duplicate admission, non-target collateral change, counterfactual input mismatch, and independent-outcome suppression.

## Security and epistemic limits

- State digests are only as trustworthy as the component that emits them. For production evidence, instrument the actual Rust consumer boundary, include accepted evidence IDs and before/after state snapshots or digests, and independently recompute digests wherever practical. A self-reported pe_influenced flag is insufficient.
- This benchmark consumes summaries; it does not execute Symthaea Rust or prove the call graph emits honest/complete events.
- Identical state digests are a strong equality check only if canonicalization, state coverage, and digest roots are defined. Hidden side effects, unrecorded channels, nondeterministic execution, or digest omissions can evade it.
- Use fixed seeds and deterministic replays for hard correctness checks. For stochastic production paths, repeat paired seeds and estimate effects with paired confidence intervals; do not demand byte equality for fields intentionally stochastic.
- Never hash raw prompts, secret values, or private content just to achieve provenance. Digest a minimized, documented state schema.

## Industry comparison and next acceptance gates

New work on agent harness evaluation separates task success from harness reliability, because task success can coexist with a broken harness (TraceTrial-Bench, https://arxiv.org/abs/2610.32285). Current drift-evaluation work likewise emphasizes timing-aware detection and leave-one-dataset-out tuning instead of single synthetic scores (KDD 2026 framework, https://arxiv.org/abs/2606.07789).

For Symthaea, report separate scorecards:

- **Integrity:** invalid/sentinel leakage, duplicate/stale/future/cross-reset admissions, orphan state transitions, and mutation detection.
- **Causal channel isolation:** invalid PE equals no-PE outputs; valid PE is admitted in positive controls; protected channels and outcome calibration remain invariant.
- **Learning utility:** held-out decision quality, adaptation/regret, recovery delay, retention/forgetting and calibration against matched baselines.
- **Efficiency:** CPU/GPU time, memory, trace volume, and audit overhead at fixed task/quality budgets.

Proposed release gates—not industry standards or achieved results—are zero integrity violations in deterministic adversarial/property tests, 100% detection of the registered mutation suite, no positive-control regressions, and task-utility superiority assessed only after hard gates pass. A candidate target is at least 5% relative improvement on a pre-registered dataset-macro utility measure with paired 95% confidence intervals excluding zero, under matched compute/memory budgets. Publish all datasets, worst-task regressions, confidence intervals, and negative controls.

## Next production step

Add Rust tests around the actual PredictionErrorEvidence path that run the same observation/action/seed from the same initial cognitive state in no-PE, invalid-PE payload, and valid-PE positive-control modes. Emit machine-readable summaries from the real CodingAgent and PosthocOutcome consumer boundary. Run this independent auditor against those files on the exact PR head, including reset and checkpoint/restore tests. Keep the research PR draft until formatting, compilation, focused tests and real trace audit execute successfully.
