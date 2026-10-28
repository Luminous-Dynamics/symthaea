# VIS-DVPC-027 — Counterfactual support audit v1

**Status:** standalone structural-support and overlap diagnostic. It is not an off-policy estimator certification and never authorizes a policy.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8713

## Why this exists

In selective-feedback campaigns, an action outcome is observed only when that action was selected. The outcome for an unselected action is counterfactual. Scoring all policies as though the logger had observed every action can silently turn missing counterfactual outcomes into an unjustified policy ranking.

This audit checks structural positivity using declared target-policy and logging-policy probabilities, verifies that reward values appear only on selected rows, and reports overlap diagnostics. File-based runs require an expected SHA-256 digest and emit the digest of the exact bytes audited; the bundled known-answer fixtures are hash-pinned. It deliberately returns `NOT_IDENTIFIED` whenever target-policy mass lies on an action whose declared logging propensity is zero. Weak but nonzero overlap is reported separately.

The tool does **not** verify that logging propensities are true, repair unobserved confounding, derive confidence intervals, adjust for adaptively collected/clustered observations, or establish target transportability. Declared cluster sizes are descriptive only. A supported result is explicitly diagnostic-only.

## Input contract

UTF-8 TSV, one row per available context/action pair, with these columns in this order:

```text
context_id action_id target_probability logging_probability selected reward cluster_id
```

The file uses tab separators. For every context, target probabilities and logging probabilities must each sum to one, and exactly one logged action is marked `yes`. Only the selected row may contain a numeric reward normalized to [0,1]; all unselected rewards must be `NA`. Propensities must come from a logging-policy/evaluator receipt, not be guessed from outcome frequency.

## Verdicts

- `NOT_IDENTIFIED`: at least one action has positive target probability and zero logging probability. The IPS point estimate is suppressed.
- `SUPPORTED_BUT_WEAK_OVERLAP`: no structural zero was found, but effective sample size is below 5 or the maximum supported target/logging weight exceeds 10. These v1 thresholds are diagnostic defaults, not universal cutoffs.
- `SUPPORTED_DIAGNOSTIC_ONLY`: the structural support and fixed overlap screens pass. This is **not** a guarantee of unbiasedness, correct interval coverage, or policy safety.

The reported IPS quantity is a diagnostic point estimate only when there is no structural support gap. It is not a certification. Effective sample size is `(sum observed weights)^2 / sum(observed weight^2)`; it is a descriptive diagnostic, not an independence-adjusted sample size.

## Run

```sh
rustc --edition=2024 --test tools/vision-dvpc-support-audit.rs -o /tmp/vis-dvpc-027-tests
/tmp/vis-dvpc-027-tests
rustc --edition=2024 -O tools/vision-dvpc-support-audit.rs -o /tmp/vis-dvpc-027
/tmp/vis-dvpc-027 --audit tests/fixtures/vision-dvpc-027-supported.tsv $(sha256sum tests/fixtures/vision-dvpc-027-supported.tsv | cut -d ' ' -f1)
/tmp/vis-dvpc-027 --audit tests/fixtures/vision-dvpc-027-zero-support.tsv $(sha256sum tests/fixtures/vision-dvpc-027-zero-support.tsv | cut -d ' ' -f1)
```

## Known-answer controls

- A supported 12-context fixture: three actions per context; target probabilities (0.2, 0.5, 0.3), logging probabilities (0.6, 0.3, 0.1), seven observed A actions, four B actions and one C action. This passes the v1 structural and overlap diagnostics.
- A zero-support fixture: target mass 0.3 lies on action C while its logging probability is zero in every context. Expected verdict: `NOT_IDENTIFIED`; the IPS point estimate is suppressed.
- A weak-overlap fixture: target/logging ratio reaches 50 despite no exact-zero support. Expected verdict: `SUPPORTED_BUT_WEAK_OVERLAP`.
- Malformed inputs, leaked unselected outcomes, inconsistent probability sums, impossible selected rows, duplicate context/action rows, and repeated declared clusters are tested fail-closed or reported.

## Evidence boundaries

This is a structural preflight only. Before using it in VIS-DVPC-025, bind trace bytes, target policy, logging-policy revision, propensity provenance, and evaluator version to a receipt. The expected digest must come from a trusted upstream manifest or receipt; calculating the expected hash from the same untrusted file immediately before auditing detects intervening changes but does not authenticate the source. Real-trace policy-superiority claims require adequate overlap and a justified off-policy inference procedure, or prospective controlled confirmation. Do not reinterpret synthetic fixtures as real-world evidence or SOTA results.
