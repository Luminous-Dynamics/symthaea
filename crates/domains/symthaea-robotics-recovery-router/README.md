# symthaea-robotics-recovery-router

A small, deterministic reference router for selecting among compatible recovery policies using training-only, cross-fitted competence evidence.

## Contract

The router receives a runtime observation-context bucket derived from deployable observations/history; exact feature-space, embodiment, action-interface and inference-runtime identities; candidate policy identities and pinned compatibility receipts; mutually exclusive outcome counts from a frozen training-only, cross-fitted aggregate; and p95 inference latency measured on the exact runtime environment.

It returns either a selected policy identity with confidence bounds and evidence receipt digests, or an explicit abstention with per-candidate rejection reasons.

It never returns an action, trajectory, torque, actuator command, or safety authority. Selected policy proposals still require independent action projection and viability/safety admission.

## Confidence and risk rule

For each candidate, form Wilson confidence intervals using the configured z value:

    safe recovery lower bound = WilsonLower(safe recoveries, attempts)
    adverse-or-unverified upper bound = WilsonUpper(adverse or unverified, attempts)

The outcome denominator is the sum of three mutually exclusive classes:

1. safe_recoveries: policy-attributable task completion, physical continuity preserved, and upstream safety evidence verified compliant;
2. adverse_or_unverified: any adverse event or unknown/unverified safety or physical continuity;
3. known_non_completions: known task non-completion without a separately recorded adverse event.

Infrastructure-indeterminate rows are excluded from the classified-attempt denominator but remain explicitly counted and bound into the aggregate-evidence receipt. They must not be silently dropped or relabelled as policy failures.

A candidate is eligible only if all provenance, training-only, cross-fit, context, embodiment, action-interface, runtime, sample-count, latency and statistical gates pass. Among eligible policies, selection is lexicographic: highest safe-recovery lower confidence bound, lowest adverse-or-unverified upper confidence bound, lowest measured p95 latency, then lexical policy ID for deterministic ties.

No eligible policy means abstention. No hidden weighted safety score is introduced.

The conservative v0 preset uses 30 minimum attempts, 0.60 safe-recovery lower bound, 0.05 adverse-or-unverified upper bound and z=1.96. These are initial research gates, not regulatory thresholds or a safety certification. With zero observed adverse outcomes, the 95% Wilson upper bound remains above 5% at 40 attempts and falls below 5% at 73 attempts. Sparse evidence should therefore produce abstention, not unjustified confidence.

## Leakage and evidence boundaries

- The profile evidence scope must be TrainingOnlyCrossFitted.
- training_manifest_sha256, crossfit_receipt_sha256 and aggregate_evidence_sha256 bind the training sample, cross-fitting procedure and resulting counts.
- Context keys must be generated from observations and permitted history—not evaluator-only Stage/Type, hidden task predicates, source-policy labels, or held-out outcomes.
- The crate checks digest shape and exact identity agreement. It does not independently verify that the producer actually performed cross-fitting or that upstream safety labels are valid; those receipts require independent audit.
- Profiles are per policy version and context cell. Different checkpoints cannot be pooled by giving them the same display name.

## Research placement and benchmark ladder

This is the first deterministic reference baseline for Symthaea #11545. Compare it under matched observations, policy portfolio, inference budget and execution semantics against the strongest fixed policy; a training-selected static policy; a fixed heuristic; a calibrated learned selector; an observation/history-conditioned HDC/CfC selector; Backtrack-style failure prediction plus physical retracing for frozen policies; published monitor/correction baselines where exact implementations are available; and the retrospective per-scenario oracle as evaluator-only upper bound.

A positive result requires paired held-out improvement in Overall RSR and preservation of Macro-9 RSR, no increase in adverse outcomes, correct selective abstention under out-of-support contexts, and matched latency/compute. The router itself does not establish physical recovery; the RoboRecover audit/scorer or an equivalent independently reviewed evidence pipeline must produce its inputs.

## Validation status

- Twenty-one unit tests are included.
- They have not been executed in this environment because rustc, cargo and rustfmt are unavailable here.
- The workspace lockfile contains the new dependency-free package entry; exact-head CI/format/test evidence is still pending.

## Nonclaims

This crate does not train a router, establish calibration, validate the cross-fit receipt, verify robot sensors, implement backtracking, control an arm, or certify any safety property. It is a deterministic selection baseline over supplied evidence.
