# HARDSEC-009: Validation-Frozen Selective Risk and OOD Triage

**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/9202  
**Evaluator:** `examples/hardsec_selective_eval.rs`  
**Cargo feature:** `hardsec-selective-eval`

## Research question

Can a hardware-Trojan detector maintain low false-alert and false-clear rates on genuinely held-out host/source families while abstaining on ambiguous cases and routing those cases to deeper evidence collection?

This is a change in the unit of success. A binary classifier is not required to label every design. It is required to:
- clear only within an explicitly bounded false-clear budget;
- alert only within an explicitly bounded false-alert budget;
- abstain when neither conclusion is justified;
- report its calibration and per-family behavior under shift;
- keep the expensive formal/simulation/human-review path inside a measurable escalation budget.

The evaluator is not a detector, model trainer, probability calibrator, or formal proof engine. It evaluates a frozen score ledger.

## Why this benchmark is needed

Two recent studies make random or instance-level performance an inadequate confirmatory target:

1. Xiao et al., *Protocol effects on feature-based hardware-Trojan detection across Trust-Hub families* (2026), evaluate 49,124 gates from 16 netlists and five base-circuit families. Their Random Forest F1 changes from 0.914 with pooled validation to 0.636 leave-one-circuit-out and 0.460 leave-one-base-family-out. This demonstrates that apparent performance can substantially depend on the split protocol.
   - https://research.google/pubs/protocol-effects-on-feature-based-hardware-trojan-detection-across-trust-hub-families/

2. Xu et al., *SeqCube-HT: state-cube-guided generation of sequential hardware Trojan benchmarks* (published 9 October 2026), report graph-probe F1 changing from 69.2% on instance-random evaluation to 47.8% on insertion-source-held-out evaluation; localization recall changes from 55.7% to 31.4%. Its covariate-matched protocol is different and must be reported separately.
   - https://doi.org/10.1186/s42400-026-00676-2

These are not directly interchangeable experiments. They motivate the same methodological constraint: report transfer conditions explicitly and do not hide distribution shift inside a pooled score.

A third 2026 study on explainability reports that generic per-gate feature attribution can have only moderate agreement between methods and may lack circuit-level context. HARDSEC therefore treats localization, explanation actionability, and selective decision quality as separate endpoints.
   - https://doi.org/10.1007/s10836-026-06245-2

## Input contract

The evaluator accepts a strict JSON ledger with schema version `hardsec-score-ledger-v1`:

```json
{
  "schema_version": "hardsec-score-ledger-v1",
  "dataset_id": "dataset-version",
  "grouping_key": "insertion_source_family",
  "score_semantics": "probability_of_trojan",
  "model_sha256": "<64 hex characters>",
  "config_sha256": "<64 hex characters>",
  "predictions": [
    {
      "sample_id": "netlist-0001",
      "group_id": "source-family-A",
      "split": "validation",
      "label": 0,
      "score": 0.012
    }
  ]
}
```

Each row contains a unique sample ID, one held-out group ID, a split, a binary label and a score in [0,1]. The grouping key should describe the unit being held out (for example host family, Trojan family, insertion source, standard-cell library, technology node, or a declared combination).

Fail-closed rules:
- duplicate sample IDs are rejected;
- the same group ID may not appear in more than one of train, validation and test;
- all three splits must be populated and contain both classes;
- labels must be integer 0 or 1;
- scores must be finite and in [0,1];
- model and configuration digests must be valid SHA-256 values;
- unknown split names or score semantics are rejected.

Group exclusivity prevents an evaluator from silently accepting an instance-random split while the run claims to be family-held-out. It cannot prove that the declared group labels are truthful or sufficiently granular.

## Validation-frozen three-way decision

The operator supplies two constraints before evaluation:
- `max_false_alert_rate`: maximum fraction of benign samples incorrectly classified as suspect;
- `max_false_clear_rate`: maximum fraction of Trojan samples incorrectly classified as clear.

The evaluator searches an operating point on validation only:
- score <= clear threshold -> **clear**;
- score >= alert threshold -> **suspect**;
- between thresholds -> **abstain / escalate**.

It maximizes validation coverage subject to both constraints. A deterministic tie-break prefers fewer total errors, then fewer false alerts, fewer false clears, and a wider abstention band. The all-abstain operating point is always a feasible fallback. Scores tied in the abstention band are not forced into a decision.

The selected thresholds are frozen and applied once to test. Test-set violations are recorded as failed operating constraints; the threshold is never retuned against test data.

The threshold search uses score histograms, prefix counts and suffix counts. Its complexity is O(n log n) for sorting and histogramming, plus a linear scan over unique validation scores; it avoids rescanning all samples for every candidate threshold pair.

## Reported measures

The receipt binds the raw ledger SHA-256, model/configuration digests, grouping key and exact selected thresholds. It reports:

| Dimension | Measures |
|---|---|
| Decision coverage | coverage, abstention rate, clear count, alert count |
| Errors | false-alert rate among benign, false-clear rate among Trojans, selective risk among decided samples |
| Security utility | alert precision, Trojan recall |
| Probability quality | Brier score, log loss, 10-bin expected calibration error |
| Distribution shift | per-held-out-group coverage and error rates, macro group coverage, worst group false-alert and false-clear rates |
| Gate status | aggregate test constraints met, all observed group constraints met |

A score labelled `probability_of_trojan` is not presumed calibrated. Brier, log loss and ECE are reported on the untouched test scores as diagnostics. ECE uses 10 equal-width bins; the final bin includes score 1.0. Calibration does not establish future calibration under a new domain or adaptive adversary.

The evaluator's run result is `evaluated`, not `pass`. An evaluated run may fail its operating constraints. The generalization gate is reported independently.

## Proposed benchmark targets

These are project research targets, not published standards, guarantees or measured achievements:

- Zero group overlap across train/validation/test; thresholds selected exclusively on validation.
- Aggregate test false-alert rate <= 0.5% and false-clear rate <= 1%.
- Macro held-out-group coverage >= 80% while respecting both aggregate caps; group-level violations remain visible even when aggregate rates pass.
- Test ECE <= 0.03, with Brier and log loss reported.
- At a fixed review/escalation budget, improve macro held-out-family coverage by >=10 absolute points over a binary threshold-only baseline without increasing the false-clear rate. Claim improvement only when a paired group-level bootstrap interval excludes zero.
- Publish the complete coverage–false-alert–false-clear Pareto frontier whenever the requested operating point is infeasible or fails on test.

The 0.5% / 1% constraints are stringent development gates. Small test sets cannot substantiate such low rates. Report denominators and confidence intervals; do not imply that zero observed errors means the true error rate is zero.

## Comparison matrix

Every baseline must use identical host/source groups, validation/test partitions, score ledger schema and review budget:

1. **Binary threshold-only:** conventional classifier at one validation-selected threshold.
2. **Selective structural baseline:** RF/XGBoost/GCN/GAT/GraphSAGE with the same clear/suspect/abstain rule.
3. **Symthaea detector:** HDC/CfC or other proposed scoring model, no special split or threshold privileges.
4. **Calibrated selective model:** any probability calibration fitted only on training/validation, then frozen before test.
5. **Evidence cascade:** inexpensive detector first; only abstained/suspect cases escalate to directed replay, bounded formal analysis, equivalence checks, or human review.

Compare at equal false-alert rate, equal false-clear rate, equal coverage, equal escalation/review budget and equal end-to-end compute budget. Report mean and p95 escalation cost, wall time, peak memory and review volume in the cascade study. The present evaluator measures prediction quality, not escalation cost; those costs must be joined through a separate exact-run receipt.

## Industry alignment

This should complement established assurance practice rather than replace it.

OpenTitan's design-verification methodology combines constrained-random simulation and formal property verification, with verification at IP and chip levels. The appropriate Symthaea integration is an auditable triage layer that helps prioritize those established checks—not a classifier claiming to replace them:
https://opentitan.org/book/doc/contributing/dv/methodology/index.html

NIST IR 8615 (published September 2026) emphasizes lifecycle evidence, provenance/traceability, interoperable assurance and scalable verification/validation. The score ledger and evidence packet can contribute portable records, but neither is a certification scheme:
https://csrc.nist.gov/News/2026/next-generation-secure-hardware-into-standards

## Reproduction

```bash
cargo test --locked \
  --manifest-path tools/hardsec-audit-harness/Cargo.toml \
  --features hardsec-selective-eval \
  --example hardsec_selective_eval

cargo run --locked \
  --manifest-path tools/hardsec-audit-harness/Cargo.toml \
  --features hardsec-selective-eval \
  --example hardsec_selective_eval -- \
  --ledger /path/to/predictions.json \
  --receipt /path/to/new-receipt.json \
  --max-false-alert-rate 0.005 \
  --max-false-clear-rate 0.01
```

No detector training or benchmark results are created by running the evaluator alone. A valid score ledger and exact model outputs from independently run baselines are required for comparative claims.

## Current implementation boundary

Implemented:
- strict ledger schema and hash validation;
- group-exclusive split enforcement;
- deterministic validation-only threshold selection;
- frozen clear/suspect/abstain evaluation;
- aggregate and per-group error/coverage metrics;
- Brier, log loss and 10-bin ECE diagnostics;
- immutable, unsigned receipt creation with raw ledger digest.

Not implemented by this evaluator:
- model fitting or probability calibration;
- bootstrap confidence intervals;
- cost accounting for the formal/simulation/human escalation cascade;
- evidence that model scores predict real-world Trojan prevalence;
- proof of Trojan presence or absence.

Those belong in distinct, preregistered experiments rather than being silently folded into one score.
