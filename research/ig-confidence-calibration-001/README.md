# IG-CONFIDENCE-CALIBRATION-001 — held-out probability calibration and selective risk

**Status:** synthetic evaluation fixture / research proposal; no real model calibration executed  
**Owner:** [Symthaea #11367](https://github.com/Luminous-Dynamics/symthaea/issues/11367)  
**Composes with:** [revision ledger #11340 / PR #11350](https://github.com/Luminous-Dynamics/symthaea/pull/11350), [stance protocol #11320](https://github.com/Luminous-Dynamics/symthaea/issues/11320)

## Research question

When an agent communicates a probability that its factual answer is correct, does that probability predict held-out correctness for the exact model revision, prompt family and task scope—and does using it improve selective decision quality without making confidence an authority signal?

A confidence number is not evidence. A calibration report digest is a commitment to an external artifact, not proof that the artifact is valid.

## Evaluation-only v0.1

This first slice evaluates post-calibration predictions on a synthetic known-answer test set. It **does not fit** Platt scaling, isotonic regression, beta calibration or temperature scaling. The method and external calibration report are profile metadata; the scorer checks their structural binding and evaluates independent held-out outcomes.

The 50 synthetic observations form two confidence groups:
- 25 forecasts at 0.8, of which 20 are correct;
- 25 forecasts at 0.2, of which 5 are correct.

The declared metric oracle is:
- Brier score: 0.16;
- log loss: approximately 0.50040242;
- 10-bin equal-width ECE: 0.0;
- reliability bins: 25 observations each, observed frequencies 0.8 and 0.2.

These are code-verification fixtures, not real model results.

## Provenance and split boundaries

The profile is bound to:
- provider/model/revision;
- prompt-family digest;
- exact factual correctness task scope and probability scale;
- calibration method and immutable profile/report digests;
- fit, validation and evaluation dataset commitments;
- disjoint task-group IDs for fit, validation and test;
- external validator identity/version and calibration sample count.

The test evaluator recomputes SHA-256 over the canonical compact JSON observation array. It verifies that each test row belongs to an explicitly declared held-out group. A profile that crosses model revision, prompt family, task scope, or split boundaries is rejected.

The fit/validation samples and report are external artifacts at this stage. Their hashes and sample-count declarations are not self-authenticating; a future verifier must inspect the referenced files, check the training procedure, recompute their digests and establish no leakage.

## Selective-risk ties and uncertainty boundary

Selective-risk calculations must be invariant to row order when forecasts have equal confidence. The evaluator now groups equal-probability observations and calculates fractional expected inclusion within a tied block. This avoids an arbitrary top-k result when a coverage cutoff falls inside a tie group. The AURC and risk-at-coverage outputs are therefore tie-aware and deterministic.

The checked-in synthetic evaluation has 50 rows but only **two independent task groups** (high-confidence and low-confidence). The CLI reports that group count and marks the uncertainty interval as `NotEstimated`; it does not treat the 50 rows as 50 independent clusters or emit a misleading confidence interval. A confirmatory real-model evaluation must preregister its independent sampling unit, cluster structure, sample-size/power plan and uncertainty method before evaluation.

## Metrics

The Rust evaluator reports independent measures:
- **Brier score** (proper score for binary correctness probability);
- **log loss** (penalizes confident wrong forecasts);
- **equal-width ECE10** plus per-bin reliability counts/means;
- **risk–coverage points** at 25%, 50%, 75% and 100%;
- **AURC** (area under the discrete risk–coverage curve).

No single combined confidence score is produced. ECE depends on binning; v0.2 should add pre-registered equal-mass ECE and reliability slope/intercept with interval estimates. Calibration performance should be reported with sample counts and confidence intervals on real held-out tasks.

The Rust qualifier can be run from the repository root:

```bash
cargo fmt --manifest-path research/ig-confidence-calibration-001/qualifier/Cargo.toml --check
cargo test --manifest-path research/ig-confidence-calibration-001/qualifier/Cargo.toml
cargo run --manifest-path research/ig-confidence-calibration-001/qualifier/Cargo.toml -- \
  research/ig-confidence-calibration-001/CONFIDENCE_EVALUATION_V0_1.json \
  research/ig-confidence-calibration-001/CONFIDENCE_NEGATIVE_CONTROLS_V0_1.json \
  research/ig-confidence-calibration-001/CONFIDENCE_EVALUATION_V0_1.schema.json \
  research/ig-confidence-calibration-001/CONFIDENCE_NEGATIVE_CONTROLS_V0_1.schema.json
```

## Executable schema contract

The Rust CLI takes both data artifacts and both schema files. It recursively audits the schema trees for unsupported keywords (including optional, currently absent properties), then applies the declared bounded schema subset to the evaluation bundle and negative-control manifest before running the metric and negative-control logic. Additional data fields are rejected where the schema closes the object.

The negative-control mutations are then checked against the semantic validator's expected violation codes. A mutated test case is not reclassified as a valid evaluation; the mutation is only testing that the semantic gate emits the declared failure. The implementation is a bounded subset, not a complete JSON Schema engine.

## Twenty-four deterministic controls

The fixture manifest exercises:
- self-report versus calibrated status;
- malformed/missing profile and report commitments;
- model revision, provider, prompt-family and task-scope mismatch;
- fit/validation/test overlap and dataset digest reuse;
- insufficient sample count;
- unsupported calibration method or scale;
- missing validator identity/version;
- duplicate sample IDs and out-of-range probabilities;
- missing correctness outcomes and unknown test groups;
- normative-pluralism misuse;
- altered test-data digest and single-outcome-class evaluation.

## Benchmark arms to add next

On matched task instances, agent count, model/token/tool budget and communication topology compare:
1. no confidence message;
2. raw self-reported confidence;
3. externally calibrated confidence;
4. shuffled confidence labels (negative control);
5. adversarial confidence inflation / strategic overclaim;
6. confidence hidden from peers;
7. diversity-aware versus homogeneous initial answer sets;
8. confidence-aware updates versus evidence-only updates.

Confidence-based updates must be compared against evidence-only baselines. A calibrated value is valid only within its registered model/prompt/task scope; out-of-scope requests return NotCalibrated/Unknown.

## Claim ceiling and civic boundary

A PASS means only that the synthetic held-out scoring/provenance implementation meets this bounded contract. It does not demonstrate calibration of any production model, calibration transfer, human confidence calibration, causal improvement in debate, or safe use in real decision-making.

Neither self-reported nor calibrated confidence changes civic base vote weight, rights, eligibility, authority, mandate, credibility or moderation priority. Normative-pluralism tasks are excluded from binary correctness calibration.

The Rust CLI takes both data artifacts and both schema files. It recursively audits the schema trees for unsupported keywords (including optional, currently absent properties), then applies the declared bounded schema subset to the evaluation bundle and negative-control manifest before running the metric and negative-control logic. Additional data fields are rejected where the schema closes the object.

The negative-control mutations are then checked against the semantic validator's expected violation codes. A mutated test case is not reclassified as a valid evaluation; the mutation is only testing that the semantic gate emits the declared failure. The implementation is a bounded subset, not a complete JSON Schema engine.

## Twenty-four deterministic controls

The fixture manifest exercises:
- self-report versus calibrated status;
- malformed/missing profile and report commitments;
- model revision, provider, prompt-family and task-scope mismatch;
- fit/validation/test overlap and dataset digest reuse;
- insufficient sample count;
- unsupported calibration method or scale;
- missing validator identity/version;
- duplicate sample IDs and out-of-range probabilities;
- missing correctness outcomes and unknown test groups;
- normative-pluralism misuse;
- altered test-data digest and single-outcome-class evaluation.

## Benchmark arms to add next

On matched task instances, agent count, model/token/tool budget and communication topology compare:
1. no confidence message;
2. raw self-reported confidence;
3. externally calibrated confidence;
4. shuffled confidence labels (negative control);
5. adversarial confidence inflation / strategic overclaim;
6. confidence hidden from peers;
7. diversity-aware versus homogeneous initial answer sets;
8. confidence-aware updates versus evidence-only updates.

Confidence-based updates must be compared against evidence-only baselines. A calibrated value is valid only within its registered model/prompt/task scope; out-of-scope requests return NotCalibrated/Unknown.

## Claim ceiling and civic boundary

A PASS means only that the synthetic held-out scoring/provenance implementation meets this bounded contract. It does not demonstrate calibration of any production model, calibration transfer, human confidence calibration, causal improvement in debate, or safe use in real decision-making.

Neither self-reported nor calibrated confidence changes civic base vote weight, rights, eligibility, authority, mandate, credibility or moderation priority. Normative-pluralism tasks are excluded from binary correctness calibration.
