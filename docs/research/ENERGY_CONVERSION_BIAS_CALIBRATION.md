# Differential-Bias Calibration and Coverage Contract

**Status:** proposed calibration protocol; no physical calibration dataset is bundled  
**Related implementation:** [candidate-decision benchmark](../../crates/domains/symthaea-energy-bench/src/candidate_decision_benchmark.rs)  
**Acceptance targets:** [Energy-Conversion Benchmark Targets](ENERGY_CONVERSION_BENCHMARK_TARGETS.md)

## The key distinction

The comparator's registered `maximum_unmodeled_differential_bias_efficiency` is a worst-case absolute bound on a **candidate-minus-baseline** efficiency discrepancy. It is not a standard deviation, a confidence interval, or evidence that the bound was calibrated. The comparator subtracts the bound linearly after the expanded measurement-uncertainty term:

```text
conservative lower bound
  = observed(candidate - baseline efficiency difference)
  - expanded measurement uncertainty
  - registered worst-case differential-bias bound
```

A calibration basis string helps a reviewer understand a claim, but it does not authenticate the referenced evidence or prove coverage. A zero bound is not justified by a lack of observed bias. If a relevant effect is known and can be corrected, document the correction and the uncertainty remaining after correction. NIST distinguishes the systematic error from uncertainty about its correction; conflating them can make an uncertainty statement misleading ([NIST TN 1297, combined standard uncertainty](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-5-combined-standard-uncertainty), [NIST, measurement uncertainty and uncorrected bias](https://www.nist.gov/publications/measurement-uncertainty-and-uncorrected-bias)).

## Oracle benchmark: make coverage observable

The candidate-decision negative-control harness has known synthetic truth and an independently selected observed efficiency delta. For each scenario define the injected synthetic discrepancy:

```text
injected_discrepancy = observed_efficiency_difference
                     - oracle_true_efficiency_difference
covered = abs(injected_discrepancy)
          <= registered_bias_bound + floating_point_tolerance
```

This is a synthetic oracle quantity, not an inferred property of a physical device. Because the generator deliberately injects the discrepancy, it can score whether the configured bound covers it.

Report the false-superiority rate separately for oracle-non-superior cases where the bound covers the discrepancy and where it does not. Under the benchmark's stated assumptions, **covered non-superior cases must have zero false-superiority claims**: when the bias is bounded and the measurement-uncertainty allowance is applied correctly, the conservative lower bound cannot exceed the true difference. Any such claim is an implementation or benchmark-contract failure. Undercovered cases are deliberate adversarial controls; they must remain visible to show that a bound which is too small can leave false claims possible.

This property is narrower than real-world safety. It does not prove that physical systematic bias is known, that the supplied reference truth is authentic, that stochastic uncertainty is calibrated, or that model discrepancy is stationary. The scenario truth and all digests are test inputs, not externally authenticated evidence.

## Physical calibration procedure

### 1. Define the differential measurand

Register the technology-appropriate decision metric, energy boundary, baseline, candidate, mission schedule, instrument chain, and operating envelope before viewing results. The bias quantity of interest is the differential error in the candidate-minus-baseline comparison. Shared sensors and common environmental effects may cancel in a paired comparison—or can interact differently with the two candidates—so neither independence nor cancellation should be assumed without evidence.

Do not compare engine efficiency to heat-pump coefficient of performance. A common discovery workflow may orchestrate multiple technology families, but each family requires an explicit metric, boundary, reference condition, and acceptance rule.

### 2. Characterize and correct recognized effects

Identify likely influences such as calibration drift, sensor nonlinearity, thermal soak, hysteresis, flow-meter range, transient lag, installation geometry, auxiliary loads, fuel/electrical input basis, and ambient conditions. Apply justified corrections where practical and preserve the uncorrected observations, correction, method, and residual uncertainty. Keep known systematic effects separate from random repeatability and from model-form discrepancy.

### 3. Calibrate on paired, independent reference cases

Collect candidate and baseline results under the same registered mission and comparable boundaries, against an independently traceable reference where available. Include the intended operating range, transients, load changes, ambient conditions, instrument changes, and aging states that could materially change the differential residual.

Split data by independent unit or condition (for example, device, test campaign, sensor chain, or operating regime) rather than randomly splitting near-duplicate samples only. Freeze held-out reference cases before fitting a correction or choosing the bound. Report residual structure and stratified coverage; a good global average must not conceal a failure in a safety- or performance-critical stratum.

### 4. Set a defensible bound

The bound must be a preregistered conservative envelope for absolute differential systematic discrepancy over the stated operating envelope, with a documented construction method and evidence. Do not simply call the largest residual in a small sample a worst-case physical bound. If the bound is statistical, state its coverage probability, confidence level, sampling assumptions, treatment of clustered observations, and limitations. An empirical maximum is evidence about observed cases, not proof about unobserved conditions.

Propagate measurement uncertainty with justified covariance assumptions. If the measurement model is nonlinear, distributions are non-Gaussian, or first-order approximation is inadequate, compare against Monte Carlo propagation as described in [JCGM 101:2008](https://www.bipm.org/en/doi/10.59161/jcgm101-2008). Unknown or unsupported correlations must remain indeterminate; do not force them to independence for convenience.

### 5. Bind the claim to auditable evidence

The manifest policy now requires `differential_bias_calibration_artifact_id` and `differential_bias_calibration_artifact_sha256`; the verifier requires the ID to appear in `data_provenance` and the declared digest to match that provenance entry. This is **reference consistency**, not retrieval or verification of the artifact bytes, a digital signature, inspection of calibration data, or independent proof that the bound is justified. A future machine-enforced calibration record should also bind at least:

- calibration/reference dataset IDs and content digests (the bias-calibration ID and SHA-256 are now required, but actual byte retrieval/verification remains the caller's responsibility);
- baseline and candidate identities, source revisions, measurement-chain and calibration-standard identities;
- the exact metric, boundary, mission schedule, units, operating envelope, and excluded conditions;
- correction model and version, held-out split definition, residual calculation, and bound derivation;
- sample counts and independent-unit counts, coverage/confidence target, stratified results, and uncertainty/covariance model;
- calibration date, expiration or revalidation triggers, and the conditions that invalidate transport to a new regime.

A free-text basis remains helpful for reviewers but is not a substitute for structured evidence links or independent calibration review. The current verifier implements only the provenance-ID/digest equality check; it does not resolve the URI, retrieve the artifact, recalculate its hash, validate calibration statistics, or authenticate the producer.

## Proposed benchmarks and release gates

1. **Synthetic logical invariant:** zero false-superiority claims across all oracle-non-superior scenarios whose registered bound covers the injected discrepancy. This is a deterministic implementation property, not a field-risk estimate.
2. **Retained negative control:** preserve undercovered cases, including at least one scenario in which an apparent observed win produces a false-superiority result. The test must fail if this adversarial case silently disappears.
3. **Unknown correlation:** 100% of cases requiring unsupported covariance propagation abstain as indeterminate; no affirmative superiority claims.
4. **Held-out decision-risk study:** target a one-sided 95% upper confidence bound of at most 1% on false-superiority probability among representative non-superior reference cases. Zero false claims in 300 independent, identically distributed trials gives an approximate upper bound just below 1%; that arithmetic is not valid unchanged for clustered, repeated-device, or distribution-shifted observations. Size the study and interval method to the actual sampling design.
5. **Calibration coverage:** report bound coverage and residual distributions on held-out cases for each materially distinct operating stratum. For probabilistic uncertainty intervals, report empirical coverage at nominal 90% and 95% and investigate systematic departures. Do not collapse stratified coverage into one flattering aggregate.
6. **Utility / false-negative tradeoff:** report false negatives, indeterminate fraction, and cost to resolution alongside false positives. An enormous bound that prevents every false positive by rejecting every candidate is not a useful discovery system.
7. **Revalidation:** trigger new calibration when the metric, boundary, solver/model revision, instrumentation, operating envelope, reference population, or relevant physical regime changes.

These are proposed gates, not demonstrated results or universal standards. Thresholds and analysis methods must be frozen before held-out outcomes are inspected.

## Implementation status

The candidate-decision harness now emits the oracle-injected discrepancy, coverage of the registered bound, and false-superiority statistics for covered versus undercovered oracle-non-superior cases; the report protocol was bumped for those fields. The benchmark manifest now requires a bias-calibration artifact ID and SHA-256 and checks that both match a provenance entry. The verifier does not fetch the referenced bytes, recompute the artifact hash, authenticate a producer, or assess physical calibration quality. Rust tests have been authored but remain unexecuted in this environment because `cargo` is unavailable. No physical calibration data, calibrated bound, or field-risk estimate is supplied here.
