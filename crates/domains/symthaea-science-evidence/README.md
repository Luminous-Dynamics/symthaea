# symthaea-science-evidence

A cross-domain experiment manifest and verification-receipt contract for
Symthaea research. Current schema: `symthaea-science-evidence-v2`.

## What it binds

- Full source commit SHA and whether the source tree was clean.
- Toolchain and execution platform labels.
- Coordinate and unit conventions.
- Equations, assumptions, configuration and RNG seed policy.
- Content-addressed input and output artifacts using BLAKE3.
- Explicit baseline, primary metrics and uncertainty method.
- A frozen, exact set of required checks.
- Claims, non-claims, and conditions that would falsify the claim.
- Verification status and exact-head log evidence.

## Hard integrity rules

- A source revision must be a full 40-character Git SHA.
- An artifact digest must be a full lowercase BLAKE3 digest over non-empty bytes.
- `ArtifactDigest::verify_bytes` recomputes the digest and size against actual
  bytes; shape validation alone does not authenticate an artifact's contents.
- Input and output artifact paths must be unique within their respective sets.
- Every check must identify the exact manifest source SHA.
- Every receipt must contain exactly the required check names frozen in the
  manifest; missing, duplicated or unexpected check names are rejected.
- A passed check must carry a log artifact digest.
- A passed run must come from a clean source tree, match the manifest toolchain
  and platform, contain every required check and output artifact, and have
  every required check passed.
- Queued, running, skipped, not-run and failed are distinct from passed.

The manifest digest is computed from the versioned serialized schema. Preserve
the schema identifier and exact manifest bytes when archiving the run. The
digest binds the serialized manifest; it does not establish who supplied it.
The required-check list is part of that digest, so a caller cannot silently
remove a check while reusing the same manifest identity.

## Demonstration

The example refuses to infer a commit or environment from hidden defaults.
Supply the exact values explicitly:

    SYMTHAEA_SOURCE_COMMIT=<full-40-char-sha> \
    SYMTHAEA_TOOLCHAIN="rustc 1.96.0" \
    SYMTHAEA_PLATFORM="x86_64-unknown-linux-gnu" \
    SYMTHAEA_SOURCE_TREE_CLEAN=true \
    cargo run -p symthaea-science-evidence --example manifest_demo

The demonstration records every preregistered check as `NotRun`, attaches no
logs or outputs, and validates the receipt structure. It must report
`receipt_valid=true` and `qualifies_as_pass=false`; the example does not
pretend that an experiment was executed.

## Theory prediction coverage and comparability

The crate also exposes a versioned TheoryPredictionLedger for cross-domain
physics/theory comparison. Every theory/observable pair must be recorded as
one of:

- Quantitative
- QualitativeOnly
- NotDerived
- OutsideValidatedDomain
- Unresolved

A quantitative record includes a finite value, units, declared validity domain,
model specification digest, source revision and derivation reference. A
non-quantitative record cannot carry a numeric value. Every ledger must include
an explicit OTHER_UNRESOLVED model identity; it cannot be assigned a convenient
zero prediction by this schema.

Pairwise comparison is deliberately limited to quantitative predictions for
the same operational observable and domain, with calibrated Gaussian
measurement noise. Schema v1 explicitly marks prediction uncertainty as
`PointValueOnly`, `Unspecified`, or `NotApplicable`. A record whose
predictive uncertainty is `Unspecified` is not admitted to pairwise
separation. A `PointValueOnly` record means no model-prediction dispersion
has been supplied; it does not imply certainty that the model describes
nature. Distribution-valued model predictions and correlated predictive
uncertainty require a future schema extension; do not combine variances by
assuming independence without an explicit covariance model.

The reported normalized separation is a point-value screening diagnostic
against the calibrated measurement sigma. It is **not** expected information
gain, a Bayes factor, model posterior probability, or a claim that one theory
is true. The generic EIG/EVSI/Fisher design utilities remain separate and
should consume full calibrated likelihoods only after those are available.

## Model-identity EIG and measurement availability

The `model_discrimination` module computes conditional expected information
gain about a finite set of named model hypotheses from calibrated Gaussian
predictive distributions. It preserves:
- each prior scenario's conditional EIG;
- availability-adjusted EIG;
- expected posterior entropy;
- coarse/refined quadrature error and convergence status;
- worst-case and weighted-mean EIG across prior scenarios.

A candidate carries `availability_probability` in [0, 1]. Under the current
explicit assumption that an unavailable measurement is uninformative,

```
expected EIG = availability_probability * conditional EIG
expected posterior entropy = prior entropy - expected EIG
```

The cost-normalized ranking uses availability-adjusted EIG per cost. It does
not overwrite the conditional EIG. A candidate with zero availability remains
visible with its conditional discrimination score but contributes zero
expected information.

The scalar availability factor is not a general missing-data model. In real
instruments, non-detection, occlusion, visibility, instrument health or radar
cross-section may themselves carry information. Such outcomes must be included
in an explicit likelihood in a future model rather than treated as
uninformative dropout.

The model-recovery reference now applies the same availability probability:
unavailable trials enter the confusion matrix as abstentions, remain in the
accuracy/coverage denominator, and are counted separately per generator.
The recovery report therefore distinguishes performance conditional on
obtaining a measurement from end-to-end performance under the declared
availability model.

Run the synthetic model-identity EIG / predictive-KL reference with:

    cargo run --locked -p symthaea-science-evidence --example model_discrimination_reference

This is a synthetic decision-rule fixture. It is not a physical theory
validation, an empirical sensor-availability estimate, or proof that a named
model is true.

The synthetic known-answer fixture is designed to make two failure modes
visible:

1. The 1+1D Minkowski causal-set ordering-fraction and continuum reference
   both predict 0.5. This is a useful sampler/reference check, but it does not
   discriminate those ontologies.
2. Chain Theory's candidate inverse fine-structure value is recorded as a
   quantitative candidate readout, while the photon propagator remains
   NotDerived. Because the measurement-noise model is not calibrated, the
   candidate value cannot be turned into a likelihood or experiment ranking.

Run the report with:

    cargo run --locked -p symthaea-science-evidence --example theory_prediction_coverage

The example is synthetic/analytic contract validation only. It is not an
independent reproduction of Chain Theory, a physical comparison against QED,
or evidence for a new theory.

## Important boundary

The crate records locally supplied provenance. It does not authenticate a
runner, sign results, establish physical truth, or replace Mycelix attestation.
BLAKE3 proves byte identity only. A passed local simulation is still a local
simulation; a passed numerical benchmark is not an experimental confirmation.

## Tests

    cargo test -p symthaea-science-evidence
