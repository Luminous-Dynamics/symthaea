# Two-wheeler validation scorecard — matched, held-out energy pairs

**Status:** v0.1 contract and synthetic oracle only; no physical measurements added.  
**Related work:** PR #8045; wheel-demand and signed-work traces in the same benchmark family.  
**Claim ceiling:** arithmetic/identity-control conformance is not model validation, vehicle qualification, or a measured performance result.

## Why this is the next tranche

A published 2026 South African electric-motorcycle digital twin reports validation against six measured drive cycles. It reports battery-power correlation of 0.905–0.981, motor-speed correlation of 0.912–0.996, and energy within 2.71% on five cycles; one short downhill/high-throttle-transient cycle has a 10.65% relative error. The paper also describes scope exclusions, including mechanical braking inputs. It is an existing baseline, not our experiment, and its paper summary is not a substitute for rights-cleared raw telemetry and component maps. See https://doi.org/10.3390/wevj1708432 and https://www.mdpi.com/2032-6653/17/8/432.

The next step is not to duplicate its digital twin. It is to make the evidence comparison hard to game.

## Metric contract

A pair is scoreable only when:
- observed and predicted cycle IDs match;
- vehicle-configuration digests and mission digests match;
- the model input trace digest equals the measured trace digest;
- the measured and predicted energy boundaries are identical;
- observed and predicted distances are positive and agree within the declared numerical tolerance;
- all quantities are finite and both reported consumption values are nonnegative under this metric contract.

Each scoreable pair reports:
- measured and predicted Wh/km;
- signed error (prediction minus observation) in Wh and Wh/km;
- absolute error in Wh and Wh/km;
- absolute relative energy error percentage, only when observed energy is greater than zero.

When measured energy is zero, relative percentage error is undefined and emitted as null; absolute errors remain available. Never average across mismatched energy boundaries (for example, battery-terminal energy vs positive mechanical wheel work).

## Evidence eligibility is separate from arithmetic

A fixture pair can be scored for test purposes without being eligible as validation evidence. An eligible validation pair additionally requires a physical evidence class (InstrumentedPhysical or FieldObserved), a HOLDOUT split, non-empty measurement-protocol, source-record, source-rights-receipt and split-manifest identities, and a calibration receipt for instrumented measurements. The reserved SYNTHETIC_PLACEHOLDER value always disqualifies a row. Identity digests bind labels/bytes; they do not prove source authenticity, correct sensors, legal rights, or independent custody. The receipts must be independently audited; their presence in JSON is not proof that they are valid.

The summary aggregates **only eligible pairs** and reports count, mean absolute Wh/km error, median absolute error, nearest-rank P90, maximum error plus its cycle ID, and signed mean bias. A zero-eligible-pair result emits null metrics; it does not produce a vacuous pass. Keep every per-cycle residual beside any summary: means and correlations can obscure a short-trip outlier.

Do not tune thresholds after observing the holdout. Pre-register the observable, energy boundary, route inclusion rules, error threshold, uncertainty handling, number of independent cycles and rules for investigating outliers. The present fixture has no physical rows and therefore cannot estimate model validity or propose an achieved error number.

## New files and command

- xtask/src/two_wheeler_validation_scorecard.rs: pair validation, residual metrics, physical-holdout eligibility, eligible-only summary, six unit tests authored.
- docs/research/data/two_wheeler_validation_scorecard_v0_1.json: 11 synthetic cases, four positive controls and seven negative controls.
- xtask two-wheeler-validation-scorecard: recomputes the frozen expected cases and prints a report.

From the workspace root, run:
    cargo test -p xtask two_wheeler_validation_scorecard
    cargo run -p xtask -- two-wheeler-validation-scorecard --input docs/research/data/two_wheeler_validation_scorecard_v0_1.json --out reports/two_wheeler_validation_scorecard.json

These commands have not been executed in this environment. A successful run would show synthetic fixture conformance only.

### Scoring a rights-cleared measured dataset

Use a separate input manifest with `status: "measured_dataset"`, root `evidence_class` set to either `InstrumentedPhysical` or `FieldObserved`, and the same `metric_contract`. Each case uses a non-empty class such as `measurement`, the same pair fields as the fixture, and omits all oracle fields (`expected_status`, `expected_error`, `expected_output`, `absolute_tolerance`, and `relative_tolerance`). Every row’s evidence class must match the root class. A split other than `HOLDOUT` can be scored diagnostically but does not enter the validation summary. A rejected row is retained in the report; only scoreable rows that satisfy the physical-holdout gate contribute to summary statistics. Split instrumented and field-observed evidence into separate manifests.

The measured-data mode does not certify the authenticity of a measurement or the authority to share it. Preserve source rights, calibration evidence, raw-data custody, and independent evaluation before calling any result a physical validation.

## Physical benchmark protocol to target next

1. Obtain rights-cleared raw telemetry and full model/component maps from a suitable existing study or instrument one reference vehicle. Preserve source licenses/consent and raw checksums.
2. Freeze vehicle/BOM, tires, rider/posture, payload, ambient conditions, grade/wind, route, start/end state, sampling rate, and calibration receipts.
3. Declare the energy boundary before collecting data. Meter energy at the same boundary the model predicts; record auxiliaries and both electrical and mechanical braking where applicable.
4. Split by **entire drive cycle/route and preferably day/rider**, not by random rows from the same trace. Keep model tuning/calibration and the final holdout separate.
5. Publish per-cycle energy error, Wh/km error, correlation only as a supplemental time-series metric, uncertainty, sample count, and all outliers. Short trips must not disappear because percentage error is unstable at low denominators.
6. Expand only after baseline validity: temperature/battery state-of-charge, ageing, repeated runs, payload, wind, surface, auxiliary loads, mechanical braking and component tolerances should each be added as separately identifiable dimensions.

## Proposed benchmark gates—not standards

- **Identity gate:** zero accepted pairs with configuration, mission, trace or energy-boundary mismatches in the frozen negative controls.
- **Evidence gate:** zero synthetic/source-summary/training pairs counted as physical holdout validation.
- **Residual gate:** report per-cycle absolute Wh and Wh/km; do not publish an overall accuracy claim until an adequately powered, independently held-out measurement campaign and uncertainty budget exist.
- **Outlier gate:** disclose and investigate every route above the pre-registered tolerance, even if aggregate median passes.
- **Reproducibility gate:** frozen raw-input digest, model commit/configuration, split manifest, measurement protocol and report digest.

## Relevant standards and limitations

ISO 4210-2:2023 covers specified city/trekking, youth, mountain and racing bicycle classes, but it explicitly excludes several specialized types including delivery bicycles and tandems. Do not silently claim that a cargo-bike, e-bike, or motorcycle profile is covered; select the category-specific standards and market requirements before setting physical acceptance limits: https://www.iso.org/standard/78077.html. UN Regulation No. 136 is the UN regulation for electric vehicles of category L; verify the current revision and target-market adoption for an actual motorcycle approval programme: https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-121-140.

This scorecard measures prediction residuals only. It is not a safety, structural, braking, roadworthiness, homologation, battery-safety, or type-approval certificate.
