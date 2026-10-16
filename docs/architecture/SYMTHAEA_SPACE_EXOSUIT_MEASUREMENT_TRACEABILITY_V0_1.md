# SX-036: Calibration Traceability and Measurement Completeness v0.1

**Status:** experimental policy checker; synthetic declarations only  
**Implementation:** \`suit_measurement_traceability\`  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/7592  
**Predecessor:** SX-035 evidence-aware peak resource usage measurements, PR #7579

## Paradigm shift: an evidence link is not a provenance chain

SX-035 makes resource forecasts answerable to observations, but explicitly leaves instrument authenticity, calibration, timestamps, uncertainty and trace completeness outside its scope. SX-036 makes those omissions machine-checkable policy inputs.

The new question is not just *“does observed usage exceed the forecast?”* but also *“what declared chain connects this observation to a reference, is that chain valid for the measurement window, how much uncertainty can the chain contribute, and is the sample sequence complete?”*

This follows metrology principles: traceability depends on an unbroken calibration chain whose links contribute uncertainty; traceability by itself does not guarantee fitness for purpose. SX-036 is a software contract inspired by those principles, not a claim of compliance, accreditation or calibrated instrumentation.

## Contract

A traceability plan pins the exact SX-035 measurement-plan digest and measurement-receipt digest. The caller must independently verify the SX-035 receipt before trusting the pin; SX-036 checks the declared digest binding and does not reimplement the upstream mission/task/reservation verifier.

For each SX-035 observation, exactly one claim must bind:
- observation ID, pool, resource kind, provider component and provider hardware revision;
- mission-relative observation window and separate claimed absolute start/end timestamps;
- instrument ID, instrument artifact digest, instrument hardware revision, firmware digest and raw-trace digest;
- first/last sample sequence, sample count, dropped count and maximum observed inter-sample gap;
- an ordered calibration chain, from instrument toward a policy-declared reference standard.

Every calibration link declares the calibrated artifact, reference artifact, certificate digest, resource kind, hardware revision, validity interval, expanded-uncertainty upper bound and caller-declared evidence level.

The evaluator fails closed on missing/extra/duplicate observation coverage, claim-to-observation mismatches, malformed digests, invalid intervals, discontinuous/cyclic chains, missing/mismatched roots, calibration expiry, impossible sample sequence arithmetic, disallowed dropped samples, excessive gaps, uncertainty overflow and uncertainty above the declared ceiling. Findings are sorted deterministically. The receipt uses domain-separated SHA-256 encoding, not a digital signature.

## Uncertainty aggregation

SX-036 uses checked integer arithmetic and sums the declared upper bounds of expanded uncertainty across calibration links. This is a conservative linear sum; it does **not** assume statistical independence or combine uncertainties with root-sum-of-squares. It also does not convert units: each link must already express uncertainty in the same unit/domain as the relevant SX-035 resource kind.

The uncertainty ceiling is a caller policy. Choosing the ceiling and deciding whether the measurement method is fit for a mission remain engineering decisions. An over-ceiling chain is blocked rather than called compliant.

## Time and sample completeness

SX-035 windows are mission-relative milliseconds. SX-036 calibration-validity timestamps are claimed Unix milliseconds and live in a distinct field; the implementation does not silently compare those clocks. Every calibration link must cover the complete declared absolute measurement interval. Timestamps are caller inputs, not trusted-clock attestations.

Sample sequence arithmetic must satisfy:

\`last - first + 1 = reported samples + dropped samples\`

with checked arithmetic. Strict policy blocks dropped samples and any maximum observed inter-sample gap beyond policy. These checks detect declared gaps, not invisible losses that the acquisition system fails to report. SX-036 cannot establish that the raw-trace digest corresponds to physical sensor output without authenticated acquisition and trusted evidence handling.

## Explicit trust boundary

SX-036 does **not** verify:
- digital signatures, certificate issuer identity, trusted timestamping, key rotation or revocation;
- that an artifact digest was issued by the claimed instrument, lab or standards body;
- physical calibration quality, sensor authenticity, measurement accuracy or SI traceability;
- sample acquisition completeness beyond the declared sequence/count/gap evidence;
- whether a measurement is fit for a specific task, or whether a suit/mission is safe.

Hashes make changes detectable relative to an agreed digest. They do not make an untrusted claim true. Evidence levels are caller-declared and never promoted. The report always sets \`represents_physical_qualification = false\`. The implementation is report-only; no scheduling, actuation, load shedding or authorization behavior is introduced.

## Synthetic regression matrix

The module tests cover valid chains, broken links, wrong roots, expired calibration, uncertainty ceiling and overflow, dropped and inconsistent samples, gap violation, provider/revision mismatch, observation coverage, stale receipt pins, low evidence floors, order invariance, deterministic receipt re-evaluation and a deterministic 2,000-case uncertainty-boundary sweep.

The fixtures use synthetic identifiers, digests, values and timestamps. They are not real instrument records. Tests are declared in source; they must not be described as executed until an exact-head CI run or local Rust toolchain has actually run them.

## Proposed benchmark targets (not measured results)

| Benchmark | Target |
|---|---|
| Broken, cyclic, discontinuous or wrong-root calibration chain | 100% blocked across negative controls |
| Calibration expiry covering any part of a measurement | 100% blocked |
| Invalid/mutated instrument, firmware, trace or certificate digest shape | 100% blocked |
| Missing/extra/duplicate observation coverage and provider/window/resource mismatch | 100% rejected |
| Sequence arithmetic mismatch or strict-policy sample drop | 100% blocked |
| Inter-sample gap above declared policy | 100% blocked |
| Uncertainty arithmetic overflow / over-ceiling sum | 100% blocked |
| Below-floor calibration evidence | Never promoted; cannot report policy-compliant |
| Input permutations | Identical report/receipt across 1,000 randomized permutations |
| Fixed boundary corpus | At least 2,000 seeded cases, zero unexpected panics and 100% expected classifications |
| Scale target | 10,000 observation claims × chain depth 4, p95 <250 ms on a documented reference runner |

For every 100% claim, record numerator, denominator, seed and corpus-manifest digest. Preserve positive controls and boundary cases; never weaken a negative control merely to obtain a pass. Timing is advisory until runner, Rust toolchain, profile, seed and corpus are frozen. These are targets, not measured results.

## Research basis

- NIST Policy on Metrological Traceability: https://www.nist.gov/calibrations/traceability
- NIST Metrological Traceability: Frequently Asked Questions and Policy: https://www.nist.gov/metrology/metrological-traceability
- NIST Technical Note 1297, guidelines for evaluating and expressing measurement uncertainty: https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-foreword

The implementation adopts the *structure* of traceability and explicit uncertainty accounting as a software validation concept. It does not assert NIST endorsement or traceability to NIST standards.
