# symthaea-chemosensation

Hardware-independent foundations for artificial olfaction and gustation in Symthaea.

## Status

**Active research / draft foundation.** The current sensor models are simplified research fixtures. They exist to make evidence handling, representation, cognition, and experiment design testable before vendor-specific hardware drivers and real calibration datasets are introduced.

They are not calibrated analytical instruments, and this crate does not make claims about subjective smell or taste experience.

No physical performance evidence level is claimed ahead of held-out sensor characterization.

## Evidence boundary

The central invariant is that interpretation never overwrites measurement:

```text
physical or simulated transducer
            |
            v
    SamplingContext (optional)
            |
            v
    ChemicalObservation
      raw values + units
      calibration identity
      sensor health
      environment
      provenance/source
            |
            v
       calibration
            |
            v
   ChemicalFingerprint
      16,384D HDC vector
      confidence
      used/ignored channels
            |
            v
      ChemicalPercept
       + exact evidence
            |
      +-----+------+----------------+
      |            |                |
   temporal      novelty          flavor
    context      assessment       binding
      |            |                |
      +------------+----------------+
                   |
                   v
        later multimodal cognition
```

Raw observations remain available as evidence even after derived representations are produced.

## Current capabilities

### Shared observation and acquisition layer

- typed olfactory/gustatory modality
- physical measurement units
- raw-value preservation
- calibration identity, baseline, gain, and drift
- saturation/contamination health metadata
- temperature, humidity, and pressure context
- pessimistic handling of NaN/infinite/corrupt values
- optional typed `SamplingContext` with protocol/run/sample/phase/step/replicate metadata
- validated `ChemicalTrace` sequences with monotonic timestamp and protocol-step invariants
- transactional trace append so rejected evidence never partially mutates a run

### Continuous HDC encoding

`ScalarHdcEncoder` interpolates between deterministic anchor hypervectors so nearby finite measurements remain locally similar. Non-finite values are rejected instead of being mapped onto a valid range endpoint.

`ChemicalFingerprintEncoder` role-binds channel identity and modality, validates units, rejects duplicate configured/observed channels, ignores unknown channels for forward compatibility, and attenuates unhealthy/drifting channels by confidence.

This is a starting representation, not a claim that anchor interpolation is the optimal continuous HDC code. Level/thermometer and conventional dense baselines are preregistered comparison targets before any representation-superiority claim.

### Olfaction fixture

`MoxArraySimulator` models:

- cross-sensitive channels
- logarithmic concentration drive
- humidity confounding
- asymmetric rise/recovery time constants
- resistance response
- transactional validation before temporal state mutation

It is not a vendor-specific transfer-function model and does not identify arbitrary chemicals.

### Gustation fixture

`ElectronicTongueSimulator` models:

- direct pH
- conductivity
- cross-sensitive potentiometric electrodes
- temperature-dependent Nernst response using Symthaea's existing biophysics implementation
- explicit latent-species dimensionality

It does not directly encode human labels such as sweet, bitter, salty, sour, or umami. Those remain learned interpretations over chemical evidence rather than primary sensor coordinates.

### Cognitive chemical context

The crate now provides:

- `ChemicalPercept`, which preserves the exact source observation beside its derived fingerprint
- modality-specific temporal change tracking
- confidence-gated temporal anchors
- traceable novelty assessment and bounded novelty memory
- explicit memory admission rather than implicit learning on exposure
- a transactional `ChemicalCognitionPipeline`
- conservative smell+taste `FlavorBinder` with time/confidence gates and both source observations retained

Absence of trustworthy evidence remains `None`; it is never manufactured into a zero-valued percept.

### Preregistered experiment decisions

`ChemicalDecisionProtocol` encodes confirmatory metric gates before outcome-bearing evaluation. Each gate has a confirmation threshold and may have a separate practical-failure threshold.

Aggregate decisions are intentionally asymmetric:

- `Confirmed`: every required confirmation gate passes
- `NotConfirmed`: at least one preregistered practical-failure boundary is crossed
- `Inconclusive`: neither condition is established

Failing to confirm is therefore not automatically a negative result.

Evidence source and dataset/session partition must match the frozen protocol exactly; development or simulated results cannot silently satisfy a held-out physical claim.

The program-level rules live in `docs/CHEMOSENSATION_PILOT_PREREGISTRATION.md`.

## Validation

The dedicated GitHub Actions workflow runs:

```bash
cargo test -p symthaea-chemosensation
cargo clippy -p symthaea-chemosensation --all-targets -- -D warnings
cargo test -p symthaea-chemosensation --doc
```

Current tests cover, among other invariants:

- calibration arithmetic and invalid numeric handling
- HDC determinism, saturation, locality, and anchor-boundary continuity
- unit mismatch and duplicate-channel rejection
- confidence weighting and dead-channel exclusion
- MOX temporal response, recovery, humidity confounding, transactional failure, and an analytical one-time-constant reference value
- potentiometric mixture/temperature behavior and the monovalent Nernst known answer at 25 C
- evidence-preserving percept construction
- temporal timestamp and confidence gates
- novelty admission and non-implicit learning
- conservative flavor binding
- sampling-context and trace consistency
- asymmetric experiment decisions, threshold ordering, evidence/partition admission, and rejection of non-finite metrics

Passing these tests establishes internal/model correctness only. Real olfactory/gustatory performance requires independent sensor characterization and held-out physical data.

## Preregistered research path

The current pilot families include:

- `OD-001`: odor identity under concentration shift
- `OD-002`: humidity nuisance robustness
- `OD-003`: temporal response utility
- `OD-004`: open-set novelty
- `GT-001`: gustatory concentration shift
- `GT-002`: mixture discrimination
- `GT-003`: temperature robustness
- `GT-004`: rinse/carryover
- `FL-001`: smell+taste complementarity
- `FL-002`: cross-modal contradiction preservation

Physical pass/failure thresholds are not invented from software fixtures. They are selected using calibration-only data, frozen in a versioned decision protocol, and only then evaluated on sealed holdout evidence.

## Deliberate next boundaries

Before live chemosensation enters the root cognitive loop, the generic multimodal presence and weighted-binding integrity repairs should be validated independently.

After that, follow-on work can add:

1. canonical `Olfactory`, `Gustatory`, and `Chemesthetic` modalities
2. an evidence-preserving bridge into root multimodal cognition
3. active sniff/sampling policy
4. gas-sensor hardware adapters
5. ADC/electrode and microfluidic tongue hardware
6. drift/recalibration studies across sessions, days, sensors, and environments
7. preregistered HDC-vs-conventional representation comparisons on held-out real data

The intended path is **measurement -> calibrated evidence -> uncertain percept -> preregistered evaluation -> multimodal cognition**, not sensor reading -> hard-coded semantic label.
