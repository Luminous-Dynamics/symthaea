# symthaea-chemosensation

Hardware-independent foundations for artificial olfaction and gustation in Symthaea.

## Status

**Active research / draft foundation.** The initial simulators are simplified research fixtures. They exist to make the perception pipeline testable before vendor-specific hardware drivers and real calibration datasets are introduced.

They are not calibrated analytical instruments, and this crate does not make claims about subjective smell or taste experience.

No crate evidence level is claimed here ahead of CI. The workspace truth registry may leave this crate unclassified until its evidence has actually been assessed.

## Evidence boundary

The central invariant is that interpretation never overwrites measurement:

```text
physical or simulated transducer
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
 future odor/taste/flavor hypotheses
```

Raw observations remain available as evidence even after derived representations are produced.

## Current capabilities

### Shared observation layer

- typed olfactory/gustatory modality
- physical measurement units
- raw-value preservation
- calibration identity, baseline, gain, and drift
- saturation/contamination health metadata
- temperature, humidity, and pressure context
- pessimistic handling of NaN/infinite/corrupt values

### Continuous HDC encoding

`ScalarHdcEncoder` interpolates between deterministic anchor hypervectors so nearby finite measurements remain locally similar. Non-finite values are rejected instead of being mapped onto a valid range endpoint.

`ChemicalFingerprintEncoder` role-binds channel identity and modality, validates units, rejects duplicate configured/observed channels, ignores unknown channels for forward compatibility, and attenuates unhealthy/drifting channels by confidence.

This is a starting representation, not a claim that anchor interpolation is the optimal continuous HDC code. Alternative level/thermometer encodings should be compared empirically before replacing it.

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

It does not directly encode human labels such as sweet, bitter, salty, sour, or umami. Those should remain learned interpretations over chemical evidence rather than primary sensor coordinates.

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

Passing these tests establishes internal/model correctness only. Real olfactory/gustatory performance requires independent sensor characterization and held-out physical data.

## Deliberate next boundaries

This foundation should become green before later PRs add:

1. canonical `Olfactory`, `Gustatory`, and `Chemesthetic` cognitive modalities
2. temporal chemical percepts, novelty, memory, and flavor binding
3. active sniff/sampling policy
4. gas-sensor hardware adapters
5. ADC/electrode and microfluidic tongue hardware
6. drift/recalibration studies across sessions, days, sensors, and environments
7. HDC-vs-dense/level-encoding ablations on held-out real data

The intended path is **measurement -> calibrated evidence -> uncertain percept -> multimodal cognition**, not sensor reading -> hard-coded semantic label.
