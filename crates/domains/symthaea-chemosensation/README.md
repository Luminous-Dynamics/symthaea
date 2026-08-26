# symthaea-chemosensation

Hardware-independent foundations for artificial olfaction and gustation in Symthaea.

## Status

**Active research / draft sensorium stack.** The current simulators are simplified research fixtures. They make the evidence and cognition pipeline testable before vendor-specific hardware drivers, real calibration datasets, and held-out physical characterization are introduced.

They are not calibrated analytical instruments, and this crate does not make claims about subjective smell or taste experience.

No crate evidence level is claimed here ahead of CI and physical validation. The workspace truth registry may leave this crate unclassified until its evidence has actually been assessed.

## Core epistemic boundary

Interpretation never overwrites measurement, and evidence identity is separate from representation identity:

```text
physical or simulated transducer
            |
            v
    ChemicalObservation
      raw values + units
      calibration identity/state
      sensor health
      environment
      provenance/source
            |
            +----> ChemicalObservationId
            |        content-addressed raw evidence
            v
       calibration
            |
            v
   ChemicalFingerprint
      16,384D HDC vector
      confidence
      used/ignored channels
      ChemicalEncodingSpaceId
            |
            v
      ChemicalPercept
            |
       +----+---------+
       |              |
       v              v
 temporal context   novelty memory
                     (explicit admission)
       |              |
       +------+-------+
              |
              v
        modality cognition
              |
      smell + taste may form
        derived FlavorPercept
              |
              v
    ChemicalModalBridge
      one input per modality/cycle
      conflict-aware confidence
      ChemicalEvidenceBundleId
      ChemicalEncodingSpaceId
      exact component percepts retained
              |
              v
      future canonical root wiring
```

The two content identities answer different questions:

- `ChemicalObservationId` / `ChemicalEvidenceBundleId`: **what raw evidence was observed?**
- `ChemicalEncodingSpaceId`: **under what HDC coordinate system was that evidence represented?**

The same raw evidence can therefore survive a representation migration with the same evidence receipt while receiving a new encoding-space ID.

## Current capabilities

### Shared observation and evidence layer

- typed olfactory/gustatory modality
- physical measurement units
- raw-value preservation
- calibration identity, baseline, gain, and drift
- saturation/contamination health metadata
- temperature, humidity, and pressure context
- pessimistic handling of NaN/infinite/corrupt values
- channel-order-invariant content IDs for raw observations
- order-invariant evidence-bundle receipts that preserve duplicate observations

### Continuous HDC encoding

`ScalarHdcEncoder` interpolates between deterministic anchor hypervectors so nearby finite measurements remain locally similar. Anchors and interpolated outputs are L2-normalized so exact grid points cannot dominate a later bundle purely because of vector magnitude. Non-finite values are rejected instead of being mapped onto a valid range endpoint.

`ChemicalFingerprintEncoder` role-binds channel identity and modality, validates units, rejects duplicate configured/observed channels, ignores unknown channels for forward compatibility, and attenuates unhealthy/drifting channels by confidence.

Every fingerprint carries a content-addressed `ChemicalEncodingSpaceId` derived from the actual scalar anchors, channel-role vectors, units/ranges, and modality-role vectors. Geometric comparison across different space IDs is rejected by downstream temporal, novelty, flavor, and bridge layers.

This remains a starting representation, not a claim that anchor interpolation is the optimal continuous HDC code. Alternative level/thermometer encodings and dense baselines should be compared empirically on held-out data.

### Temporal and novelty cognition

`ChemicalTemporalTracker` keeps smell and taste anchors separate, rejects non-monotonic evidence, confidence-gates anchor replacement, and treats an encoding-space change as an explicit migration boundary rather than chemical change.

`ChemicalNoveltyMemory` separates assessment from admission. Merely perceiving a novel pattern does not teach the system that the pattern is normal. Memory is namespaced by modality and encoding space, bounded both per space and across retained representation generations, and old representation spaces are evicted deterministically when the configured history bound is exceeded.

### Flavor binding

`FlavorBinder` creates a derived flavor representation only from one trustworthy olfactory and one trustworthy gustatory percept that are temporally compatible and share a source encoding space.

The flavor vector receives its own derived encoding-space identity, because equal dimensionality does not make raw odor/taste fingerprints and flavor representations interchangeable. The original olfactory and gustatory evidence remains attached.

Flavor is deliberately not emitted as a third root sensory modality; doing so alongside smell and taste would double-count the same evidence.

### Same-cycle multimodal bridge

`ChemicalModalBridge` collapses multiple comparable same-modality percepts into one root-ready current-cycle input so several noses or tongue devices do not masquerade as temporal progression.

It:

- requires one modality and one encoding space per aggregate
- rejects non-finite vectors, invalid confidence, zero-trust components, wrong dimensions, and excessive timestamp skew
- orders components deterministically before floating-point accumulation
- uses confidence-weighted HDC fusion
- does not increase confidence merely because redundant agreeing sensors exist
- allows strong contradictory evidence to collapse aggregate confidence
- prevents very weak contradictory sensors from obtaining veto power over strong evidence
- retains all component percepts
- carries a `ChemicalEvidenceBundleId` for the exact raw observations summarized by the aggregate
- targets the reserved root IDs `Olfactory = 13` and `Gustatory = 14`

`Chemesthetic = 15` remains reserved but intentionally unimplemented until there is a genuine chemesthetic observation path rather than a relabeling of ordinary taste.

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

## Validation

The dedicated GitHub Actions workflow runs:

```bash
cargo test -p symthaea-chemosensation
cargo clippy -p symthaea-chemosensation --all-targets -- -D warnings
cargo test -p symthaea-chemosensation --doc
```

Current tests cover, among other invariants:

- calibration arithmetic and invalid numeric handling
- HDC determinism, saturation, locality, unit norm, and anchor-boundary continuity
- unit mismatch and duplicate-channel rejection
- confidence weighting and dead-channel exclusion
- encoding-space identity determinism and migration separation
- raw-observation and evidence-bundle receipt stability
- temporal ordering and cross-space comparison rejection
- explicit novelty admission and bounded migration history
- flavor temporal/space compatibility and distinct derived representation identity
- same-cycle aggregation order invariance, conflict handling, and evidence retention
- MOX temporal response, recovery, humidity confounding, transactional failure, and analytical one-time-constant reference behavior
- potentiometric mixture/temperature behavior and the monovalent Nernst known answer at 25 C

Passing these tests establishes internal/model correctness only. Real olfactory/gustatory performance requires independent sensor characterization and held-out physical data.

## Deliberate next boundaries

The next layers should remain separate PRs with explicit validation gates:

1. validate and land the generic stable-modality, presence-semantics, and weighted-binding prerequisites
2. add canonical root `Olfactory` and `Gustatory` variants plus a thin adapter from `ChemicalModalBridgeInput`
3. add a real chemesthetic evidence path before enabling `Chemesthetic`
4. add active sniff/sampling policy and deterministic record/replay protocols
5. add gas-sensor hardware adapters
6. add ADC/electrode and sample/rinse hardware boundaries for electronic tongue work
7. characterize drift/recalibration across sessions, days, sensors, and environments
8. run HDC-vs-dense/level-encoding ablations on held-out real data
9. define persisted-memory migration receipts before novelty memory is stored across software upgrades

The intended path remains **measurement -> evidence receipt -> calibrated uncertain representation -> cognition -> multimodal binding**, never sensor reading -> hard-coded semantic label.
