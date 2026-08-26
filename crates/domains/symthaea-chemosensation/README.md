# symthaea-chemosensation

Hardware-independent foundations for artificial olfaction and gustation in Symthaea.

## Status

**Active research / draft sensorium stack.** The current simulators are simplified research fixtures. They make the evidence and cognition pipeline testable before vendor-specific hardware drivers, real calibration datasets, and held-out physical characterization are introduced.

They are not calibrated analytical instruments, and this crate does not make claims about subjective smell or taste experience.

No crate evidence level is claimed here ahead of CI and physical validation. The workspace truth registry may leave this crate unclassified until its evidence has actually been assessed.

## Core epistemic boundary

Interpretation never overwrites measurement, evidence identity is separate from representation identity, and a timestamp is never assumed comparable merely because it is an integer:

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
      timestamp_us
      optional ChemicalClockDomainId
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
 (clock-gated)       (explicit admission)
       |              |
       +------+-------+
              |
              v
        modality cognition
              |
      smell + taste may form
        derived FlavorPercept
        only on a shared clock
              |
              v
    ChemicalModalBridge
      one input per modality/cycle
      conflict-aware confidence
      shared-clock proof for multi-source skew
      ChemicalEvidenceBundleId
      ChemicalEncodingSpaceId
      exact component percepts retained
              |
              v
      ChemicalRootProjector
      BinaryHV + explicit projection diagnostics
      clock domain retained, never invented
```

The content identities answer different questions:

- `ChemicalObservationId` / `ChemicalEvidenceBundleId`: **what raw evidence was observed, including its declared clock domain or explicit lack of one?**
- `ChemicalEncodingSpaceId`: **under what HDC coordinate system was that evidence represented?**
- `ChemicalRootProjectionPolicyId`: **which ContinuousHV -> BinaryHV projection rule was applied?**
- `ChemicalRootBinarySpaceId`: **which resulting root BinaryHV comparison space is this representation in?**

The same raw evidence can therefore survive a representation migration with the same evidence receipt while receiving a new encoding-space ID. Conversely, identical sensor values and numeric timestamps under different declared clock domains are different raw evidence.

Observation and evidence-bundle hash/content-address namespaces are **v2** because clock-domain provenance changes the meaning of those receipts. Old v1 digests must never be silently reinterpreted as v2. Legacy serialized observations that lack a `clock_domain` field can still deserialize, but they become explicitly **clock unspecified** rather than implicitly Unix-time evidence.

## Clock-domain contract

`ChemicalClockDomainId` is a bounded canonical identifier for a timestamp-comparison domain. It states only that timestamps are intended to be interpreted against the same declared timebase.

It does **not** prove:

- clock accuracy
- synchronization error bounds
- authenticity
- monotonicity
- agreement with wall-clock time

`ChemicalClockDomainId::unix_epoch()` is the explicit well-known domain for values claimed to be microseconds since the Unix epoch. `None` never means Unix time.

The rules are intentionally asymmetric:

- a single chemical observation may remain clock-unspecified because no cross-source timestamp comparison is required
- unclocked chemistry remains valid chemical evidence and may be encoded, assessed for novelty, and retained
- unclocked chemistry does not create or advance temporal anchors, elapsed-time estimates, or change rates
- a positively different declared clock is a temporal migration boundary rather than a chemical change
- two or more same-cycle sensor inputs must declare the same clock domain before their timestamp skew is interpreted
- olfactory/gustatory flavor binding must declare the same clock domain before smell/taste temporal skew is interpreted

## Current capabilities

### Shared observation and evidence layer

- typed olfactory/gustatory modality
- physical measurement units
- raw-value preservation
- calibration identity, baseline, gain, and drift
- saturation/contamination health metadata
- temperature, humidity, and pressure context
- optional canonical clock-domain identity with no implicit epoch
- pessimistic handling of NaN/infinite/corrupt values
- channel-order-invariant content IDs for raw observations
- clock-domain-aware v2 observation/evidence identities
- order-invariant evidence-bundle receipts that preserve duplicate observations
- backward wire compatibility for pre-clock observations as explicitly unclocked evidence

### Continuous HDC encoding

`ScalarHdcEncoder` interpolates between deterministic anchor hypervectors so nearby finite measurements remain locally similar. Anchors and interpolated outputs are L2-normalized so exact grid points cannot dominate a later bundle purely because of vector magnitude. Non-finite values are rejected instead of being mapped onto a valid range endpoint.

`ChemicalFingerprintEncoder` role-binds channel identity and modality, validates units, rejects duplicate configured/observed channels, ignores unknown channels for forward compatibility, and attenuates unhealthy/drifting channels by confidence.

Every fingerprint carries a content-addressed `ChemicalEncodingSpaceId` derived from the actual scalar anchors, channel-role vectors, units/ranges, and modality-role vectors. Geometric comparison across different space IDs is rejected by downstream temporal, novelty, flavor, and bridge layers.

This remains a starting representation, not a claim that anchor interpolation is the optimal continuous HDC code. Alternative level/thermometer encodings and dense baselines should be compared empirically on held-out data.

### Temporal and novelty cognition

`ChemicalTemporalTracker` keeps smell and taste anchors separate, requires explicit clock provenance before making elapsed-time claims, rejects non-monotonic evidence within one clock domain, rejects clock-domain migration, confidence-gates anchor replacement, and treats an encoding-space change as an explicit migration boundary rather than chemical change.

A percept without a clock domain still receives a valid chemical representation and novelty assessment, but its temporal context deliberately contains no elapsed time, prior comparison, or change rate and it does not create a temporal anchor.

`ChemicalNoveltyMemory` separates assessment from admission. Merely perceiving a novel pattern does not teach the system that the pattern is normal. Memory is namespaced by modality and encoding space, bounded both per space and across retained representation generations, and old representation spaces are evicted deterministically when the configured history bound is exceeded. Stored reference timestamps retain their optional clock-domain identity so traceability never invents an epoch.

### Flavor binding

`FlavorBinder` creates a derived flavor representation only from one trustworthy olfactory and one trustworthy gustatory percept that share one declared clock domain, are temporally compatible within that domain, and share a source encoding space.

The resulting `FlavorPercept` retains the common clock domain alongside its temporal skew. That timebase identity is provenance for the temporal claim, not proof that the clocks were accurately synchronized.

The flavor vector receives its own derived encoding-space identity, because equal dimensionality does not make raw odor/taste fingerprints and flavor representations interchangeable. The original olfactory and gustatory evidence remains attached.

Flavor is deliberately not emitted as a third root sensory modality; doing so alongside smell and taste would double-count the same evidence.

### Same-cycle multimodal bridge

`ChemicalModalBridge` collapses multiple comparable same-modality percepts into one root-ready current-cycle input so several noses or tongue devices do not masquerade as temporal progression.

It:

- requires one modality and one encoding space per aggregate
- requires a shared declared clock domain before comparing timestamps from multiple components
- permits a single component to remain clock-unspecified because no cross-source skew comparison is needed
- rejects non-finite vectors, invalid confidence, zero-trust components, wrong dimensions, and excessive admissible timestamp skew
- orders components deterministically before floating-point accumulation
- uses confidence-weighted HDC fusion
- does not increase confidence merely because redundant agreeing sensors exist
- allows strong contradictory evidence to collapse aggregate confidence
- prevents very weak contradictory sensors from obtaining veto power over strong evidence
- retains all component percepts
- carries the aggregate clock domain forward when one is declared
- carries a `ChemicalEvidenceBundleId` for the exact raw observations summarized by the aggregate
- targets the reserved root IDs `Olfactory = 13` and `Gustatory = 14`

`Chemesthetic = 15` remains reserved but intentionally unimplemented until there is a genuine chemesthetic observation path rather than a relabeling of ordinary taste.

### Root projection boundary

`ChemicalRootProjector` revalidates publicly constructible bridge inputs before any ContinuousHV -> BinaryHV projection. It independently checks evidence-bundle identity, encoding-space consistency, modality consistency, clock-domain consistency, timestamp envelope, confidence bounds, vector dimensionality, finite numeric content, and non-degenerate geometry.

The resulting `ChemicalRootProjection` retains the evidence bundle ID, input encoding-space ID, optional clock domain, timestamp envelope, confidence, source agreement, component count, and projection-quality diagnostics. The BinaryHV is a derived integration representation, not replacement evidence.

Projection policy and output BinaryHV space have distinct content identities. Threshold quality/stability studies remain descriptive until held-out physical experiments preregister acceptance gates.

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
- v2 clock-domain-aware evidence identity and legacy unclocked wire migration
- temporal ordering, clock migration, unclocked temporal abstention, and cross-space comparison rejection
- explicit novelty admission, clock-preserving traceability, and bounded migration history
- flavor clock/temporal/space compatibility and distinct derived representation identity
- same-cycle aggregation order invariance, conflict handling, evidence retention, and shared-clock enforcement
- public root-projection revalidation of clock/timestamp/evidence/representation invariants
- MOX temporal response, recovery, humidity confounding, transactional failure, and analytical one-time-constant reference behavior
- potentiometric mixture/temperature behavior and the monovalent Nernst known answer at 25 C

Passing these tests establishes internal/model correctness only. Real olfactory/gustatory performance requires independent sensor characterization and held-out physical data.

## Deliberate next boundaries

The next layers should remain separate PRs with explicit validation gates:

1. validate and land the generic stable-modality, presence-semantics, and weighted-binding prerequisites
2. compose the chemical root handoff while preserving clock provenance and refusing to relabel device-local time as root Unix time
3. add a real chemesthetic evidence path before enabling `Chemesthetic`
4. add active sniff/sampling policy and deterministic record/replay protocols with explicit acquisition-clock identity
5. add gas-sensor hardware adapters
6. add ADC/electrode and sample/rinse hardware boundaries for electronic tongue work
7. characterize clock synchronization error, sensor drift, and recalibration across sessions, days, sensors, and environments
8. run HDC-vs-dense/level-encoding ablations on held-out real data
9. define persisted-memory migration receipts before novelty memory is stored across software upgrades

The intended path remains **measurement -> evidence receipt -> calibrated uncertain representation -> cognition -> multimodal binding**, never sensor reading -> hard-coded semantic label, and never bare integer timestamp -> assumed shared timebase.
