# symthaea-chemosensation-bridge

Evidence-preserving adapter between `symthaea-chemosensation` and Symthaea's root multimodal cognition.

## Boundary

This crate owns the one-way composition seam between the chemical sensing domain and the root `symthaea` package. It does not move chemical sensing into the root package and does not make the root package depend on `symthaea-chemosensation`.

```text
ChemicalModalBridgeInput
        |
        v
ChemicalRootProjector
  validates evidence / spaces / clocks / timestamps / geometry
        |
        v
ChemicalRootProjection
  retains source acquisition time + clock domain
        +
ChemicalRootContentLineage
        |
        v
ModalLineageReceipt
        +
ModalInput
  root timestamp =
    explicit Unix acquisition time, or
    root ingestion time otherwise
        |
        v
LineagedModalInput
```

## Identity is not activation

`Olfactory` and `Gustatory` can be named at the root boundary, but this crate does not add them to the default root channel set or amodal convergence topology. A chemical input can therefore be observed and retain exact lineage while still being excluded from root fusion.

Chemesthesis remains distinct and has no mapping here because the chemical domain does not yet expose a genuine chemesthetic transduction target.

## Evidence semantics

The root lineage preserves four roles:

1. raw chemical evidence bundle identity;
2. ContinuousHV encoding-space identity;
3. projection-policy identity;
4. BinaryHV output-space identity.

The bridge carries the chemical aggregate's conflict-adjusted confidence into root cognition and retains the complete `ChemicalRootProjection` alongside the generic input so agreement, projection-quality diagnostics, source acquisition timestamp, and source clock-domain provenance are not discarded.

Content identity is not authenticity, trust, synchronization quality, or subjective experience.

## Acquisition time is not ingestion time

The chemical domain may report a timestamp against an explicit acquisition clock such as a device-monotonic domain, Unix epoch, or no declared domain at all. Root `ModalInput::new(...)`, by contrast, assigns the current root ingestion time relative to Unix epoch.

The bridge therefore uses asymmetric timestamp translation:

- if `ChemicalRootProjection.clock_domain == ChemicalClockDomainId::unix_epoch()`, the source acquisition timestamp is copied exactly into `ModalInput.timestamp`;
- for a device-local clock, the original `(clock_domain, latest_timestamp_us)` remains authoritative source provenance on `ChemicalRootProjection`, while `ModalInput.timestamp` remains the root ingestion time;
- for an unspecified source clock, the numeric source timestamp is still retained by the projection but is never relabeled as Unix time; root ingestion time is used instead.

This distinction is deliberate. A value like `123456 us` from a sensor uptime counter must never silently become `1970-01-01T00:00:00.123456Z` merely because the root field also uses `Duration`.

No mutable field is added to the public root `ModalInput` API for source-clock metadata. The detailed source-time contract stays on the chemical projection, avoiding a source-breaking root struct change.

## Validation intent

The crate's tests require that:

- chemical target IDs agree with root stable modality IDs;
- projection validation runs before root input construction;
- confidence, BinaryHV, and lineage roles survive handoff;
- unclocked source time remains source provenance while root uses ingestion time;
- device-local source time remains source provenance while root uses ingestion time;
- explicitly Unix-epoch source time is preserved exactly as the root timestamp;
- an olfactory input is visible as current-cycle evidence while unconfigured;
- an unconfigured chemical input cannot alter legacy visual fusion;
- gustation never aliases chemesthesis or a derived flavor representation.

The bridge should remain draft until the root multimodal prerequisite stack, chemical clock/evidence stack, and this exact composed head have executable CI evidence.
