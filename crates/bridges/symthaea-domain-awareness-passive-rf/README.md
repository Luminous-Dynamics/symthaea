# symthaea-domain-awareness-passive-rf

Receive-only passive RF evidence bridge for domain awareness.

Symthaea already has `SpectrumManager` and RTL-SDR receive-only support. This crate
adds the missing assurance boundary between measured spectrum data and the shared
`ObservationEnvelope` contract without creating another SDR stack.

## Scope

The bridge accepts the measured fields already present on `SpectrumObservation`:

- center frequency
- noise floor
- SNR

plus explicit timing, receiver identity, calibration, integrity, health, and raw
evidence provenance.

The existing `SpectrumObservation::jammed` interpretation is deliberately **not**
a constructor input. A subsystem may form an interference/jamming hypothesis
elsewhere, but domain awareness receives the underlying measurement evidence rather
than silently promoting that interpretation to fact.

The normalized `ObservationEnvelope` publishes RF modality + SNR and retains a
reference to the richer raw spectrum record containing exact frequency/noise-floor
data.

## Correlation-aware provenance

Different processors fed by one SDR can publish different logical observations,
but they retain the same `physical_receiver_id`. `symthaea-sensor-trust` therefore
counts them as one physical witness rather than allowing software duplication to
manufacture corroboration.

## Silence semantics

`PassiveRfScanCoverage` records that a receiver actually scanned a declared band.
A scan with no observed emission means only:

> no RF emission was observed by this receiver under this scan/calibration/health context.

It does **not** establish:

- that no physical object exists
- object identity
- intent
- safety
- physical authority

Autonomous/non-emitting systems therefore remain visible to the architecture as a
known limitation of passive RF rather than being misclassified as safe.

## Deliberate API absence

This crate defines no transmit, jamming, spoofing, protocol manipulation, or
electronic-attack API.

## Verification

```bash
cargo test -p symthaea-domain-awareness-passive-rf
```
