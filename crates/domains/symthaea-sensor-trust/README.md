# symthaea-sensor-trust

Replay-resistant observation admission and source-trust supervision for Symthaea.

This crate generalizes the defensive sensor-ingress ideas already present in the
subterranean domain into reusable infrastructure for `ObservationEnvelope`.
It deliberately stops before semantic fusion: it decides whether evidence is
admissible, not what an observed object is or what anyone should do about it.

## Admission checks

For every observation batch the supervisor checks:

- hard batch bounds
- duplicate observation IDs
- duplicate source IDs within one batch
- canonical observation validation
- freshness and clock envelope
- integrity failure
- source availability
- monotonic source sequence numbers / replay and reorder resistance
- source-isolation state

Accepted evidence is then summarized by distinct physical source lineage and
sensor modality. Multiple processors derived from the same physical sensor can
be accepted as separate processing records but count as only one physical
witness.

## Fail-closed quorum

A deployment policy defines minimum independent physical sources and minimum
independent modalities. `requires_fail_closed` is true whenever:

- no evidence is admitted, or
- physical-source diversity is below policy, or
- modality diversity is below policy.

The crate does not translate this result into an actuator command.

## Source reliability

Source reliability begins at 1.0. An independent verifier can submit normalized
quality feedback. Negative feedback reduces reliability at the configured
penalty rate; positive feedback recovers only at the configured bounded recovery
rate. Sources at or below the isolation threshold are rejected.

The API documentation explicitly forbids feeding a source's own confidence back
as its quality signal: otherwise a compromised source could self-rehabilitate.
A maintenance reset exists for commissioning/review workflows and is intentionally
explicit.

## Important distinctions

```text
admissible evidence != true claim
quorum              != identity
identity            != intent
risk                != authority
```

This crate contains no targeting, jamming, firing, interception, weapon-control,
or engagement-optimization logic.

## Verification

```bash
cargo test -p symthaea-sensor-trust
```
