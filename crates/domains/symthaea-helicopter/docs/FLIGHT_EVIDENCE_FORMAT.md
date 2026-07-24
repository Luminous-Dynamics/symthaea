# Flight Evidence Format

`FlightRecorder` emits `symthaea-helicopter-flight-log-v1` records containing:

- scenario/controller identity, seed, and physics rate;
- requested and actually applied actuator commands;
- full helicopter state, active perturbation effects, fuel/powertrain state;
- rotor kinetic energy and latched landing evidence;
- ordered authority, navigation, perturbation, reserve, and operator events.

The canonical JSON representation contains no maps, so field and vector order is
stable for a fixed schema. `fnv1a64` is provided only for deterministic replay
comparison and accidental-corruption detection. Publication-grade evidence must
also be hashed and signed by the repository's cryptographic evidence tooling.

## Replay chain and segmentation

Frames and events share a single monotonic sequence and ordered replay chain.
Each FNV-1a link commits to the previous link, record domain, and canonical
record payload. Segment seals carry the parent chain tip and sequence boundary;
continuations must preserve manifest identity and strictly increase sequence.
The chain detects accidental mutation and discontinuity but is not a
cryptographic signature or adversarial tamper-proofing mechanism.

## Cryptographic authenticity boundary

The recorder's FNV-1a links are deterministic replay checksums only. They are
not signatures and do not establish authorship or tamper resistance.
`SignedFlightSegmentSeal` binds canonical recorder bytes and the segment seal
to an explicit external `EvidenceCryptoProvider`. The default provider is
unavailable and fails closed; production deployments must supply and validate a
real digest/signature implementation, key identifier, and trust policy. Test
providers must never be accepted as production evidence.
