# Symthaea HAL v7 Continuation Patches

The v7 continuation starts from hardened v6 and adds signed authority,
cryptographic evidence identity, remote-command ingress containment, clock
integrity, supervisor redundancy, safety-case admission, and portable audit
exports.

## Ordered patches

- **0057** Canonical domain-separated SHA-256 evidence digests.
- **0058** Detached signature provider boundary with no HAL-held private keys.
- **0059** Signed operator authority quorum for critical actions.
- **0060** Authority-signed startup seals.
- **0061** Local queue, rate, and sequence-jump limits for remote commands.
- **0062** Suspend/resume and clock-discontinuity detection.
- **0063** Runtime fail-stop integration for time-integrity faults.
- **0064** Redundant external supervisor delivery quorum.
- **0065** Portable digest-bound operational audit exports.
- **0066** Deployment safety claims with evidence freshness.
- **0067** Operational audit-export CLI.
- **0068** Deployment safety-case verifier CLI.
- **0069** Signed authority, audit, and deployment operating guidance.

## Compatibility

Legacy calibration, HIL, and fault-ledger fingerprints intentionally retain
their existing `fnv1a64:` representation. New v7 artifacts use `sha256:`.
Applications may adopt the new modules independently. Runtime clock integrity is
optional until explicitly installed.

## Remaining physical validation

Source review and deterministic artifact replay do not establish physical
safety. Before deployment, run the HIL shutdown campaign on the exact robot,
verify independent OE/e-stop power removal, exercise watchdog and supervisor
failure modes, and produce current safety-case evidence with reviewed expiry.
