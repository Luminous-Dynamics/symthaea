# symthaea-domain-awareness-common-cause

Monotonic common-cause diversity gate for domain-awareness track assurance.

The vision/domain-awareness layer can establish that a track is persistent and appears corroborated by multiple physical sensors/modalities. `symthaea-sensor-common-cause` separately evaluates whether those accepted physical sensors are genuinely diverse across deployment-reviewed fault domains.

This bridge composes the two results conservatively:

- `Unassessed`, `Tentative`, and `Persistent` can never be upgraded here.
- `Corroborated` and `IdentityEvidenceSupported` are preserved only when common-cause diversity is usable and its physical-source count agrees with the track-assurance report.
- otherwise they are downgraded to `Persistent`.

The bridge cannot create identity, risk, intent, or physical authority. It only removes assurance that is not supported by diverse witnesses.

## Verification

```bash
cargo test -p symthaea-domain-awareness-common-cause
```
