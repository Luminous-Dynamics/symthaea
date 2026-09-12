# symthaea-sensor-common-cause

Common-cause fault-domain diversity assurance for admitted sensor evidence.

`symthaea-sensor-trust` already prevents multiple software processors on one physical sensor from counting as multiple witnesses. This crate addresses the next layer: two *different* physical sensors may still share a common failure point.

Examples include shared:

- power feed
- clock source
- network segment
- compute host
- enclosure/mount
- site/environmental exposure

A deployment supplies evidence-backed `PhysicalSourceFaultProfile` records and an explicit `CommonCausePolicy`. The assessor then checks only the physical sources that were actually accepted by the sensor-trust admission report.

Missing profiles, missing required fault-domain categories, duplicate profiles, or insufficient diversity fail closed.

The crate deliberately does not infer fault domains from sensor names or topology guesses.

## Verification

```bash
cargo test -p symthaea-sensor-common-cause
```
