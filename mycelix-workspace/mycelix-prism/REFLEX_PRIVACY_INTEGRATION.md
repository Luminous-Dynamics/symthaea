# Adaptive Security Pipeline (Implementation Plan)

## Overview
The `prism-reflex` arc currently classifies content into static `ContentZone` tags. We will hook this output into the `encoding_gate_adaptive` privacy system, enabling real-time zone downgrading for high-threat content.

## Integration Plan
1. **Reflex Verdict Refinement:** In `prism-reflex/src/lib.rs`, when generating a `PostParseVerdict`, we will call `encoding_gate_adaptive(zone, threat_score)`.
2. **Zone Downgrading:** The result will be used to modify the `ContentZone` of the `PostParseVerdict` before it is emitted to the system.
3. **Chrome Telemetry:** The updated `ContentZone` will flow naturally into the UI telemetry circuit I implemented earlier, meaning the browser chrome will immediately show "LOCAL ONLY" instead of "PUBLIC" if the sanitization gate decides a threat is present.

## Security Rationale
- **Proactive Sanitization:** The engine now acts as a reactive firewall. It doesn't just display the threat; it fundamentally changes the data's permission level (downgrading from Public to Local-only) so it cannot be shared via DHT.
- **Unified Policy:** All zones—whether determined by user intent or automated Reflex Arc scanning—now pass through the same unified privacy enforcement logic.
- **Atomic Security:** The transition from detected threat to enforced zone downgrade happens within a single tick of the Reflex Arc engine, minimizing the exposure window.
