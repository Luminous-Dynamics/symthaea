# Integrity Framework

Hardware integrity and tamper detection for Symthaea's cognitive loop.

**Feature gate**: `integrity`
**Location**: `src/integrity/`
**Tests**: 43 unit + 8 proptests + 20 chronobiology

## Architecture

```
IntegrityManager (src/integrity/mod.rs)
├── AttestationRegistry (attestation.rs)    — BLAKE3 hash verification
├── TemporalConsistencyMonitor (temporal.rs) — wall clock vs CfC delta_t
├── CanaryRunner (behavioral_canaries.rs)   — known-answer tests
├── EventLog (VecDeque<IntegrityEvent>)     — 64-entry ring buffer
└── PanicSnapshot (global Arc<Mutex>)       — crash forensics
```

### Data Flow

1. **Startup**: CLS constructor registers 4 attestations (safety thresholds, consciousness weights, receptor sensitivities, moral topology) with BLAKE3 baselines. Each registration runs a self-test assertion.
2. **Each cycle**: `tick(cycle, cfc_delta_t, full_sweep)` runs:
   - Attestation verification (jittered ~101 cycles ±10%)
   - Temporal consistency (every cycle)
   - Behavioral canaries (co-prime intervals)
3. **Severity pipeline**: All anomaly sources feed `global_failure_streak` → 1-2 = Warning, 3+ = Critical
4. **Consciousness gating**: `integrity_confidence` (1.0/0.5/0.1) multiplied into `consciousness_level`
5. **Telemetry**: `IntegrityTelemetry` in `CycleMetadata.integrity` → Pulse TUI

## Attestation Registry

BLAKE3 hashes of safety-critical constants, verified periodically.

| Registration | Constants | Risk if Tampered |
|---|---|---|
| `safety_thresholds` | Moral concern/benefit thresholds | Silent ethical drift |
| `consciousness_weights` | Dominance/policy thresholds | Wrong consciousness scores |
| `receptor_sensitivities` | Neuromodulator baselines | Personality corruption |
| `moral_topology_constants` | Harmony weights, MORAL_DIM | Distorted moral evaluations |

### Self-Test

`register()` immediately calls the hasher closure and asserts it matches the provided hash. Catches registration bugs at startup, not 101 cycles later. Test-only `register_tampered()` skips this for tamper simulation.

### Live Verification

`verify_live_thresholds()` re-hashes current const values each cycle (in addition to the periodic frozen-copy check). Catches runtime mutation, not just binary patching.

### Jitter

Attestation interval is nominally 101 cycles but jittered ±10% deterministically via `hash(cycle + seed) % 21 - 10`. An attacker who knows the base interval cannot time manipulation between checks.

## Behavioral Canaries

Known-answer tests at co-prime intervals. Each canary tracks consecutive failures.

| Canary | Interval | Tests | Severity |
|---|---|---|---|
| `threshold_ordering` | 103 | Safety threshold ordering invariants | Corruption |
| `blake3_determinism` | 107 | BLAKE3 known-answer test | Corruption |
| `fpu_sanity` | 109 | FP arithmetic (1+2=3, sin(pi/6)=0.5, NaN, exp) | Corruption |
| `consciousness_equation` | 113 | ConsciousnessEquationV2 known-answer range | Corruption/Drift |
| `moral_algebra_determinism` | 127 | Proportional justice HV norm and self-similarity | Corruption/Drift |
| `hdc_encoding_consistency` | 131 | HDC encoder determinism (same input → same HV) | Corruption |

## Unified Severity Pipeline

All anomaly sources (attestation, canary, temporal) feed a single `global_failure_streak`:

```
Clean tick (no anomalies from any source) → streak = 0
Any anomaly → streak += 1
streak 1-2 → all anomalies tagged Warning
streak 3+  → all anomalies escalated to Critical
```

This means: 2 canary failures + 1 attestation failure across 3 consecutive ticks = Critical, even though no single source had 3 consecutive failures.

## Integrity Confidence

| Value | Meaning | Effect |
|---|---|---|
| 1.0 | All checks pass | Full consciousness trust |
| 0.5 | Warning-level anomalies | Consciousness halved |
| 0.1 | Critical anomalies | Consciousness reduced to 10% |

Applied in `cycle_phase_output/`: `metadata.consciousness_level *= integrity_confidence`

Flows to `SafetyMetrics.integrity_critical` for NRC-level safety escalation.

## Snapshot Export

`export_snapshot() -> Vec<u8>` produces a BLAKE3-signed binary blob:

```
[u32: record_count]
  per record: [u16: name_len][bytes: name][32: baseline_hash][u32: consecutive_failures]
[u32: global_failure_streak]
[f32: integrity_confidence]
[u32: event_count]
  per event: [u16: source_len][bytes: source][u8: severity][u32: cycle]
[32: blake3_signature]
```

`verify_snapshot(blob)` checks the trailing BLAKE3 signature.

### Panic Hook

`install_panic_hook()` chains a custom hook that dumps the most recent snapshot to `/tmp/symthaea-integrity-dump-{unix_secs}.bin` on any panic. The snapshot is updated every 10 cycles via a global `Arc<Mutex<Option<Vec<u8>>>>`.

## Event Log

64-entry ring buffer (`VecDeque<IntegrityEvent>`) recording all anomalies with:
- Cycle number and timestamp
- Source component (attestation, temporal, canary, live_attestation)
- Description and severity

Accessible via `event_history()` for dashboards and post-mortem analysis.

## Pulse TUI Display

The Integrity Shield pane shows:
- Status indicator (green VERIFIED / yellow WARNING / red CRITICAL)
- Attestation, temporal, canary pass/fail checkmarks
- Registered count (attestations and canaries)
- Confidence percentage and failure streak count
- 60-cycle confidence sparkline (full block = 1.0, half = 0.5, low = 0.1)
- Anomaly count when non-zero

## Default Behavior (Feature Off)

When `integrity` feature is disabled:
- `IntegrityTelemetry::default()` returns all-pass, confidence 1.0
- `integrity_critical` defaults to `false`
- No performance overhead — all integrity code is compiled out

## Running Tests

```bash
# Unit tests (43)
cargo test -p symthaea --lib --features integrity integrity

# Property tests (8)
cargo test -p symthaea --test proptest_integrity --features integrity

# All tests including chronobiology
cargo test -p symthaea --lib --features integrity
```
