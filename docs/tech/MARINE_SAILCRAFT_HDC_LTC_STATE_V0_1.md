# Evidence-aware sailcraft HDC/LTC state representation v0.1

**Owner:** MARINE-SAIL-COGNITION-001 #12218  
**Parent:** MARINE-SAIL-AUTONOMY-001 #12153 / P0 sailcraft kernel PR #12158  
**Status:** authored Rust representation contract; exact-head runner pending  
**Evidence class:** synthetic known-truth only

## Purpose and boundary

This crate introduces a sailcraft-specific observation representation using the existing Symthaea HDC core and HDC/LTC temporal network. It does not reuse the automotive 20-channel vehicle encoder: its channels are physically and semantically tied to sailcraft measurements, and angular quantities are represented circularly.

The crate has no policy head and does not import the vehicle controller or its command type. Its outputs are:
1. an HDC representation of the available evidence and its per-channel status; and
2. an HDC/LTC latent temporal state over those representations.

It does not output a route, tack, gybe, rudder angle, sheet angle, reef/furl operation, actuator command or operating-mode transition. A latent state is not a sailing skill or safety assurance.

## Paradigm: values + provenance + missingness

The input uses the existing `symthaea-sailcraft::Evidence<f64>` and `EvidenceState` types. It preserves each channel's:
- explicit physical meaning, unit and declared range;
- evidence state (`QUALIFIED`, `MISSING`, `STALE`, `UNVERIFIED`, `CONFLICTED`);
- source artifact digest;
- timebase identity;
- age and maximum allowed age.

For a `QUALIFIED` value, the encoder requires a nonzero digest, a matching declared common-timebase identity, finite nonnegative freshness values with age no greater than the bound, and a finite in-domain measurement. A digest is an identity pointer, not signer authentication, publisher authenticity or proof of sensor truth.

For any non-qualified state, the encoder emits only a status symbol. It does **not** encode the numeric payload, even when a stale or conflicted record happens to carry a value. The per-channel receipt metadata remains available for review, separate from the latent HDC.

Out-of-domain values are rejected. They are never clamped or saturated into a plausible physical state.

## Initial channel contract

| Signal | Unit / convention | Domain | Encoding |
|---|---|---:|---|
| True-wind speed | m/s | 0–40 | bounded scalar |
| True-wind direction | degrees FROM, clockwise from true north | [0, 360) | sin/cos |
| Apparent-wind speed | m/s | 0–50 | bounded scalar |
| Apparent-wind direction | degrees FROM, clockwise from true north | [0, 360) | sin/cos |
| Heading | degrees clockwise from true north | [0, 360) | sin/cos |
| Speed through water | m/s | 0–20 | bounded scalar |
| Speed over ground | m/s | 0–20 | bounded scalar |
| Current speed | m/s | 0–10 | bounded scalar |
| Current direction | degrees TOWARD, clockwise from true north | [0, 360) | sin/cos |
| Heel | signed degrees | −60–60 | bounded scalar |
| Leeway | signed degrees | −60–60 | bounded scalar |
| Rudder feedback | signed degrees | −90–90 | bounded scalar |
| Sheet feedback | degrees | 0–180 | bounded scalar |
| Waypoint range | m | 0–100,000 | bounded scalar |
| Waypoint bearing | degrees clockwise from true north | [0, 360) | sin/cos |
| Actuator-response latency | ms | 0–5,000 | bounded scalar |
| Significant wave height | m | 0–20 | bounded scalar |

These are v0.1 **input-domain checks for synthetic representation tests**, not universal vessel limits, operating envelopes or sensor specifications. Actual vessel configuration, calibration, units, reference frames and validated ranges must be source-bound before any physical use.

Circular encoding is important: bearings 359° and 1° should be adjacent in representation. A linear 0–360 scalar incorrectly makes them appear almost one full revolution apart.

## Admission summary

The encoder may represent partial evidence so the monitor can perceive *what is absent*. It also emits an admission summary:
- `COMPLETE`: all 17 channels are qualified;
- `DEGRADED`: noncritical channels are missing/stale/unverified/conflicted, while the declared navigation-critical channels remain qualified;
- `CRITICAL_EVIDENCE_INCOMPLETE`: at least one critical channel is not qualified.

The critical set in v0.1 is true-wind speed/direction, heading, speed through water, current speed/direction and waypoint bearing. This summary is not authority to act. The ODD, qualification and encounter-safety owners still decide whether a given downstream function is supportable.

## HDC and temporal layer

The encoder uses Genesis-seeded semantic channel bases, a level codebook, bound value vectors, explicit per-channel state vectors, separate sine/cosine features for circular bearings, and a weighted bundle into a 16,384D `ContinuousHV`.

`SailcraftTemporalState` evolves those representations through `HdcLtcUnifiedNetwork`. The temporal layer:
- accepts only a finite positive timestep bounded to 2 seconds in v0.1;
- preserves the observation's admission and per-channel status metadata;
- rejects a change in timebase identity during one temporal sequence;
- checks output dimension and finite latent values.

A new timebase domain requires a new/reset temporal-state instance; recurrent state is not silently continued across clock discontinuities. The timebase identity check does not itself qualify clock synchronization.

## Falsification suite

The authored unit tests cover:
- circular 359°/1° bearing proximity relative to 180°;
- stale numeric payload non-leakage (8 m/s stale and 18 m/s stale yield the same HDC input when status/metadata match);
- distinct missing/stale/unverified/conflicted symbols;
- invalid source digest and mismatched timebase rejection;
- stale freshness and non-finite measurement rejection;
- out-of-domain wind/bearing rejection without clamping;
- configuration mismatch rejection;
- deterministic Genesis replay;
- critical missingness preserving a representation while blocking policy-input admission;
- per-channel source/timebase/freshness metadata retention;
- finite HDC/LTC latent output and timebase-discontinuity rejection;
- invalid timestep rejection.

## Future benchmark matrix

The representation contract is not an accuracy leaderboard. Later learning/evaluation work should compare matched inputs and compute across:
1. typed scalar vector with explicit masks/status flags;
2. one-hot status + normalized scalar baseline;
3. circular sin/cos encoding;
4. HDC bound-channel representation;
5. HDC with derivative channels;
6. HDC/LTC temporal representation;
7. ablations removing status symbols, circular encoding, configuration binding and common-timebase gating.

Later task-level metrics should include circular angular error, missing/stale/conflict discrimination, out-of-domain detection, source/timebase/configuration misbinding, temporal prediction residual and calibration on held-out episodes, resilience under sensor dropout/common-cause faults, representation stability, encoding latency, memory and compute. Do not infer policy competence from embedding similarity or finite latent output.

## Reproduction

From the Symthaea workspace root:

```sh
cargo fmt --check -p symthaea-sailcraft-cognition
cargo test --locked -p symthaea-sailcraft-cognition
cargo clippy --locked -p symthaea-sailcraft-cognition --all-targets -- -D warnings
```

## Claim ceiling

Passing establishes deterministic evidence-aware sailcraft representation and HDC/LTC temporal-state software conformance on synthetic inputs only. It does not establish a trained sailing policy, autonomous sailing, hydrodynamic validity, collision avoidance, COLREG compliance, vessel stability, seaworthiness, class/flag/MASS approval or permission to operate.
