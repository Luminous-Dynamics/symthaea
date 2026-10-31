# MEL-MR-EVAL-035 — signed, topology-specific tail-profile registry

**Status:** Rust registry and replay-verification code authored. Twelve unit tests are authored. No compilation, test execution, FluidSynth render, topology inspection, profile issuance or runtime qualification is claimed.

## Problem

EVAL-017 v0.2 / EVAL-034 can measure influence tails and apply a conservative guard. A digest field by itself does not prove:
- which renderer executable, SoundFont and effects settings were measured;
- whether the full downstream route/master/output graph was known;
- whether the selected guard covers held-out compositions;
- whether the profile publisher was trusted;
- whether a profile is appropriate for the current render.

EVAL-035 makes the tail guard a signed, renderer-bound statistical artifact.

## Implementation

Module: `src/tail_profile_registry.rs`

### 1. Signed renderer-topology attestation

`RendererTopologyAttestation` commits to:
- renderer identity and executable SHA-256;
- SoundFont SHA-256;
- renderer and render-config digests;
- FluidSynth effects-group assignment;
- effect, reverb and chorus settings digests;
- complete downstream route-graph digest and explicit completeness flag;
- output-chain digest;
- sample rate, channels, bit depth and PCM format.

An Ed25519 authority signs the domain-separated topology digest. Verification checks the authority key ID, signature, digest fields, effects-group assignment and complete-route flag. The returned topology is opaque.

The signer is an explicit trust boundary: signing must only happen after independently inspecting the actual renderer build, SoundFont, settings, plugins, buses, master processing and output chain. A signature authenticates the declaration; it cannot make a false declaration true.

### 2. EVAL-031 lineage binding

The builder accepts only:
- replay-verified `VerifiedCounterfactualSupportAdapterReport` tokens;
- a clean, replay-verified EVAL-031 benchmark manifest/audit;
- a verified topology attestation;
- a build specification and profile signing key.

Each source bundle names one benchmark sample ID. Its score digest and supported operation family must match that exact EVAL-031 manifest row. Composition lineage and split are derived from the manifest, never caller-entered.

Partition policy:
- EVAL-031 `train` → profile calibration;
- EVAL-031 `validation` → held-out profile qualification;
- EVAL-031 `test` → refused.

Cross-split lineage leakage is rejected. Multiple edited events within one composition are aggregated to one independent lineage observation.

### 3. Tail statistics and confidence

For each event:
- final note-off is taken from the verified performed MIDI event lineage;
- raw measured-support end and hysteresis-support end are retained separately;
- observed tail duration is the maximum of raw and hysteresis tail durations;
- all-channel trailing quiet-window counts are computed by EVAL-025 during replay of the EVAL-017 v0.2 window-segmentation trace, stored in each event's digest-bound support evidence, and consumed by the profile builder.

An event is rejected as right-censored unless it has at least the preregistered minimum trailing quiet windows after the final influence window. A clipped output horizon or tail guard is never accepted as a complete tail observation.

For each composition lineage, the registry uses the maximum raw tail, maximum hysteresis tail and maximum of both across all its events. This prevents repeated events from one composition from inflating the independent sample count.

The default research target is:
- 99th-percentile tail duration;
- 95% one-sided distribution-free confidence;
- sample maximum as the upper tolerance bound;
- a separately declared safety margin;
- at least 30 independent held-out qualification lineages with zero guard misses.

The calibration minimum is derived mathematically:

`P(max(X_1,…,X_n) ≥ Q_q) = 1-q^n`

For q=0.99 and confidence=0.95, n must be at least 299 independent composition lineages. This is a distribution-free order-statistic guarantee under the independence/representativeness assumption; it is not proof that future compositions follow the calibration distribution.

Calibration and qualification lineage sets must be disjoint. The profile also reports the empirical raw-tail and hysteresis-tail quantiles separately from the conservative maximum-based upper bound.

### 4. Replayable signed profile

The signed profile retains:
- exact renderer topology/context and attestation digest;
- segmentation configuration digest and operation family;
- benchmark manifest/audit digests;
- full event-level observation references;
- per-lineage cluster summaries;
- calibration and qualification lineage-set digests;
- builder specification and digest;
- empirical quantiles, maximum-based confidence bound, confidence coverage, safety margin and final guard;
- held-out lineage count, tail-miss count, maximum held-out tail and zero-miss upper-rate statistic.

Verification recomputes the event-set digest, lineage summaries, builder digest and all statistics from the retained observations before checking the profile signature. A publisher cannot simply sign a different guard number while retaining contradictory observations.

### 5. Registry and exact resolution

`TailProfileRegistry`:
- loads an explicit trusted publisher key set;
- accepts only verified, signed and qualified profiles;
- rejects duplicate profile digests;
- resolves only when the caller supplies the exact expected `TailProfileContext`;
- rejects revoked profiles;
- returns an opaque `ResolvedTailProfile` that applies the guard and profile digest to EVAL-017 v0.2 segmentation configuration.

Revocation does not rewrite historical profiles or reports.

## Current barrier — deliberately not papered over

EVAL-025/EVAL-034 adapter reports currently set `routing_complete=false` and do not include a complete route-graph digest. They cannot create a usable profile by themselves.

EVAL-035 requires a separate trusted topology attestation that names the exact same renderer executable, SoundFont, configuration, effects assignment and output format. Until a genuinely complete graph can be independently inspected and signed, tail-profile issuance remains blocked. The current source tree contains no such qualified production attestation, and this PR does not fabricate one.

The only currently accepted operation family is `Remove`, because the underlying EVAL-017 counterfactual renderer path is not yet qualified for every intervention family. Other operations fail closed.

## Tests authored

1. 99th-quantile / 95%-confidence calibration needs 299 independent lineages.
2. Nearest-rank quantile and sample-maximum bound are distinct.
3. Topology signature binds the complete route digest.
4. Incomplete route graph cannot mint verified topology.
5. Multiple events from one composition aggregate to its maximum tail.
6. Missing qualification / insufficient calibration fails.
7. Incomplete trailing quiet-window evidence rejects profile construction.
8. Raw-or-hysteresis tail maximum and worst trailing quiet count are retained.
9. Signed registry requires exact context and supports revocation.
10. Changed context is rejected.
11. Re-signing a forged aggregate guard is rejected by replayed statistics.
12. Re-signing modified event observations is rejected by recomputed profile evidence.

Tests are source-authored and **not compiled or executed** in this environment.

## Qualification metrics for the eventual renderer campaign

For each exact topology and supported operation family, report:
- calibration lineage count, upper quantile and confidence coverage;
- held-out lineage count, misses and one-sided upper miss-rate interval;
- empirical raw/hysteresis quantiles;
- excess guard frames and extra render duration;
- censored-output rejection rate;
- p50/p95/p99 observed tail duration;
- false-negative influence energy on adversarial long-tail fixtures;
- render time, CPU and memory overhead.

Do not pool SoundFonts, effects-group counts, output chains or renderer versions into one profile. Each exact context gets its own profile. Do not promote a profile to the default solely because it has a valid signature; require held-out qualification and listener/preservation non-inferiority for user-facing use.

## Limitations

- The 299-lineage guarantee assumes independent, representative composition clusters. EVAL-031 helps prevent known leakage but cannot prove curator lineage declarations are truthful.
- A signed route graph is only as trustworthy as the authority's inspection and key custody.
- The all-channel quiet-window requirement is a measurement completion check, not a proof of metaphysical zero influence.
- The current adapter does not have a verified production route graph, so this PR creates the registry but does not issue a real profile.
- The module is available only with the `theory` feature on non-WASM targets.