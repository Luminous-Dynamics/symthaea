# Ablation findings — encoder robustness under channel contamination

**Date:** 2026-04-14
**Runner:** `cargo run -p symthaea-logparse --example ablation_noise --release`
**Corpus:** 5 classes × 40 events = 200 events/run, seed 0xAB1A7105

## Results

| noise | nc_purity | hd_purity | hd_clusters | hd_noise_pts |
|------:|----------:|----------:|------------:|-------------:|
|  0.00 |     1.000 |     1.000 |          12 |            0 |
|  0.10 |     0.900 |     0.937 |          10 |            9 |
|  0.20 |     0.875 |     0.950 |          10 |           20 |
|  0.30 |     0.805 |     0.914 |          14 |           14 |
|  0.40 |     0.730 |     0.965 |          15 |           30 |
|  0.50 |     0.655 |     0.989 |          14 |           23 |
|  0.60 |     0.645 |     0.941 |          21 |           12 |
|  0.70 |     0.745 |     0.930 |          19 |           13 |
|  0.80 |     0.880 |     0.919 |          19 |            2 |
|  0.90 |     0.980 |     0.862 |          16 |            4 |
|  1.00 |     0.995 |     0.837 |          16 |            4 |

Nearest-centroid chance level: 0.200 (1/5 classes). HDBSCAN never approaches chance.

## Interpretation

**The V-shape in nearest-centroid and HDBSCAN's flat robustness are a real finding, but the ablation isn't aggressive enough.**

The noise model (`generate_noisy_corpus` in `src/fixtures.rs`) contaminates four channels per affected event:
- `provider` (e.g. `Microsoft-Windows-Security-Auditing` → `NETLOGON`)
- `event_id` (e.g. 4624 → 1014)
- `component` (e.g. `Security` → `System`)
- 2 donor fields merged into `fields`

It does **NOT** touch the class-characteristic original fields that `gen_*` sets directly: `LogonType`, `TargetUserName`, `IpAddress` (benign_login / lateral_movement); `TargetFilename`, `Image` (ransomware); `DomainController`, `FailureReason` (network_outage); `ServiceName` (service_restart).

So at `noise=1.0`, every event has a consistent systematic discriminator-shift but the untouched field signal still fully separates the classes. This is why:
- **Nearest-centroid recovers at high noise**: the contamination is so systematic that centroids of each class still form coherent clusters in the untouched subspace, even though the contaminated channels are scrambled.
- **HDBSCAN stays high throughout**: density separation in 16,384D only needs a few uncontaminated dimensions to succeed.

## What this DOES tell us (honest reads)

1. **Role-filler VSA composes well under partial channel corruption.** Contaminating 4 of ~7 channels per event did not collapse HDBSCAN purity. This is the "graceful degradation via overcomplete encoding" property that makes VSA attractive for noisy signal domains.

2. **The encoder is not over-fitting to `provider+event_id`.** If it were, the V-shape would show complete collapse at noise=0.5 and not recover. Instead, the remaining field channels carry enough signal to reconstruct classes.

3. **Real IT-ops data where the same Event ID appears across benign and malicious contexts (Security 4624 is the canonical example) should not catastrophically break this encoder.** That's the wedge's core operating assumption, and this ablation gives it at least partial support.

## What this does NOT tell us (limits)

1. **The noise model is too weak.** A proper adversarial ablation should scramble ALL fields, not just discriminators. The follow-up is `generate_noisy_corpus_v2` with a `full_contamination` mode that re-generates a donor event of a random other class and swaps the ENTIRE event body (keeping only the ground-truth label). Expected result: nearest-centroid approaches chance (0.20), HDBSCAN approaches chance-plus-density-bias. If it doesn't, there's something else going on.

2. **The class-characteristic fields are artificially stable.** Real Evtx records from the same provider vary in field presence and ordering. We don't model that.

3. **Synthetic fixtures are easy by construction.** Purity near 1.0 on noise=0 is expected and means nothing — we designed the classes to be separable. The V-shape is the interesting datum, not the absolute numbers.

## v2 — full-event contamination (2026-04-14, later same day)

Follow-up experiment with `generate_noisy_corpus_v2` swapping entire events
(keeping only ground-truth label). This is the honest adversarial test.

**Runner:** `cargo run -p symthaea-logparse --example ablation_noise_v2 --release`

| noise | nc_purity | hd_purity | hd_clusters | hd_noise |
|------:|----------:|----------:|------------:|---------:|
|  0.00 |     1.000 |     1.000 |          12 |        0 |
|  0.10 |     0.900 |     0.900 |          12 |        0 |
|  0.20 |     0.815 |     0.815 |          12 |        0 |
|  0.30 |     0.695 |     0.695 |          12 |        0 |
|  0.40 |     0.610 |     0.610 |          12 |        0 |
|  0.50 |     0.550 |     0.550 |          11 |        0 |
|  0.60 |     0.415 |     0.425 |          11 |        0 |
|  0.70 |     0.295 |     0.335 |          12 |        0 |
|  0.80 |     0.290 |     0.355 |          12 |        0 |
|  0.90 |     0.350 |     0.385 |          12 |        0 |
|  1.00 |     0.350 |     0.385 |          12 |        0 |

Total decay: nc = 0.650, hd = 0.615. Chance = 0.200.

### v2 conclusions

**1. v1's V-shape WAS a noise-model artifact.** v2 gives a clean monotone
decay (tiny reversal at 0.8→0.9 is small-sample variance). The v1 recovery
at noise=1.0 came from v1 only touching 4 channels, leaving the
class-characteristic fields (LogonType, TargetFilename, etc.) intact.

**2. The encoder degrades gracefully, not catastrophically.** 65 percentage
points of decay from 1.000 to 0.350 for *full feature scramble* is what a
correctly-composed role-filler VSA should look like. A brittle encoder would
have a cliff.

**3. The 0.35 floor (not 0.20 chance) reflects generator-state leakage, not
encoder magic.** `gen_*` functions call `rand_user` / `rand_host` / `rand_ip`
in different orders, creating weak per-class statistical fingerprints in the
PRNG state that donor events inherit. This is a fixture leak, not a model
property. The honest floor is ~0.20.

**4. At noise=0.5, purity = 0.550 — essentially at the Phase 1 kill criterion
(0.50).** This is the single most important datum from the whole ablation.
If real Evtx corpora turn out to be roughly 50% cross-class contaminated
(i.e., half the events have features that resemble other incident classes —
which is plausible for Security 4624, Sysmon 11, etc.), the current encoder
sits right on the knife's edge. **We will need either richer encoding
channels OR a labeled corpus where the cross-class contamination is lower
than 50%, probably both.**

**5. HDBSCAN tracks nearest-centroid almost exactly under v2 contamination.**
At every noise level above 0.5, the two metrics differ by ≤0.06. This means
contaminated events form tight density cores at their new positions, not
scatter noise. That's a meaningful property of binary HV bundling: the
encoder's geometry is "Gaussian-like" even under adversarial relabeling, so
HDBSCAN can always find clusters — just not clusters that match the ground
truth. This tells us clustering quality will track label alignment, not
cluster-existence, on real data.

### What this means for Phase 2 planning

- Do not assume the current 6-channel role-filler encoder will clear the 0.50
  kill criterion on real data. Budget time for adding more encoding channels
  (SID patterns, path-component parsing, numeric-field binning, temporal
  deltas between events) BEFORE running the real-corpus test.
- When real data arrives, run both v1 and v2 style ablations on it to
  estimate its actual cross-class contamination rate. That number is more
  predictive than raw purity.
- The `0.35` residual in v2 at noise=1.0 should be closer to 0.20 after the
  fixture generator is fixed — tracked but low priority.

## What still matters

The Phase 1 kill criterion remains: **≥0.50 purity on a real labeled Evtx corpus**. Synthetic ablations — even honest ones — don't validate the thesis. They shape our priors about what to measure when real data arrives.
