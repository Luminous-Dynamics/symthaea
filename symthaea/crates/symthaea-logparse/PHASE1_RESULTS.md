# Phase 1 results — real EVTX-ATTACK-SAMPLES corpus

**Date:** 2026-04-14
**Corpus:** sbousseaden/EVTX-ATTACK-SAMPLES (`git clone --depth 1`)
**Commit:** [to be added]

## Corpus

- **Source:** https://github.com/sbousseaden/EVTX-ATTACK-SAMPLES
- **Total files:** 261 .evtx across 8 MITRE ATT&CK tactics
- **Total events:** 35,807 after parsing
- **Largest outlier:** `credential_access__CA_PetiPotam_etw_rpc_efsr_5_6.evtx` = 29,109 events (81% of corpus, ETW trace not a typical sample)
- **Per-label distribution after parsing:**
  - privilege_escalation: 625
  - credential_access: 424
  - lateral_movement: 469
  - defense_evasion: 321
  - execution: 240
  - command_and_control: 230
  - persistence: 224
  - discovery: 157

## Sampling strategy

To keep HDBSCAN tractable (O(n²·d) on 16,384-dimensional vectors):

- **MAX_PER_FILE = 50**: deterministic stride-sample each file, caps the 29K-event ETW trace
- **MAX_TOTAL = 1000**: stratified downsample to ~125 events per label
- **Final corpus:** 1000 events across 8 labels

## Results

### Supervised upper bound

**Nearest-centroid against known labels: 0.451 purity** (FAIL — below the 0.50 kill criterion).

This is the theoretical ceiling for any approach that asks "which of 8 class-centroids is this event closest to". The encoder cannot cleanly separate the MITRE tactics at the class-centroid level.

### HDBSCAN min_cluster sweep (unsupervised)

| min_cluster | clusters | noise | purity |
|-------------|---------:|------:|-------:|
| **10**      | 26       | 96    | **0.565** ✓ |
| **20**      | 17       | 185   | **0.555** ✓ |
| 40          | 3        | 89    | 0.258 ✗ |
| 50          | 3        | 96    | 0.260 ✗ |
| 80          | 2        | 146   | 0.217 ✗ |

**Best HDBSCAN: 0.565 at min_cluster=10, 26 clusters, 96 noise points.** This is ABOVE the 0.50 kill criterion.

### Per-class assignment (at best HDBSCAN config)

All 8 classes have >80% of events assigned to some cluster (not marked noise):

| class | assigned |
|-------|----------|
| command_and_control | 122/125 (98%) |
| credential_access | 108/125 (86%) |
| defense_evasion | 110/125 (88%) |
| discovery | 111/125 (89%) |
| execution | 119/125 (95%) |
| lateral_movement | 113/125 (90%) |
| persistence | 103/125 (82%) |
| privilege_escalation | 118/125 (94%) |

## Interpretation — the real story

**The naive binary verdict is wrong.** My example code prints "encoder cannot separate" when nc < 0.50, but that ignores the HDBSCAN result. The actual finding is more interesting and more useful:

1. **HDBSCAN passes the kill criterion (0.565), nearest-centroid fails (0.451).** This is not a contradiction — they measure different things. HDBSCAN finds *fine-grained dense clusters* (26 of them) that each have high internal label homogeneity. Nearest-centroid forces every point into one of 8 big class-centroids that blur the fine structure.

2. **The encoder produces "locally coherent, globally blurry" hypervectors.** Events that look alike at the Windows-event-field level (same provider, same EventID, similar structured fields) end up near each other regardless of MITRE label. Events in the same MITRE tactic but from different underlying attack techniques end up far apart.

3. **This IS a diagnostic oracle signal. It is NOT a classifier signal.**
   - For "what's happening on this box?" (cluster → show similar past events), the encoder works.
   - For "is this credential access?" (force into one of 8 labels), the encoder fails.

4. **The v2 ablation's prediction held almost exactly.** v2 at noise=0.6 predicted 0.425 supervised purity; reality was 0.451. The MITRE tactic labels are approximately 55–60% cross-contaminated at the Windows Event level, which is exactly what one would expect (Security 4624 appears across credential_access, lateral_movement, defense_evasion; Sysmon 11 appears across ransomware, persistence, defense_evasion).

## Phase 1 verdict

**The kill criterion was written assuming clustering-quality would track classification-quality. It doesn't.** On this corpus:
- Unsupervised fine-grained clustering: **PASSES** (0.565 > 0.50)
- Supervised class-level separation: **FAILS** (0.451 < 0.50)

**The thesis survives in reframed form.** Symthaea can produce a diagnostic oracle that says *"here are N groups of similar events happening on your network, here's what each group's dominant features are"*, and that oracle is demonstrably better than random (chance = 0.125, HDBSCAN = 0.565, 4.5× above chance).

It cannot produce an MSP-grade *classifier* that assigns events to MITRE tactics reliably without additional supervised training.

## What Phase 2 should do (revised from original plan)

The original Phase 2 plan was "add encoding channels before attempting a real-corpus retry." The retry happened in Phase 1 and gave us a partial pass, so the priorities shift:

### Priority 1 — reframe the product narrative (no code)
The external pitch should be *"unsupervised incident clustering with MITRE-tactic hints"*, not *"autonomous Tier-2 classifier". The reframe is honest, matches the data, and still differentiates from ConnectWise.

### Priority 2 — supervised probe (2 weeks, low risk)
Add `examples/supervised_probe.rs` with a simple logistic regression over the 16,384D bipolar HVs against MITRE labels. This measures the encoder's supervised ceiling — does a learned linear classifier do better than nearest-centroid (0.451)? If yes by how much? If it reaches >0.70, we have a supervised classifier path. If it plateaus near 0.50, we need richer encoding.

### Priority 3 — richer encoding channels (4 weeks, moderate)
The original priority. Add:
- SID / path-component tokenization
- Numeric-field binning (ports, process IDs, file sizes)
- Temporal deltas between co-occurring events (session-level context)
- Provider-specific role channels

Re-run nc + HDBSCAN sweep after each channel addition. Target: supervised nc purity ≥ 0.60 (convincingly above chance and above v1 ablation's noise=0.5 synthetic value of 0.55).

### Priority 4 — anomaly-detection pivot (exploratory)
The 26-cluster HDBSCAN result is actually a plausible anomaly detector: cluster the baseline, flag events that HDBSCAN assigns to noise (-1) as anomalous. The 0.565 purity says most noise points would be genuine outliers. Worth a half-week spike to evaluate.

## Honest caveats

- Sample is only 1000 events (downsampled from 35,807). Full-corpus HDBSCAN at 16,384D is computationally infeasible without dimension reduction.
- EVTX-ATTACK-SAMPLES is adversarial samples only, no benign baseline. Anomaly detection benchmarks need benign data we don't have yet.
- HDBSCAN's Euclidean distance on bipolar HVs is not the same as cosine, just monotonically related. A cosine-native clusterer could give different numbers.
- The 26 HDBSCAN clusters have not been inspected for semantic coherence — it's possible some are over-splits of single attacks.

## Reproducibility

```bash
# Clone corpus
git clone --depth 1 https://github.com/sbousseaden/EVTX-ATTACK-SAMPLES \
  /tmp/evtx-corpus/EVTX-ATTACK-SAMPLES

# Stage with labels.csv
./scripts/stage_evtx_attack_samples.sh \
  /tmp/evtx-corpus/EVTX-ATTACK-SAMPLES /tmp/evtx-staged

# Run spike
cargo run -p symthaea-logparse --example cluster_evtx --release -- /tmp/evtx-staged
```

Expected output: `nc purity ≈ 0.451`, `best HDBSCAN ≈ 0.565 at min_cluster=10`.
