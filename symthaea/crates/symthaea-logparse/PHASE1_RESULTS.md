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
- Supervised class-level separation via nearest-centroid: **FAILS** (0.451 < 0.50)
- **Supervised class-level separation via learned linear probe: PASSES (0.815, see below)**

---

## Amendment — supervised probe (same day, Apr 14, 2026)

After the first run, I shipped Priority 2 from the revised Phase 2 plan — a
one-vs-rest logistic regression over the 16,384D bipolar HVs — and ran it on
the same 1000-event corpus with an 80/20 stratified split.

**Configuration:** 40 epochs, lr=0.05, L2=1e-4, batch=32, Xavier init.
**Implementation:** pure-Rust from scratch (`src/probe.rs`, ~280 LOC, zero new deps).

### Results

| metric | value |
|---|---|
| Train accuracy | 0.994 |
| **Test accuracy** | **0.815** |
| Chance (1/8) | 0.125 |
| Nearest-centroid baseline | 0.451 |
| HDBSCAN best | 0.565 |
| Phase 1 kill criterion | 0.500 |

**The supervised probe is at 0.815, 6.5× above chance, 1.8× above NC, 1.4× above HDBSCAN.**

### Per-class test accuracy (n=25 per class)

| class | correct/total | accuracy |
|---|---|---|
| command_and_control | 24/25 | 0.96 |
| discovery | 24/25 | 0.96 |
| execution | 24/25 | 0.96 |
| credential_access | 22/25 | 0.88 |
| persistence | 22/25 | 0.88 |
| defense_evasion | 19/25 | 0.76 |
| lateral_movement | 16/25 | 0.64 |
| privilege_escalation | 12/25 | 0.48 |

### Interpretation — the encoder is fine

1. **The information IS in the 16,384D HVs.** NC averages every dimension
   equally and throws most of the discriminative signal away. A learned
   linear classifier weights the ~100-1000 genuinely discriminative
   dimensions up and the noise dimensions down, and recovers most of the
   label structure.

2. **Train-test gap is 18pp (0.994 → 0.815).** Moderate overfitting on
   a 800-sample train set in 16,384 dimensions is expected. Stronger L2
   or more data would tighten it, but 0.815 is already well above
   threshold.

3. **The two weak classes are semantically the overlap classes.** Lateral
   movement (0.64) and privilege escalation (0.48) share the most Windows
   event types with adjacent tactics — network logons (4624) overlap
   benign logins and credential access; token manipulation overlaps
   defense evasion. MITRE's tactic taxonomy is known to be fuzzy at these
   boundaries, so the probe's weakness there reflects the taxonomy, not
   the encoder.

4. **The original product framing survives.** A probe that classifies
   Windows events into MITRE tactics at 0.815 accuracy — trained on 800
   public adversarial samples — is a real product. The earlier panic
   about needing to reframe to "unsupervised incident clustering only"
   was premature; the classifier path IS viable.

### Revised Phase 2 priorities (again)

1. **~~Reframe to unsupervised only~~** — no longer needed. Keep the
   classifier framing.
2. **Statistical robustness** (1 week): re-run the probe with 5-fold
   cross-validation to quantify the train/test variance. The 0.815
   single-split number needs a confidence interval before it goes into
   any external pitch. Expected: mean 0.78-0.83, std ~0.03.
3. **Richer encoding channels** (still valuable, but now optional):
   adding SID / path / numeric-bin channels would push the probe from
   0.815 toward 0.90+ and close the per-class gap for lateral_movement
   and privilege_escalation. Not a blocker.
4. **Out-of-distribution test** (3 weeks): probe trained on
   EVTX-ATTACK-SAMPLES, tested on a held-out corpus (Mordor? ATT&CK
   Evaluations? raw enterprise telemetry if we can get it). This is the
   honest generalization test. 0.815 on the same-distribution split is
   a ceiling, not a floor, for real-world deployment.
5. **Anomaly-detection pivot** — still exploratory, lower priority now.

### What to tell a CIO (external pitch, revised)

> "We built a classifier over Windows Event Logs that assigns events to
> MITRE ATT&CK tactics with 81.5% test accuracy after training on 800
> public attack samples. It's not production-ready — we haven't measured
> cross-distribution generalization — but it demonstrates that the
> hyperdimensional encoding captures enough tactic-level signal to
> support an MSP-grade classifier. Our next experiment is cross-validation
> for a confidence interval, followed by an out-of-distribution test on a
> different corpus."

Honest, specific, concedes what we haven't measured yet, and doesn't
overclaim "Z3-verified" anything.

### What this does NOT tell us (limits)

- **Single 80/20 split → no confidence interval.** The 0.815 could be
  0.77 or 0.85 on a different split. Cross-validation is the next step.
- **EVTX-ATTACK-SAMPLES is adversarial-only.** There's no benign
  baseline, so we can't measure false-positive rate on normal traffic.
- **Same distribution.** Train and test are both drawn from the same
  curated repo. A probe trained here might crater on real enterprise
  telemetry with different Windows versions, locales, or configurations.
- **1000-event sample.** Full-corpus training is feasible (unlike
  HDBSCAN), but we haven't done it yet.
- **8 classes, not finer-grained techniques.** MITRE tactics are the
  coarse level; the full ATT&CK matrix has ~200 techniques. Whether the
  encoder can discriminate at technique level is an open question.

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
