# Connectome gait outcome bridge V1

**Tracking:** #11007 — SCI-CONNECTOMICS-103  
**Composes:** #9934 (Pugliese source reproduction), #10919 / PR #10922 (phase-reset transfer), Symtropy #3353 / PR #3354 (raw gait metrics), #2145 (animal replay scorecard).  
**Status:** source-bound typed endpoint envelope and synthetic coverage oracle; no biological metric reproduction is claimed.

## 1. Problem: one “gait score” is scientifically unsafe

The pinned Pugliese Anipose notebook computes local per-joint autocorrelation rhythmicity and per-leg stepping-frequency outputs. Those do not measure whether the six legs are coordinated. Pairwise phase locking, tripod overlap, phase-reset transfer and physical forward motion are different estimands.

The source's two Anipose inputs also differ materially:
- DNg100 input includes FicTrac forward-velocity fields.
- DNb08 was recorded off-ball and has no `fictrac_*` columns.

Absence of a sensor channel is NOT_AVAILABLE, not zero velocity. The notebook's millions of frame rows are time samples nested within trials/fly/session, not independent biological replicates.

## 2. Endpoint lattice

V1 retains six separate endpoint classes:
- `LOCAL_JOINT_RHYTHMICITY`: autocorrelation score and/or dominant frequency for one named leg/joint channel.
- `PER_LEG_STEPPING`: per-leg cycle frequency/event statistics.
- `PAIRWISE_INTERLEG_COUPLING`: an explicit leg pair and named phase metric/pipeline.
- `TRIPOD_COORDINATION`: one of the two alternating tripod groups under a named TCS formula.
- `PHASE_RESET_TRANSFER`: dynamic sender/recipient perturbation response from #10922; fractional phase and integer cycle slips stay separate.
- `EMBODIED_BEHAVIOUR`: velocity/contact/stability under the sensor modalities actually available.

These are not promotion levels. Local rhythmicity cannot certify inter-leg coordination; PLV cannot certify tripod overlap; TCS cannot certify physical locomotion; a phase-reset response is not steady-state PLV.

## 3. Source and metric ownership

- Pugliese Anipose and neural oscillation source processing: #9934.
- Phase-reset/cycle-winding measurement algebra: #10919 / PR #10922.
- Raw TCS, phase-vector strength and duty factor: Symtropy #3353 / PR #3354.
- Paired inference/scorecard and animal-level aggregation: Symtropy #2145.
- This crate validates **provenance, metric identity, endpoint compatibility and coverage only**. It does not compute any raw metric or aggregate biological replicates.

Symtropy PR #3354's conventional phase-vector strength is not automatically Sapkal v2's paper-specific phase pipeline. The latter requires separate preprocessing and phase-estimator digests. A pairwise coverage family is complete only when all 15 pairs share the same metric, method, preprocessing and phase-estimator digests. A phase-reset probe is counted only when its fractional-phase and cycle-slip outputs share the same probe key and exact pipeline identity. TCS NatVar 2024 and TCS Sapkal 2026 v2 use distinct denominators and must never be pooled into one median. The Sapkal v2 ratio is not clamped.

## 4. Packet boundary

One packet is one immutable dataset revision × raw trajectory × animal × session × trial × condition × stimulus. Each measurement row repeats and must match that packet source-binding tuple. A packet cannot pool DNg100 and DNb08 rows or distinct conditions. Cross-trial and cross-animal statistics belong to the qualified scorecard, never this trial-level bridge.

Every row carries:
- endpoint, exact metric ID, unit and aggregation level;
- method and optional preprocessing/phase-estimator digests;
- exact probe class for intervention rows: local phase kick, load/contact, proprioceptive channel, descending command, or mechanical body/substrate;
- named leg, pair, tripod group or transfer probe key where relevant;
- availability state, value, support/sample counts and frame window;
- repeated dataset/revision/raw digest plus animal/session/trial/condition/stimulus binding.

Frame-level rows are rejected as inferential outcomes. Frame count remains source/timebase metadata; animal/session/trial identity is the biological nesting.

## 5. Exact synthetic known answer

Two independent synthetic packets contain six local-rhythm rows, six per-leg stepping rows, one paired phase-reset probe (fractional response plus cycle winding), no pairwise phase metrics and no TCS rows.

| Packet | Rhythm | Stepping | Pairwise | Tripod groups | Phase-reset probes | FicTrac |
|---|---:|---:|---:|---:|---:|---|
| DNg100, ball contact | 6/6 | 6/6 | 0/15 | 0/2 | 1/8 (partial) | MEASURED |
| DNb08, air stepping | 6/6 | 6/6 | 0/15 | 0/2 | 1/8 (partial) | NOT_AVAILABLE |

Both summaries must be `RHYTHM_ONLY`; neither is permitted to claim coordination or walking certification. The report always includes `emits_global_gait_certification = false`.

## 6. Required negative controls

- high local rhythmicity promoted to inter-leg coordination;
- generic vector strength relabelled as Sapkal paper-specific PLV without the filter/phase-estimator receipt;
- the two TCS metric IDs pooled into one summary;
- Sapkal TCS silently clamped to [0,1];
- one tripod group represented as both groups;
- 14/15 leg pairs reported as complete;
- mixed pairwise extraction-method hashes treated as one complete grid;
- same trial packet with a row from another animal/session/stimulus/dataset revision;
- frame-level records treated as independent trials;
- negative fractional phase-reset or cycle-slip values rejected despite being valid signed responses;
- a sender-to-same-recipient phase-reset row confused with an invalid PLV self-pair;
- DNb08 velocity missing column converted to numeric zero;
- out-of-window and missing/unavailable outcomes silently imputed.

## 7. Qualification and claim ceiling

V1 qualifies schema, source/subject identity consistency, metric compatibility and endpoint-coverage classification on synthetic packets. It does not prove the upstream signal processor, verify the original Parquet/Zenodo source bytes, establish animal-level statistics, reproduce Pugliese/Sapkal results, identify a neural mechanism, demonstrate SOTA superiority or certify a walking controller.

Biological intake remains blocked until #9934 resolves the paper-versus-README Zenodo record discrepancy (22260923 vs 22260924), verifies source rights, and binds actual data artifacts.
