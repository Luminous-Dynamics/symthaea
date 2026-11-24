# symthaea-logparse

**Status:** Research prototype. Not production; not a qualified threat-detection product.

This crate explores whether structured Windows Event Log and syslog records can be normalized and analysed using Symthaea's hyperdimensional computing (HDC) representations. It is part of the MSP / IT-operations research programme.

> [!CAUTION]
> ## Generalization is currently poor
>
> The historical supervised probe achieved **0.817 ± 0.018 event-level 5-fold CV on the same sbousseaden corpus**, but only **0.168 accuracy on OTRF Security-Datasets**, against **0.143 chance across seven labels**. Several classes were near zero. The existing report concludes that the model learned corpus/collection fingerprints rather than robust attack-tactic semantics.
>
> The 0.815/0.817 results are in-distribution only. They are **not** enterprise detection performance, production readiness or SOTA evidence. See [PHASE1_RESULTS.md](PHASE1_RESULTS.md) for the full record and caveats.

## Current evidence

| Experiment | Result | Interpretation |
|---|---:|---|
| EVTX corpus parsed | 261 files, 35,807 events | Research corpus |
| Phase 1 labeled sample | 1,000 events, 8 tactic labels | Stratified event sample |
| Nearest-centroid purity | 0.451 | Below original 0.50 criterion |
| HDBSCAN purity | 0.565 at min cluster 10 | Passes that corpus-specific clustering criterion only |
| Linear probe, single event split | 0.815 | In-distribution only |
| Linear probe, 5-fold event CV | 0.817 ± 0.018 | Same-corpus in-distribution |
| Cross-corpus OOD | 0.168, chance 0.143 | Severe generalization failure |

These numbers come from different experiment designs and should not be combined into a single performance claim.

## Implementation status and limitations

- **Ingestion:** offline EVTX parsing and RFC5424/RFC3164 syslog parsing exist.
- **Normalization:** `LogEvent` captures source, severity, provider, event ID, component, fields, host and optional research label.
- **Encoding:** current `encoder.rs` uses a local, deterministic 16,384-dimensional bipolar MAP-I reference implementation. The `hdc-encoder` feature is still a placeholder; actual integration with `symthaea-core` is not established.
- **Clustering:** nearest-centroid and HDBSCAN evaluation code exist.
- **Supervised probe:** a pure-Rust logistic probe exists. The updated `supervised_probe` runner carries source EVTX filename as split-only provenance and supports grouped file-disjoint 80/20 holdout and five-fold CV. The grouping information is not encoded as a model feature.
- **Not implemented:** production Windows ETW collector, production SIEM/EDR pipeline, live endpoint response, independently qualified detection logic, enterprise benign baseline, or SOC workflow.

A file-disjoint split reduces one source of leakage. It does not establish host-, incident-, campaign-, time- or cross-corpus generalization. The known OOD failure remains until independently reproduced and repaired.

## Running the experiments

Run from the Symthaea workspace in its pinned Nix development environment.

```bash
# Pipeline sanity only; synthetic data is not a benchmark result.
cargo run -p symthaea-logparse --example cluster_evtx --release -- --synthetic

# Historical corpus clustering study.
cargo run -p symthaea-logparse --example cluster_evtx --release -- /path/to/evtx-corpus

# File-disjoint supervised holdout and grouped five-fold CV.
# labels.csv maps each filename to one class; each class needs at least five
# independent source EVTX files for five-fold grouping.
cargo run -p symthaea-logparse --example supervised_probe --release -- /path/to/evtx-corpus

# Historical scale-up, event-level CV and cross-corpus OOD experiment.
# Treat the event-level CV as in-distribution only.
cargo run -p symthaea-logparse --example phase2_experiments --release -- \
  /path/to/evtx-staged /path/to/otrf-staged
```

Expected corpus layout:

```text
evtx-corpus/
  labels.csv
  incident_a.evtx
  incident_b.evtx
  ...
```

`labels.csv` format:

```csv
filename,label
lateral_movement_01.evtx,lateral_movement
ransomware_03.evtx,ransomware
benign_baseline.evtx,benign
```

The file-disjoint runner refuses to score if any label has fewer than five independent file groups or a provenance group appears under multiple labels. This is intentional: do not silently fall back to event-level splits.

## Next research priorities

1. Reproduce the 0.168 cross-corpus result from frozen source corpora, parser versions and split manifests.
2. Add a third independently collected corpus and benign enterprise-like background telemetry; measure false alerts per endpoint-day.
3. Add incident/campaign-, host- and temporal-disjoint split manifests, retaining source lineage without exposing it as a model feature.
4. Normalize provider names, EventIDs and source-specific field layouts; explicitly measure collection-methodology leakage.
5. Compare rules, TF-IDF/linear models, conventional ML, nearest-centroid, the local HDC reference encoder and the real Symthaea core encoder on identical held-out splits.
6. Integrate and reproduce the actual Symthaea HDC encoder; remove the placeholder only when the adapter is working and tested.
7. Report per-class precision/recall/F1, macro-F1, PR-AUC, calibration, uncertainty and false-positive rates—not accuracy alone.
8. Evaluate incident triage and analyst usefulness only after event-level generalization is demonstrated.

## Claim discipline

A synthetic fixture validates wiring, not the thesis. Same-corpus CV does not establish cross-corpus robustness. Clustering purity does not establish attack detection. A detection score does not establish safe containment or verified recovery.

The current honest product description is: **an experimental log-normalization and HDC research pipeline with a documented cross-corpus generalization failure and new file-disjoint evaluation support.**
