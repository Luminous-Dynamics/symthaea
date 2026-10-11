# Domain-invariant encoder ablation

## Why this experiment exists

The recorded Phase 2 results in [PHASE1_RESULTS.md](PHASE1_RESULTS.md) report 0.817 ± 0.018 five-fold in-corpus accuracy and 0.168 cross-corpus accuracy on OTRF (seven classes; uniform-chance reference 0.143). These are prior repository-recorded results, not measurements reproduced by this change. The historical five-fold split was event-level: events from one EVTX capture could appear in both train and test, so 0.817 ± 0.018 is not leakage-resistant and may overstate generalization. The OTRF test is a separate-corpus evaluation, but it does not erase the in-corpus split problem. This PR switches future in-corpus evaluation to source-file-grouped splits and grouped folds; do not compare the new grouped numbers directly to the historical event-level CV as if the protocols were equivalent.

The gap is the key failure signal: the representation may be learning collection and serialization fingerprints more than portable security semantics.

The baseline encoder directly binds raw `provider`, `event_id`, and `component`. The new `encode_domain_invariant` is an additive, explicitly experimental ablation intended to test whether removing some producer-specific channels changes cross-corpus generalization. It does not replace `encode`, modify existing benchmark results, or establish that the new representation is better.

## What v1 changes

- Omits raw provider and component strings.
- Maps a bounded list of known Windows EventIDs into coarse families: authentication, process, file, network, registry, service/task, scripting, and object access.
- Maps common structured-field aliases to canonical field names (for example, `TargetUserName` and `user` to `identity_user`).
- Preserves source, severity, and raw field values. High-cardinality value normalization and free-text message encoding are intentionally not included in v1 so this remains a narrow, interpretable ablation.
- Uses an `unknown` family for unmapped numeric IDs. Event IDs can collide across providers, so the mapping is a heuristic, not a universal taxonomy.

The implementation is **not OCSF-compliant**. The long-term direction should be to normalize inputs through a versioned, vendor-neutral schema such as [OCSF](https://github.com/ocsf/ocsf-schema), rather than growing a large ad-hoc alias table indefinitely.

## Run the paired experiment

Stage the same EVTX-ATTACK-SAMPLES and OTRF corpora as described in the example and run:

```bash
cargo run -p symthaea-logparse --example phase2_experiments --release -- \
  /tmp/evtx-staged /tmp/otrf-staged --domain-invariant-ablation
```

The ordinary baseline experiments now use source-file-grouped train/test and 5-fold evaluation. The EVTX loader retains a stable group ID per source file, and OTRF recursive JSON-file traversal is sorted for repeatable corpus ordering. The optional ablation then uses the same grouped train/test seed as the baseline, trains a second probe on the domain-invariant vectors, and reports:

- in-distribution test accuracy;
- OOD accuracy;
- OOD macro recall, averaged equally across the represented classes;
- the corresponding raw-encoder OOD accuracy and macro recall.

Accuracy alone can be dominated by common classes; read it alongside macro recall and the per-class breakdown. The script prints measurements from the currently staged corpora; do not copy historical values into a new result as if it had been reproduced.

## Decision discipline

This first run is exploratory. A candidate result should be treated as promising only if OOD macro recall improves by at least 5 absolute percentage points over the raw-encoder baseline and in-corpus accuracy drops by no more than 5 points. This threshold is a predeclared screening rule, not a production acceptance criterion.

Even if it passes that screen, it still needs multiple deterministic seeds, a held-out third corpus, benign telemetry and false-positive measurement, per-class confusion analysis, calibration/abstention evaluation, and a grouped split by source file or attack scenario. The existing event-level split can place events from the same capture in train and test; it is not sufficient evidence of deployment generalization.

Do not report a security detection pass from classification accuracy alone. A robust evaluation needs detection quality, false alarms, alert precision, diagnostic quality, uncertainty behavior, and an independent holdout. These dimensions are consistent with the direction of the [MITRE ATT&CK Evaluations Enterprise 2026 methodology](https://attackevals.mitre.org/enterprise/er8), which moves beyond a binary detected/not-detected verdict.

## Safety and scope

This experiment operates only on offline, staged telemetry corpora. It does not run attacks, perform live host collection, or authorize automated containment. Labels are used only by the evaluator and are not part of the encoded event fields.
