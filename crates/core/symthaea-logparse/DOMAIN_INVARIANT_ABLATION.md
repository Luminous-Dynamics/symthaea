# Domain-invariant encoder ablation

## Why this experiment exists

The recorded Phase 2 results in [PHASE1_RESULTS.md](PHASE1_RESULTS.md) report 0.817 ± 0.018 five-fold in-corpus accuracy and 0.168 cross-corpus accuracy on OTRF (seven classes; uniform-chance reference 0.143). These are prior repository-recorded results, not measurements reproduced by this change. The historical five-fold split was event-level: events from one EVTX capture could appear in both train and test, so 0.817 ± 0.018 is not leakage-resistant and may overstate generalization. The OTRF test is a separate-corpus evaluation, but it does not erase the in-corpus split problem. This PR switches future in-corpus evaluation to source-file-grouped splits and grouped folds; do not compare the new grouped numbers directly to the historical event-level CV as if the protocols were equivalent.

The gap is the key failure signal: the representation may be learning collection and serialization fingerprints more than portable security semantics.

The baseline encoder directly binds raw `provider`, `event_id`, and `component`. The new `encode_domain_invariant` is an additive, explicitly experimental ablation intended to test whether removing some producer-specific channels changes cross-corpus generalization. It does not replace `encode`, modify existing benchmark results, or establish that the new representation is better.

## What v1 changes

- Omits raw provider and component strings.
- Uses provider/channel tokens only to select a recognized namespace (Sysmon, Security Auditing, Service Control Manager, or PowerShell), then maps only known IDs within that namespace into coarse families. Raw provider/component strings are discarded from the vector; unknown providers or IDs map to an `unknown` family.
- Maps common structured-field aliases to canonical field names (for example, `TargetUserName` and `user` to `identity_user`).
- Preserves source, severity, and raw field values. High-cardinality value normalization and free-text message encoding are intentionally not included in v1 so this remains a narrow, interpretable ablation.
- Uses an `unknown` family for unmapped IDs and unknown namespaces. Known semantic families can be shared across recognized providers (for example, Sysmon process-create and Security process-create events); a colliding numeric ID from an unknown provider does not inherit that meaning. This mapping remains a bounded heuristic, not a universal taxonomy.

The implementation is **not OCSF-compliant**. The versioned `DOMAIN_INVARIANT_ENCODER_VERSION` constant is included in the hash namespaces and printed with ablation output; bump it whenever provider recognition, family mapping, or field canonicalization semantics change, and keep each result tied to the exact commit.

The long-term direction should be a version-pinned, schema-driven normalization layer rather than growing this alias table indefinitely. OCSF models events as semantic event classes with class-specific attributes and activity IDs; for example, its [Process Activity](https://github.com/ocsf/ocsf-schema/blob/main/events/system/process_activity.json) and [Authentication](https://github.com/ocsf/ocsf-schema/blob/main/events/iam/authentication.json) classes carry distinct semantics. This ablation only maps to coarse families, does not populate OCSF class/category/activity identifiers, and does not validate required attributes. It must not be presented as an OCSF implementation. See the [OCSF event-class model](https://github.com/ocsf/ocsf-docs/blob/main/overview/understanding-ocsf.md) for the fuller mapping contract.

## Run the paired experiment

Stage the same EVTX-ATTACK-SAMPLES and OTRF corpora as described in the example and run:

```bash
cargo run -p symthaea-logparse --example phase2_experiments --release -- \
  /tmp/evtx-staged /tmp/otrf-staged --domain-invariant-ablation
```

The ordinary baseline experiments now use source-file-grouped train/test and 5-fold evaluation. The EVTX loader retains a stable group ID per source file, and OTRF recursive JSON-file traversal is sorted for repeatable corpus ordering. The optional ablation then uses the same grouped train/test seed as the baseline, trains a second probe on the domain-invariant vectors, and reports:

- in-distribution test accuracy;
- OOD accuracy;
- macro precision, macro recall, and macro F1;
- per-class support, predicted count, TP/FP/FN, precision, recall, F1, and the full confusion matrix;
- the corresponding raw-encoder metrics for direct comparison;
- a margin-ranked selective-prediction curve at 50%, 75%, 90%, and 100% target coverage, reporting accepted accuracy and macro recall for both encoders.

The selective curve uses the top-1 minus top-2 model-logit margin to rank examples, then describes performance when the least-separated classifications are withheld. It reports the margin threshold used; if several examples tie at the boundary, all are retained, so actual coverage can exceed target coverage rather than split ties according to row order. For macro recall, any class with no accepted examples contributes zero, a conservative convention that makes class starvation visible. This margin is **not a calibrated probability**; the curve does not establish a safe production threshold. Any actual abstention policy needs a separate validation split, per-class coverage analysis, and explicit review of the cost of false negatives versus analyst workload.

Accuracy alone can be dominated by common classes; read it alongside macro metrics and the confusion matrix. Class-level false positives here mean confusion among the labeled attack classes—not false alarms on benign enterprise activity. The staged OTRF corpus does not provide the benign control needed to measure operational false-positive rates. The script prints measurements from the currently staged corpora; do not copy historical values into a new result as if it had been reproduced.

## Ingestion integrity

The same PR also hardens the OTRF/EVTX ingestion path used by this benchmark:

- Event IDs are narrowed with checked integer conversion. An out-of-range number or string maps to the existing unknown/default ID instead of wrapping into a valid Windows EventID.
- OTRF high-cardinality field values are truncated at a valid UTF-8 boundary, so a multibyte character at the byte cap cannot panic the parser.
- Recursive OTRF JSON files and EVTX inputs are traversed in sorted path order for deterministic sampling.

These checks improve parser robustness but do not make the corpus trustworthy by themselves: skipped malformed records should still be monitored, source manifests should be pinned, and source-label assignments require independent audit.

## Decision discipline

This first run is exploratory. A candidate result should be treated as promising only if OOD macro recall improves by at least 5 absolute percentage points over the raw-encoder baseline and in-corpus accuracy drops by no more than 5 points. This threshold is a predeclared screening rule, not a production acceptance criterion.

Even if it passes that screen, it still needs multiple deterministic seeds, a held-out third corpus, benign telemetry and false-positive measurement, per-class confusion analysis, calibration/abstention evaluation, and stronger scenario-aware holdouts where one capture is not equivalent to one attack scenario. Grouping by source file is a minimum leakage control, not sufficient evidence of deployment generalization.

Do not report a security detection pass from classification accuracy alone. A robust evaluation needs detection quality, false alarms, alert precision, diagnostic quality, uncertainty behavior, and an independent holdout. These dimensions are consistent with the direction of the [MITRE ATT&CK Evaluations Enterprise 2026 methodology](https://attackevals.mitre.org/enterprise/er8), which moves beyond a binary detected/not-detected verdict.

## Safety and scope

This experiment operates only on offline, staged telemetry corpora. It does not run attacks, perform live host collection, or authorize automated containment. Labels are used only by the evaluator and are not part of the encoded event fields.
