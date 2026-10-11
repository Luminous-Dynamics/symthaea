# Capture/scenario-level LogParse evaluation

## Why this is a separate measurement

A grouped event split prevents events from the same EVTX file leaking into both train and test, but its accuracy is still event-weighted: a capture with many retained events can contribute many predictions. This experiment adds a second unit of analysis where one EVTX source file or one OTRF scenario directory produces one example.

The library helper `scenario::aggregate_scenarios` validates that every group has one non-empty ground-truth label, bundles that group’s already-encoded event vectors, records the event count, and returns scenarios sorted by stable group ID. Labels and group IDs are evaluation metadata; neither is encoded in the vector.

## How aggregation works—and what it does not do

The initial implementation uses an orderless HDC bundle over event vectors. It tests whether the collection of signals in a capture carries transferable class information after reducing each capture to one sample. It **does not preserve event order, time gaps, causal relationships, or attack-chain transitions**. It must not be described as temporal correlation, incident reconstruction, or an autonomous SOC analyst.

The aggregate may be dominated by a large number of unrelated or repeated events. The 50-event cap keeps the per-file contribution bounded, but the result still depends on the source collection and labeling scheme. A capture label is assumed to apply to the retained events as a whole; mixed-technique captures require more precise scenario annotations.

## Run the paired comparison

Stage the same EVTX-ATTACK-SAMPLES and OTRF directories described by the Phase 2 example. Then run:

```bash
cargo run -p symthaea-logparse --example phase2_experiments --release -- \\
  /tmp/evtx-staged /tmp/otrf-staged \\
  --scenario-level-evaluation --domain-invariant-ablation
```

The example still runs the grouped event-level baseline and optional domain-invariant ablation. The scenario flag additionally reports, for the raw and domain-invariant variants:

- source-capture/scenario grouped five-fold CV accuracy;
- a separate 80/20 source-scenario holdout report;
- OTRF scenario-level OOD metrics;
- per-class support, predicted counts, TP/FP/FN, precision, recall, F1, and confusion matrix.

Each model is trained on one bundled vector per EVTX file and evaluated on one bundled vector per OTRF scenario directory. If any class has fewer than five independent source groups, grouped cross-validation fails closed rather than silently falling back to an event-level split.

## Interpretation discipline

Compare results from the same exact commit and the same staged data, and record the printed `DOMAIN_INVARIANT_ENCODER_VERSION` for the experimental representation. Do not treat the raw and domain-invariant results as reproduced or improved until the staged run actually completes.

Scenario-level accuracy and macro metrics still do not measure benign operational false-positive rates because the current OTRF test set is an attack-labelled corpus. They also do not establish transfer to a third collection or a production environment. Retain the independent OTRF test, add benign business activity, repeat with multiple seeds, and report uncertainty and per-class errors.

## Next research step

The next meaningful model extension is a separately measured order-aware scenario encoder or temporal event graph with explicit time-window and relation features. It should be tested against this orderless bundle baseline on the same capture-level splits. Multi-event correlation and signal/noise discrimination are emphasized in the [MITRE ATT&CK Evaluations Enterprise 2026 methodology](https://attackevals.mitre.org/enterprise/er8), but this PR does not implement those capabilities.

This experiment only parses offline staged telemetry. It does not execute attacks, collect live host data, or authorize containment actions.
