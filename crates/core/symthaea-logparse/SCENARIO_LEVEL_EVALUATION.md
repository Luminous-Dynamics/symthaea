# Capture/scenario-level LogParse evaluation

## Why this is a separate measurement

A grouped event split prevents events from the same EVTX file leaking into both train and test, but its accuracy is still event-weighted: a capture with many retained events can contribute many predictions. This experiment adds a second unit of analysis where one EVTX source file or one OTRF scenario directory produces one example.

The library helper `scenario::aggregate_scenarios` validates each group, bundles its event vectors without encoding the label or group ID, records its event count, and returns scenarios sorted by stable group ID. This is the orderless baseline.

The companion `scenario::aggregate_ordered_scenarios` binds each event vector to one of eight coarse relative record-position roles before bundling. This is an experimental ordered variant; both produce exactly one sample per capture/scenario, so the statistical unit stays the same while representation changes. Each `ScenarioEmbedding` also carries an explicit `ScenarioEncodingMode`; the runner rejects a source/held-out mix of modes rather than training on one representation and evaluating on another.

## What these representations do—and do not—encode

The orderless HDC bundle tests whether the collection of signals in a capture carries class information after reducing it to one sample. It intentionally discards event order.

The relative-record-position variant retains coarse position within the parser-provided record order. It does **not** preserve absolute timestamps, elapsed time, time gaps, causal relationships, or reliable attack-chain transitions. It must not be described as a temporal graph or causal model. The caller is responsible for passing deterministic source order within each scenario.

This distinction matters because the current EVTX and OTRF parsers fill missing or invalid timestamps with `Utc::now()`. Sorting such events by parsed timestamp could inject nondeterministic order. This PR therefore does not sort on timestamp; it uses source-record order and labels it accordingly. Source record order is not proof of chronology, especially if upstream records were reordered or missing.

Both representations can be dominated by unrelated or repeated events. The 50-event cap bounds contributions but does not solve mixed-technique labels. A source-file or OTRF-directory label is assumed to apply to all retained events in that group; more precise scenario annotations remain an important limitation.

## Run the paired comparison

Stage the same EVTX-ATTACK-SAMPLES and OTRF directories described by the Phase 2 example. Then run:

```bash
cargo run -p symthaea-logparse --example phase2_experiments --release -- \
  /tmp/evtx-staged /tmp/otrf-staged \
  --scenario-level-evaluation --domain-invariant-ablation
```

The scenario flag reports two raw-encoder scenario models by default, and four when combined with the domain-invariant flag:

- raw encoder with orderless bundling;
- raw encoder with relative-record-position encoding;
- domain-invariant encoder with orderless bundling;
- domain-invariant encoder with relative-record-position encoding.

Each model reports three repeats of source-scenario grouped five-fold CV using fixed split seeds (`0xDEADBEEF`, `0x51CE0002`, and `0x51CE0003`), a separate 80/20 source-scenario holdout report, and OTRF scenario-level OOD precision/recall/F1 and confusion matrix. The mean and population standard deviation across the three repeat means are printed; this is a descriptive stability measure, not a confidence interval. The same seeds are reused across encoder variants, keeping split comparisons paired. Class distributions are printed before training. The ordered mode version is printed as `ORDERED_SCENARIO_ENCODER_VERSION`; the domain-invariant representation separately prints `DOMAIN_INVARIANT_ENCODER_VERSION`. The encoding mode is retained in typed scenario metadata and source/held-out mode mismatches are rejected.

If any class has fewer than five independent source groups, grouped five-fold CV fails closed instead of falling back to event-level splitting. Compare all variants on the same exact commit and staged data; none should be considered an improvement until the run actually completes.

## Interpretation discipline

Capture-level metrics answer a different question than event-level metrics. They should not be compared as though each event were an independent sample, nor directly compared with historical event-level results. Record source-corpus identities, group counts per class, event caps, exact commit, encoder version, and seeds with every result.

Scenario-level accuracy and macro metrics still do not measure benign operational false-positive rates because the current OTRF set is attack-labelled. They also do not establish transfer to a third collection or a production environment. Add benign business activity, another independent corpus, repeated seeds, calibration/abstention assessment, and confidence intervals before making deployment claims.

## Next research step

The next meaningful step is not simply to add more hand-written event IDs. It is to replace the heuristic with a version-pinned, schema-validated event normalization layer, then test time-aware relations using trustworthy timestamps and explicit missingness rather than substituting wall-clock time. That temporal/causal model should be independently compared against both the orderless and relative-position baselines on identical capture-level splits.

Multi-event correlation and signal/noise discrimination are emphasized in the [MITRE ATT&CK Evaluations Enterprise 2026 methodology](https://attackevals.mitre.org/enterprise/er8), but this PR does not implement production-grade correlation.

This experiment only parses offline staged telemetry. It does not execute attacks, collect live host data, or authorize containment actions.
