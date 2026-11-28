# Symthaea Event-Triggered Temporal Foveation v0.1

**Date:** 2026-11-28  
**Status:** implemented policy precursor; runtime and benchmark qualification pending  
**Scope:** event-triggered ROI sampling for audiovisual interaction, built on the existing dorsal/ventral foveation bridge

## 1. Research motivation

A naive way to capture more nonverbal interaction is to raise the full-frame video frame rate. That can increase pixel/encoder work and can overload cross-modal attention rather than improve understanding.

VideoFDB v1.4 reports two relevant observations:
- Some nonverbal cues unfold over roughly 1–2 seconds, making very sparse visual sampling liable to miss them.
- Raising one evaluated system's controllable video rate beyond 2 FPS was associated with a drop in conversational fluency from 3.55 to 2.33; the authors attribute this to overload of cross-modal attention.

This motivates a more selective hypothesis:

> Preserve low-rate global context, then temporarily increase analysis of a small number of decision-relevant regions when an independently identified interaction event makes the additional evidence valuable.

The goal is not maximum FPS. It is the greatest useful temporal evidence per pixel/encoder/compute token, without destabilizing spoken interaction.

Reference: [VideoFDB v1.4 leaderboard and paper](https://research.nvidia.com/labs/amri/projects/video-fdb/leaderboard.html).

## 2. Existing Symthaea substrate

The existing `symthaea-foveation` subsystem already supplies:
- dorsal-stream saliency and surprise-driven ROI candidates;
- a priority queue of salient patches;
- bounded in-flight requests and a bounded background channel;
- high-resolution cropping with a maximum crop-pixel limit;
- asynchronous ventral recognition;
- source frame IDs and timestamps for temporal binding;
- neuromodulated surprise threshold and concurrency capacity.

Its previous dispatch policy was primarily a surprise threshold plus fixed cooldown. The new temporal budget adds a third, independent control: *when ROI analysis is temporally worth doing and how much pixel work may be spent*.

No new camera, sensor, eye-tracking or raw-media capture permission is introduced. The policy only decides how to process frames already supplied to the foveation bridge. Sensor capture and storage permissions remain upstream.

## 3. Current implementation

### 3.1 Pure scheduling policy

`crates/domains/symthaea-foveation/src/temporal_budget.rs` implements:
- baseline ROI interval;
- event-triggered burst interval and duration;
- urgency threshold;
- bounded rolling pixel-work token bucket;
- max in-flight budget;
- source-clock regression, future-trigger and stale-trigger rejection;
- source-frame ID matching and maximum source-age checks;
- trigger ID exactly-once protection;
- fractional token refill without floating-point accounting;
- separate Dispatch, Defer, BudgetExhausted and ConcurrencyLimited outcomes.

Default proposal values are:
- baseline ROI analysis interval: 1,000,000 µs (1 Hz);
- triggered ROI interval: 250,000 µs (4 Hz);
- burst duration: 1,500,000 µs;
- trigger threshold: 0.70;
- pixel-token capacity: 1,000,000 pixels;
- refill: 500,000 pixel tokens/s;
- maximum in-flight requests: 2;
- maximum source ROI age: 500,000 µs (500 ms).

These are initial engineering defaults, not empirically optimized parameters.

### 3.2 Trigger vocabulary

The typed trigger set includes:
- visual surprise;
- motion onset;
- audio prosody change;
- turn-boundary candidate;
- backchannel candidate;
- cross-modal disagreement;
- user-directed attention.

The trigger must be supplied by an upstream component that has already received and is authorized to process its source. The foveation policy itself does not infer a trigger from unavailable data.

A repeated trigger ID cannot extend the burst. Trigger timestamps are checked against the foveation clock high-water mark. Low-urgency triggers are recorded but do not start a burst.

### 3.3 Source-clock freshness and frame identity

The scheduler rejects a source ROI when:
- its timestamp is in the future relative to the manager's current source-clock sample;
- its age exceeds the configured 500 ms default horizon;
- its source frame ID no longer matches the one full-resolution frame buffer currently retained by the manager.

This frame-ID guard fixes a subtle temporal-binding failure in the original manager: it retained only the newest full-resolution frame, but queued saliency requests could remain from older frames. Cropping an old request's coordinates from a new frame and returning the old frame ID would falsely bind recognition to pixels it was not actually computed from. The updated manager drops such queued requests and increments a stale-drop counter.

When the temporal policy is enabled, completed recognition results are checked again at collection time and at `drain_results_at(now_us)`, because a result can become stale while waiting in the ready queue. Legacy `drain_results()` remains available for backward compatibility; interactive consumers using the temporal policy should use `drain_results_at`.

The stale horizon is a decision-evidence bound, not proof that all tasks have the same temporal needs. Task-specific horizons require a preregistered calibration study.

### 3.4 Manager integration

`FoveationManager` exposes:
- `enable_temporal_budget(config, now_us)`;
- `disable_temporal_budget()`;
- `notify_temporal_trigger(trigger, now_us)`, which checks future/stale timestamps against the current source-clock sample;
- a read-only temporal-budget accessor.

The policy is disabled by default, preserving legacy cooldown-only behavior. When enabled, it also rejects expired source frames/results and refuses to relabel an old saliency request as a crop from a newer frame.

When enabled, the manager gates ROI dispatch against the temporal interval, pixel estimate and in-flight budget. The estimated token cost is the padded source ROI pixel area *before downscaling*, not the smaller resized output. The caller supplies the current source-clock sample when registering triggers; valid events between scheduler ticks are not incorrectly rejected merely because the last token refill occurred earlier. The budget is committed only after the bounded background channel accepts the request. On an unexpected commit invariant failure, the optional policy is disabled and the accepted request is still tracked; it is not duplicated. Source-frame ID mismatch or stale timestamps drop pending items; completed results are checked again at collection and at GWT injection.

### 3.4 Authored tests

The new policy has fourteen unit tests, and the foveation manager has five integration tests. They cover:
- baseline interval;
- event burst interval;
- low-urgency no-burst;
- duplicate trigger protection;
- source-clock regression;
- pixel token exhaustion and exact fractional refill;
- concurrency cap;
- invalid configuration and urgency;
- optional enable/disable behavior;
- manager baseline-to-burst dispatch;
- charging pre-downscale source ROI pixels.

Tests are authored but not compiled or executed.

## 4. Research paradigm shift: expected value of visual information

The next step should replace a fixed urgency threshold with a calibrated expected-value policy.

For candidate ROI i, define:
- `p_i`: calibrated probability the ROI contains task-relevant evidence;
- `v_i`: expected decision value if that evidence is acquired;
- `c_i`: expected pixel/encoder/compute cost;
- `l_i`: expected result latency;
- `r_i`: privacy and authorization risk cost;
- `s_i`: staleness penalty based on source frame age and predicted motion.

A first policy hypothesis is:

```
priority_i = (p_i * v_i - r_i - s_i) / max(c_i, cost_floor)
```

Dispatch only when priority exceeds a preregistered threshold and the rolling token budget permits it.

Important restrictions:
- `p_i` must be calibrated on held-out data; surprise is not automatically probability.
- `v_i` must be estimated from decision improvement, not subjective saliency alone.
- `r_i` is a hard veto where capture/output permissions prohibit processing; it is not merely a cost that high value can outweigh.
- expired/stale evidence is rejected even if its estimated information value is high.
- unknown cost or uncalibrated probability should trigger a conservative fallback rather than optimistic dispatch.
- no trigger may increase full-frame capture cadence by itself.

This creates a decision-theoretic scheduler rather than a saliency-only queue.

## 5. Benchmark protocol: matched-compute temporal attention

Build a clean-room, separately consented or suitably licensed interaction corpus with exact media rights for the planned transformations. Do not assume VideoFDB's non-commercial evaluation license permits edited participant-video counterfactuals. Do not place raw audio/video in Git.

Compare at least these arms:

| Arm | Global context | ROI analysis | Purpose |
|---|---|---|---|
| A: sparse fixed | 1 FPS | none | low-compute baseline |
| B: fixed 2 FPS | 2 FPS | none | stronger fixed baseline |
| C: fixed 4 FPS | 4 FPS | none | overload / high-sampling control |
| D: hybrid static | 1 FPS | fixed 4 Hz ROI schedule | ROI benefit without event triggers |
| E: event-triggered | 1 FPS | bursts up to 4 Hz | current implemented policy |
| F: value-of-information | 1 FPS | calibrated priority and token budget | research target |
| G: oracle ROI | 1 FPS | human/oracle event windows | upper reference only |

For arms D–G, hold the global context stream constant at 1 FPS. Only the ROI policy changes. The current code implements ROI dispatch budgeting; it does not implement or change the global-context capture scheduler. Arms A–C are separate full-frame sampling baselines.

Compare under two regimes:
1. **Matched pixel/encoder budget:** total decoded/encoded pixels and model visual tokens matched across arms.
2. **Matched end-to-end compute budget:** named hardware, same model, same precision, same inference concurrency and equal wall-clock/energy envelope.

Do not compare different total frame/pixel budgets and call the result an architecture win.

## 6. Task set and metrics

Use the same frozen audio waveform and matched visual variants where the experimental task requires visual evidence.

Task families:
- turn-yield and floor-holding cues;
- backchannels and acknowledgements;
- gaze shifts toward task-relevant shared referents;
- gesture onset/offset;
- motion onset and object interaction;
- visual cue disappearance / stale-frame rejection;
- audiovisual temporal alignment.

Measure separately:
- cue recall and precision;
- onset-to-first-useful-ROI latency;
- cue-to-decision latency;
- missed early cues and stale-evidence decisions;
- task accuracy and groundedness;
- counterfactual sensitivity and irrelevant-visual invariance;
- interruption and end-of-turn metrics;
- conversation fluency / response quality;
- total global pixels, ROI pixels, encoder tokens and GPU/CPU time;
- p50/p95/p99 processing latency;
- dropped/deferred ROI count;
- energy per correct decision on named hardware.

Use confidence intervals clustered by conversation/speaker/session. Report every task family and difficult cohort separately.

## 7. Proposed targets (not results)

Initial goals for a pilot, to be preregistered after feasibility:
- >=95% recall on clear, annotated critical nonverbal cues;
- >=90% precision for event-triggered ROI dispatches;
- p95 trigger-to-ROI-decision <=300 ms on a named local hardware profile;
- >=10 percentage-point cue/task accuracy improvement over sparse 1 FPS at matched pixel budget;
- >=5 percentage-point improvement over fixed 2 FPS at matched compute budget;
- no material decline in spoken-task accuracy, fluency or interruption handling;
- zero use of expired/unauthorized visual evidence;
- <=10% of visual encoder tokens spent on non-task-relevant ROI analysis.

Do not promote these to claims until the confirmatory held-out run and hardware manifest exist.

## 8. Experimental controls and negative cases

- randomize condition order and keep model/prompt/decoding fixed;
- hash audio and video payloads and preserve exact audio bytes across visual counterfactual conditions;
- freeze ROI expansion, crop dimensions, resampling method and encoder preprocessing;
- include repeated-trigger IDs, stale triggers, clock regressions, empty ROIs, edge-of-frame ROIs, dropped frames and background channel saturation;
- inject model stalls and ROI results that arrive after the cue becomes irrelevant;
- ensure the budget is charged by pre-downscale pixel work and not the resized result alone;
- verify that a high urgency trigger cannot bypass consent or privacy restrictions;
- verify that burst expiry returns to baseline without increasing full-frame FPS;
- verify that disabling the optional temporal policy restores legacy manager behavior.

## 9. Industry positioning

VideoFDB demonstrates why a fixed higher frame rate is not necessarily the best design: some nonverbal information is temporally sparse, while excess visual sampling can compete with speech processing. The event-triggered ROI hypothesis should therefore be tested against matched-cost fixed-rate baselines.

- VideoFDB remains the external full-duplex perception/generation comparison; use its exact official evaluation protocol and license scope.
- InteractionBench is useful for spoken tasks, timing, whisper/volume control and grounding; its own license and model-provider terms still apply.
- StreamEval is useful for following and A/V synchronization; a synchronized video is not proof of useful visual information.
- The new temporal-foveation lane is an internal diagnostic benchmark, not a replacement leaderboard and not a claim of superiority to any of the external systems.

## 10. Conservative expected-value ranking prototype (2026-11-28)

A separate, pure ranking prototype is now available at `crates/domains/symthaea-foveation/src/roi_information_value.rs`.

For an already-authorized ROI candidate, it consumes:
- expected task decision loss without the ROI;
- expected task decision loss with the ROI;
- a conservative lower confidence bound on loss reduction;
- confidence level;
- explicit latency and staleness penalties;
- source pixel cost before downscaling;
- source clock domain and timestamp;
- calibration ID, report digest, model digest, task family, sample count and held-out expected calibration error.

The net value is:

```
net_benefit = loss_reduction_lower_confidence_bound
              - latency_penalty
              - staleness_penalty

value_per_1000_pixels = 1000 * net_benefit / source_pixel_cost
```

The ranking requires a task-family-matched calibration report, at least 100 calibration samples, expected calibration error <=0.10, confidence level >=0.95, a matching source clock, and a source frame within the configured freshness horizon. Rejected candidates remain in a separate rejection list. Duplicate candidate IDs are rejected as a group; score ties use candidate ID as a deterministic tie-break.

This is deliberately not integrated into `FoveationManager` yet. The manager still uses the urgency threshold and pixel-token budget. The new module only ranks caller-supplied, calibration-backed estimates. It cannot authorize media capture or override privacy/permission checks.

Fourteen Rust tests are authored in the temporal/foveation package (the temporal budget and EVI modules together), and the manager has five new integration tests. They remain uncompiled and unexecuted. No calibration report, held-out ECE result, ROI utility study or EVSI leaderboard result exists.

## 11. Qualification status

The Rust temporal policy, optional `FoveationManager` integration and independent conservative EVI ranking prototype have been committed on the polyembodiment research branch. Source ROI freshness is checked against the caller's monotonic time; task-specific age limits still require empirical calibration. The implementation is source-level only:
- no cargo build or test run;
- no camera/stream runtime qualification;
- no GPU/CPU profiling;
- no energy measurement;
- no matched VideoFDB, StreamEval or InteractionBench evaluation;
- no consented clean-room temporal dataset has been created.

The optional policy is disabled by default. It changes only the dispatch policy for frames already supplied to the foveation subsystem. It does not open capture, change camera permissions, or establish that an external sensor feed is authorized.
