# MEL-PRE-001 — predictive forecast integrity v1

**Date:** 22 November 2026  
**Parent:** #10190 — predictive transparency and musician anticipation  
**Related:** #10088 RMI-Bench, #10210 RMI expressive-performance v2, #10391 runner/CPAL clock bridge, #11319 co-creative agency, #13186 agency transaction traces

## Research objective

Make one narrow component of predictive transparency executable: every forecast used as a collaboration preview must be source-bound, issued before its target, explicitly displayed or left undisplayed, and linked to the eventual realization without rewriting its history.

A natural-language explanation after the music has played is not a forecast. A forecast that was generated but never shown is not evidence that the musician could anticipate it. A prediction silently replaced after new input must not be counted as the forecast that was originally presented.

## External motivation

The 2026 Agents in Concert case study reports two months of musician co-design and a public concert with 97 audience members. It describes visual forecasts of agent predictions as an intervention that helped musicians anticipate responses. This is a design opportunity, not a controlled effect estimate for Muse.

The 2026 NIME jam_bot paper provides a stronger engineering baseline: it models velocity, supports free-form call-and-response across varying tempi, increases throughput and compensates for external MIDI output latency. In its player-piano experiment, beat alignment without compensation was reported at F1 0.0678; fixed-upper-bound compensation had median F1 0.3386; average-latency compensation median F1 0.7638; and per-note/per-velocity compensation median F1 0.7834. These are results for the paper's specific piano, playback chain, modified-velocity MIDI and five non-overlapping windows—not a universal live-music threshold.

## Paradigm shift

Move from reactive generation plus retrospective explanation to a **precommitted, uncertainty-aware, steerable forecast**.

The system should be able to issue a compact prediction of its next musical commitment, display that prediction early enough to be useful, let the musician accept/steer/veto/branch/ignore it, and bind the actual output back to the exact preview. Revised predictions supersede earlier ones; they never rewrite them.

Keep four separate clocks/outcomes:

1. **Input observation:** the exact performer event and the clock domain in which it was observed.
2. **Forecast issue:** when the policy committed to a prediction.
3. **Preview display:** when the forecast became available to the human-facing interface.
4. **Realization:** when the first bound musical event actually began and when its artifact completed.

This prevents output-generation latency, display latency, performer anticipation lead time and MIDI/audio scheduling error from being collapsed into one number.

## Implemented reference slice

The isolated Rust 2024 crate at tools/mel-forecast-integrity provides:

- a strict, versioned JSON event contract;
- source-input and policy/prediction digest references;
- forecast target and expiry windows;
- explicit display events and partner dispositions;
- explicit supersession/invalidation;
- realization-start and realization-completion links;
- monotonic event ordering and a replayable SHA-256 event chain;
- a JSON report with separate input-to-issue latency, display lead time, input-to-realization latency and issue-to-realization latency;
- 13 deterministic tests covering positive lifecycle, post-hoc display, late issue, unresolved realization, supersession, missing display, outside-window realization, invalidation, event tampering and ordering controls.

The checked-in positive fixture is an unsealed trace seed; the CLI seals it deterministically before validation. The seal operation is a fixture utility, not an authorization operation.

## Benchmark plan

### A. Integrity conformance (offline, deterministic)

Positive and negative controls must cover:

- forecast-before-target versus forecast-after-target;
- pre-output display versus post-hoc explanation;
- display receipt present versus absent;
- disposition before/after execution;
- superseded/cancelled/expired versus active forecasts;
- forecast-to-realization linkage and missing output;
- correct versus altered source, policy and prediction digests;
- duplicated, reordered, non-monotonic and tampered events;
- visible versus generated-but-never-displayed forecasts.

Qualification targets (not achieved performance claims):

- 100% rejection of post-hoc previews and forecasts issued after the target window begins;
- 100% rejection of realization against a superseded, cancelled or expired forecast;
- 100% preservation of forecast history under supersession;
- 0 forecasts counted as human-visible without a preceding display event;
- 100% deterministic replay from frozen traces.

### B. Live timing (separate hardware panel)

Integrate, rather than duplicate, #10391's runner→CPAL callback→predicted-playback→output-frame clock chain. Capture:

- input-to-forecast issue latency p50/p95/p99;
- issue-to-display latency p50/p95/p99;
- display-to-target lead time;
- output scheduling error in milliseconds and beat fractions;
- event late/miss/drop rate, jitter and clock-drift residuals;
- latency by tempo, velocity, patch, device and buffer configuration.

Compare uncorrected scheduling, fixed upper-bound compensation, average compensation and per-note/per-velocity compensation, following the jam_bot design. Use matched hardware, renderer, MIDI instrument, velocity schedule, output path and task schedule. The jam_bot RTX 4090 / Ryzen 9 7950X throughput figures must not be directly compared with Muse on different hardware. Report throughput and musical event timing separately.

### C. Human anticipation and agency (confirmatory study)

Use the six-arm structure in #10190: no preview, post-hoc explanation, yoked preview, single forecast, uncertainty-aware alternatives and steerable preview. Counterbalance task and arm order; keep generation model, action budget, task, renderer, volume and response time matched.

Measure independently:

- human anticipation accuracy before the response is played;
- preparation time and entry/join timing;
- missed entries, collisions, repairs and interruption recovery;
- forecast trust calibration and ability to detect uncertainty;
- acceptance, steering, veto, branching and ignore rates;
- perceived agency, ownership, workload and desire to continue;
- whether forecasts induce anchoring, conformity or reduced spontaneity.

Do not set a universal latency threshold or infer improved agency from synthetic traces. Select live timing margins from instrument/device-specific pilot distributions. Use musician/item clustered analysis, blinded conditions where feasible, and preregistered primary outcomes.

## Competitive comparison

- MIREX 2026 reward-model leader CloudEar scores 73.9% pairwise preference accuracy on the 249-example core set. That is a reward-model metric, not a live-collaboration result.
- DAMN's MIREX symbolic continuation scores are useful generation anchors, but the reported study uses eight prompts and 12 participants. They do not measure reciprocal interaction.
- jam_bot is the direct live-performance comparator for expressive velocity, varying tempo, throughput and output latency compensation.
- Agents in Concert is the human-centered design reference for forecast visibility and musician anticipation.
- ICLR 2026 work on adversarial post-training for live human-AI music interaction warns that coherence-based rewards can collapse output diversity. Muse must measure creative variation and partner responsiveness separately rather than optimize one coherence score.

Muse should aim to exceed SOTA on the capabilities it can demonstrate under matched conditions: forecast integrity, latency-tail predictability, response timing, interruption recovery, collaborator control, non-assimilation and useful alternative branching. We should report a Pareto profile, not a single weighted “collaboration score”.

## Claim ceiling

A passing offline trace establishes that the evaluator can distinguish valid and invalid event histories. It does not establish that a model forecast is musically accurate, that a preview was actually rendered to or understood by a musician, that a musician anticipated the response, or that a live system beats jam_bot. Those claims require the clock bridge, a real display adapter, matched live baselines and a preregistered musician study.
