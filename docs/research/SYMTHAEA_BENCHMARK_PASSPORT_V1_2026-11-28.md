# Symthaea Benchmark Passport v1 — protocol and qualification strategy

**Status:** proposed protocol / schema. Not an executed benchmark, certification, signed attestation, or claim of SOTA.

## 1. Why a passport

A leaderboard score without a reproducible execution record is a weak engineering artifact. The passport binds a result to the exact benchmark protocol, model/configuration, dataset split and license record, evaluator, runtime environment, measurement boundary, raw outputs and per-item records. It makes it harder to accidentally compare unlike systems or silently change the measurement definition.

The passport is evidence packaging, not a trust oracle. A digest proves byte identity only when the bytes are available; a signature proves control of a key, not that a model, dataset, clock, or machine is honest. Dataset permission must still be independently established. Hardware claims require an attestation system if they are to be trusted.

Machine-readable schema: `SYMTHAEA_BENCHMARK_PASSPORT_V1_2026-11-28.schema.json`.

## 2. Paradigm shifts

1. **From scoreboards to evidence-carrying results.** Every reported number resolves to per-item outputs, evaluator version, denominator, exclusions and uncertainty.
2. **From one composite score to a capability vector.** Report perception, content quality, turn control, latency, grounding, privacy, reliability and cost separately. A strong content score cannot compensate for unauthorized output.
3. **From model-centric to closed-loop system evaluation.** Evaluate sensor → semantic interpretation → policy → action schedule → renderer/actuator → independent observer. Measure the actual user-heard/user-seen result, not merely model token emission or ECS acceptance.
4. **From average latency to deadline reliability.** Report p50/p90/p95/p99, deadline-miss rate, early/late distributions, stop-command acceptance and independently observed stop latency. Never substitute command enqueue time for physical stop time.
5. **From static test sets to causal counterfactuals.** Pair each scenario with minimal changes to the relevant cue and irrelevant nuisance variables. Score whether the action changes only when causally warranted.
6. **From consent as metadata to consent as an executable invariant.** Test revocation, expiry, destination changes, intensity tightening, replay and clock discontinuity as adversarial events. Any post-revocation output is a hard failure.
7. **From a single avatar to polyembodiment invariance.** The same canonical cue must preserve semantic digest, privacy classification and authorization across all forms; only renderer grammar may differ.
8. **From benchmark chasing to capability-per-cost.** Report quality and reliability against GPU-seconds, energy, bandwidth, memory, human annotation hours and dollars per qualified interaction.

## 3. Run protocol

- Freeze protocol, model, configuration, dataset manifest, split and evaluator before execution. Hash all of them.
- Keep dev and test subjects/sessions disjoint. Freeze thresholds on dev; never tune on test.
- Use the same input recordings, client, network conditions and measurement boundary for all systems in a comparison stratum.
- Distinguish full-duplex, cascaded, audio-only, audio-visual, native avatar and post-hoc avatar strata. Do not rank across incomparable strata.
- Record raw outputs and time-aligned traces. Missing output is a scored failure where the scenario requires an action; never silently drop it.
- Publish per-axis metrics, denominators, confidence intervals, exclusions, failure taxonomy and hardware cost. Preserve the exact raw-record digest.
- Require a second independent evaluator or human audit for headline claims. LLM-as-judge is a diagnostic, not sole ground truth.
- Treat licensing, consent and redistribution as separate gates. A manifest declaration is not proof of legal permission.

## 4. Benchmark portfolio

| Lane | Public reference | Symthaea target (aspirational) |
|---|---|---|
| AV perception / interaction | VideoFDB v1.4 | Matched full-duplex stratum: perception >=4.0/5; TOR alignment >=92%; median TOR <=150 ms |
| AV cue generation | VideoFDB v1.4 | Matched generation stratum: >=4.35/5; TOR alignment >=60%; median TOR <=900 ms |
| Continuous engagement | DIVE / Full-Duplex-Bench | Response-window success >=75%; p95 reaction <=500 ms; quality >=4/5 |
| Multi-party floor control | MultiPartyTurnBench / InteractionBench | EOT recall >=95% at FPR <=5%; selective interruption balanced accuracy >=80%; AUROC >=0.90 |
| Grounded spoken tasks | InteractionBench / tau-Voice | >=95% on supported held-out task slice; report abstention and missing replies separately |
| Cross-form invariance | Internal paired-form suite | 100% semantic digest preservation; >=95% held-out human meaning agreement |
| Visual causal grounding | VideoFDB counterfactual extensions | >=10 pp gain on visually-required items vs audio-only; irrelevant visual invariance >=98%; captioning collapse <=3% |
| ROI attention | Internal matched-budget suite | >=5 pp decision-quality gain over surprise-only at equal pixels/compute; regret <=10% vs oracle |
| Output safety | Internal adversarial lifecycle suite | Zero unauthorized, expired, replayed or post-revocation output |
| Runtime cost | Matched hardware trace | Quality/cost frontier; publish energy, GPU-seconds, memory, bandwidth and latency |

Targets are hypotheses, not present capabilities or achieved results. Report each lane separately; do not collapse them into a single headline number.

## 5. Critical measurement definitions

- **TOR (time-of-response):** state the triggering event and observed output event. For a live system, prefer client-side first audible/visible output. Include network delay in product-facing latency and additionally report server-only latency when clocks permit.
- **Stop latency:** from the annotated interruption/revocation event to the last independently observed audible/visible output. Report adapter-accepted stop separately.
- **Silence preservation:** correct no-output decisions divided by all scenarios requiring silence; also report false activations per hour.
- **Causal visual contribution:** paired AV vs audio-only performance on items where visual evidence is necessary; include irrelevant-video and counterfactual-video controls.
- **Calibration:** report Brier/ECE plus held-out cluster-level coverage and confidence intervals. Split by speaker/session, not random frames.
- **Human quality:** blinded multi-rater protocol, inter-rater reliability and adjudication. Keep model-judge results separate.

## 6. Failure taxonomy

At minimum: missed cue, false intervention, wrong addressee, missed interruption, false floor takeover, backchannel interruption, late action, early action, hallucinated visual inference, captioning collapse, visual-stream ignorance, stale-frame action, unauthorized output, output after expiry, stop not observed, cross-form semantic drift, grounding error, privacy leakage, dropped frame/audio chunk, resource exhaustion, and missing evidence.

## 7. Qualification ladder

- **L0 — authored:** protocol/schema/tests written; no execution claim.
- **L1 — locally executed:** exact source head and environment recorded; tests and fixtures actually run.
- **L2 — reproducible:** clean independent rerun yields matching per-item outputs or declared tolerances.
- **L3 — externally replicated:** independent operator reproduces headline claims on held-out data.
- **L4 — deployment qualified:** live device/compositor/actuator observation, privacy/safety review, resource budgets and failure recovery qualified.

A skipped workflow, queued runner, uncompiled test, schema parse, mock adapter acknowledgement or self-reported result cannot advance the ladder.

## 8. Immediate experiment sequence

1. Run existing unit tests and schema validation on the exact PR head once Rust tooling is available; preserve failures rather than editing around them.
2. Create a small, licensed, consented 20–50 scenario internal pilot with multi-party addressee, EOT, selective interruption, backchannels, ambient speech, human resolution, visually-required cues and irrelevant visual controls.
3. Instrument client-heard/client-seen timestamps and a separate observer. Keep server and client clock domains distinct; estimate offset and uncertainty.
4. Compare audio-only, AV at fixed frame rates, and event-triggered ROI at matched compute/pixels.
5. Run baseline systems in the same stratum and measurement harness. Publish per-item traces and passports.
6. Red-team policy revocation, expiry, clock rollback, duplicate events, queue overload, dropped acknowledgements and output-stop failure before any user-facing deployment.

## 9. Explicit non-claims

This protocol does not establish model quality, SOTA, consciousness, emotional experience, legal permission, device safety, trusted time, secure key custody, or independent observation. Those require separate evidence.
