# MEL Closed-Loop Music Collaboration Benchmark v1

**Status:** research protocol and target proposal. No new benchmark run, external baseline reproduction, loopback measurement, listener study or SOTA result is claimed by this document.

**Integration owner:** [#13480 — forecast-to-intervention-to-observed-effect lineage](https://github.com/Luminous-Dynamics/symthaea/issues/13480).  
**Implementation:** [#13481 — episode manifest](https://github.com/Luminous-Dynamics/symthaea/pull/13481) and the stacked Rust reducer PR.  
**Existing benchmark owners:** #10190 predictive transparency; #10088 reciprocal musical interaction; #13313 scoped musical agency; #10265 grounded musical event perception; #10417 external benchmark registry; #12949 paired edit evidence; #9184 human-grounded edit evaluation.

## 1. Research thesis

The benchmark unit is not a generated clip or a single response. It is a **closed-loop interaction episode**:

\`input \u2192 forecast \u2192 musician disposition \u2192 authorized effect \u2192 independent capture \u2192 qualified event observation \u2192 replayable outcome\`

The programme asks five distinct questions:

1. **Musical competence:** did the system understand and produce the intended musical material?
2. **Causal responsiveness:** did the musician's actual intervention change the next action in the intended dimension, beyond sham and yoked replay?
3. **Agency preservation:** could the musician steer, veto, branch, ignore and recover without silent override or preference assimilation?
4. **Physical execution:** what did the output path actually emit, enqueue, admit, predict and independently capture?
5. **Epistemic validity:** is the observation complete, is the detector qualified, and what remains unknown?

No overall collaboration score may let musical quality compensate for unauthorized output, poor veto compliance, incomplete capture or unqualified negative evidence.

## 2. Paradigm shifts worth testing

### 2.1 Forecasts become negotiated, expiring commitments

A forecast is a pre-output commitment with a version, input watermark, time window, uncertainty, immutable model/runtime identity and expiry. It is not a post-hoc explanation and not an instruction to the performer.

A musician can accept, steer, veto, branch, ignore, clarify or abstain. Each disposition attaches to the exact forecast revision shown. A successor forecast does not rewrite the one previously displayed.

### 2.2 Silence and withholding become measured musical actions

WAIT, YIELD, intentional rests and non-response are first-class actions. Missing output is not retrospectively reclassified as WAIT. Silence is useful only when the system's decision and the independently observed output are both accounted for.

### 2.3 Editing becomes an evidence transaction

The same control philosophy extends to composition editing: an exact source, target change set, protected musical invariants, candidate, independent verification, bounded repair budget and explicit commit decision. A no-op that preserves everything but fails to edit must not pass. A beautiful regeneration that damages protected content must not pass either.

### 2.4 The evaluator is itself a system under test

Use no-op similarity traps, wrong-direction edits, yoked forecasts, output substitutions, detector-miss injections, answer-order swaps and best-of-N selection pressure. Qualify judges and selectors against adversarial candidate distributions, not only random examples.

### 2.5 Personalization is scoped adaptation, not assimilation

Keep present instructions, contextual preferences, agent proposals and persistent identity updates distinct. A current hard instruction outranks stale soft preference. A local session preference does not silently become a global taste model. A participant may disagree with the agent and retain a separate branch.

### 2.6 Evidence is a musical capability

A musician should be able to inspect what was proposed, what was authorized, where execution reached, what an independent observer captured, and why the outcome is present, absent, ambiguous or unobserved.

## 3. Benchmark tracks

### Track A — Causal reciprocal improvisation

Tasks: motif response/transformation; phrase completion after unexpected continuation; tempo drift and rubato; gap/entry negotiation; role exchange; preserve-motif/change-texture; deliberate rest; interruption; recovery after mismatch.

Conditions, counterbalanced within musician where feasible:

1. no preview;
2. post-hoc explanation (anticipation negative control);
3. yoked preview from an unrelated matched session (contingency negative control);
4. one precommitted forecast;
5. uncertainty-aware alternative forecasts;
6. steerable forecast with accept/steer/veto/branch/ignore/abstain.

Baselines: fixed/scripted accompaniment; reactive-rule accompaniment; imitation-only; one-way prompt-response; yoked replay; current Melothaea without relationship memory; Melothaea with authorized contextual memory; human-human reference on a matched subset. JAM_BOT is a compatible MIDI/live comparator only when instrument, tempo, clock and output measurement can be matched. Audio-stream systems belong in a separate controls-only adapter track.

### Track B — Human agency and intervention

Use registered races and musician sessions for:
- veto before scheduling, after enqueue, after callback admission and after predicted playback;
- steer before versus after forecast expiry;
- ignore/non-response;
- branch/merge disagreement;
- deliberate musician departure from the preview;
- preference reversal and collaborator swap;
- revocation between retrieval and commit.

Measure veto propagation at each actual boundary. Do not claim already-admitted callback frames or device buffers were physically recalled. A human deviation from a prediction is agency, not a forecast violation.

### Track C — Independent output and event observation

Report separately:
- MIDI message emitted;
- note-queue operation applied;
- application output enqueued;
- callback admission observed;
- backend/CPAL playback prediction;
- independently loopback-observed audible onset.

The negative-evidence rule is absolute: a complete capture may support a detector-reported absence; an incomplete or unqualified capture is unobserved. Complete capture is not detector correctness. Detector correctness is not physical capture completeness.

### Track D — Musical editing and preservation

Reproduce the compatible native operations from SMEBench and the score-native editing suite. Keep score-native edits, real multitrack edits and arbitrary mixture editing as separate strata. Preserve native task definitions, splits, metrics and denominators.

### Track E — Evaluator and selector robustness

Compare human judgments and independent observers against automatic judges, then measure how candidate selection pressure changes the chosen output. Use candidate-set sizes 1/2/4/8/16/32/64/128 and matched render budgets. Retain every failed, unparseable, timed-out and unselected candidate in the attempt denominator.

## 4. External benchmark alignment

External systems measure different things. Reproduce compatible protocols rather than combine scores.

| External reference | Published scope | How Melothaea should use it |
|---|---|---|
| SMEBench | 12 paired edit operations; balanced 3,600-example test set; separate 416-pair real-audio complement | Reproduce operation-specific target/preservation/text metrics and disclose metric-ranking reversals |
| ICASSP 2027 Audio Editing Challenge / MMAE | 2,000 public examples with 17,741 atomic rubrics; separate 500-example hidden sets for Single Model and Agent tracks | Reproduce official IFR, CR and conjunctive EMR where eligible; add score/lineage/agency diagnostics separately |
| HEARTBEAT 2027 | 132,571 training/development samples across online/offline speech/audio-agent tracks; online protocol logs silent chunks and causal response times | Reuse causal streaming and silence-accounting ideas; do not label it a music benchmark |
| JAM_BOT (NIME 2026) | Live symbolic call-and-response, variable tempo, expressive velocity and MIDI latency compensation; paper reports median beat-alignment F1 0.7834 with per-note/per-velocity compensation in its player-piano experiment | Matched live MIDI timing comparator, not universal improvisation SOTA |
| Daydream research | Vendor reports parameter-to-changed-sound 81 ms, chord-root accuracy 69%, melody pitch-class accuracy 53%, groove accent F1 80%, accompaniment onset match 33% | Task-specific real-time audio-control comparator; preserve vendor protocol and hardware caveats |
| Agents in Concert (IUI 2026) | Two-month musician co-design and public co-improvised performance with 97 audience members | Human-centred and longitudinal study design reference, not a numerical leaderboard |
| MIREX 2026 CMI-RewardBench | Best core pairwise human-preference agreement 73.9%; vocal extension 60.9%; instrumental extension 72.6% | Matched human-judge calibration target; not an edit-success or agency metric |

Primary references:
- SMEBench: https://arxiv.org/abs/2610.36715
- Audio Editing Challenge: https://audio-editing-challenge.github.io/
- HEARTBEAT: https://heartbeat-challenge.github.io/
- JAM_BOT: https://nime.org/proc/nime2026_73/index.html
- Daydream: https://daydream.live/research
- Agents in Concert: https://www.media.mit.edu/publications/agents-in-concert/
- MIREX: https://music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench_Results

### Important competition timing

The Audio Editing Challenge organizers announced on 23 November 2026 that final submission and leaderboard freeze moved to **28 November 2026, 11:59 PM U.S. Pacific Time**. Registration closed 8 October; the Agent Track public model/weights cutoff remains 1 November. The public MMAE development set remains useful for local reproduction even when competition registration or release-date eligibility is unavailable. Do not imply Luminous Dynamics is registered or eligible without organizer confirmation.

## 5. Scorecards and prospective targets

All thresholds below are proposed gates, not results. Each must be reported by task family, instrument, source type, performer, renderer/device and musical-density stratum where applicable.

| Dimension | Initial target | Stretch target / hard rule |
|---|---:|---:|
| Unauthorized effect after valid pre-admission veto | 0 | 0 |
| Stale/expired/superseded forecast executed | 0 | 0 |
| False absence from incomplete/unqualified capture | 0 | 0 |
| Exact forecast/disposition/decision/effect/capture join coverage on scored episodes | 100% | 100% |
| Synthetic sequence/frame reconciliation | 100% | 100% |
| Qualified live window yield | \u226599% | \u226599.9%, per device/backend |
| Detector sensitivity lower 95% confidence bound on declared held-out scope | \u22650.98 | \u22650.995 |
| Detector false-positive upper 95% confidence bound | \u22640.02 | \u22640.005 |
| Musician steer/veto/branch disposition honoured | \u226599% | 100% on deterministic controls |
| Partner-to-agent intervention effect beyond sham/yoked controls | Positive, pre-registered CI | Replicate across held-out performers/tasks |
| Preference judge agreement on matched MIREX-compatible protocol | \u226578% | \u226582%, report vocal/instrumental separately |
| Deterministic edit target adherence | \u226599.5% | 100% on registered exact controls |
| Deterministic protected-event preservation | \u226599.9% | 100% on registered exact controls |

Do not set an arbitrary universal latency threshold. First measure musician-input-to-independent-loopback p50/p95/p99 with calibrated clock uncertainty, under matched hardware, buffer, sample rate, route and task. Then derive task- and musician-specific timing tolerances in a pilot and preregister them before confirmation.

### Independent dimensions to publish

- Forecast presence calibration (Brier/log loss only where probabilities are meaningful), onset uncertainty calibration, revision rate and expired/stale forecast rate.
- Anticipation accuracy, useful lead time and entry/turn repair.
- Partner-to-agent causal effect and agent-to-partner causal effect, separately.
- Intent execution, protected-content preservation and correspondence coverage.
- Veto/stop correctness, unauthorized mutation, steering success, branch retention and rollback.
- Capture closure yield and reason-coded missingness.
- Detector sensitivity/specificity, calibration, risk/coverage and event-scope support.
- p50/p95/p99 latency for each distinct execution stage.
- Listener-rated audibility, musical validity, preference, ownership, autonomy, workload and willingness to continue.
- Candidate/render/repair budgets, cost per qualified episode and human correction time.

## 6. Statistical protocol

- Split by source work/composition lineage, not excerpt; hold out performers, motifs, instruments and at least one renderer/backend family.
- Randomize and counterbalance condition order within musician where feasible.
- Use matched model version, input history, response deadline, output budget and compute budget in counterfactual arms.
- Include no-op, sham, unrelated-feature, yoked and hidden-preview negative controls.
- Pilot variance and within-musician correlation first; preregister the powered sample size for primary outcomes.
- Use mixed-effects or hierarchical models with musician, session, task and source-work effects. Bootstrap by source-work and listener, not by individual audio frames or repeated turns.
- Report missing sessions, dropped windows, detector abstentions, timeouts, unsupported operations and failed runs; never remove them silently.
- Evaluate preference with A/B order swaps, randomized display order, explicit ties and unsure responses. Report position bias and swap consistency.
- Freeze human rubrics and automatic evaluator identities before held-out scoring. Keep adjudication disagreement and plural valid targets visible.

## 7. Route to surpassing the field

1. Reproduce a compatible published baseline faithfully before optimizing.
2. Improve one registered task family at a time; do not tune a global composite.
3. Use paired counterfactual interventions and source-lineage-disjoint held-out data.
4. Require improvement over the strongest reproduced baseline with source-clustered 95% intervals and no material regression on primary subgroups.
5. Qualify automatic judges against humans and test the selector under increasing best-of-N pressure.
6. Demonstrate live control on matched hardware and independently observed audio, not only a fast generation API.
7. Run longitudinal musician sessions to test whether transparency improves coordination without anchoring, fatigue or reducing creative diversity.
8. Publish nulls, negative controls, uncertainty and claim ceilings alongside wins.

A proposed superiority claim requires at least a five-percentage-point absolute improvement on a preregistered primary matched metric, a confidence interval excluding zero, replication on held-out source/performance strata and no unacceptable hard-gate regression. This threshold is a study design target, not a promise of attainable improvement.

## 8. Immediate no-runner execution

- Keep the v1 schema and synthetic fixtures as the contract baseline.
- Implement and locally reason about the deterministic Rust reducer; run exact-head pinned Rust format/tests/Clippy when the workflow becomes available.
- Add adversarial tests for veto races, sequence/frame loss, identity collapse, payload substitution, forecast expiry and complete-capture detector misses.
- Extend the reducer only through existing upstream verifier APIs and replay receipts; do not accept self-reported \`verified=true\`.
- Build a public MMAE/SMEBench materialization plan with source-license and rights review before acquiring audio.
- Start symbolic Track A counterfactual experiments with public-domain, owned or synthetic material while real loopback/human access is arranged.

## 9. Claim boundary

A schema or reducer pass proves only its specified structural rules. Synthetic fixtures do not prove musical quality, detector sensitivity, acoustic latency, human agency, reciprocal causal gain or SOTA superiority. An authenticated receipt proves who asserted/bound an observation, not that the physical observation was correct. Each of those claims needs its own evidence and independent validation.
