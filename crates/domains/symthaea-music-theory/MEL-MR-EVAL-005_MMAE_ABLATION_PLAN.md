# MEL-MR-EVAL-005 — MMAE Interop and Verifier-Ablation Plan

**Protocol draft:** v0.1.0  
**Tracking:** https://github.com/Luminous-Dynamics/symthaea/issues/8799 and https://github.com/Luminous-Dynamics/symthaea/issues/8814  
**Status:** research plan only; no model evaluation has been run.

## 1. Question

Does a typed edit contract plus independent post-edit verification measurably improve audio-editing reliability, compared with the same editing system without that guard? If repair is added, does it improve exact completion enough to justify its extra latency and tool calls?

This question is more useful than comparing an internal verifier score to an audio-generation leaderboard. The validator is not an editor, so its value must be measured as a component in a controlled editing pipeline.

## 2. Published context as of 2026-10-29

MMAE comprises 2,000 audio-editing examples and 17,741 atomic rubrics. The benchmark reports:
- **IFR:** whether requested edits were executed;
- **CR:** whether unrelated content and audio quality were preserved;
- **EMR:** the fraction of examples satisfying every required instruction-following and consistency rubric.

The ICASSP 2027 Audio Editing Challenge leaderboard ranks primarily by EMR, then IFR, then CR. It plans to release 500 hidden test examples for each of its Single Model and Agent tracks on November 10, 2026. The public MMAE set is available for development beforehand.

Reported reference results come from different evaluation scopes and should not be blended into a fictional single ranking:
- Original full-MMAE single-model comparison: Step-Audio-EditX leads IFR (44.86%) and CR (58.88%); Ming-UniAudio leads EMR (3.20%).
- Challenge agent baseline on all 2,000 cases: 7.45% EMR, 44.06% IFR, 74.63% CR.
- AuK without Prompt Enhancer: 7.58% EMR on a 1,003-case single-operation subset.
- Audio-Omni: 4.99% EMR on a separate 801-case subset of examples up to 10 seconds.

Sources, checked 2026-10-29:
- MMAE paper: https://arxiv.org/abs/2606.07229
- Public dataset: https://huggingface.co/datasets/BoJack/MMAE
- Survey and benchmark table: https://github.com/MM-Speech/AudioEditSurvey#benchmarks
- Official challenge protocol and dates: https://audio-editing-challenge.github.io/leaderboard/

These are context, not Symthaea results. The challenge's registration deadline was extended to October 8, 2026. Unless the team already registered or receives explicit organizer permission, plan on a reproducible public-set evaluation rather than assuming challenge eligibility.

## 3. Experimental arms

Evaluate the **same base editing agent**, with the same versioned tools and a matched resource budget, under these arms:

| Arm | Pipeline | Purpose |
|---|---|---|
| A — Baseline | Editor/agent as supplied; no MEL guard | Establish task success and unintended-change baseline |
| B — Contract gate | Same agent + contract generation + independent post-edit audit; fail closed on critical violations | Measure audit benefit without repair |
| C — Verify and repair | Same agent + contract gate + a bounded repair loop; re-audit after each repair | Measure whether feedback turns detection into better final edits |

Use the official challenge agent baseline implementation if its license, interfaces and resource requirements permit. If not, pin a clearly documented alternative baseline. Do not compare Arm C using a more capable model or a larger tool budget than Arm A and then attribute all gains to the verifier.

The same sample must be evaluated under each arm where possible. Randomize order, cache only source-side analyses that are identical across arms, and prevent any arm from seeing hidden test labels or rubrics.

## 4. Eligibility gate: do not overclaim what MEL can audit

The current MEL verifier supports index-preserving monophonic symbolic event sequences. The new semantic contract validator checks contract structure and scope, but it does **not** yet implement calibrated acoustic detectors or general rendered-audio preservation.

Therefore:
1. First evaluate the contract validator on synthetic positive/negative fixtures; that tests validator correctness, not audio-editing capability.
2. Build a controlled MIDI-to-audio pilot with pinned rendering and known output ground truth.
3. Add calibrated audio-side detectors and abstention when evidence is outside their validated domain.
4. Only then run a broader MMAE-compatible agent ablation. Until then, label any public-set experiment as a limited pilot rather than a full MMAE result.

A valid contract or replayable receipt proves neither good sound quality nor human-perceived preservation.

## 5. Primary outcomes and statistics

**Primary endpoint:** EMR delta between Arm A and Arm C, computed at the sample level on a preregistered evaluation set. Arm B isolates the effect of the gate from the repair loop.

**Secondary outcomes:**
- IFR and CR separately;
- contract rejection / abstention rate and reasons;
- critical collateral false-accept and false-reject rates on cases with deterministic ground truth;
- repair success conditional on an identified violation;
- regression rate: initially passing outputs made worse by repair;
- model/tool calls, wall-clock p50/p95, compute cost, and failed/unsupported tasks;
- judge instability and disagreement on rubric-scored cases.

For paired examples, report sample-level paired differences and 95% confidence intervals. Use paired bootstrap resampling by source/composition family where family metadata exists. For binary per-sample EMR transitions, report the paired discordance table and an appropriate paired test such as exact McNemar. Publish denominators and unrounded counts; do not infer reliability from a percentage alone.

MMAE's rubric judge is useful for broad semantic and quality checks, but deterministic ground truth should control exact symbolic assertions. Report judge sensitivity and disagreements; never use an MLLM majority vote as the sole authority for claims such as exact note identity or content digest integrity.

## 6. Preregistered hypotheses

These are hypotheses to test, not promised outcomes.

- **H1:** Arm B reduces critical collateral false accepts relative to Arm A on deterministic fixtures.
- **H2:** Arm C improves EMR over Arm A on matched held-out edits, with a 95% confidence interval for the paired difference reported.
- **H3:** Arm C improves consistency without an unacceptable reduction in IFR.
- **H4:** The gain persists under composition-family, renderer, effect-chain, instrumentation and complexity holdouts.
- **H5:** Any gain is not explained solely by more tool calls or a larger compute budget; report quality-versus-budget curves.

No “better than SOTA” claim is justified unless the matched, preregistered comparison supports it. If the confidence interval crosses zero, say the evidence is inconclusive.

## 7. Adversarial and distribution-shift matrix

Include:
- requested edit missing or only partially applied;
- successful target change plus collateral corruption;
- wrong-region and wrong-track edits;
- boundary-tolerance violations;
- overlapping sources, polyphony, sustained notes, reverb/shared buses and transient-heavy instruments;
- renderer/version mismatch and nondeterministic replay;
- no-op and contradictory instructions;
- detector out-of-domain and alignment ambiguity;
- multi-operation and multi-round tasks;
- same composition rendered through held-out instrument banks or effects where licensed assets allow.

Keep source/composition families and renderer configurations disjoint across development, calibration and final evaluation. Deduplicate by source fingerprint before splitting, freeze thresholds on calibration data, and do not tune based on the final test.

## 8. Proposed go/no-go gates

These are targets, not attained results:
- 100% expected verdicts on deterministic contract conformance controls after Rust tests are executed;
- zero accepted cases in the predeclared critical mutation panel, with sample counts and confidence bounds;
- at least 95% requested-property success and at most 5% protected-property violations on independent annotated edits before a reliability claim;
- positive paired EMR delta for Arm C versus Arm A with the 95% confidence interval excluding zero before claiming measured end-to-end improvement;
- report the full quality / latency / cost trade-off rather than a single ranking.

A zero-error finite panel is not a proof of zero population risk. A positive EMR delta on a small or selected subset is not a general industry-superiority result.

## 9. Execution record required

Every final report must pin:
- commit SHA and exact commands;
- model weights, licenses, code revision, decoding settings and tool versions;
- source dataset version, sample IDs, split manifest and evaluator revision;
- contract, source, output and receipt digests;
- per-arm tool calls, failures, abstentions, duration and resource budget;
- per-sample outputs or lawful reproducibility references;
- raw counts, confidence intervals, exclusions, and negative-control results.

At the time this plan was written, no Arm A/B/C study, acoustic detector calibration, human listening study, Rust test run, or challenge leaderboard submission had been executed by this work.
