# MEL-BENCH-001 — Reproducible music benchmark registry v1

**Status:** typed Rust profile/run manifest, rights and failure-accounting validator, direct-comparability classifier, two strict JSON schemas, one synthetic profile fixture, a CLI profile-check example and 15 authored Rust tests are committed. No external model run, SOTA reproduction, Rust compilation or CI result is claimed.

## Purpose

Give every Melothaea/Muse benchmark result a reproducible identity and prevent an aggregate leaderboard from concealing incomparable protocols, model versions, rights restrictions, inference budgets or excluded failures.

The module is a **music benchmark registry and adapter contract**. The included CLI checks one synthetic profile only; it emits no model or benchmark score. It does not implement evaluator isolation, candidate/oracle separation, signing or replay protection. Those belong to the shared EVAL-ORACLE work (#9437 / #9621). A run manifest references the independent evaluator receipt digest produced by that authority.

## Profile contract

A `MusicBenchmarkProfileV1` pins:
- benchmark ID/title/family and immutable release/version. The family taxonomy includes a distinct `music_editing` value for MMAE/SMEBench rather than forcing editing tasks into `other`;
- the native task/track ID and protocol reference;
- exact source manifest, split manifest and task-protocol SHA-256;
- native scorer ID/version/SHA-256 and the exact metric IDs it emits;
- a comparison-scope ID;
- intended data use and explicit rights status/reference/evidence;
- permitted uses listed by the rights holder or license.

Unknown or unresolved rights fail closed. A restricted asset can be used only for the uses explicitly named in its permission record. Cleared/restricted profiles require a rights-evidence digest. Do not train, redistribute, publish, or even use an externally sourced audio asset for evaluation solely because a copy is accessible.

## Run manifest contract

A `MusicBenchmarkRunManifestV1` binds:
- the exact profile digest;
- provider/model ID, immutable model/API version, observation timestamp, endpoint and optional model-weight digest;
- prompt-template and system-prompt digests, adapter and input-preprocessing digests, decoding configuration, runtime environment, runner commit, precision and seed;
- inference budget: requests, audio seconds, input/output tokens, retries/request and concurrency;
- terminal counts for success, failure, timeout, unparseable output and abstention;
- observed resource use: requests, audio, tokens, retries, concurrency, wall time and optional independently metered energy;
- native metrics with task scope, direction, unit and denominator;
- content digests for candidate output and raw output;
- the independent EVAL-ORACLE receipt digest;
- a full run-manifest SHA-256 commitment.

All attempted items must remain in the denominator. The contract does not permit silently dropping parser failures, timeouts, failed generations or abstentions. Energy remains absent unless a measurement record and digest accompany it; simulated/estimated energy is not represented as measured energy.

## Direct comparison rule

The registry first validates both run manifests, then marks a pair **directly comparable** only when benchmark ID/version, native track, source/split/protocol/scorer digests, metric IDs, comparison scope, inference budget, prompts, adapters, preprocessing, decoding configuration, precision, runner commit, runtime environment and seed all match.

Otherwise the result is **descriptive only**, with explicit mismatch reasons. Different benchmarks, different tracks, different scorers, different budgets, different prompts, or different preprocessing do not become comparable because their headline metric names look similar. In particular, never pool TAG-Bench mIoU, MusTBench Hit@3s, Arena Elo and a bespoke Melothaea score.

## Current external benchmark map (checked November 4, 2026)

### Temporal grounding and temporal reasoning

**TAG-Bench** (Dai et al., 2026; arXiv:2609.01542) evaluates every matching interval for a natural-language query. It contains 1,750 human-verified query/recording pairs across 149.5 hours; 22.1% have multiple ground-truth intervals. Across 21 systems, the top published mIoU is 31.2 and top recall at IoU ≥0.7 is 21.5%; no system exceeds 13.2% one-to-many count accuracy. The paper includes parse failures in the core localization denominators as empty predictions. This exposes a concrete gap in multi-occurrence grounding, not proof that every model fails every music task. https://arxiv.org/abs/2609.01542

**MusTBench** (Kwon et al., 2026; arXiv:2605.29300) has 1,264 QA examples over 875 tracks in five native tasks:
- TSG — Temporal Source Grounding (400);
- LTR — Local Timestamp Reasoning (208);
- TAD — Timestamp-Aware Description (208);
- GTO — Global Temporal Ordering (198);
- MTR — Music Temporal Region Grounding (250).

Use its native task metrics and evaluator, not an invented single score. https://arxiv.org/abs/2605.29300 ; https://github.com/sony/MusTBench

**MusicGroundingBench** (Fang, Wang & Fujinaga, 2026; arXiv:2608.29480) renders algorithmically generated piano MIDI to audio with exact symbolic/audio alignment. Its note-level and two-bar tasks are a strong controlled cohort for exact event/interval lineage, but are not representative of dense real-world polyphony. https://arxiv.org/abs/2608.29480

**TEMPO** (EMNLP 2026) provides unified audio/speech/music timestamping, with 119K training examples and a 10K evaluation set across five tasks. The authors report that SFT accounts for most improvement and GRPO supplies more modest refinement. This is a direct method comparator for timestamping, not proof that RL has no role in collaborative agency or music generation. https://aclanthology.org/2026.emnlp-main.694/

### Global music understanding: benchmark caution

**MUB** (arXiv:2610.22989, October 2026) covers 500 identity-masked recordings and 4,676 structural intervals. Its current public report is not a modern multimodal leaderboard: it notes a directional legacy key-metric bug, strongly imbalanced metre labels (492/500 are 4/4), unverified independent annotation agreement, unresolved recording rights and no contemporary LALM comparison runs. Treat MUB as a protocol to audit and reproduce after corrections, not as a current SOTA result. https://arxiv.org/abs/2610.22989

### Industry music-generation preference

A dated Artificial Analysis Instrumental Music Arena snapshot (September 2, 2026) reports blind-preference Elo: Suno V5.5 1186, Mureka V9 1177, Mureka V8 1166, Suno V5 1160 and Lyria 3 Pro 1117. This is a **narrow instrumental preference** comparison and does not measure structure control, temporal grounding, collaboration, rights, vocal generation or cultural validity. The leaderboard can change; capture a fresh immutable snapshot and task definition before any claim. https://artificialanalysis.ai/music/leaderboard/instrumental

Google's Lyria 3 product announcements describe Gemini-app 30-second generation and developer API access to Lyria 3 Pro/Clip. These are product capabilities, not independent quality scores. https://blog.google/innovation-and-ai/products/gemini-app/lyria-3/ ; https://blog.google/innovation-and-ai/technology/developers-tools/lyria-3-developers/

AgentAbstain (NeurIPS 2026 Oral) reports 263 paired near-identical agent tasks across 42 executable environments, with a best paired score of 59.5% across 17 frontier agents. It is a tool-agent abstention benchmark, not a music baseline; use its paired-task methodology only as an analogy for #10402. https://agentabstain.github.io/

## Qualification targets (targets, not results)

1. **TAG-Bench reproduction first.** Re-run the official top baseline with exact code/splits/scorer. Stretch goals on the unchanged official task: ≥40 mIoU, ≥35% R@IoU≥0.7 and ≥50% one-to-many count accuracy (published best currently 13.2%). Keep parse failures and under-counted cases in their native denominators.
2. **MusTBench.** Reproduce each of TSG/LTR/TAD/GTO/MTR separately. Stretch goal: ≥10-point improvement over the strongest independently reproduced baseline on the relevant task aggregate, with no task falling by more than 2 points. Do not invent missing baseline values.
3. **MusicGroundingBench / exact synthetic.** Target 100% source-event identity, interval and occurrence-count correctness on deterministic unambiguous fixtures; zero acceptance of wrong artifact/source lineage.
4. **Selective participation (#10402).** Pilot musician-defined costs first. Target ≥90% paired act/withhold correctness on a confirmatory cohort, with separate false-entry, missed-invitation, unauthorized-yield and abstention metrics. Deterministic veto/expiry hard gates target zero accepted violations. No claim that an abstention score numerically beats AgentAbstain unless task protocols are equivalent—which they are not.
5. **Instrumental human preference.** Reproduce a fresh blind arena snapshot with matched prompt, length, rendering and level controls. A stretch target is Elo >1200 with a 95% interval excluding the dated 1186 point estimate; separately report controllability, structure, editability, rights and cohort preferences. Do not optimize solely for Elo.
6. **Cultural validity (#10399).** Report separately per musician-governed cohort, performer and annotator familiarity. Do not infer African coverage from cross-cultural datasets that did not evaluate an African tradition.
7. **Provenance.** 100% of public claims link to pinned profile/run manifests and an evaluator receipt; zero unknown-rights training or redistribution; no hidden denominator changes.

All thresholds above are aspirational qualification goals. They are not current Melothaea results or claims of being better than SOTA.

## Execution sequence

1. Reproduce native benchmark baselines in their original environments and scoring scripts.
2. Import each benchmark into a separate profile/track with source/split/protocol/scorer digests and rights.
3. Run model candidates under a matched inference budget and fixed prompt/preprocessing/decoder protocol.
4. Submit candidate output to the independent EVAL-ORACLE process (#9437/#9621); bind its receipt digest in the run manifest.
5. Report native metrics, denominator accounting, per-task/per-cohort uncertainty and resource use.
6. Mark results as official-reported, locally-reproduced, exploratory or provisional; never blur these states.
7. Only after exact reproduction, set the next target based on the observed bottleneck (multi-occurrence count, long-context recall, calibration, control, rights or cost).

## Claim boundary

The registry makes benchmark evidence reproducible and comparisons more honest. It does not make a weak model stronger, certify a benchmark's rights, establish musician agreement, guarantee the evaluator process is secure, or establish universal musical intelligence. A leaderboard lead is a narrow task result; broader claims require the distinct temporal, generative, participatory, cultural and governance evidence above.
