# Melothaea competitive frontier — 23 November 2026

## Strategic framing

The field is advancing on at least four mostly separate fronts:

1. high-preference full-song generation;
2. controllable audio rendering and chunk-level editing;
3. symbolic planning and score-aware generation;
4. live interactive musical response.

Melothaea should not collapse those into one synthetic leaderboard. The stronger product strategy is to become a **renderer-independent musical director and collaborator**: preserve musical intent, score/event identity, authorship and reversible edits, then direct the strongest available renderer while measuring the live collaboration separately.

## Current external reference points

| System / benchmark | Current published scope | How to compare fairly |
|---|---|---|
| Artificial Analysis AA-Music Vocal v1.1 (current 23 Nov) | ACE Music v1: 1160±20 (2,128 votes); Suno v6: 1145±18 (3,073); Suno v6-mini: 1133±18; Mureka V9.5: 1101±18; Lyria 3.5: 1050±18; Lyria 3 Pro: 1048±18; Eleven Music v2.5: 1017±18 | Blind finished-track preference, vocals board only; 95% intervals shown |
| Artificial Analysis AA-Music Instrumental v1.1 (current 23 Nov) | Suno v6: 1148±22 (2,208 votes); Suno v6-mini: 1114±21; Mureka V9.5: 1084±21; Mureka V9: 1083±21; Lyria 3 Pro: 1080±21; Lyria 3.5: 1078±21; Eleven Music: 1021±21; Stable Audio 3 Large: 1014±21 | Separate instrumental board; do not merge with vocal Elo; 95% intervals shown |
| Taan TimingBench | 96.1% instruction success; 91.3% timed events within ±500 ms | Prompted temporal control, 250 thirty-second clips; vendor-published |
| Daydream research | 81 ms parameter-change-to-sound; 69% chord-root accuracy; 53% melody pitch-class accuracy; 80% groove-accent F1; 33% accompaniment onset match; 79% human-mix proximity; 100% note-density direction | Vendor-published task-specific realtime control; latency is parameter-to-sound, not musician-to-sound |
| JAM_BOT / NIME 2026 | Live symbolic MIDI call-and-response, variable tempo and latency compensation | Matched device/tempo/latency experiment; paper's player-piano result is narrow |
| MIREX 2026 CMI-RewardBench | CloudEar-overall 73.9% pairwise agreement on 249 core items; always-select-B baseline 59.0%; vocal/instrumental extensions separate | Reward-model/human-preference alignment; candidate order and label prevalence must be controlled |
| HEARTBEAT 2027 | Streaming WAIT/RESPOND/CONTINUE/YIELD protocol; complete chunk logs including non-responses | Causal turn-taking protocol reference, not a musician benchmark |
| ICASSP 2027 Audio Editing Challenge (MMAE) | 2,000 public dev examples / 17,741 rubrics; separate 500-example hidden Single Model and Agent tracks; IFR, Consistency Rate and Exact Match Rate | Direct comparator for localized edits, preservation and multi-step editing agents; registered teams' final submission/freeze extended to Nov 28, 2026 (11:59 PM Pacific); registration closed Oct 8 |
| YuE2 | Readable melody/chord plan expanded into audio tokens and full-song realization | Symbolic planning/audio integration; compare score editability and realized-score consistency |
| Eleven Music v2.5 | Structured composition plans of up to 30 chunks; audio-reference chunks can be kept unchanged while other chunks regenerate | Chunk-level edit baseline; score exact preservation and audio preservation separately |
| Suno MIDI-in | Upload a MIDI file to guide the generated track; available on desktop Pro/Premier | MIDI-conditioned generation baseline; not a reciprocal live collaborator |
| Lyria RealTime | Persistent stream with musical controls such as BPM, density, brightness and scale; 48 kHz stereo output | Continuous audio steering baseline; control responsiveness is not event-level reciprocal agency |

### Evidence notes

- Artificial Analysis's current 23 November board separates vocal and instrumental Elo. Current top vocal: ACE Music v1 1160±20 (95% interval, 2,128 votes), Suno v6 1145±18 (3,073), Suno v6-mini 1133±18, Mureka V9.5 1101±18. Current top instrumental: Suno v6 1148±22 (2,208), Suno v6-mini 1114±21, Mureka V9.5 1084±21, Mureka V9 1083±21, Lyria 3 Pro 1080±21 and Lyria 3.5 1078±21. These are not cross-modality comparable. The official methodology says ratings use blind pairwise votes on same-prompt, same-modality tracks and Bradley-Terry maximum-likelihood estimation rescaled to Elo, with 95% CIs; voters listen at least 10 seconds, and tracks are loudness-normalized toward -16 LUFS with static gain (boost capped at -1 dBTP), then re-encoded to 320 kbps MP3 at 44.1 kHz with provider metadata/artwork removed. This is a strong matched *finished-track preference* track, not causal interaction or agency. https://artificialanalysis.ai/music/leaderboard/vocals https://artificialanalysis.ai/music/leaderboard/instrumental https://artificialanalysis.ai/methodology/music
- Taan's numbers are published by Taan. Its task is 250 prompts for 30-second instrumental clips, with timed entries/exits and tempo instructions. ±500 ms is a broad tolerance and should not be conflated with beat-relative phase or real-time human interaction.
- Daydream describes its latency as time from parameter change to changed sound. That is not musician-input-to-acoustic-onset and is not directly comparable to the loopback endpoint Melothaea is designing.
- **MIREX 2026 CMI-RewardBench:** the official 249-example core reports CloudEar-overall at 73.9% pairwise preference agreement, with SongEval at 72.7% and an always-select-B baseline at 59.0%. Vocal and instrumental extension results are separate. Any Melothaea pairwise reward-model study must counterbalance A/B position, publish label prevalence, report balanced accuracy and intervals, and compare against position-only and majority-label baselines. https://music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench_Results
- **ICASSP 2027 Audio Editing Challenge:** the official challenge page posted a 23 November news update extending the final submission deadline/leaderboard freeze by three days, from November 25 to **November 28, 2026 at 11:59 PM U.S. Pacific Time**. Team registration closed October 8, so we can submit only if already registered. MMAE has 2,000 public development examples and 17,741 atomic rubrics; the hidden Single Model and Agent test tracks each contain 500 examples. The official judge uses a frozen Qwen3-Omni model three times per rubric with shuffled choices and majority vote; metrics are Instruction Following Rate, Consistency Rate and Exact Match Rate (primary). The challenge page says existing systems on MMAE score below 5% EMR. The Single Model baseline uses AuK end-to-end; the Agent baseline routes via DeepSeek-V4-Flash to DSP tools, SAM-Audio-Large and AuK. Keep hidden test inputs/rubrics sealed and report the challenge's judge score separately from independent loopback and musician evidence. https://audio-editing-challenge.github.io/ https://audio-editing-challenge.github.io/leaderboard/ https://2027.ieeeicassp.org/sp-grand-challenges/
- **JAM_BOT / NIME 2026:** the paper's first-minute *Für Elise* player-piano experiment reports beat-alignment F1 0.0678 without compensation, median 0.3386 with a fixed upper-bound estimate, 0.7638 with average latency compensation and 0.7834 with per-note/per-velocity compensation across five non-overlapping windows. Its throughput table includes a 170M model at 2 ms / 499.5 tokens/s in ggml CUDA on RTX 4090, and a 416M model at 4.04 ms / 247.65 tokens/s. These results are narrow, hardware-specific and not general-ensemble superiority, but they strongly motivate per-note/per-velocity output calibration and matched hardware reporting. https://nime.org/proc/nime2026_73/index.html https://nime.org/proceedings/2026/nime2026_73.pdf
- YuE2's paper reports that symbolic planning is preferred over no planning in its same-checkpoint expert preference setup (49.3% vs 34.6%). The accompanying WildSongBench site reports best-of-eight selection and a 192-prompt evaluation; best-of-eight must not be compared as though it had the same sample budget as one-shot systems.
- ElevenLabs composition plans give structured per-section text/style/duration. Its inpainting path can insert stored-song audio reference chunks unchanged while regenerating other chunks. The correct comparator is the preservation/edit task, not a generic preference score.
- Suno's own help page says a MIDI file can guide its Create flow while a prompt supplies genre/instrumentation/feel. This is a direct baseline for score/MIDI-conditioned generation.

## What Melothaea should do differently

### 1. Treat score and musical intent as the stable source of truth

A model-generated waveform should be a realization, not the only authoritative representation of what the system intended. Keep stable IDs for score events, motifs, parts, roles and edits. Bind every render to its source score/version and renderer identity.

### 2. Make every edit reversible and preservation-aware

For an edit request, define an explicit target region and an explicit preservation set. Compare:
- exact untouched MIDI/event preservation;
- beat-relative timing drift at preserved boundaries;
- audio similarity in unchanged regions;
- target instruction satisfaction;
- musical coherence at generated/retained transitions;
- undo/branch/rollback success.

A high-quality regeneration that silently changes a musician's approved passage is not a successful edit.

### 3. Separate continuous controls from reciprocal agency

Lyria RealTime and Daydream show that continuous audio controls and fast parameter response are competitive product capabilities. Melothaea's additional question is whether she responds to the musician's *actual* musical events, negotiates role/phrase changes, respects silence and veto, and can explain what input caused the response through an auditable receipt.

### 4. Make model and renderer identity part of each counterfactual trial

RMI-Bench already isolates mutable episode state. The next requirement is immutable model/adapter/runtime identity, now addressed in [#10206](https://github.com/Luminous-Dynamics/symthaea/issues/10206) and its implementation PR. Without this, same ID/version labels can hide a changed checkpoint or adapter between paired arms.

### 5. Treat control requests as hypotheses, not proof of control

A realtime API accepting a BPM, key/scale, density or brightness field does not prove the rendered audio followed it. Lyria RealTime's official guide documents a WebSocket workflow with weighted prompts and a live music-generation config. Separate community reports against the experimental endpoint describe cases where BPM changes had no audible effect and one scale setting returned audio identical to the unspecified-scale case. These are reports, not a definitive evaluation of every current Lyria deployment, but they motivate a strong benchmark rule: every control intervention must be tested against an independently observed audio effect, with a sham/no-op control and a randomized reversal.

For each claimed control:
- record the exact request, accepted response and effective model/deployment identity;
- randomize the control and sham conditions with the same prompt/seed/material where the endpoint permits;
- observe tempo, pitch-class/key, density, onset, dynamics or other claimed output dimensions from rendered audio;
- score the intended effect, collateral drift and persistence after control release;
- do not mark an accepted API configuration as a successful musical intervention unless the output changed in the intended way.

This is closely related to the observation-closure work: an instruction receipt proves what was sent, while an independent event/audio observation is needed to establish what changed.

## Proposed benchmark portfolio

Do not produce one universal “music intelligence” score. Maintain distinct tracks:

| Track | Primary metrics | Prospective target |
|---|---|---|
| Finished-track preference | Listener-stratified Elo/Bradley-Terry, CI, order balance | Reach parity with strongest matched systems; report separate vocal/instrumental results |
| Symbolic plan quality | Musician preference, motif/harmony/structure correctness, score-to-audio adherence | Held-out composition and musician cohorts; exact plan/audio lineage |
| Targeted editing | Preserved-region event accuracy, target edit satisfaction, transition quality, rollback | 100% exact preservation for explicitly immutable symbolic spans; all deviations surfaced |
| Live timing | Median/p95 beat-phase error, drift, recovery after tempo changes | ≤0.10 beat median / ≤0.25 beat p95 initial; ≤0.05 / ≤0.15 stretch |
| Reciprocal interaction | Target-specific causal response, yoked/unrelated controls, response directionality | Zero false causal positives on fixed/one-way/yoked controls; pass all positive controls |
| Turn-taking | WAIT/YIELD F1, false interruption, deadline success | ≥0.95 initial / ≥0.98 stretch; false interruptions ≤5% / ≤2% |
| Human agency | Veto, steering, branch acceptance, fatigue, authorship, diversity | Zero unauthorized output after veto; no material agency/ownership regression |
| Evidence integrity | Immutable identity, capture closure, manifest provenance, replay | 100% of scored episodes carry complete identity and observation receipts |

All numeric values above are proposed targets, not present results. Sample sizes and non-inferiority margins for human studies should be set by a pilot and power analysis, not selected after looking at confirmatory data.

## Matched baselines to add to the RMI-Bench programme

For the live/symbolic programme, compare:
- human-human musician reference;
- fixed accompaniment;
- reactive rule-based MIDI accompaniment;
- context-ablated Symthaea;
- one-way prompt generation;
- yoked replay from a separate matched session;
- direct MIDI-conditioned generation (Suno MIDI-in);
- chunk/inpainting composition (Eleven Music v2.5);
- score-first symbolic planning (YuE2-like pipeline);
- renderer-steered continuous audio control (Lyria/Daydream where the exposed interfaces permit matched tasks);
- full Melothaea reciprocal interaction.

Keep input material, action budget, number of revisions, instrument, tempo schedule, output route and measurement hardware matched. Do not count "number of accepted edits" as collaboration quality.

## Source links

- Artificial Analysis vocal leaderboard: https://artificialanalysis.ai/music/leaderboard/vocals
- Artificial Analysis instrumental leaderboard: https://artificialanalysis.ai/music/leaderboard/instrumental
- Artificial Analysis methodology: https://artificialanalysis.ai/methodology/music
- Taan TimingBench: https://taan.ai/research/
- Daydream research: https://daydream.live/research
- JAM_BOT / NIME 2026: https://nime.org/proc/nime2026_73/index.html
- HEARTBEAT 2027: https://heartbeat-challenge.github.io/
- YuE2 paper: https://arxiv.org/abs/2609.33757
- ElevenLabs composition plans: https://elevenlabs.io/docs/eleven-api/guides/how-to/music/composition-plans
- ElevenLabs inpainting: https://elevenlabs.io/docs/eleven-api/guides/how-to/music/inpainting
- Suno MIDI upload: https://help.suno.com/en/articles/14594177
- Lyria RealTime official streaming guide: https://ai.google.dev/gemini-api/docs/realtime-music-generation
- Lyria RealTime community report of BPM control not changing tempo: https://discuss.ai.google.dev/t/bpm-parameter-in-musicgenerationconfig-has-no-effect-on-tempo-lyria-realtime-exp/139309
- Lyria RealTime community report of an identical output for one scale value: https://discuss.ai.google.dev/t/lyria-realtime-scale-e-flat-major-c-minor-has-no-effect-output-bit-identical-to-scale-unspecified/186831

All competitors are comparators for their disclosed task, not proof that Melothaea already matches or exceeds them.
