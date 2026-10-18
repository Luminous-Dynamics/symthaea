# Creative Intent Contracts v1

Status: additive engineering prototype; no claim that the contract is connected to production selection or feedback.

## Why this tranche exists

The creative-quality audit in CREATIVE_METRIC_VALIDITY_AUDIT_2026-07-31.md records a concrete failure: the current melodic-coherence proxy prefers a one-pitch drone over a more varied generated composition, while the current form proxy can reward a concentration of notes followed by silence. These are not repaired by adding more arbitrary weights.

The underlying problem is category error. A repeated-pitch drone may be exactly right for an ambient brief and a failure for a melody brief. A single scalar with no declared intent cannot tell those cases apart.

## Paradigm shift: intent contract first, measured features second, taste evaluation separately

The first vertical slice adds CreativeIntentContractV1 in symthaea-muse. A caller explicitly declares mechanical constraints; the evaluator reports observations and contract violations. It does not infer that the candidate is beautiful, emotionally successful, original, culturally grounded, or fun.

The contract initially measures:
- valid note count;
- distinct rounded MIDI pitches;
- pitch range in semitones;
- onset coverage across equal-duration windows;
- share of notes occupying the dominant pitch.

Malformed contracts, durations, and notes fail closed. Invalid notes are reported by index and cannot be silently discarded to make a candidate pass. The module deliberately leaves the legacy CreativeQualityScore and live reward loop unchanged until the new seam has been calibrated and integrated in a separately reviewed change.

## How this generalizes to games and cross-modal creative work

A future shared project contract should declare, by modality:
- intention and audience;
- immutable requirements and editable degrees of freedom;
- machine-verifiable constraints;
- observed feature vector, with no universal aesthetic scalar;
- provenance, seed, build and generator identity;
- human review questions and failure cases.

For a game, hard constraints might cover launchability, reachability of an ending, controller/input coverage, save/replay integrity, and content restrictions. Separate feature descriptors could capture navigation density, risk/reward profile, mechanics, pacing, visual palette, and sound dynamics. A candidate can be mechanically admissible but artistically weak; those outcomes must not be collapsed into one label.

For music, candidate search should retain a set of outputs along different useful axes instead of repeatedly selecting the current scalar maximum. An ambient drone, a lyrical melody, a percussion-led groove, and an experimental piece should be judged against their respective briefs. The system should preserve surprising but admissible candidates rather than allowing the current taste model to erase all alternatives.

## Benchmark programme

All numbers below are proposed gates, not achieved results.

### Gate A — contract correctness
- 100% of seeded invalid-contract and invalid-note cases are rejected.
- 100% of seeded task-constraint violations are detected by the independent test oracle.
- Identical input and contract produce identical reports.
- No malformed candidate is accepted because its invalid fields were dropped before scoring.

### Gate B — generator usefulness
Compare, on matched briefs and held-out seeds:
1. current generator without candidate filtering;
2. random candidates filtered only by the contract;
3. generator plus contract;
4. a quality-diversity search that maintains alternatives across declared feature dimensions.

Report acceptance rate, runtime, memory, contract-specific feature coverage, pairwise novelty, and every rejection reason. Do not compare best-of-N selection with one-shot generation without reporting candidate budget and both results.

### Gate C — human evaluation
- Blind, randomized pairwise ratings on held-out briefs and evaluators.
- Target: more than 60% preference over the strongest matched baseline, with the 95% confidence interval's lower bound above 50%, replicated on a separate set of briefs before any superiority claim.
- Independently rate intent adherence, coherence, novelty, diversity, playability/listenability, and overall preference. Preserve disagreements; do not merge all dimensions into one score.
- Report within-system and across-system diversity. A system can make each individual user's output more pleasing while narrowing collective diversity.

### Gate D — cross-modal integration
For a playable prototype, require:
- all critical interaction paths pass deterministic smoke tests;
- sound, visuals, mechanics, and narration remain bound to one versioned project intent;
- at least 90% of blinded reviewers can identify the declared high-level intent from the assembled experience;
- revision improves a preregistered human-rated dimension without worsening hard validity or collapsing diversity;
- every claim is tied to the exact source/build and evaluation corpus.

The 90% intent-recognition target is an engineering aspiration, not a literature standard. Calibrate the task and reviewer protocol before confirmatory trials.

## Additive temporal-support measurements

The separate module `src/creative_temporal_support.rs` addresses a known blind spot without changing `CreativeIntentContractV1` acceptance semantics or the existing QD archive:

- **Onset-window coverage**: how many windows contain attacks.
- **Sustain-window coverage**: how many windows overlap at least one sounding note.
- **Occupied-time ratio**: the union of sounding note intervals divided by clip duration, so overlapping notes are not double-counted.
- **Maximum silent gap**: the longest silent interval, including leading and trailing silence.

This separates a single sustained drone from a sequence of staccato attacks that happens to touch every window. Notes are fully validated; note tails crossing the declared clip end are clipped only for these measurements. Empty scores are representable and report zero occupancy with the full clip duration as silence. Excessive window counts fail before allocation.

The measurements are descriptive and are **not** currently used to accept/reject V1 candidates, rank outputs, or modify production selection. They should become QD dimensions only after calibration against human judgments. The tests cover long sustain, short attacks, interval union, clip-edge clipping, malformed notes, empty scores, window limits, note-order invariance, and duplicate-note invariance.

## Research basis

- Khalifa et al., The Procedural Content Generation Benchmark: An Open-source Testbed for Generative Challenges in Games (FDG 2025). It explicitly separates quality, diversity and controllability over 12 game-content problems: https://arxiv.org/abs/2503.21474
- MIREX 2026 Symbolic Music Generation uses double-blind subjective ratings for prompt coherence, creativity, structuredness and overall musicality, and separates an eight-candidate generation/selection budget: https://www.music-ir.org/mirex/wiki/2026%3ASymbolic_Music_Generation
- Ding et al., Quality Diversity through Human Feedback (ICML 2024). It explores inferring diversity dimensions from human similarity judgments instead of relying only on hand-authored descriptors: https://proceedings.mlr.press/v235/ding24h.html
- Bradley et al., Quality-Diversity through AI Feedback (ICLR 2024). It studies combining AI-based critique with diversity-driven search for qualitative generation tasks: https://proceedings.iclr.cc/paper_files/paper/2024/hash/5b9bef4eae0f574cedbf9f4bf29d8ae7-Abstract-Conference.html
- Doshi & Hauser, Generative AI enhances individual creativity but reduces the collective diversity of novel content (Science Advances, 2024): https://doi.org/10.1126/sciadv.adn5290

## Non-claims and follow-up

This prototype is not a creativity test, aesthetic critic, consciousness measurement, music-theory validator, player-experience test, or proof that an output is good. It is a fail-closed mechanical contract boundary. Before production integration, separately audit and repair the known legacy score defects, add human-calibrated criteria, and run the benchmark matrix above against declared baselines.


## Executable contract / descriptor-diversity slice

The first frozen corpus and runner are available as a deterministic benchmark fixture:
- Corpus: crates/domains/symthaea-muse/fixtures/creative_intent_contract_v1.json
- Rust runner: crates/domains/symthaea-muse/examples/creative_qd_benchmark.rs
- Reusable receipt/archive implementation: src/creative_qd_bench.rs

Run it with:

    cargo run -p symthaea-muse --example creative_qd_benchmark -- <source-commit-sha>

The corpus has five task families and 17 fixed candidates: ambient drone, lyrical melody, rhythmic groove, formal development, and experimental texture. It contains adversarial controls for one-pitch melodies, clustered events, invalid note values, and onsets at the duration boundary, and a zero-velocity silent-event trap. Each candidate has a preregistered expected acceptance value. The runner emits a SHA-256-bound JSON receipt, candidate-level audits, per-family acceptance counts, and a simple descriptor archive.

The archive uses pitch-diversity, temporal-coverage, onset-density, and inter-onset-interval-CV bins. The benchmark runner receipt is now schema version 2 (the frozen input corpus remains schema version 1). Every candidate receipt includes a temporal-support outcome: measured onset coverage, sustain coverage, occupied-time ratio, and maximum silent gap, or an explicit error when the notes are malformed. This evidence is recorded side-by-side with contract acceptance. It does not yet affect the contract result or the archive's four descriptor bins. For the frozen steady ambient drone, the receipt can show 100% onset-window coverage and 100% sustain-window coverage while occupied time is only 40%; this is precisely why attack coverage is not interchangeable with sustained support or time occupancy. It keeps the lexicographically smallest candidate ID in a colliding cell so archive membership is independent of candidate enumeration order. It does not score aesthetic quality and does not wire into production selection. The onset-interval CV is only a statistical feature; it cannot tell whether an irregular pattern is intentional syncopation because meter, beat phase, and instrument role are not in this contract.

Proposed first gates: all preregistered fixture outcomes match; unknown corpus fields fail closed; receipts bind the exact fixture bytes; and reversing candidate enumeration leaves occupied cells and representatives unchanged. The 16 fixtures are a synthetic correctness corpus, not a human preference study, generator benchmark, or measured creative-quality result.
