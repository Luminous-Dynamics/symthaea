# MEL-FRONTIER-001: Capability-Backed Musical Range Benchmark v1

**Status:** M0 source/identity frontier and symbolic smoke audit; not an audio-quality result.  
**Initial cohort:** the public Muse 152 catalog. **Growth policy:** additive; 152 is a cohort boundary, never a code limit.  
**Implementation:** `examples/capability_frontier.rs`, registered behind the existing `theory` feature.

## 1. Paradigm shift: capability frontier, not catalog size

A catalog entry, callable composer route, valid symbolic score, instrument identity, selected renderer, source-backed sample, playable gesture, and convincing musical result are separate evidence classes. Do not combine them into one `supported` boolean.

MEL-FRONTIER-001 models the search space as a versioned capability graph:

```text
canonical style ID
  -> callable grammar/composition route
  -> symbolic score contract and control evidence
  -> instrument identity + role/range/gesture requirements
  -> actual per-note renderer receipt
  -> selected sample/preset and source terms, if any
  -> signal and playable-range evidence
  -> matched audio baseline
  -> blind listener evaluation
  -> expert/tradition review where required
```

A style is not complete because a score was generated. A sound is not sample-backed because a folder is declared. A non-silent WAV is not proof of accurate pitch, instrument identity, or perceptual fidelity. A model-based style classifier is not a substitute for listening or qualified tradition review.

The larger goal is **combinatorial generalization**: when a new style grammar, instrument, expressive gesture, or production treatment is registered, Melothaea should be able to test unseen compatible combinations without adding a bespoke hand-written benchmark for every pair. The candidate set must be pruned by explicit compatibility/constraint rules, not by silently assuming every cross-product cell is playable.

## 2. M0 report implemented here

The Rust example produces one versioned JSON report with the source-derived registry state and a fixed-seed symbolic smoke result.

### Source-derived inventory

- Enumerates the complete `CATALOG` rather than a parallel hard-coded list.
- Audits each catalog ID, constellation, maturity/review label, declared `composer_style`, and whether that exact route name resolves to a member of `Style::ALL`.
- Records every canonical instrument ID from `Instrument::ALL`, Rust variant identity, declared renderer family, and General MIDI compatibility program.
- Emits a candidate frontier cell for every catalog-style × registered-instrument pair.
- Records the running executable SHA-256, catalog digest, and a declared source revision. The revision must be a full 40–64 character hexadecimal ID supplied through `SYMTHAEA_SOURCE_REV`, `GITHUB_SHA`, or `GIT_COMMIT`, in that order.

Counts are derived from live registries. The initial-cohort marker is 152, but the code uses the current registry length for cross-products and reports entries above 152 instead of rejecting them. No public ID should be renumbered when the catalog grows.

### Symbolic smoke

The benchmark calls the production `composer::compose_styled` path for every executable `Style::ALL` member and eight development seeds:

`[0, 1, 7, 13, 29, 42, 73, 101]`

For each style/seed it composes twice and compares a SHA-256 fingerprint over explicit score fields. It also checks style-independent mechanical properties: nonempty output, finite positive tempo and total duration, nonzero meter, non-negative onset, positive note duration, and velocity in `[0, 1]`.

This is a **development smoke test**. Passing it only says the symbolic output met these bounded mechanics and replay checks. It does not certify a distinct style identity, responsive controls, expressive performance, renderer behavior, sample coverage, or listener preference. It deliberately does not claim that the cross-axis candidate cells have already been tested.

### Candidate-cell states

- `style_not_callable`: catalog entry has no declared composer route.
- `declared_route_unresolved`: catalog contains a route name that does not resolve to the executable style roster.
- `candidate_unqualified`: the style route resolves and an instrument identity exists, but the style×instrument pair has not yet passed the paired evidence gates.
- `measured_pair_evidence=false`: always explicit in M0; registry presence is not pairwise qualification.

The report attaches required downstream gates to each pair: route/score contract, per-note render receipt, register coverage, dynamics/articulation/gesture coverage, signal-health/replay evidence, and blind audio evaluation for quality claims. Cultural or tradition-specific claims retain a separate expert-review requirement.

## 3. Running M0

Run from the repository root on a native target:

```sh
SYMTHAEA_SOURCE_REV="$(git rev-parse HEAD)" \
  cargo run -p symthaea-muse --features theory --example capability_frontier -- \
  /tmp/melothaea-capability-frontier.json
```

To inspect just the JSON on stdout, omit the output path. The run is intentionally offline and does not download datasets, soundfonts, or sample libraries. The report stores the exact declared revision and executable digest so results can be compared without relying on a mutable branch name.

The digest of the running executable binds the binary bytes, not the build process. Environment-provided source revision is caller/build-supplied metadata, not an authenticated provenance attestation. Preserve the report beside the exact source checkout and independently observed build/test evidence.

## 4. What comes next: M1–M4

### M1 — Connect actual instrument evidence

Join this frontier to the existing instrument capability inventory, semitone render sweeps, per-note renderer receipts, selected-sample digests, and source-terms verifier. These belong to the existing work under [instrument support #7416](https://github.com/Luminous-Dynamics/symthaea/issues/7416) and related work such as [per-note renderer evidence #7914](https://github.com/Luminous-Dynamics/symthaea/pull/7914) and [sample/source provenance #8019](https://github.com/Luminous-Dynamics/symthaea/pull/8019); do not fork a second authority for those facts.

Produce a versioned compatibility graph for pitched/unpitched semantics, instrument range, dynamics, articulations, tuning system, ensemble role, renderer availability, production assumptions, and source-term evidence. Unknown stays unknown. Incompatible or unavailable cells count as explained outcomes, not hidden omissions.

### M2 — Controlled cross-axis execution

For each declared-compatible cell, freeze the exact style/grammar, instrument, register, dynamics, gestures, performance dialect, renderer settings, and seed. Start with a bounded pilot across the currently callable style roster, then cover the full 152-entry catalog as styles become callable.

Each attempt must retain:
- requested capability identity and resolved production route;
- exact style recipe, instrument ID, control values, seed, and output digest;
- per-note selected-render-path receipt and selected asset/preset evidence, when applicable;
- all rejected, unsupported, failed, silent, clipped, or resource-limited cells;
- replay result and resource cost (latency, memory, output duration/size);
- versioned compatibility and qualification rules.

Do not mark a cell qualified by borrowing a neighboring instrument's or style's result. A shared mechanism can be reused, but the exact declared contract must be shown to survive it.

### M3 — Held-out compositional generalization

Reserve held-out combinations, not only held-out random seeds. Hold back entire style×instrument pairs, gesture×instrument pairs, style×production combinations, prompts, and—for specialist traditions—practitioners/repertoire items where feasible. Train/tune no model or selector on confirmatory items.

Compare:
1. current fixed presets;
2. a style-agnostic/common-renderer baseline;
3. the declared capability-guided resolver;
4. the same resolver with one defining constraint removed (negative control);
5. a human-authored or professionally arranged reference where rights permit.

Match duration, loudness, instruments, hardware, candidate budget, and post-processing. Publish one-shot results separately from best-of-2/4/8 selection. Never compare a best-of-N candidate with a one-shot baseline.

### M4 — Open-ended registry growth beyond 152

New styles should be append-only, versioned profiles rather than a growing pile of enum branches. Keep stable catalog IDs and separate:
- structural grammar owner;
- phrase/form contract;
- harmonic or modal syntax;
- rhythmic-cycle and accent contract;
- melodic vocabulary/ornaments;
- ensemble and instrument constraints;
- performance dialect and expressive gestures;
- production environment;
- cultural/rights provenance and required reviewers;
- supported control axes and measured response curves.

Register a new profile only when its declared fields validate. Unknown behavior remains a capability gap. A new combination may receive an honest “hybrid / inspired by” identity instead of being represented as a historical tradition it has not demonstrated.

## 5. Proposed qualification targets

These are **proposed targets, not achieved results**. Establish and publish a frozen baseline first. Report per-style/per-instrument tails and failed cases; averages alone are insufficient.

| Dimension | Proposed target | Evidence boundary |
|---|---|---|
| Registry integrity | 100% unique, stable IDs; 100% of declared routes resolve or carry an explicit reason-coded unavailable status | Source audit, never a musical-quality claim |
| Initial catalog cohort | Reach 152/152 callable routes over time, but mark quality/foundation maturity separately | Route availability does not establish style identity |
| Symbolic mechanics | 100% valid output over at least 10,000 fixed-seed/configuration cases; zero unexplained universal invariant failures | Exact source head, seeds, configurations, full denominator, and validator fault-injection controls |
| Replay | 100% exact symbolic replay for declared deterministic configurations | Hashes over canonical score fields |
| Control contract | At least 95% of declared-live style/control pairs exhibit the preregistered target effect, with non-target effects reported separately | Held-out seeds; do not count intentionally unsupported controls as live |
| Cross-axis coverage | Every compatible pilot cell has an explicit terminal status; zero missing/hidden attempts; unsupported cells carry reason codes | Candidate count is not the denominator for claimed perceptual quality |
| Instrument range and gestures | At least 95% coverage inside each evidence-backed declared playable range; no unreported holes | Separate pitch, velocity/dynamics, articulation, tuning and gesture layers |
| Generalization | Establish scores on held-out style×instrument and gesture×instrument combinations; target no more than 5 percentage points degradation from matched in-scope performance, reported per family | Prospective research target; never average away tradition-specific failures |
| Style identity | After expert-label acquisition, broad-family macro-F1 ≥0.80 and exact-label top-1 ≥0.65 on the reviewed subset | Classifier diagnostics corroborated by blind listening and confusion matrices |
| Audio preference | Target >60% pairwise preference against the strongest matched baseline with a 95% CI lower bound >50%, then replicate on held-out prompts/listeners | Separate vocal/instrumental, style, instrument, task, and subgroup results |
| Objective metric validity | Target Spearman `ρ ≥ 0.50` against held-out expert judgments for each preregistered dimension | Metric is frozen before confirmatory evaluation; correlation is not proof of quality |
| Cost | Report p50/p95 generation and rendering latency, memory, audio duration, candidate budget and failure rate on named hardware | No SLO before measurement baseline |

The numeric thresholds are aspirational engineering targets. We should revise them only through a versioned rationale and fresh evidence—not after seeing disappointing results.

## 6. Industry/SOTA comparison without a misleading leaderboard

External benchmarks use different tasks and populations; their scores should not be treated as directly comparable to Melothaea's symbolic-style route or cross-axis coverage.

- [MIREX 2026 Symbolic Music Generation](https://www.music-ir.org/mirex/wiki/2026%3ASymbolic_Music_Generation) evaluates piano continuation from a four-measure prompt to a 16-measure piece, with blinded human ratings for prompt coherence, creativity, structuredness, and overall musicality. It allows a fixed eight-candidate selection stage. Melothaea should report matched one-shot and best-of-8 results separately.
- [Artificial Analysis Music Arena methodology](https://artificialanalysis.ai/methodology/music) uses blind pairwise listening votes, Bradley–Terry-derived Elo-like ratings and confidence intervals, normalizes loudness toward −16 LUFS with a −1 dBTP true-peak ceiling, and breaks down results by genre and vocal/instrumental mode. We should adopt the fairness principles and add per-instrument and control-fidelity analysis—not imply a score match without running a matched study.
- [MIREX 2026 music audio generation](https://music-ir.org/mirex/wiki/2026%3AMusic_Audio_Generation) separates melody, harmony, key consistency, style consistency, and arrangement/production in expert listening. Its task is audio-cover generation and is not a direct substitute for the broader style-completion experiment.
- The published [MIREX 2026 CMI-RewardBench results](https://music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench_Results) evaluate **preference predictors**, not music generators, against human pairwise choices. On the 249-pair core dataset, the best reported submitted system achieved 73.9% accuracy, SongEval's average 72.7%, and the CMI-RM baseline 65.9%; the extended set reports separate vocal/instrumental subsets. Those numbers provide an external target for a future Melothaea reward model only after matching the dataset and protocol. They are not Melothaea music-quality scores and must not be compared directly with style-route or composition metrics.
- The open-access [2026 survey on music-generation methods and evaluation](https://doi.org/10.1007/s10462-026-11582-x) describes fragmented evaluation and the absence of a single metric covering the many dimensions of music-generation quality. This is a reason to publish a multi-axis scorecard, not to invent a single universal “Melothaea score.”

Our differentiator should be the **auditable capability frontier**: explicit constraints, true route/render evidence, compositional tests, held-out cross-axis generalization, and cost/quality trade-offs. It is reasonable to aim to outperform specific baselines on preregistered tasks. It is not yet justified to claim we exceed state of the art overall.

## 7. M0 verification boundary

The example and schema are authored on a feature branch stacked on the canonical instrument-roster work. This connector environment has no local Rust checkout or compiler, so formatting, compilation, and tests have not been executed here. No CI/benchmark PASS is claimed. The eight-seed smoke is a generated measurement once run—not an evidence result until its exact-head output is preserved and independently inspected.

No third-party recordings, stems, or lyrics are included. Any subsequent listening corpus must use original works, verified public-domain material, or explicitly licensed/authorized assets.
