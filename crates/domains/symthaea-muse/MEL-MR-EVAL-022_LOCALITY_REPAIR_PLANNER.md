# MEL-MR-EVAL-022 — Locality-Constrained Musical Repair Planner

**Status:** first Rust proposal-selector tranche authored on 12 November 2026. The module is feature-gated behind Muse's existing \`theory\` feature. Nine deterministic unit tests are authored, but Rust compilation/test execution, formatting and Clippy are not yet confirmed. No real-render, human-listener or SOTA improvement is claimed.

**Parent:** [#9185 — Build locality-constrained musical edit repair planner](https://github.com/Luminous-Dynamics/symthaea/issues/9185)

**Evidence chain:** [#9128 EVAL-017](https://github.com/Luminous-Dynamics/symthaea/pull/9128) → [#9133 EVAL-018](https://github.com/Luminous-Dynamics/symthaea/pull/9133) → [#9151 EVAL-019](https://github.com/Luminous-Dynamics/symthaea/pull/9151) → [#9155 EVAL-020](https://github.com/Luminous-Dynamics/symthaea/pull/9155) → this proposal selector.

## Why this layer exists

The preceding work produces distinct kinds of evidence:

- **EVAL-017:** a replay-verified, renderer-specific counterfactual influence estimate for one performed event, with exact MIDI/event-map binding and repeatability controls.
- **EVAL-018:** a four-coalition pairwise interaction measurement, separating PCM and spectral residuals.
- **EVAL-019:** a deterministic sparse probe plan under a render budget. It selects pairs to measure; a selected pair is not yet a measured interaction.
- **EVAL-020:** exhaustive small-N synthetic coalition outputs and PCM/spectral Harsanyi dividends. This is a known-answer reference engine, not a FluidSynth or perceptual model.
- **EVAL-010/EVAL-012:** frozen edit contracts and source/target symbolic/audio binding.

Those components do not select a minimal repair candidate. EVAL-022 is that bridge. It does not create a second influence model, interaction estimator, generic DAG planner, or mutation authority.

## Paradigm shift

Replace generate-and-hope with a constrained sequence:

1. Compile the requested musical edit into a frozen operation contract.
2. Validate a bounded symbolic candidate against the exact source Score, target Score, symbolic-locality receipt, renderer manifest and preservation profile.
3. Replay EVAL-017/018 reports against the exact artifacts before consuming their summaries.
4. Use EVAL-019's replayed probe plan to distinguish selected-but-not-yet-rendered probes from pairs that were not selected.
5. Exclude candidates that mutate protected events, lack influence evidence, exceed the declared probe-render budget, or have unresolved changed-to-protected interactions beyond the allowed threshold.
6. Return the Pareto frontier with separate objective values, plus a deterministic suggested candidate.
7. Render the chosen candidate under the exact plan and independently qualify its actual output. Planning evidence never authorizes a mutation and never certifies preservation.

## Implemented first tranche

\`src/locality_repair_planner.rs\` adds:

- A typed request binding the source Score, instruction, renderer manifest, preservation profile, renderer execution profile, sparse-probe profile, protected event inventory and hard render budget.
- A symbolic candidate qualification function that consumes the existing frozen edit contract and EVAL-012 symbolic/audio contract validation. The supported operation subset is intentionally limited to \`PitchAt\` and \`Transpose\`; unsupported edits reject rather than falling back to a broader global rewrite.
- Opaque summary types for EVAL-017 influence, EVAL-018 interaction and EVAL-019 probe-plan evidence. Constructors first call the existing report replay verifiers.
- Exact source Score / instruction / renderer manifest / preservation profile / renderer identity / renderer configuration matching across candidates and evidence.
- Per-candidate changed-event identity and protected-event checks.
- Separate accounting for missing event influence, no-detectable-influence outcomes, unmeasured changed-to-protected pairs, selected-but-unmeasured pairs, and unselected pairs.
- A hard render-budget status using EVAL-019's cached-render estimate.
- A multi-objective vector retaining changed-event count, symbolic pitch magnitude, EVAL-017 measured support, EVAL-018 PCM RMS/peak and spectral interaction, affected interaction windows, estimated render count and unknown protected-pair count.
- Deterministic Pareto-frontier construction and stable tie-breaking for a preferred *proposal*. It is not a claim that a scalar objective captures musical value.
- A domain-separated plan digest and replay verifier over the same opaque qualified evidence summaries.

### Important distinctions preserved

**Selected pair ≠ measured pair.** EVAL-019 may schedule a pair, but until an actual four-coalition EVAL-018 report replays, that pair remains unknown.

**No detected influence ≠ zero influence.** EVAL-017's \`NoEffectAboveNoiseFloor\` is retained as an unresolved target-audibility condition. It does not become a zero-risk score and does not enter the ready-to-render frontier.

**Measured influence ≠ sample ownership.** EVAL-017 support frames are an empirical exposure descriptor. They do not identify exclusive sample ownership or prove audibility.

**Pairwise interaction ≠ nonlinear renderer verdict.** EVAL-018's PCM and spectral residuals are retained as separate diagnostics. The planner does not call a renderer additive or nonlinear.

**Pareto choice ≠ edit acceptance.** The frontier recommends which candidate to render. Only independent post-render verification can accept the actual changed artifact.

## Synthetic tests authored

Nine tests exercise:

1. Missing EVAL-017 evidence is not interpreted as zero influence.
2. Missing changed-to-protected EVAL-018 reports become probe debt.
3. An EVAL-019-selected pair is still unmeasured until its four-coalition report exists.
4. Complete event and protected-pair evidence produces a candidate-render proposal.
5. An insufficient hard render budget blocks the proposal.
6. Pareto selection retains non-dominated low-edit/high-spill and more-edit/low-spill alternatives while removing a dominated candidate.
7. A candidate that mutates a protected event rejects.
8. The report replays exactly and tampering is rejected.
9. A calibrated EVAL-017 no-detect result is not treated as zero target risk.

These are authored unit tests; no test pass is claimed.

## Relationship to existing architecture

- **CQEP-001 / #11525 / PR #11528:** generic capability-qualified edit DAG. EVAL-022 consumes its plan/capability boundary; it does not replace the generic compiler.
- **MUSE-RUNTIME-001 / #11540:** one-shot content-bound authorization, attempt journaling and unknown-outcome reconciliation. EVAL-022 does not authorize or dispatch a mutating renderer call.
- **VMEA-001 / #11503 / PR #11506:** task-aware request execution and preservation rubric evaluation.
- **MMAE-TRACE-001 / #11508 / PR #11517:** multi-turn constraint persistence and exact output chaining.
- **EVAL-021 / #9184:** human calibration on held-out renderers and instruments. The planner's acoustic summaries are engineering evidence, not a listener-quality metric.

## Benchmark protocol

Compare the planner under matched candidate sets and hard render budgets:

- **Full regeneration:** regenerate the full score/audio for every instruction.
- **Single-event greedy repair:** choose the smallest one-event candidate without influence evidence.
- **Influence-only repair:** use EVAL-017 support summaries but ignore pairwise interactions.
- **Pairwise repair:** use replayed EVAL-018 reports and fixed probe allocation.
- **Budgeted repair:** use EVAL-019 selection and preserve unprobed uncertainty.
- **Exhaustive oracle bound:** use EVAL-020 coalition truth on small synthetic scenes only; it is not a production renderer.
- **No-op / abstention:** for impossible, under-specified or unsupported requests.

Split by composition lineage, not random windows or repeated seeds. Hold out renderer, SoundFont/instrument family, effect topology and at least one interaction family. Keep real-audio items and synthetic exact-ground-truth fixtures separate.

## Metrics and initial research targets

All targets below are proposed goals, not results:

| Dimension | Initial target |
|---|---:|
| Protected symbolic event mutation | 0 accepted deterministic negative controls |
| Candidate receipt / plan replay | 100% exact replay on frozen deterministic fixtures |
| Unknown event influence | Never silently converted to zero |
| Selected-but-unmeasured pair | Never treated as a measured interaction |
| Hard render budget | 100% rejection on enumerated over-budget controls |
| Pareto correctness | 100% agreement with brute-force dominance on bounded fixtures |
| Collateral vs full regeneration | Lower collateral at matched edit success, with paired cluster-aware intervals |
| Render efficiency | Fewer renders to independently verified acceptance than greedy/full-regeneration baselines |
| Risk coverage | Publish abstention and unsupported-operation rates, not forced acceptance |
| Listener alignment | Demonstrate improvement on EVAL-021 blinded judgments before product-quality superiority claims |

Keep requested-edit success, protected-content violations, human preference, interaction calibration, runtime/cost and abstention as separate outputs. Report p50/p95 render cost and time-to-verified-acceptance. Do not claim a single composite score is a universal measure of musical quality.

## Known limits / next work

- The current symbolic candidate qualifier is deliberately limited to pitch-at and whole-score transposition. Instrument, articulation, rhythm, tempo, duration, phrase structure and insertion/deletion operations need their own typed contract comparators and tests.
- The current candidate objective uses EVAL-017/018 summaries as risk descriptors. It is not an acoustic collateral prediction model with calibrated probabilities.
- EVAL-019's render estimate is consumed as a planner budget input; this module does not dispatch the renders.
- The request binds the preservation-profile digest, but the runtime still needs to resolve that profile against its trusted registry and verify the exact registry snapshot.
- Human perceptual quality and unseen-renderer generalization remain unmeasured.
- No CI pass is claimed. The first qualification workflow must compile and execute the new module at an exact head; legacy Cargo.lock mismatch is an existing blocker and must not be bypassed by removing \`--locked\`.

Next increment: connect the planner output to the runtime's one-shot authorization/reconciliation state machine, then add a real end-to-end campaign that compiles a pitch edit, performs EVAL-017/018 measurements, chooses a candidate within budget, renders it, and independently checks the exact output.
