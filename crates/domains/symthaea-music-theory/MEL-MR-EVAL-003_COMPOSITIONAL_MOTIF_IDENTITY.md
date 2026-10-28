# MEL-MR-EVAL-003 — Compositional motif identity and alignment

**Date:** 2026-10-28  
**Status:** research/design increment with synthetic smoke fixtures. This is not evidence that the Rust verifier passes the corpus.

## Implementation status at prototype revision

The PR now includes an isolated Rust reference implementation in `src/compositional_motif.rs`, exported by the crate root. It has versioned descriptors, independent per-axis evidence, monotonic edit alignment, coverage/gap gates, invalid-input rejection, and a 1,000,000-cell DP budget. Unit tests are authored for compound transformations, adversarial axis violations, bounded insertions, invalid/range claims, the computation budget, temporal-origin invariance, and the committed smoke corpus.

**Execution boundary:** those Rust tests have not been compiled or executed in this environment. Exact-head GitHub CI records are skipped. An independent local Python simulation of the intended alignment/verdict rules produced zero mismatches on the 108-case smoke corpus; this is algorithm-level fixture checking, not Rust execution or scientific validation.

Current API scope is deliberately monophonic: events must have strictly increasing onsets. Simultaneous notes, explicit rests, retrograde, ornament operators, audio realization, and historical restoration lineage are not supported by this contract and must not be inferred.

## Motivation: move beyond a single transformation label

The v5 motif-return contract improves fail-closed identifiability but exposes a single ReturnTransformation label. Musical transformations are compositional: a return can be inverted, transposed, augmented, ornamented and presented as a fragment at once. A winner-takes-one label discards information and makes obligations brittle when they specify multiple simultaneous properties.

The proposed shift is an auditable transformation descriptor. The verifier emits per-axis results and alignment evidence. No aggregate similarity score can compensate for a failed required axis. “Restored” remains a temporal-lineage claim and cannot be inferred from a pairwise comparison alone.

## Descriptor contract

Each claim declares pitch orientation (preserve or invert), signed semitone offset, onset scale, duration scale, and a source-selection descriptor (full phrase or a contiguous fragment with explicit start and length).

For a selected source region with anchor pitch pₐ and onset tₐ, orientation s ∈ {+1, −1}, semitone offset Δ, target start u₀, onset factor α, and duration factor β:

    target_pitch_i = pₐ + Δ + s × (source_pitch_i − pₐ)
    target_onset_i = u₀ + α × (source_onset_i − tₐ)
    target_duration_i = β × source_duration_i

This represents inversion + transposition + augmentation in one claim. Classical augmentation/diminution must satisfy both onset and duration scaling. Exceptions such as a fermata or inserted rest must be explicit operators, not hidden tolerance.

## Alignment contract

Replace blind positional zipping in the next implementation with monotonic sequence alignment and interpretable costs for substitutions, insertions, deletions and onset deviations. Report matched-event coverage, gap budget, pitch/interval residuals and temporal residuals independently. A permissive alignment must not rescue unrelated material through excessive gaps. Full transformations require full coverage; fragment claims declare their source window and minimum event count.

Each axis returns support status, estimate, residual, tolerance and evidence count. Accept only when every required axis passes; reject when evidence contradicts a supported claim; return unsupported for operators outside the verified contract. Calibrate thresholds on the calibration split and freeze them before the test split.

## Smoke corpus

The adjacent JSONL corpus has 108 deterministic cases from 12 motif families: 60 positives and 48 negatives. Family-level splits are development (6 families / 54 cases), calibration (2 / 18) and test (4 / 36). Per family, cases cover identity, transposition, transposition + augmentation, inversion + diminution, a transposed fragment, a false inversion claim, duration-only false augmentation, a single pitch outlier and onset jitter.

The JSON Schema and corpus are fixtures for a future Rust adapter. Their current QA checks schema, case IDs, split isolation and deterministic oracle consistency only. This smoke corpus is not human annotated and is not a representative generalization result.

## Proposed benchmark gates — targets, not results

1. At least 99.9% correctness on a frozen, diverse synthetic factorial suite; report each axis and each supported composition separately.
2. Kill 100% of predeclared safety-critical mutants with matched valid positives preserved. A mapped mutation matrix is not a mutation score unless mutants are actually executed.
3. Initial held-out annotated-data target: ≥95% precision and ≥90% recall per axis and common composed operator, with family-clustered confidence intervals and annotation agreement.
4. Report alignment precision/recall, coverage, insertion/deletion rate and residual distributions; never tune thresholds on test data.
5. Edit locality target: ≥95% requested-property success and ≤5% protected-property violations.
6. Blind motif recognition and full-piece pairwise preference with prespecified sampling and confidence intervals.
7. p50/p95 verifier latency, memory by event count, and deterministic replay/hash stability.

These are goals, not achieved performance. Grow to at least 50 distinct families per split, document licensing/provenance, and collect independent annotations before generalized claims.

## Industry and research comparisons

The 2026 music-generation survey notes that no single metric captures musical quality, diversity, faithfulness and control. TimingBench is a useful adjacent reference for timestamped event control and detector-error audits, but its audio-generation task is not directly comparable to symbolic motif-verifier accuracy. Artificial Analysis uses blind pairwise listening and Bradley–Terry-style aggregation for finished audio; that measures perceptual preference, not symbolic transformation correctness.

Do not claim “better than Suno/Udio” from this benchmark. The direct differentiator is verifiable compositional control and edit locality. A fair product comparison should use identical prompts, candidate budgets, rendering/normalization and blind listening, while separately reporting symbolic constraint correctness.

## Remaining implementation and evaluation sequence

1. Execute the authored Rust suite on an exact commit SHA with `cargo test --locked -p symthaea-music-theory`, formatting and Clippy gates; never infer a pass from a skipped workflow.
2. Generate a factorial suite across orientation, transposition, onset scale, duration scale, source-window selection, event insertions/deletions, and local perturbations. Run executable mutation tests against the Rust verifier and preserve matched valid positives.
3. Add a score adapter only after the standalone contract is validated, keeping the legacy ReturnTransformation wire format unchanged.
4. Grow to at least 50 independent motif families per split, validate annotation agreement, then calibrate tolerance and gap thresholds without test leakage.
5. Add the edit-locality benchmark and measure preservation of non-targeted pitch, rhythm, harmony, voice-leading, and form.
6. Run blinded motif-recognition and full-piece preference studies only after symbolic controls are stable. Compare these results separately from verifier correctness and report confidence intervals.

## References

- Wang et al., Recent advances in music generation: methods, evaluation, and challenges (2026): https://link.springer.com/article/10.1007/s10462-026-11582-x
- Bhake et al., TimingBench (August 2026): https://taan.ai/research/rushing-or-dragging/
- Carone, Roman & Ripollés, LLMs can read music, but struggle to hear it (PMLR 303, 2026): https://proceedings.mlr.press/v303/carone26a.html
- Artificial Analysis, Music Generation Benchmarking Methodology: https://artificialanalysis.ai/methodology/music
