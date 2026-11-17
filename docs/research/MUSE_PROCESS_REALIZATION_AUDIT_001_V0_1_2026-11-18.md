# MUSE-PROCESS-REALIZATION-AUDIT-001 — Additive Process Score Verification (v0.1)

Date: 2026-11-18  
Status: source increment authored; compile/test/CI evidence pending  
Repository: `Luminous-Dynamics/symthaea`

## Research question

When a ProcessAdditive composer declares a sequence of growing and shrinking cell prefixes, does the realized symbolic score actually contain those prefixes?

The existing planner emits `ProcessPlan { trajectory_kind, prefix_lengths, cycle_starts }`. The existing grammar trace labels regions (for example, `additive-prefix-3`), but those labels are assertions made by the producer. They are not an independent test that the notes in each region match the declared process. Plan-only tests can therefore pass while a realized score is corrupted.

This tranche closes that narrow gap without changing the grammar's output format, the generic obligation ledger, the renderer, or the audio pipeline.

## Implemented contract

`verify_additive_process_score_v1(plan, score)` is a fail-closed score-side verifier. It:

1. Requires a non-empty prefix plan and a one-to-one correspondence between prefix lengths and cycle starts.
2. Requires the first cycle to start at exact rational beat zero, positive prefix lengths, strictly increasing cycle starts, and positive half-open cycle windows.
3. Independently derives the canonical prefix contour for the declared `ProcessTrajectoryKind` and maximum prefix length. It does not call the planner's contour construction.
4. Assigns every melody event to exactly one planned cycle using rational beat comparisons, not floating-point tolerance.
5. Rejects melody events outside the trajectory, events crossing a cycle boundary, an incorrect cycle event count, and a cycle whose first event does not begin at its declared start.
6. Checks that consecutive melody events are contiguous and that the final melody event ends exactly at `Score.total_beats`.
7. Uses a maximal-prefix cycle as the realized cell template and requires each other cycle's melody to match the corresponding exact pitch-and-duration prefix.

The return value is a small versioned report: cycle count, observed melody event count, and maximum prefix length. The error enum reports specific rejection causes.

## Evidence boundary

This establishes symbolic score-to-plan realization only. It does **not** establish:

- that the process is audible or perceptually recognized as additive;
- that velocity and section-intensity dynamics match an expressive profile (not checked by v1);
- that accompaniment, timbre, orchestration, or renderer output is correct;
- that the Minimalism style is culturally authentic, artistically valuable, or preferred;
- persistent part identity (current ProcessAdditive notes use `PartId::UNASSIGNED`);
- that the score is superior to commercial audio generators.

In particular, event-count growth is a structural fact, not a listener-experience result. Audio/perceptual evaluation remains a separate scorecard.

## Adversarial fixtures authored

Six new tests are present in `process_grammar.rs`:

- accept unmodified realizations for all four declared trajectory kinds;
- reject a plan with a false cycle count / contour;
- reject a mutation to a note's rhythm/content within a cycle;
- reject a pitch mutation that breaks the repeated cell prefix;
- reject a shifted cycle boundary;
- reject a score with a missing melody event.

Existing process tests remain in place. No test has been compiled or executed in the current environment. Do not describe these tests as passing.

## Benchmark plan

Report separate measurements; do not collapse them into a single music-quality score.

| Dimension | Metric | Qualification target |
|---|---|---|
| Deterministic positive fixtures | Exact plan-to-score audit acceptance | 100% |
| Frozen adversarial fixtures | Invalid score rejection | 100%; zero false semantic accepts |
| Generated holdout cohort | Exact adherence across trajectory kinds, seeds, and bar counts | 100% for the pinned deterministic generator |
| Mutation robustness | Detection rate by mutation class, plus false rejection rate | 100% detection on the frozen matrix; report every class separately |
| Coverage | Accepted examples / all examples; abstentions by reason | Report explicitly; never hide abstentions in accuracy |
| Runtime | p50 / p95 / p99 verifier latency, by event count | Establish measured baseline before target setting |
| Perceptual validity | Blind listening recognition of growth/release and musical preference | Separate preregistered human study; no score-audit proxy |

A useful first holdout matrix is all four process kinds × seeds 0–255 × 2, 4, 8 and 16 input bars, with engine revision pinned. Then add mutation families: missing/extra notes, pitch substitution, duration change, wrong cycle start, wrong contour declaration, gap, overlap, cross-boundary note, and trailing/leading events. Preserve each case's plan digest, score digest, expected disposition, actual disposition and verifier version.

## Industry comparison and differentiation

Commercial music systems compete heavily on generated-audio quality, prompt adherence, iteration speed, editing experience and listener preference. This score-side verifier does not compete on those dimensions and must not be presented as doing so.

The differentiated proposition is **verifiable musical control**: a generator can make a formal promise, then an independent score-side component can reject a score that violates it. This is a complement to high-quality audio generation, not a replacement for it. A credible SOTA claim would require matched blind audio evaluations, clearly defined tasks and confidence intervals in addition to symbolic correctness.

## Paradigm shifts this increment supports

- **From plan assertions to proof-carrying scores:** a plan is not its own evidence; the realized events must independently satisfy it.
- **From global genre tags to time-indexed process:** growth, plateau, release and transition become explicit, testable temporal structures.
- **From scalar quality to a Pareto scorecard:** symbolic fidelity, editing control, identity, interaction, perceptual preference and runtime remain distinct.
- **From silent correction to fail-closed realization:** a mismatch rejects with a typed reason; the verifier does not silently retime, rewrite or reinterpret the performance.
- **From benchmark anecdotes to reproducible cohorts:** freeze inputs, seeds, engine revisions, mutation fixtures and exact outputs.

## Integration path

The next M1 integration should invoke this verifier on the Minimalism segment and retain its report beside the existing trace-integrity report. The verifier should remain narrowly scoped to ProcessAdditive until another grammar family has its own independent semantic contract. The MUSE Analyst typed-obligation work in #9797/#9835 remains the owner for general compositional obligations; this audit does not create a parallel obligation protocol.

## Validation status

- Source authored on a branch based on current `main` at `67939c904adcfafd246848b74cd53c52a2cca244`.
- Six new Rust unit tests authored.
- Cargo/rustc/rustfmt execution unavailable in this environment; compilation, formatting and test results are pending.
- No workflow pass, audio result, human preference result, or SOTA superiority is claimed.
