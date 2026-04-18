# Phase 6 scoping — Cognition-to-Lean Bridge

**Status:** plan draft. 2026-04-18.
**Prereq:** this doc was commissioned after option #1 (invariant-discovery benchmark) produced a first measurement of Symthaea's GP on closed-form sequences (6/13 = 46.2% closed, `docs/phase6-scoping/invariant_discovery_seed42.csv`).

## The gap

Phases 1-5 built a **Lean-bridge**: static Rust code that compiles `FolFormulaExt` goals into hand-designed Mathlib tactic cascades. The bridge has reached 96.9% on hand-curated miniF2F and 44% on auto-ingested. But:

- **The cognitive loop never sees the goals.** `src/cognitive_loop/cycle.rs:38` takes `&str` input, produces a `CycleResult` with HDC hypervectors and active-inference telemetry — but the Lean bridge bypasses the loop entirely. HDC, CfC, Φ, and all the machinery measured by Psych-Bench are irrelevant to the 96.9% number.
- **The cascade selection is hand-coded.** `synthesize_arith_tactic` in `fol_ext_bridge.rs` writes out `first | rfl | norm_num | ring | omega | linarith | nlinarith[…] | positivity | …` for every goal. That ordering was tuned by reading individual miniF2F failures in sessions P3→P5a. Every future cascade improvement is another round of hand-tuning.
- **There is no learning.** The cascade compiler is a pure function. It doesn't accumulate. It can't notice that "goals of shape X close better with tactic ordering Y" and adapt.

Meanwhile Symthaea's `conjecture_engine` / `SymbolicRegressor` — inside the same repo — solved Kepler's third law at machine-epsilon (3.7e-15 test error on held-out data) and recovered Stefan-Boltzmann exactly (`n^4`) in 1.7 seconds of CPU. That's real *discovery*, on known-ground-truth physics sequences, using HDC-adjacent GP machinery. It's just not talking to Lean.

Phase 6 closes this gap.

## The core idea

For each miniF2F goal φ:

1. **Encode φ as cognitive input.** Convert the `FolFormulaExt` to a string representation, feed into `CognitiveLoopService::cycle(goal_text)`. Let the loop run its normal HDC→CfC→Φ pipeline.
2. **Extract a goal signature.** `CycleResult.wisdom_hv: BinaryHV` (16,384D) or `thought_vector: Vec<f32>` (32D) is the loop's compressed cognitive state after processing the goal. That's the signature.
3. **Pick a cascade variant from the signature.** Instead of one hand-coded cascade, we have K cascade variants (current: 1). A learned mapping `BinaryHV → cascade_id` picks the right variant per goal.
4. **Learn the mapping from Lake outcomes.** For each training goal, try all K cascades, record which closed Lake. The mapping is the argmax table.

Unlike the current bridge, this system *learns* from the 32 curated + ~35 auto-ingested fixtures. Future goals that look cognitively similar (high HDC cosine) pick the same cascade without manual tuning.

## Why this is honestly hard

- **What does "close in HDC" mean for math goals?** The current HDC encoder was tuned for sequence observations (Kepler orbits, Fibonacci ratios) and natural language. Math goals are neither. The similarity metric might collapse (all goals map to a blob) or spread too finely (every goal is its own singleton). Only measurement tells us.
- **Sample efficiency.** We have ~67 Lake-verified goals (32 curated + 35 auto-ingested). That's nothing for a learned classifier. Either the HDC signature is already structured enough that kNN works on 67 examples, or we need data augmentation / synthetic goals.
- **Cascade variance.** Phase 4a showed that cascade ordering matters a lot (heartbeat timeouts from hint bloat). A learned system that picks from K cascades needs each cascade to be independently stable, not just individually tuned. Current cascade has 11 alternatives in `first | …`; making K of those is nontrivial.

## Concrete plan — scoped for 3 sessions

### Session 1 (data): fingerprint the 67 goals

- Add a helper `cognitive_signature(phi) -> BinaryHV` that feeds `formula_to_lean(phi)` into `CognitiveLoopService::cycle()` and returns `wisdom_hv`.
- For each of the 67 goals, record the signature.
- Output: CSV with (goal_name, signature_hash, Lake_outcome). ~100 LOC example.
- **Measurement:** are the signatures stable across reruns (should be — HDC is deterministic)? Do Lake-accepted goals cluster by HDC cosine? Do Pattern-D goals cluster together? This is the "go / no-go" for the learned bridge.

### Session 2 (variants): write 3 cascade variants, measure head-to-head

Current cascade is `C0` (the shipped `first | rfl | … | polyrith`). Write:
- `C1` — nlinarith-first (skip the early rfl/norm_num/ring, hit nlinarith early)
- `C2` — field-first (field_simp + linarith before trying nlinarith)
- `C3` — induction-aware (for goals with `∀ n : ℕ`, emit `induction n` scaffolding)

For each of the 67 goals, record `Lake(Ci, goal) ∈ {accept, reject}`. Output: 67×4 outcome matrix. Gives us empirical evidence of whether cascade variance matters and which goals reward which cascade.

### Session 3 (bridge): learned cascade selection

- Train a kNN-over-HDC classifier on (signature → best-cascade) pairs from session 2.
- New emitter: given goal φ, compute signature, pick nearest-K cascade, emit that cascade's Lean file.
- Re-measure: does the learned emitter beat C0 on a held-out split?
- If yes: ship. If no: document why (signatures collapse? too few samples? cascades too correlated?) and close the phase.

## What this does NOT do

- No training of HDC itself. The cognitive loop stays frozen; we just query it.
- No gradient-based learning. kNN-over-signatures is the simplest-possible learned mapping. If *that* doesn't work we'll know the bottleneck is the signature, not the learner.
- No new AST features. Abs, mod, Finset, function abstraction all stay out of scope. The benchmark set is the same 67 goals.

## Honest risk: this may not help

The null hypothesis is "goal signatures don't correlate with the best cascade — kNN-selection doesn't beat C0." That result would be published negatively; it tells us that in the current regime, the cognitive loop doesn't have useful structural information about math goals. A negative result would shift Phase 7+ toward either (a) training HDC specifically on math goals (expensive), or (b) giving up on the bridge direction and investing in symbolic mechanisms (like Pattern D's RREF solver).

Both negative-result futures are acceptable. The null is *discovering the loop doesn't know math*, which is more informative than any cascade polish.

## Prior art inside Symthaea

- `CognitiveLoopService::cycle(&str)` → `CycleResult` — the entry point.
- `CycleResult.wisdom_hv: BinaryHV` — 16,384D per-cycle cognitive state. Already used for mesh broadcast.
- `CycleResult.thought_vector: Vec<f32>` — 32D projection, intended for visualization. Probably cleaner signature than raw wisdom_hv.
- `CycleResult.metadata.reasoning_context: String` (per `cycle_consciousness.rs`) — human-readable trace of what the reasoning_engine did this cycle. Useful for post-hoc analysis of "why did the loop pick this cascade?".
- `conjecture_engine::SymbolicRegressor::fit(seq, top_k)` — GP that actually does *discovery*. Not directly usable here (math goals aren't sequences), but reference for how HDC-adjacent code touches Symthaea's cognitive state.

## File-level changes expected

- New: `symthaea-lean-bridge/src/cognitive_signature.rs` — ~60 LOC, `signature_for(phi: &FolFormulaExt) -> BinaryHV`.
- New: `symthaea-lean-bridge/src/cascade_variants.rs` — ~200 LOC, 4 `emit_cascade_Cx()` functions.
- Modified: `fol_ext_bridge::render_fol_ext_file` — pick cascade from signature instead of hard-coding.
- New: `symthaea-lean-bridge/examples/fingerprint_goals.rs` — session-1 harness.
- New: `symthaea-lean-bridge/examples/cascade_tournament.rs` — session-2 harness.
- New: `docs/phase6-results/` — CSV artifacts.

## Sequencing

Session 1 output determines whether sessions 2 and 3 are worth running. If signatures already cluster meaningfully on the 67 goals (clear Lake-accepted vs Lake-rejected separation in HDC space), sessions 2+3 have a good prior. If not, document and stop.
