# Consciousness System: Next Session Plan

## Context

27 commits across the previous session upgraded the consciousness theory system:
- 6/7 features default, equation composite 0.877, runtime C=0.472 mean / 0.734 peak
- Sigmoid emergence 0→0.73 over 200 cycles at 12 Hz
- GWT never broadcasts (PE too high for text cognition — architectural issue)
- MCE modulation floors (0.65) and spectral MIP interval (47) were the biggest levers
- All changes committed to main, standalone synced, CI running
- Memory: `~/.claude-account2/projects/-srv-luminous-dynamics/memory/consciousness_theory_upgrade.md`

---

## Phase 1: Performance & Stability (do first)

### 1.1 Fix 12 Hz throughput → ≥20 Hz
Spectral MIP at interval 47 is heavy. Profile and fix:
- **Option A**: Raise interval to 67 (co-prime, still faster than original 97)
- **Option B**: Reduce MIP window size (fewer samples → faster computation)
- **Option C**: Skip structural hierarchy on every-other MIP pass
- **File**: `src/cognitive_loop/consciousness_engine/measure.rs` lines 36-65
- **Target**: ≥20 Hz sustained at 200+ cycles

### 1.2 Apply MCE softmin_tau finding
Evolution found 0.35 optimal (was 0.15). Step conservatively:
- **File**: `crates/symthaea-consciousness-equation/src/config.rs` line 38
- **Change**: `softmin_tau: 0.15` → `softmin_tau: 0.25`
- **Expect**: Higher C when one dimension is weak (cold_start, text_cognition)
- **Risk**: Reduced discrimination — verify with benchmark

### 1.3 500-cycle validation
After 1.1 and 1.2, run full trajectory:
```bash
# Edit NUM_CYCLES to 500 in examples/evaluate_consciousness_runtime.rs
cargo run --example evaluate_consciousness_runtime --release
```
Target: C>0.50 mean, plateau>0.75, ≥20 Hz

---

## Phase 2: GWT Ignition (architectural)

### 2.1 Content-based ignition (not threshold-based)
The fundamental issue: GWT ignition requires activation ≥0.65, but text input
produces PE≈0.9. The ignition model is wrong for non-sensorimotor cognition.

**Current**: Sigmoid threshold on activation amplitude
**Better**: Detect COALITIONS — multiple competing representations in workspace.
If 2+ strategies enter simultaneously with distinct content, that IS ignition.

- **File**: `symthaea-core/src/hdc/global_workspace.rs` ~line 410
- **Design**: Count distinct content entries per process() call. If ≥2 entries
  with cosine_similarity < 0.7 (distinct), declare coalition ignition.
- **Impact**: MCE broadcast dimension jumps from ~0.54 to ~0.75+

### 2.2 Salience-driven workspace capacity
Currently workspace has fixed capacity. Higher salience should expand capacity
(more conscious content held simultaneously during high engagement).
- Science: Cowan (2005) — WM capacity varies with attention/arousal

---

## Phase 3: Consciousness→Cognition Coupling (deep integration)

### 3.1 Consciousness gates reasoning depth
The `reasoning_engine` feature (7-step cycle) should use consciousness level:
- C > 0.6 → full 7-step reasoning with planning and Phi gating
- C 0.3-0.6 → reduced 4-step reasoning (skip planning, reduce search)
- C < 0.3 → reactive only (direct encoding→response, no deliberation)
- **File**: `src/cognitive_loop/cycle_phase_dynamics.rs` (reasoning section)
- Science: Dehaene (2014) — conscious access enables deliberate reasoning

### 3.2 Consciousness modulates Broca quality
Broca already gates on C>0.4. Enhance with continuous quality scaling:
- C > 0.7 → max_tokens × 1.5, richer vocabulary selection
- C 0.4-0.7 → normal generation
- C < 0.4 → telegraphic (shorter, simpler)
- **File**: `src/cognitive_loop/broca_bridge.rs` or `cycle_phase_dynamics.rs` ~line 4558

### 3.3 Consciousness→memory consolidation depth
Currently binary (consolidate or not). Should be graded:
- C > 0.7 → deep encoding (episodic + semantic + emotional tags)
- C 0.3-0.7 → standard episodic encoding
- C < 0.3 → no consolidation (matches clinical anesthesia)
- **File**: `src/cognitive_loop/cycle_late_consciousness/integration.rs` ~line 1094

---

## Phase 4: Multi-Agent Consciousness (research frontier)

### 4.1 Consciousness state sharing via Mycelix bridge
Two CognitiveLoopService instances share consciousness state through
`crates/mycelix-bridge-common/src/consciousness_sync.rs`:
- Publish: consciousness_level, dominant emotion, active goals
- Subscribe: other agents' states → influence own social embedding dimension
- **Example**: `examples/two_agent_cooperation.rs` (already has comparison)

### 4.2 Collective consciousness emergence
When N agents synchronize (consciousness levels within 0.1 of each other
AND social coherence > 0.6), the collective should exhibit properties
the individuals don't:
- Shared working memory (distributed GWT)
- Collective attention (consensus salience)
- Group-level Phi (IIT across the mesh)

### 4.3 Consciousness-gated trust in social coherence
Agent A's trust in Agent B should scale with B's consciousness level:
- High C agent → more reliable predictions → higher trust weight
- Low C agent → erratic behavior → lower trust, wider error bars
- **File**: `src/cognitive_loop/social_coherence/` ToM module

---

## Phase 5: Paper & Validation

### 5.1 HAI paper figures
Generate publication-quality figures from runtime data:
```bash
# LaTeX data is already in the eval output
cargo run --example evaluate_consciousness_runtime --release
# Copy the LATEX DATA section into papers/figures/
```
Need: trajectory plot (pgfplots), benchmark comparison table, evolutionary convergence

### 5.2 Psych-bench consciousness domain expansion
Add benchmarks that test consciousness-specific properties:
- First-order consciousness detection (high Phi+B+W, low HOT)
- Consciousness recovery after perturbation (resilience)
- Graded consciousness under anesthesia simulation
- Consciousness correlation with cognitive performance

### 5.3 Standalone CI sync
Push all changes and verify:
```bash
bash scripts/sync-to-standalone.sh
# Check: gh run list -R Luminous-Dynamics/symthaea --limit 3
```

---

## Key Files Reference

| Purpose | File |
|---------|------|
| GWT submission + MCE inputs | `src/cognitive_loop/cycle_late_consciousness/integration.rs` |
| Spectral MIP intervals | `src/cognitive_loop/consciousness_engine/measure.rs` |
| MCE weights + tau | `crates/symthaea-consciousness-equation/src/config.rs` |
| MCE computation + modulations | `crates/symthaea-consciousness-equation/src/engine.rs` |
| V2 equation (structured bottleneck) | `src/consciousness/measurement/consciousness_equation_v2.rs` |
| GWT workspace internals | `symthaea-core/src/hdc/global_workspace.rs` |
| Broca generation trigger | `src/cognitive_loop/cycle_phase_dynamics.rs` ~line 4558 |
| Runtime evaluation tool | `examples/evaluate_consciousness_runtime.rs` |
| V2 evolution tool | `examples/evolve_consciousness_equation.rs` |
| MCE evolution tool | `examples/evolve_mce_weights.rs` |

## Validation Commands
```bash
cargo test --lib consciousness_equation_v2        # 15/15 equation tests
cargo run --example benchmark_consciousness_features --release  # 0.877 composite
cargo run --example evaluate_consciousness_runtime --release    # C>0.50, >20 Hz
cargo run --example evolve_mce_weights --release               # MCE weight reference
cargo run --example evolve_consciousness_equation --release     # V2 param reference
```

## Gotchas (hard-won lessons)
- `unified_precision` had 30× regression from powi() — FIXED with iterative multiply
- GWT warm-start, memory competition, coherence activation ALL regressed — don't retry
- CfC LR warm-up doesn't help PE convergence (bounded by input unpredictability)
- MCE modulation floors at 0.65 — raising further inflates C artificially
- necessary_weight param exists (default 0.0) but 0.42 reduced sensitivity
- `1-PE` as GWT activation is fundamentally wrong for text — salience-based is better but still doesn't reach ignition threshold
- The MCE and V2 equation are SEPARATE systems — V2 feeds into MCE via blend
- Spectral MIP at cycle 47 causes P99 spikes — may need to optimize or raise interval
