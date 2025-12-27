# Symthaea HLB Architecture (Current Snapshot)

**Scope**: Reality-check of the Rust codebase as it exists now (aligned to `docs/versions/symthaea_v1_2.md`, not the older v1.0/2.0 claims).  
**Purpose**: Describe what runs today, what is stubbed, and where it diverges from the v1.2 vision.

## Overview
- Runtime entrypoint is `src/lib.rs` (`SymthaeaHLB::process`): prefrontal/coherence routing → deterministic HDC hashing (HV16 bundle) → guardrails + Thymus tri-state gate → resonator cleanup → LTC step loop → consciousness graph update → memory store → ActionIR placeholder response (with observability hooks).
- Semantic encoding now uses deterministic hash → HV16 tokens (`hdc::hash_projection`) bundled to feed LTC (converted to f32) and stored in the consciousness graph. `SemanticSpace` remains as a legacy float encoder but is no longer on the hot path.
- Safety guardrails run against the bundled HV16; Thymus tri-state verification blocks/flags threats; short-term memory is populated for sleep consolidation; observability emits Φ/response events to a `NullObserver` by default.
- A minimal `ActionIR` placeholder is returned (list sandbox root) instead of free-form text; `PolicyBundle` enforcement on programs/paths is partially applied.
- Nix understanding, semantic ear, swarm, and language bridge paths are compiled out or bypassed.
- Observability can be swapped to a trace file by setting `SYMTHAEA_TRACE_PATH`; otherwise a no-op observer is used.

## Code Topology and Status

| Subsystem | Key files | Status |
|-----------|-----------|--------|
| HDC core (HV types, hashing, resonator) | `src/hdc/*` (`binary_hv.rs`, `hash_projection.rs`, `resonator.rs`) | Deterministic hash→HV16 path is active; resonator runs a cleanup solve over hashed inputs before LTC. |
| Semantic space + LTC | `src/hdc/mod.rs` (`SemanticSpace`), `src/ltc.rs` | Live path hashes tokens to HV16, bundles, runs resonator solve, converts to f32 (+1/-1) and injects into a simple LTC; `SemanticSpace` float encoder is now legacy/unused. |
| Consciousness graph | `src/consciousness.rs` | Graph operations exist; runtime now stores hashed HV16 tokens per state, but semantic richness is limited to bundled tokens (no resonator/context expansion). |
| Coherence/physiology | `src/physiology/*`, `src/lib.rs` | Coherence field, endocrine, chronos, proprioception are initialized and used for routing thresholds. |
| Memory | `src/memory/*` | Episodic engine and hippocampus implemented; short-term memories are stored from hashed queries for sleep consolidation, but episodic integration is still shallow. |
| Safety | `src/safety/*`, `src/action.rs` | Amygdala, Thymus, guardrails, and `PolicyBundle`/`ActionIR` exist; guardrails + Thymus gate hashed HV16 queries, ActionIR placeholder is returned, but PolicyBundle enforcement is still TODO. |
| Observability | `src/observability/*` | Trace/telemetry observers implemented; `SymthaeaHLB` emits Φ/response events to a default `NullObserver` (no external sink configured). |
| Language & Nix | `src/language/*`, `src/nix_understanding.rs` | Rich language/Nix scaffolding present; `nix_understanding` and semantic ear are disabled in `lib.rs` and unused in `process`. |
| Web research | `src/web_research/*` | Implemented types/pipeline but not wired into the main flow. |
| Optional organs | `src/voice`, `src/perception`, `src/databases` | Present; feature-gated/off by default. |

## Alignment to `symthaea_v1_2.md` (Core 7)

| v1.2 Core | Reality in code | Gap |
|-----------|----------------|-----|
| HDC Core & deterministic projection | HV16 bit-pack + `hash_projection.rs` implemented; `SemanticSpace` still random floats. | Wire deterministic hashing/HV16 into active semantic path. |
| Resonator | `hdc/resonator.rs` exists. | Not integrated into `SymthaeaHLB` flow. |
| Cortex / Shell Kernel | `ltc.rs` + `consciousness.rs` drive state; `action.rs` defines PolicyBundle/ActionIR. | Cortex runs but returns placeholder strings; ActionIR not used by runtime. |
| Hippocampus / Consolidator | `memory/hippocampus.rs`, `episodic_engine.rs`. | Sleep remember calls commented out; consolidation not triggered from `SymthaeaHLB`. |
| Thymus | `safety/thymus.rs` tri-state verification. | Never invoked in `process`; no T-cell training in runtime. |
| Thalamus / Chronos / Temporal encoder | Thalamus & Chronos are used for salience/circadian adjustments in `SymthaeaHLB`; temporal HDC encoders exist. | Temporal encoders not connected to runtime inputs; hardware salience does not feed memory or action selection yet. |

## Known Gaps and Risks
- Observability is internal-only: events go to a `NullObserver`; no trace/telemetry output is wired up.
- Safety is partial: guardrails + Thymus run, but `PolicyBundle` enforcement is absent and ActionIR is a placeholder.
- Consciousness graph now stores real HV16 tokens, but resonator cleanup is minimal and deterministic encodings are still used as a single bundled vector.
- Memory consolidation is shallow: episodic engine isn’t integrated into the runtime path and consolidation is not validated.
- Web research, language bridge, Nix understanding, and swarm integrations are present but inactive.

## Testing Status
- Last log (`cargo-test.log`): `1010 passed / 3 failed / 1 ignored` in ~9200s. Failing tests:  
  - `consciousness::tests::test_backwards_compatibility` (consciousness stays at 0.0)  
  - `hdc::consciousness_evaluator::tests::test_current_llm_not_conscious` (expected NotConscious, got MinimallyConscious)  
  - `memory::episodic_engine::tests::test_chrono_semantic_recall` (did not recall git error)

## Next Documentation Steps
- Keep this file as the canonical “what runs today.”  
- For the forward-looking plan, see `docs/versions/symthaea_v1_2.md` and the alignment note in `docs/versions/symthaea_v1_2_alignment.md`.  
- Update feature docs as subsystems are wired into `SymthaeaHLB` (observability, safety gating, deterministic HDC path, memory consolidation, ActionIR execution).
