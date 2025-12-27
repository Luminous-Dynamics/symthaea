# Symthaea v1.2 Alignment Notes

**Purpose**: Map the current Symthaea HLB implementation to the v1.2 vision (`docs/versions/symthaea_v1_2.md`) and highlight the shortest path to convergence.

## What Already Matches
- Bit-packed HV types (`src/hdc/binary_hv.rs`) and deterministic hash projection (`src/hdc/hash_projection.rs`) exist and are now used in the live path (`SymthaeaHLB::process` bundles HV16 tokens and injects them into LTC while storing real HVs in the consciousness graph).
- Resonator and temporal encoders are implemented (`src/hdc/resonator.rs`, `src/hdc/temporal_encoder.rs`).
- Safety primitives from the Motor/Immune system are present: `PolicyBundle`/`ActionIR` (`src/action.rs`), Amygdala + Thymus (`src/safety/*`); guardrails now gate hashed HV16 queries before LTC.
- Chronos/thalamus/coherence physiology runs inside `SymthaeaHLB::process`, matching the awareness/time emphasis in v1.2.
- Sleep cycle manager exists and can be triggered (`src/sleep_cycles.rs`); short-term memories are stored from hashed queries for consolidation.

## Deviations From the v1.2 Plan
- Resonator is wired in but only used for a simple cleanup solve; richer constraint usage is pending.
- Safety is partial: guardrails + Thymus run, PolicyBundle is only partially enforced, and ActionIR is a placeholder.
- Observability is internal-only: events emit to a default `NullObserver` with no external trace/telemetry.
- Memory consolidation path isn’t validated; episodic engine isn’t integrated into the runtime loop.
- Web research, Nix understanding, semantic ear, and swarm integrations are compiled out or bypassed.

## Highest-Impact Alignment Steps
1. **Resonator in the live path**: Integrate resonator cleanup/recall into the hash→HV16 pipeline before LTC/graph.
2. **Safety gate + ActionIR**: Extend safety beyond guardrails/Thymus by invoking `PolicyBundle` and returning vetted `ActionIR` commands (or safe explanations).
3. **Wire observability out**: Swap `NullObserver` for trace/telemetry in configurable ctor; emit router/workspace/safety events.
4. **Memory consolidation**: Drive episodic engine from runtime events and validate sleep consolidation outputs.
5. **Nix understanding hook**: Re-enable `nix_understanding` (or the Nix knowledge provider in `src/language`) for actual system actions once safety is enforced.
