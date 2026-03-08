# Symthaea HLB Architecture (Current Snapshot)

**Last Updated**: January 2026
**Scope**: Reality-check of the Rust codebase as it exists now.
**Purpose**: Describe what runs today, what is stubbed, and the current integration status.

## Overview

Symthaea HLB is a consciousness-aware system that integrates:
- **Hyperdimensional Computing (HDC)**: 16,384-bit binary vectors (HV16) for semantic encoding
- **Eight Harmonies Value System**: Ethical alignment framework with word-level bundling
- **Multi-Database Architecture**: Specialized databases for different "mental roles"
- **Sleep-Memory Consolidation**: HDC-based memory compression during sleep cycles
- **Motor Cortex Action Inference**: Consciousness-level-gated action execution

### Core Processing Flow

1. **Prefrontal/Coherence Routing**: Input processing and attention allocation
2. **Deterministic HDC Hashing**: `hash_projection` → HV16 bundling with word-level semantic alignment
3. **Safety Gates**: Guardrails + Thymus tri-state verification + Eight Harmonies evaluation
4. **LTC Processing**: Liquid Time Constant network for temporal dynamics
5. **Consciousness Graph Update**: Φ (integrated information) computation
6. **Memory Store**: Short-term → long-term consolidation via sleep cycles
7. **Action Output**: Motor Cortex with action type inference and value-guided gating

### Recent Improvements (Session 2026-01)

- **Semantic Encoder**: Word-level bundling via BLAKE3 hash for proper semantic overlap
- **Streaming Causal Observability**: Time-based event eviction and rapid sequence alerts
- **Sleep-Memory Integration**: `SleepCycleManager` ↔ `MemoryConsolidator` with HDC compression
- **Database Integration Tests**: 8 tests for UnifiedMind multi-database architecture
- **NixOS Knowledge Provider**: configuration.nix parsing for services, packages, programs
- **Motor Cortex Action Inference**: ActionType inference (Basic/Governance/Voting/Constitutional)

## Code Topology and Status

| Subsystem | Key files | Status |
|-----------|-----------|--------|
| **HDC Core** | `src/hdc/binary_hv.rs`, `hash_projection.rs`, `resonator.rs` | Active: Deterministic HV16 (16,384-bit) vectors with BLAKE3 hashing |
| **Eight Harmonies** | `src/consciousness/seven_harmonies.rs` | Active: Word-level bundling via BLAKE3 for semantic alignment |
| **Consciousness** | `src/consciousness/*.rs` | Active: Φ computation, attention mechanisms, value evaluation |
| **Memory** | `src/memory/*`, `src/sleep_cycles.rs`, `src/brain/consolidation.rs` | Active: HDC-based memory consolidation integrated with sleep cycles |
| **Databases** | `src/databases/*` | Active: UnifiedMind with mental-role architecture (Qdrant/Cozo/Lance/Duck) |
| **Motor Cortex** | `src/brain/motor_cortex.rs` | Active: Action inference (Basic/Governance/Voting/Constitutional) |
| **NixOS Knowledge** | `src/language/nix_knowledge_provider.rs` | Active: configuration.nix parsing, package search, semantic embeddings |
| **Observability** | `src/observability/streaming_causal.rs` | Active: Time-eviction, rapid sequence alerts, causal analysis |
| **Safety** | `src/safety/*`, `src/action.rs` | Active: Guardrails, Thymus tri-state, PolicyBundle |
| **Physiology** | `src/physiology/*` | Active: Coherence field, endocrine, chronos, proprioception |
| **Language** | `src/language/*` | Partial: Rich scaffolding, semantic ear disabled |
| **Web Research** | `src/web_research/*` | Scaffolded: Types/pipeline present but not wired |
| **Voice/Perception** | `src/voice`, `src/perception` | Feature-gated: Present but off by default |

## Key Architectural Components

### Multi-Database "Mental Roles" Architecture

From Revolutionary Improvement #30, the system uses specialized databases:

| Database | Mental Role | Purpose |
|----------|-------------|---------|
| Qdrant | Sensory Cortex | Ultra-fast vector similarity (<10ms) |
| CozoDB | Prefrontal Cortex | Recursive Datalog reasoning |
| LanceDB | Long-Term Memory | Multimodal life records |
| DuckDB | Epistemic Auditor | Statistical self-analysis |

### Motor Cortex Action Inference

Actions are classified by consciousness requirement:

| ActionType | Consciousness Threshold (Φ) | Examples |
|------------|----------------------------|----------|
| Constitutional | ≥ 0.6 | NixOS rebuild, boot/kernel changes |
| Voting | ≥ 0.4 | Community decisions, approvals |
| Governance | ≥ 0.3 | Service management, policy changes |
| Basic | ≥ 0.2 | File operations, queries |

### Eight Harmonies Value System

Word-level semantic alignment using BLAKE3 hashing:
- `encode_text_wordwise()`: Bundled word vectors via majority vote
- Normalized alignment scores (0.0 = random, 1.0 = perfect alignment)
- Integrated with Motor Cortex for action gating

## Known Gaps and Future Work

### Active Gaps
- **Web Research**: Pipeline scaffolded but not wired to main flow
- **Semantic Ear**: Language bridge disabled pending integration
- **Voice/Perception**: Feature-gated, requires explicit enablement

### Potential Improvements
- PolicyBundle enforcement is now strict on command env overrides, working_dir paths, and execution budgets (shell + file writes). Remaining gaps: read/list budgeting and richer env allowlists.
- Real database connections (currently mocks for testing)
- Temporal encoder integration with runtime inputs
- Hardware salience feeding into memory selection

### Action Policy Notes
- `allowed_env` is an allowlist: if empty, all env overrides are rejected.
- An allowed value of `"*"` permits any value for that key; otherwise the value must match exactly.
- `working_dir` must resolve inside the sandbox and satisfy read allowlist patterns.

## Testing Status

Test coverage includes:
- **Eight Harmonies**: 8 tests for semantic alignment
- **Streaming Causal**: 8 tests for time eviction and alerts
- **Sleep Cycles**: 5 tests for memory consolidation
- **Database Integration**: 8 tests for UnifiedMind
- **Config Parsing**: 10 tests for NixOS configuration
- **Motor Cortex**: 19 tests including 7 for action inference

Run tests with: `cargo test --lib`

## Documentation References

- `docs/versions/symthaea_v1_2.md` - Design vision
- `docs/architecture/REVOLUTIONARY_ARCHITECTURE_IMPROVEMENTS.md` - Enhancement history
- `docs/improvements/REVOLUTIONARY_IMPROVEMENT_*.md` - Individual improvements
