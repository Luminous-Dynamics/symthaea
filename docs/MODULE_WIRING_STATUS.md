# Module Wiring Status

Which `src/` modules are integrated into the cognitive loop vs structural/standalone.

## Tier 0: Core Infrastructure

| Module | LOC | Status | Notes |
|--------|-----|--------|-------|
| `errors/` | ~200 | Wired | Used crate-wide |
| `symthaea.rs` | ~2,000 | Wired | Public facade (8-phase pipeline) |

## Tier 1: Cognitive Loop (Primary)

All directly imported by `cognitive_loop/`.

| Module | LOC | Status | Notes |
|--------|-----|--------|-------|
| `cognitive_loop/` | ~47K | Core | 43 files, ~92 fields on CLS |
| `consciousness/` | ~76K | Wired | 116 files, 84 imports from cognitive_loop |
| `dynamics/` | ~8K | Wired | CfC/LTC temporal evolution |
| `memory/` | ~44 | Wired | Facade → symthaea-memory sub-crate |
| `perception/` | ~3K | Wired | Multi-modal HDC encoding |
| `brain/` | ~4K | Wired | Affect + prefrontal cortex |
| `causal/` | ~2K | Wired | Causal structure discovery |
| `exploration/` | ~9 | Wired | Pure re-export facade |
| `safety/` | ~1K | Wired | Safety gates, error handling |
| `voice/` | ~5K | Wired | TTS, vocal tract, vocoder |

## Tier 2: Feature-Gated Integrations

Compiled and wired only when feature flags are enabled.

| Module | LOC | Feature Flag | Status |
|--------|-----|-------------|--------|
| `api/` | ~1K | `api_module` | Wired | REST server |
| `gui_bridge/` | ~400 | `gui` | Wired |
| `identity/` | ~800 | `identity` | Wired | MFDI system |
| `integration/` | ~500 | `integration_module` or `nix-mind` | Wired |
| `school/` | ~2K | `school_learning` | Wired | Curriculum |
| `web_research/` | ~1K | `web_research_module` | Wired |
| `observability/` | ~500 | `observability_module` | Wired |
| `benchmarks/` | ~300 | `benchmarks` | Wired |
| `humanoid/` | ~1K | `humanoid` | Wired | Bipedal control |

## Tier 3: Connected Support Modules

Used by cognitive loop or its direct dependencies.

| Module | LOC | Status | Notes |
|--------|-----|--------|-------|
| `partnership/` | ~2K | Wired | Multi-agent coordination |
| `shell/` | ~500 | Wired | Nix error diagnosis |
| `wisdom/` | ~12 | Wired | Stub (experience aggregation) |
| `hdc/` | ~500 | Wired | HDC extensions |
| `hdc_ltc_bridge/` | ~35K | Wired | Alternative unified HDC-LTC neuron |
| `chronobiology/` | ~1K | Wired | Circadian rhythms |
| `mind/` | ~2K | Wired | Theory of Mind |

## Tier 4: Entry Points

| Module | LOC | Status | Notes |
|--------|-----|--------|-------|
| `repl/` | ~3K | Entry point | Interactive interface |
| `action/` | ~1K | Entry point | Action dispatch |

## Tier 5: Peripheral/Low-Priority

| Module | LOC | Status | Notes |
|--------|-----|--------|-------|
| `language/` | ~2K | Feature-gated | Language generation pipeline |
| `meta/` | ~1K | Low-priority | Metacognitive utilities |
| `user_state_inference/` | ~600 | Minimal | User state estimation |
| `soul/` | ~500 | Minimal | Soul module |
| `resonant_speech/` | ~400 | Minimal | Speech resonance |
| `visualization/` | ~1K | Standalone | Debug visualization |
| `swarm/` | ~2K | Standalone | Swarm intelligence |
| `mycelix/` | ~1K | Feature-gated | Holochain bridge |

## Tier 6: Alternative Backends

| Module | LOC | Status | Notes |
|--------|-----|--------|-------|
| `unified_ltc/` | ~53K | Alternative | RK4 integration LTC (vs default CfC) |

## Key Findings

- **All 41 declared modules are wired** — no truly dead modules exist
- **9 feature flags** gate optional integrations
- **consciousness/** is the most heavily coupled module (84 imports from cognitive_loop)
- **Minimal stubs** (wisdom, exploration) are legitimate facades over sub-crates
- The ~25% "structural/disconnected" estimate from earlier was incorrect — those modules
  are feature-gated, not disconnected

## Dependency Flow

```
Input → perception → chronobiology → mind
                  ↓
            cognitive_loop ← consciousness, dynamics, memory, brain, causal, safety
                  ↓
            action → voice → repl
                  ↓
            school, language, meta (optional learning)
```

---

*Generated 2026-03-03. Update when module structure changes significantly.*
