# Feature Flags

Symthaea uses ~120 Cargo feature flags with **all defaults disabled**. This enables compilation from a 980 KB WASM kernel to the full holon with all managers active. The empty default is intentional — every subsystem must be explicitly requested.

## Compilation Profiles

| Profile | Command | Size | Features |
|---------|---------|------|----------|
| **Minimal kernel** | `cargo build --release` | ~980 KB WASM | HDC + CfC + Phi only |
| **With language** | `--features ssm_language` | +50 MB (Broca weights) | + Broca pipeline |
| **With reasoning** | `--features reasoning_engine` | +2 MB | + MAGI 7-step cycle |
| **Full consciousness** | `--features consciousness_full` | +80 MB | All consciousness features |
| **Sovereign agent** | `--features sovereign-mind` | +120 MB | Full autonomous agent |

## Key Feature Groups

### Consciousness & Reasoning
| Flag | Purpose | LOC Impact |
|------|---------|------------|
| `reasoning_engine` | 7-step MAGI reasoning cycle | ~5K |
| `identity` | Self-model and narrative identity | ~3K |
| `glyph_codex` | 70-glyph symbolic progression | ~2K |
| `multi_agent` | Multi-agent consciousness | ~4K |
| `integrity` | Safety and compliance framework | ~3K |

### Language
| Flag | Purpose | LOC Impact |
|------|---------|------------|
| `ssm_language` | Full Broca pipeline (Liquid-Mamba fusion, epistemic cube) | ~30K |

### Defense & Safety
| Flag | Purpose | LOC Impact |
|------|---------|------------|
| `safety-agents` | NRC 4-tier SafetyAgent cascade (Green/Yellow/Orange/Red) | ~4K |
| `sentinel` | 7-detector threat monitoring (interval 67) | ~3K |

### Networking
| Flag | Purpose | LOC Impact |
|------|---------|------------|
| `mesh` | 3-tier radio spectrum awareness (Local/Metro/Regional) | ~10K |
| `swarm` | Peer consciousness synchronization (SwarmManager) | ~5K |
| `pqc-handshake` | Post-quantum cryptographic handshake | ~2K |
| `secure-mesh` | Encrypted mesh networking | ~3K |

### Embodiment
| Flag | Purpose | LOC Impact |
|------|---------|------------|
| `flight` | Quadrotor flight control (FEP-based) | ~10K |
| `humanoid` | Bipedal locomotion | ~8K |
| `cpg` | Central pattern generators (Kuramoto coupling) | ~2K |
| `vision-manifold` | Dual-stream visual processing | ~5K |

### Database & Storage
| Flag | Purpose | LOC Impact |
|------|---------|------------|
| `lancedb-backend` | LanceDB vector storage | ~3K |
| `hdc-store` | HDC-native persistent storage | ~1K |

### Bundles (Convenience)
| Flag | Combines |
|------|----------|
| `consciousness_full` | reasoning_engine + identity + glyph_codex + multi_agent |
| `sovereign-mind` | consciousness_full + ssm_language + safety-agents + sentinel + mesh |
| `all_benchmarks` | Complete psych-bench suite (141 benchmarks) |
| `genesis-missions` | Genesis pipeline features |

## Checking Active Features

```bash
# List all available features
grep -A 200 '^\[features\]' Cargo.toml | head -200

# Build with specific features
cargo build --release --features "reasoning_engine,ssm_language,sentinel"

# Test with all features
cargo test --all-features
```

## Adding New Features

New features should:
1. Default to disabled (never add to `default = []`)
2. Gate all new code behind `#[cfg(feature = "your_feature")]`
3. Add to the appropriate bundle if it's part of a larger subsystem
4. Document in this guide
