# Feature Flags

Symthaea uses ~120 Cargo feature flags with **all defaults disabled**. This enables compilation from a 980 KB WASM kernel to the full holon.

## Key Feature Groups

### Consciousness & Reasoning
| Flag | Purpose |
|------|---------|
| `reasoning_engine` | 7-step MAGI reasoning cycle |
| `identity` | Self-model and narrative identity |
| `glyph_codex` | 70-glyph symbolic progression |
| `multi_agent` | Multi-agent consciousness |

### Language
| Flag | Purpose |
|------|---------|
| `ssm_language` | Full Broca pipeline (Liquid-Mamba fusion) |

### Defense & Safety
| Flag | Purpose |
|------|---------|
| `safety-agents` | NRC 4-tier SafetyAgent cascade |
| `sentinel` | 7-detector threat monitoring |

### Networking
| Flag | Purpose |
|------|---------|
| `mesh` | 3-tier radio spectrum awareness |
| `swarm` | Peer consciousness synchronization |
| `pqc-handshake` | Post-quantum handshake |

### Embodiment
| Flag | Purpose |
|------|---------|
| `flight` | Quadrotor flight control |
| `humanoid` | Bipedal locomotion |
| `cpg` | Central pattern generators |

### Database Backends
| Flag | Purpose |
|------|---------|
| `lancedb-backend` | LanceDB vector storage |

### Bundles
| Flag | Includes |
|------|----------|
| `consciousness_full` | All consciousness features |
| `sovereign-mind` | Full autonomous agent |
| `all_benchmarks` | Complete psych-bench suite |
