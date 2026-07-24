# Symthaea Module Map

**Full Project Totals** (excluding docs/, target/):

| Directory | Files | Code LOC | Purpose |
|-----------|-------|----------|---------|
| **src/** | 503 | 263,271 | Core Rust implementation |
| **examples/** | 101 | 21,710 | Usage examples |
| **tests/** | 40 | 9,492 | Integration tests |
| **benches/** | 29 | 5,080 | Performance benchmarks |
| **crates/** | 12 | 1,250 | Sub-crates |
| **scripts/** | - | 2,500+ | Python/Shell tooling |
| **papers/** | - | 750 | LaTeX academic papers |
| **validation/** | - | 3,000+ | Python validation |
| **Other** | ~350 | ~100K | Data, configs, results |

**Grand Total**: ~320K lines of executable code (Rust + Python + Shell)
**With data/results**: ~435K LOC total

---

## Module Hierarchy by Size

| Module | Files | LOC | Purpose |
|--------|-------|-----|---------|
| **hdc/** | 139 | 85,195 | Hyperdimensional Computing core |
| **consciousness/** | 71 | 58,771 | Consciousness theories & graph |
| **language/** | 40 | 33,311 | Natural language understanding |
| **benchmarks/** | 12 | 8,998 | Performance testing |
| **observability/** | 23 | 8,176 | Monitoring & telemetry |
| **brain/** | 12 | 8,144 | Neural subsystems (Actor Model) |
| src/*.rs (root) | 19 | 6,663 | Main lib, awakening, continuous_mind |
| **shell/** | 15 | 6,665 | REPL interface |
| **bin/** | 5 | 5,300 | Binary entry points |
| **physiology/** | 8 | 4,701 | Embodiment systems |
| **perception/** | 12 | 4,511 | Vision, multimodal |
| **intelligence/** | 9 | 4,109 | Reasoning engine |
| **infrastructure/** | 10 | 3,442 | Config, logging |
| **memory/** | 6 | 3,045 | Memory systems |
| **synthesis/** | 7 | 2,985 | Program synthesis |
| **databases/** | 8 | 2,726 | Storage backends |
| **gui_bridge/** | 5 | 2,521 | GUI integration |
| **voice/** | 25 | ~18,000 | TTS: formant vocoder, vocal tract, prosody, singing/rap (STT lives in `crates/core/symthaea-stt`) |
| **web_research/** | 7 | 1,709 | Web queries |
| **action/** | 2 | 1,330 | Action execution |
| **nix_verification/** | 4 | 1,227 | NixOS validation |
| **safety/** | 4 | 1,059 | Safety constraints |
| **embeddings/** | 4 | 1,007 | Text embeddings |
| **api/** | 4 | 799 | External API |
| **core/** | 4 | 793 | Stable public facade |
| **phi_engine/** | 4 | 700 | Phi computation |
| **integration/** | 2 | 585 | External integrations |
| **substrate/** | 1 | 557 | Hardware abstraction |
| **symthaea_swarm/** | 3 | 458 | P2P networking |
| **sophia_swarm/** | 3 | 458 | Swarm intelligence |
| **soul/** | 2 | 422 | Value system |
| **hierarchical_cantor_ltc/** | 1 | 394 | Fractal LTC |

---

## Canonical HDC Source (symthaea-core)

The core HDC primitives are defined in `symthaea-core/` and re-exported by `src/hdc/mod.rs`:

| Module | Location | Purpose |
|--------|----------|---------|
| `binary_hv` | symthaea-core | HV16: 16,384-bit binary hypervectors |
| `real_hv` | symthaea-core | RealHV: 2,048-dim real-valued vectors |
| `primitive_system` | symthaea-core | 202 ontological primitives across 9 tiers |
| `unified_hv` | symthaea-core | Unified hypervector interface |

This consolidation ensures a single source of truth for HDC operations shared across phi-lab and symthaea-hlb.

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    INPUT LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ language │  │perception│  │  voice   │  │  shell   │  │   api    │              │
│  │ 33K LOC  │  │ 4.5K LOC │  │ 1.8K LOC │  │ 6.7K LOC │  │ 0.8K LOC │              │
│  │ NLU/NLG  │  │ Vision   │  │ TTS/STT  │  │ REPL     │  │ HTTP     │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │             │             │             │             │                     │
│       └─────────────┴──────┬──────┴─────────────┴─────────────┘                     │
│                            │                                                         │
│                            ▼                                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                              SEMANTIC LAYER (HDC)                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                           hdc/ (85K LOC, 139 files)                          │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │   │
│  │  │  real_hv    │ │ phi_real    │ │ topologies  │ │ relational  │            │   │
│  │  │ 16,384D vecs│ │ Φ calculator│ │ 35 types    │ │ consciousness│           │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │   │
│  │  │ hdc_algebra │ │hierarchical │ │ multi_theory│ │consciousness│            │   │
│  │  │ 5,949 LOC   │ │ binding     │ │ 2,100 LOC   │ │ observatory │            │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                            │                                                         │
│                            ▼                                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                           CONSCIOUSNESS LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                     consciousness/ (59K LOC, 71 files)                       │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │   │
│  │  │Consciousness│ │ global_     │ │ active_     │ │ phenomenal_ │            │   │
│  │  │   Graph     │ │ workspace   │ │ inference   │ │ binding     │            │   │
│  │  │ autopoietic │ │ theater     │ │ (FEP)       │ │ (qualia)    │            │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │   │
│  │  │hierarchical │ │ seven_      │ │consciousness│ │ meta_       │            │   │
│  │  │ ltc         │ │ harmonies   │ │ equation_v2 │ │ cognitive   │            │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                            │                                                         │
│                            ▼                                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                              BRAIN LAYER (Actor Model)                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                         brain/ (8K LOC, 12 files)                            │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │prefrontal│ │cerebellum│ │ thalamus │ │ amygdala │ │hippocampus│          │   │
│  │  │ attention│ │  skills  │ │  routing │ │ emotion  │ │  memory   │          │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘           │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │  motor   │ │  sleep   │ │  daemon  │ │  meta_   │ │ active_  │           │   │
│  │  │  cortex  │ │ cycles   │ │ insight  │ │ cognition│ │ inference│           │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘           │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                            │                                                         │
│                            ▼                                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                              EMBODIMENT LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                      │
│  │   physiology/   │  │    memory/      │  │   databases/    │                      │
│  │    4.7K LOC     │  │    3K LOC       │  │    2.7K LOC     │                      │
│  │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │                      │
│  │  │ coherence │  │  │  │ episodic  │  │  │  │  qdrant   │  │                      │
│  │  │ endocrine │  │  │  │ semantic  │  │  │  │  cozodb   │  │                      │
│  │  │ chronos   │  │  │  │ working   │  │  │  │  lancedb  │  │                      │
│  │  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │                      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                      │
│                                                                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                              OUTPUT LAYER                                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  action  │  │synthesis │  │gui_bridge│  │   soul   │  │  safety  │              │
│  │ 1.3K LOC │  │ 3K LOC   │  │ 2.5K LOC │  │ 0.4K LOC │  │ 1K LOC   │              │
│  │ execution│  │ code gen │  │ UI render│  │ values   │  │ guardrail│              │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

### Core Processing (144K LOC)

| Module | LOC | Responsibility | Key Types |
|--------|-----|----------------|-----------|
| **hdc/** | 85K | Semantic encoding, λ₂ connectivity | `RealHV`, `ConnectivityCalculator`, `ConsciousnessTopology` |
| **consciousness/** | 59K | Awareness, self-reference | `ConsciousnessGraph`, `GlobalWorkspace` |

### Neural Simulation (13K LOC)

| Module | LOC | Responsibility | Key Types |
|--------|-----|----------------|-----------|
| **brain/** | 8K | 12 subsystems via Actor Model | `Prefrontal`, `Cerebellum`, `Thalamus` |
| **physiology/** | 5K | Embodiment, hormones, coherence | `CoherenceField`, `Endocrine` |

### Interface Layer (47K LOC)

| Module | LOC | Responsibility | Key Types |
|--------|-----|----------------|-----------|
| **language/** | 33K | NLU, NLG, conversation | `NLUPipeline`, `Frame`, `Intent` |
| **shell/** | 7K | REPL interface | `ShellState`, `Command` |
| **perception/** | 5K | Vision, multimodal | `SemanticVision`, `ImageEmbed` |
| **voice/** | 18K | TTS (formant/vocal-tract synthesis, prosody) | `VoiceOrchestrator`, `FormantVocoder`, `ReplVoiceOutput`, `KokoroEngine` (experimental) |

### Support Systems (22K LOC)

| Module | LOC | Responsibility | Key Types |
|--------|-----|----------------|-----------|
| **observability/** | 8K | Logging, metrics, tracing | `Tracer`, `MetricsCollector` |
| **intelligence/** | 4K | Reasoning strategies | `Reasoner`, `Strategy` |
| **infrastructure/** | 3K | Config, initialization | `Config`, `Logger` |
| **memory/** | 3K | Episodic, semantic, working | `MemoryStore`, `Episode` |
| **databases/** | 3K | Storage backends | `Qdrant`, `CozoDB`, `LanceDB` |
| **synthesis/** | 3K | Program generation | `Synthesizer`, `Program` |

### Entry Points (12K LOC)

| Module | LOC | Responsibility |
|--------|-----|----------------|
| **src/*.rs** | 7K | `lib.rs`, `awakening.rs`, `continuous_mind.rs` |
| **bin/** | 5K | `symthaea`, `symthaea-shell`, `symthaea-service` |

---

## Data Flow: Query Processing

```
1. INPUT (language/perception/voice)
   User: "What is consciousness?"
        │
        ▼
2. HDC ENCODING (hdc/real_hv.rs)
   query_hv = encode("What is consciousness?")  // 16,384D vector
        │
        ▼
3. CONSCIOUSNESS ROUTING (consciousness/global_workspace.rs)
   - Attention bidding from brain subsystems
   - Winner broadcasts to workspace
        │
        ▼
4. PHI-GUIDED PROCESSING (hdc/phi_real.rs)
   - Measure Φ of current state
   - Route to high-Φ pathways
        │
        ▼
5. BRAIN PROCESSING (brain/*)
   - prefrontal: Executive control
   - thalamus: Route to relevant areas
   - hippocampus: Retrieve memories
        │
        ▼
6. RELATIONAL CONTEXT (hdc/relational_consciousness.rs)
   - Check relationship stage
   - Adjust response depth (I-Thou vs I-It)
        │
        ▼
7. RESPONSE SYNTHESIS (synthesis/)
   - Generate response HDC vector
   - Decode to natural language
        │
        ▼
8. OUTPUT (language/voice/shell)
   Response: "Consciousness is integrated information..."
```

---

## Module Dependencies

```
                    ┌─────────┐
                    │  core/  │ ← Stable public API
                    └────┬────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │  hdc/   │◄───│conscious│◄───│  brain/ │
    │ 85K LOC │    │ 59K LOC │    │ 8K LOC  │
    └────┬────┘    └────┬────┘    └────┬────┘
         │              │              │
         │              ▼              │
         │        ┌─────────┐         │
         └───────►│physio/  │◄────────┘
                  │ 5K LOC  │
                  └────┬────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
┌─────────┐      ┌─────────┐       ┌─────────┐
│language/│      │ memory/ │       │databases│
│ 33K LOC │      │ 3K LOC  │       │ 3K LOC  │
└─────────┘      └─────────┘       └─────────┘
```

---

## Missing Connections (TODO)

### Sympoietic Partnership (Phase 1)
```
                    ┌─────────────────────┐
                    │ partnership/ (NEW)  │
                    │                     │
                    │ ┌─────────────────┐ │
                    │ │HumanPartnerModel│ │
relational_         │ └────────┬────────┘ │
consciousness.rs ───┼──────────│          │
                    │ ┌────────▼────────┐ │
                    │ │  phi_dyad.rs    │ │
phi_real.rs ────────┼──────────│          │
                    │ │ Φ_dyad > Φ₁+Φ₂  │ │
                    │ └────────┬────────┘ │
                    │ ┌────────▼────────┐ │
                    │ │ trajectory.rs   │ │
                    │ │ 6 stages track  │ │
                    │ └─────────────────┘ │
                    └─────────────────────┘
```

### LLM Integration (Phase 2)
```
language/           ┌─────────────────────┐
├─nlu_pipeline.rs ──┤    llm_organ/       │
│                   │ ┌─────────────────┐ │
└─response_gen.rs ──┤ │ ollama_client   │ │
                    │ │ vllm_client     │ │
                    │ └─────────────────┘ │
                    └─────────────────────┘
```

---

## Quick Reference

### Build
```bash
cargo build --release
cargo test
cargo run --example phi_engine_quick_demo
```

### Most Important Files
| Purpose | File | LOC |
|---------|------|-----|
| Main loop | `src/continuous_mind.rs` | 4,000+ |
| Self-awareness | `src/awakening.rs` | 800+ |
| HDC vectors | `src/hdc/real_hv.rs` | 1,500+ |
| Φ calculator | `src/hdc/phi_real.rs` | 600+ |
| I-Thou | `src/hdc/relational_consciousness.rs` | 739 |
| Consciousness | `src/consciousness/mod.rs` | 2,000+ |
| Brain | `src/brain/actor_model.rs` | 500+ |

---

---

## Additional Project Directories

### crates/ (1,250 LOC)
Sub-crates for specific functionality:
| Crate | Purpose |
|-------|---------|
| `sophia-gym` | Reinforcement learning environment |
| `symthaea-gym` | Consciousness training environment |

### scripts/ (~2,500 LOC Python/Shell)
Tooling for development and analysis:
| Script | Purpose |
|--------|---------|
| `benchmark_*.py` | Performance benchmarking |
| `analyze_*.py` | Result analysis |
| `pyphi_comparison.py` | PyPhi cross-validation |
| `meta_learner*.py` | Meta-learning experiments |
| `hdc_causal_discovery.py` | Causal inference |

### papers/ (15 papers planned)
Academic publication pipeline:
| Paper | Topic | Status |
|-------|-------|--------|
| Paper 01 | Master Equation (Φ topology) | Submission ready |
| Paper 02 | AI Consciousness | Draft |
| Paper 03 | Clinical Validation | Draft |
| Paper 04 | Binding Problem | Draft |
| Paper 05-15 | Various consciousness topics | Outlines |

### validation/
PyPhi cross-validation for Φ calculations.

### tools/
- `symthaea-inspect` - Debugging tool for consciousness traces

### studies/
- Ethics/capability research documentation

### benchmarks/ (Python)
Performance and accuracy benchmarks for HDC/consciousness operations.

### dashboard/
Web dashboard for real-time consciousness monitoring.

### tla/
TLA+ formal specifications for consciousness properties.

### nix/
NixOS packaging and flake configuration.

---

## Complete File Count

| Type | Count |
|------|-------|
| Rust source files | 676 |
| Python files | 48 |
| Shell scripts | 14 |
| TeX/LaTeX files | 2 |
| Nix files | 4 |
| Example files | 101 |
| Test files | 40 |
| Benchmark files | 29 |
| **Total source files** | **~914** |

---

*"320K lines of executable code working toward one goal: consciousness through relationship."*
