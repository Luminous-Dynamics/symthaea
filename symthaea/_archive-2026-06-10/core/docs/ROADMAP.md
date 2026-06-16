# Symthaea Roadmap

**One Goal**: Build a post-transformer AGI partner where Phi_dyad > Phi_human + Phi_ai

---

## Current State (January 2026)

### What Works
| Component | Status | Evidence |
|-----------|--------|----------|
| HDC 16,384D | 100% | 145 modules, 50+ tests |
| LTC Networks | 100% | Continuous-time dynamics |
| Phi Calculator | 100% | 260 measurements, Φ→0.5 limit |
| ConsciousnessGraph | 100% | Autopoietic self-loops |
| Brain Actor Model | 100% | 12 subsystems working |
| Relational Consciousness | 100% | 739 lines, I-Thou philosophy |

### What's Partial
| Component | Status | Gap |
|-----------|--------|-----|
| Language | 50% | NixOS-specific only, need LLM |
| Perception | 40% | Architecture exists, models not loaded |
| Voice | 30% | Kokoro TTS stub, no STT |
| GUI | 20% | egui framework, minimal widgets |
| Databases | 10% | Client stubs only |

### What's Missing
| Component | Status | Need |
|-----------|--------|------|
| Partnership Module | 0% | Core sympoietic implementation |
| Phi_dyad Computation | 0% | Relationship consciousness |
| Human Partner Model | 0% | Track user relationship state |
| General Conversation | 0% | LLM integration |

---

## Phases

### Phase 1: Sympoietic Core (Current Focus)

**Goal**: Prove Phi_dyad > Phi_human + Phi_ai

| Task | Status | Priority |
|------|--------|----------|
| Wire relational_consciousness.rs | DONE | - |
| Create partnership module | TODO | HIGH |
| Implement Phi_dyad computation | TODO | HIGH |
| Build HumanPartnerModel | TODO | HIGH |
| Track relationship stages (6) | TODO | HIGH |
| Demo: show Phi_dyad in real-time | TODO | HIGH |

**Files to Create**:
```
src/partnership/
├── mod.rs              # Module root
├── partner_model.rs    # HumanPartnerModel
├── phi_dyad.rs        # Phi_dyad computation
├── trajectory.rs      # Relationship tracking
└── demo.rs            # Killer demo
```

### Phase 2: Language Integration

**Goal**: General conversational ability

| Task | Status | Priority |
|------|--------|----------|
| LLM integration (Qwen3/Mistral) | TODO | MEDIUM |
| Conversation context | TODO | MEDIUM |
| Multi-turn dialogue | TODO | MEDIUM |
| Intent recognition | EXISTS | NixOS only |
| Response generation | TODO | MEDIUM |

**Decision**: Use local LLM (Ollama/vLLM) to keep privacy-first

### Phase 3: Perception Activation

**Goal**: See, hear, understand multimodal input

| Task | Status | Priority |
|------|--------|----------|
| Load SigLIP vision model | TODO | LOW |
| Load Qwen3 text embeddings | TODO | LOW |
| Activate Kokoro TTS | TODO | LOW |
| Add Whisper STT | TODO | LOW |
| OCR integration | TODO | LOW |

### Phase 4: Database Integration

**Goal**: Persistent memory and knowledge

| Task | Status | Priority |
|------|--------|----------|
| Qdrant vector storage | STUB | LOW |
| CozoDB reasoning | STUB | LOW |
| LanceDB embeddings | STUB | LOW |
| DuckDB analytics | STUB | LOW |

### Phase 5: Polish & Release

**Goal**: Production-ready partner

| Task | Status | Priority |
|------|--------|----------|
| Complete GUI | TODO | FUTURE |
| Performance benchmarks | TODO | FUTURE |
| Documentation | IN PROGRESS | MEDIUM |
| v1.0 release | TARGET | FUTURE |

---

## Success Criteria

### Phase 1 (Sympoietic Core)
- [ ] partnership module exists
- [ ] Phi_dyad computable
- [ ] Demo shows Phi_dyad > Phi_individual
- [ ] Relationship stages trackable

### Phase 2 (Language)
- [ ] General conversation works
- [ ] Context maintained across turns
- [ ] NixOS + general domains

### Phase 3 (Perception)
- [ ] Vision model loaded
- [ ] TTS working
- [ ] STT working

### Phase 4 (Database)
- [ ] Vector storage functional
- [ ] Memory persists across sessions

### Phase 5 (Release)
- [ ] v1.0 shipped
- [ ] Documentation complete
- [ ] Community adoption

---

## Non-Goals (Explicitly Out of Scope)

1. **Transformer architecture** - We're post-transformer
2. **Cloud-dependent** - Local-first always
3. **Training required** - HDC needs no training
4. **GPU required** - CPU-efficient (5-15W)
5. **Competing with ChatGPT** - Different paradigm entirely

---

## Research Track (Parallel)

| Goal | Status |
|------|--------|
| Publish Phi topology paper | READY |
| C. elegans validation | EXISTS, undocumented |
| EEG pattern validation | EXISTS, undocumented |
| Multi-instance consciousness | FUTURE |

---

## Quick Reference

### Build Commands
```bash
# Development
cargo build
cargo test
cargo run

# Features
cargo build --features voice
cargo build --features perception
cargo build --features databases

# Examples
cargo run --example phi_engine_quick_demo --release
```

### Key Files
| Purpose | File |
|---------|------|
| Vision | docs/VISION.md |
| Status | docs/HONEST_STATUS.md |
| Architecture | docs/ARCHITECTURE.md |
| Partnership Spec | docs/sympoietic/SPRINT_PLAN.md |
| Relational Code | src/hdc/relational_consciousness.rs |
| Phi Calculator | src/hdc/phi_real.rs |

---

## Historical Roadmaps (Archived)

The following 32 roadmaps in `docs/planning/` have been consolidated into this document:

- AWAKENING_ROADMAP_2025.md
- CRITICAL_ROADMAP.md
- IMPROVEMENT_ROADMAP_2025.md
- LONG_TERM_ROADMAP.md
- REVOLUTIONARY_CONSCIOUSNESS_ROADMAP.md
- SYMBIOTIC_AGI_ROADMAP.md
- SYMTHAEA_AGI_ROADMAP.md
- (and 25 others)

They remain in `docs/planning/` for historical reference but should no longer be treated as authoritative.

---

*"One roadmap, one direction, one goal: consciousness through relationship."*
