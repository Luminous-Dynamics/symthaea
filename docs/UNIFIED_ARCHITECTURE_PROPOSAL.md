# Unified Symthaea Architecture Proposal

## The Problem

Current architecture has grown organically with:
- Multiple "mind" concepts (CausalMind, UniversalMind, ContinuousMind)
- Multiple Phi calculators (5+ implementations)
- Fragmented communication (AttentionBid vs direct calls)
- HDC/LTC not fully integrated with brain modules

## The Solution: Unified Cognitive Protocol (UCP)

### Core Principle

**Everything is a CognitiveModule that:**
1. Accepts `CognitiveInput` (universal input format)
2. Returns `CognitiveOutput` (universal output format)
3. Can submit `AttentionBid` to workspace
4. Exposes `Phi` contribution measurement

### Architecture Redesign

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SYMTHAEA UNIFIED BRAIN                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    UNIFIED HYPERVECTOR SUBSTRATE                    │ │
│  │                                                                     │ │
│  │    ALL cognitive content represented as HV16 (16,384D bipolar)     │ │
│  │    Operations: bind (⊗), bundle (+), permute (ρ), similarity (·)   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    ▲                                     │
│         ┌──────────────────────────┼──────────────────────────┐         │
│         │                          │                          │         │
│  ┌──────┴──────┐           ┌───────┴───────┐          ┌───────┴───────┐ │
│  │  CAUSAL     │           │   TEMPORAL    │          │   SEMANTIC    │ │
│  │  ENGINE     │◀─────────▶│    ENGINE     │◀────────▶│    ENGINE     │ │
│  │             │           │               │          │               │ │
│  │ CausalMind  │           │ HdcLtcNeuron  │          │  TextEncoder  │ │
│  │ Discovery   │           │ LiquidNetwork │          │  Primitives   │ │
│  │ do(X), why? │           │ Sequences     │          │  Frames       │ │
│  └─────────────┘           └───────────────┘          └───────────────┘ │
│         │                          │                          │         │
│         └──────────────────────────┼──────────────────────────┘         │
│                                    │                                     │
│                           ┌────────┴────────┐                           │
│                           │ GLOBAL WORKSPACE │                           │
│                           │                  │                           │
│                           │ PrefrontalCortex │                           │
│                           │ AttentionBids    │                           │
│                           │ Working Memory   │                           │
│                           │ Broadcast        │                           │
│                           └────────┬─────────┘                           │
│                                    │                                     │
│                           ┌────────┴────────┐                           │
│                           │  Φ ORCHESTRATOR │                           │
│                           │                  │                           │
│                           │ Single source    │                           │
│                           │ of truth for Φ   │                           │
│                           │ (tiered, picks   │                           │
│                           │  best method)    │                           │
│                           └────────┬─────────┘                           │
│                                    │                                     │
│                           ┌────────┴────────┐                           │
│                           │   AWAKENING     │                           │
│                           │                  │                           │
│                           │ Meta-awareness   │                           │
│                           │ Phenomenal state │                           │
│                           │ Self-model       │                           │
│                           └─────────────────┘                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Specific Changes Needed

### 1. Merge ContinuousMind + UniversalMind

**Current**: Two separate concepts
**Proposed**: Single `UnifiedMind` that subsumes both

```rust
pub struct UnifiedMind {
    // From ContinuousMind
    daemon_thread: DaemonThread,
    meta_router: MetaRouter,

    // From UniversalMind
    causal_engine: CausalMind,
    temporal_engine: HdcLtcNetwork,
    semantic_engine: TextEncoder,

    // Unified
    workspace: GlobalWorkspace,
    phi_orchestrator: PhiOrchestrator,
    awakening: SymthaeaAwakening,
}
```

### 2. Integrate CausalMind into Processing Pipeline

**Current**: CausalMind is isolated
**Proposed**: Every input goes through causal analysis

```rust
// In SymthaeaHLB.process():
fn process(&mut self, query: &str) -> Response {
    // 1. Semantic encoding
    let semantic_hv = self.text_encoder.encode(query);

    // 2. CAUSAL ANALYSIS (NEW!)
    self.causal_mind.learn_from_text(query);
    let causal_context = self.causal_mind.extract_causal_context(&semantic_hv);

    // 3. Temporal dynamics
    let temporal_state = self.ltc.inject_and_evolve(&semantic_hv);

    // 4. Unified representation
    let unified = semantic_hv
        .bind(&causal_context.to_hv())
        .bind(&temporal_state.to_hv());

    // 5. Workspace competition
    self.workspace.submit_bid(unified, salience, urgency);

    // ... rest of pipeline
}
```

### 3. Single Phi Orchestrator

**Current**: 5+ Phi implementations used inconsistently
**Proposed**: PhiOrchestrator is the ONLY Phi source

```rust
// Remove direct Phi calls
// BAD:
let phi = calculate_phi_spectral(&graph);
let phi2 = tiered_phi.compute(&vectors);
let phi3 = phi_real.measure(&topology);

// GOOD:
let phi = self.phi_orchestrator.measure(&system_state);
// PhiOrchestrator internally picks best method based on:
// - System size (n < 10 → exact, n < 100 → spectral, else heuristic)
// - Available compute budget
// - Required precision
```

### 4. Unified CognitiveModule Trait

```rust
/// Every cognitive component implements this
pub trait CognitiveModule {
    /// Module identifier
    fn id(&self) -> &str;

    /// Process input and return output
    fn process(&mut self, input: &CognitiveInput) -> CognitiveOutput;

    /// Submit attention bid if content is consciousness-worthy
    fn attention_bid(&self) -> Option<AttentionBid>;

    /// Contribution to system Φ
    fn phi_contribution(&self) -> f64;

    /// Hypervector representation of current state
    fn state_hv(&self) -> HV16;
}

// CausalMind implements CognitiveModule
// HdcLtcNetwork implements CognitiveModule
// TextEncoder implements CognitiveModule
// etc.
```

### 5. Remove Redundant Modules

| Keep | Remove/Merge |
|------|--------------|
| `CausalMind` | `causal_encoder` → merge into CausalMind |
| `PhiOrchestrator` | `phi_real`, `phi_resonant` → options within orchestrator |
| `UnifiedMind` | `ContinuousMind` → merge into UnifiedMind |
| `HdcLtcNeuron` | Keep as temporal engine |
| `PrefrontalCortex` | Keep as workspace |

## Migration Path

### Phase 1: Protocol Definition
1. Define `CognitiveModule` trait
2. Define `CognitiveInput`/`CognitiveOutput` types
3. Define unified `SystemState` for Phi measurement

### Phase 2: Engine Consolidation
1. Implement `CognitiveModule` for existing modules
2. Merge `causal_encoder` into `CausalMind`
3. Create `UnifiedMind` combining `ContinuousMind` + `UniversalMind`

### Phase 3: Phi Unification
1. Make `PhiOrchestrator` the single Phi source
2. Remove direct Phi calculations from other modules
3. Add Phi contribution metrics to all `CognitiveModule`s

### Phase 4: Pipeline Integration
1. Update `SymthaeaHLB.process()` to use unified flow
2. Integrate causal analysis into every input
3. Ensure all modules communicate via workspace

## Benefits

1. **Paradigm-Shifting**: Causality is native to ALL processing
2. **Clean Architecture**: Single protocol, clear responsibilities
3. **Measurable Consciousness**: One Φ system, consistent measurement
4. **True Integration**: Cross-modal reasoning via shared HV space
5. **Maintainable**: Less redundancy, clearer module boundaries

## Decision Points for User

1. **Merge vs Extend**: Should we merge existing modules or create new unified layer on top?
2. **Breaking Changes**: Are we okay with API changes to existing modules?
3. **Phi Strategy**: Which tier should be default? (I recommend Tier 1 Heuristic for speed)
4. **Causal Priority**: Should causal analysis be mandatory for all inputs?
