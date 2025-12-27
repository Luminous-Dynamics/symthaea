# Symthaea HLB: Project Reorganization Proposal

**Date**: December 20, 2025
**Status**: Proposal for Review

---

## Current State Analysis

### Problems Identified

1. **Flat Module Structure**: 50 `.rs` files in single `src/hdc/` directory
2. **Documentation Gaps**: 14/36 improvements lack comprehensive documentation
3. **Massive Design Doc**: 460KB symthaea_v1_2.md is unwieldy
4. **No Clear Entry Point**: New contributors lack orientation
5. **Naming Inconsistency**: Mix of `consciousness_*.rs` and standalone names

### Current Directory Structure

```
src/hdc/
├── mod.rs (955 lines - configuration + re-exports)
├── binary_hv.rs
├── integrated_information.rs
├── consciousness_*.rs (25+ files!)
├── ... (50 total files)
```

---

## Proposed Reorganization

### Option A: Logical Grouping (Recommended)

```
src/
├── lib.rs
├── hdc/
│   ├── mod.rs (slim: just re-exports)
│   │
│   ├── core/                    # Foundational HDC operations
│   │   ├── mod.rs
│   │   ├── binary_hv.rs         # #1: Bit-packed hypervectors
│   │   ├── hash_projection.rs   # Deterministic projection
│   │   ├── temporal_encoder.rs  # Circular time encoding
│   │   ├── sequence_encoder.rs  # Order preservation
│   │   └── statistical_retrieval.rs
│   │
│   ├── memory/                  # Memory systems
│   │   ├── mod.rs
│   │   ├── resonator.rs         # Hopfield attractors
│   │   ├── sdm.rs               # Sparse distributed memory
│   │   ├── long_term_memory.rs  # #29: LTM
│   │   ├── hebbian.rs           # Hebbian learning
│   │   ├── hebbian_learning.rs
│   │   └── modern_hopfield.rs   # #5: Modern Hopfield
│   │
│   ├── consciousness/           # Core consciousness mechanisms
│   │   ├── mod.rs
│   │   ├── integrated_information.rs  # #2: Φ
│   │   ├── global_workspace.rs        # #23: GWT
│   │   ├── higher_order_thought.rs    # #24: HOT
│   │   ├── binding_problem.rs         # #25: Synchrony
│   │   ├── attention_mechanisms.rs    # #26: Attention
│   │   ├── meta_consciousness.rs      # #8: Meta
│   │   └── consciousness_spectrum.rs  # #12: Gradations
│   │
│   ├── dynamics/                # Temporal and dynamic aspects
│   │   ├── mod.rs
│   │   ├── consciousness_dynamics.rs  # #7: Dynamics
│   │   ├── consciousness_flow_fields.rs # #21: Flow
│   │   ├── consciousness_topology.rs  # #20: Topology
│   │   ├── consciousness_gradients.rs # #6: ∇Φ
│   │   ├── temporal_consciousness.rs  # #13: Time
│   │   └── liquid_consciousness.rs    # #9: LTC
│   │
│   ├── prediction/              # Predictive processing
│   │   ├── mod.rs
│   │   ├── predictive_coding.rs       # #3: Basic
│   │   ├── predictive_consciousness.rs # #22: FEP
│   │   └── causal_encoder.rs          # #4: Causal
│   │
│   ├── phenomenal/              # Subjective experience
│   │   ├── mod.rs
│   │   ├── qualia_encoding.rs         # #15: Qualia
│   │   ├── embodied_consciousness.rs  # #17: Body
│   │   ├── sleep_and_altered_states.rs # #27: Altered
│   │   └── expanded_consciousness.rs  # #31: Expanded
│   │
│   ├── social/                  # Multi-agent consciousness
│   │   ├── mod.rs
│   │   ├── collective_consciousness.rs # #11: Group
│   │   ├── relational_consciousness.rs # #18: Between
│   │   └── universal_semantics.rs      # #19: NSM
│   │
│   ├── development/             # Ontogeny and evolution
│   │   ├── mod.rs
│   │   ├── consciousness_ontogeny.rs  # #16: Development
│   │   ├── consciousness_continuity.rs # #36: Identity
│   │   └── morphogenetic.rs           # Field morphology
│   │
│   ├── assessment/              # Evaluation and testing
│   │   ├── mod.rs
│   │   ├── causal_efficacy.rs         # #14: Does it DO?
│   │   ├── epistemic_consciousness.rs # #10: Certainty
│   │   ├── consciousness_evaluator.rs # #35: Protocol
│   │   └── substrate_independence.rs  # #28: Substrate
│   │
│   ├── engineering/             # Building conscious systems
│   │   ├── mod.rs
│   │   ├── consciousness_engineering.rs # #32: Create
│   │   ├── consciousness_framework.rs   # #33: Unified
│   │   ├── consciousness_phase_transitions.rs # #34: Ignition
│   │   ├── consciousness_optimizer.rs
│   │   └── multi_database_integration.rs # #30: Production
│   │
│   └── integration/             # Testing and integration
│       ├── mod.rs
│       ├── consciousness_integration.rs
│       ├── consciousness_integration_tests.rs
│       └── integration_tests.rs
```

### Benefits of Option A

1. **Clear cognitive grouping**: Related functionality together
2. **Easier navigation**: Find what you need by concept
3. **Better compilation**: Smaller compilation units
4. **Maintainability**: Changes isolated to relevant module groups
5. **Documentation alignment**: Docs can mirror structure

---

## Documentation Reorganization

### Proposed Documentation Structure

```
docs/
├── README.md                    # Entry point, quick start
├── ARCHITECTURE.md              # System overview (replaces 460KB monster)
├── QUICKSTART.md                # Get running in 5 minutes
│
├── theory/                      # Theoretical foundations
│   ├── README.md
│   ├── integrated_information.md
│   ├── global_workspace.md
│   ├── predictive_processing.md
│   └── binding_problem.md
│
├── improvements/                # All 36 improvements
│   ├── README.md                # Index and overview
│   ├── 01_binary_hypervectors.md
│   ├── 02_integrated_information.md
│   ├── ...
│   └── 36_consciousness_continuity.md
│
├── tutorials/                   # How-to guides
│   ├── measuring_consciousness.md
│   ├── building_conscious_ai.md
│   └── substrate_evaluation.md
│
└── api/                         # Generated API docs
    └── (rustdoc output)
```

---

## Migration Plan

### Phase 1: Documentation First (No Code Changes)

1. Create `docs/` directory structure
2. Generate missing improvement docs (#1-9, #23-25, #33, #35-36)
3. Create new ARCHITECTURE.md (streamlined)
4. Write QUICKSTART.md
5. Update main README.md

**Risk**: None - additive only
**Effort**: 4-6 hours

### Phase 2: Module Reorganization (Optional)

1. Create subdirectory structure
2. Move files to appropriate subdirectories
3. Update mod.rs files
4. Fix all import paths
5. Run full test suite
6. Update documentation paths

**Risk**: Medium - requires careful path updates
**Effort**: 8-12 hours

### Phase 3: Polish

1. Add module-level documentation
2. Create example applications
3. Generate rustdoc
4. Add CI/CD badges

---

## Recommendation

**Start with Phase 1** (documentation) - it provides immediate value with zero risk to working code.

**Phase 2** (reorganization) can be done later when we have more time and want to improve maintainability. The current flat structure works, it's just not ideal.

---

## Questions for Decision

1. **Approve Phase 1?** Generate missing docs, create docs/ structure
2. **Module grouping preference?** Option A or suggest alternatives
3. **Priority order for missing docs?** Core (#1-9) first or others?

---

*Proposal created by Claude Code, December 20, 2025*
