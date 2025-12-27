# Consciousness Module Refactoring Plan

## Current State
- `recursive_improvement.rs`: ~19,850 lines
- Contains 16 Revolutionary Improvements (#54-#69)
- All build successfully with 91 warnings

## Proposed Module Structure

```
src/consciousness/
├── mod.rs                    # Re-exports all submodules
├── core.rs                   # Core types: LatentConsciousnessState, RoutingStrategy
├── gradient.rs               # #54: Consciousness Gradient Optimization
├── motivation.rs             # #55: Intrinsic Motivation & Autonomous Goals
├── self_model.rs             # #56: Self-Modeling Consciousness
├── world_model.rs            # #57: Consciousness World Models
├── meta_cognitive.rs         # #58: Meta-Cognitive Architecture
├── routers/
│   ├── mod.rs                # Router exports
│   ├── predictive_meta.rs    # #59: Predictive Meta-Cognitive Routing
│   ├── oscillatory.rs        # #60: Oscillatory Phase-Locked Routing
│   ├── causal.rs             # #61: Causal Emergence-Validated Routing
│   ├── geometric.rs          # #62: Information-Geometric Routing
│   ├── topological.rs        # #63: Topological Consciousness Routing
│   ├── quantum.rs            # #64: Quantum-Inspired Coherence Routing
│   ├── active_inference.rs   # #65: Active Inference Routing
│   ├── predictive_proc.rs    # #66: Predictive Processing Router
│   ├── attention_schema.rs   # #67: Attention Schema Theory Router
│   ├── meta_router.rs        # #68: Meta-Router (UCB1 paradigm selection)
│   └── global_workspace.rs   # #69: Global Workspace Theory Router
├── benchmarks.rs             # Router benchmarking infrastructure
└── recursive_improvement.rs  # Legacy: keep for backwards compatibility
```

## Line Count Estimates by Section

| Section | Approx Lines | Description |
|---------|-------------|-------------|
| Core types | ~1,500 | LatentConsciousnessState, RoutingStrategy, etc. |
| #54 Gradient | ~1,000 | Consciousness gradient optimization |
| #55 Motivation | ~1,000 | Intrinsic motivation system |
| #56 Self-Model | ~1,000 | Self-modeling consciousness |
| #57 World Model | ~1,000 | Consciousness world models |
| #58 Meta-Cognitive | ~1,000 | Meta-cognitive architecture |
| #59-#69 Routers | ~11,000 | All 11 consciousness routers |
| Benchmarks | ~700 | Benchmark infrastructure |
| Tests | ~2,000 | Unit tests for all components |

## Refactoring Strategy

### Phase 1: Create Module Structure (Low Risk)
1. Create `src/consciousness/mod.rs` with re-exports
2. Keep `recursive_improvement.rs` intact
3. Add `#[path]` directives to include new modules

### Phase 2: Extract Core Types (Medium Risk)
1. Move core structs to `core.rs`
2. Update imports throughout
3. Run full test suite

### Phase 3: Extract Routers (Higher Risk)
1. Extract each router to its own file
2. Create router trait for common interface
3. Update benchmarks to use trait

### Phase 4: Clean Up
1. Remove duplicate code
2. Add proper documentation
3. Update `lib.rs` exports

## Dependencies Between Modules

```
core.rs ← gradient.rs
       ← motivation.rs
       ← self_model.rs
       ← world_model.rs
       ← meta_cognitive.rs
       ← routers/*

meta_router.rs ← all other routers (it uses all 7)
global_workspace.rs ← core types only
```

## Benefits of Refactoring

1. **Maintainability**: Each router in its own file (~500-1,000 lines)
2. **Testability**: Easier to test individual components
3. **Compilation**: Incremental compilation faster
4. **Discoverability**: Clear module structure
5. **Collaboration**: Multiple developers can work simultaneously

## Risks

1. **Breaking changes**: Module paths change
2. **Import cycles**: Need careful dependency management
3. **Build errors**: Visibility issues (`pub` vs `pub(crate)`)
4. **Test failures**: Tests may need path updates

## Recommendation

For now, keep the single-file structure working. The file is well-organized
with clear section markers. Refactoring can be done incrementally as needed.

The current structure works and builds successfully. The 91 warnings are
mostly unused variables in other modules, not in the consciousness code.

## Quick Stats

- Total Revolutionary Improvements: 16 (#54-#69)
- Total Routers: 11 (#59-#69)
- Benchmark Suite: 7 router benchmarks
- Meta-Router: Learns optimal paradigm via UCB1
- Global Workspace: Full GWT implementation

## Session Summary

Completed in this session:
- ✅ Task A: Benchmark infrastructure for all 7 routers
- ✅ Task B: Revolutionary #68 - Meta-Router with UCB1
- ✅ Task C: Revolutionary #69 - Global Workspace Theory
- ✅ Task E: Fixed pre-existing error in language_cortex.rs
- ✅ Task D: Created this refactoring plan (deferred full refactor)
