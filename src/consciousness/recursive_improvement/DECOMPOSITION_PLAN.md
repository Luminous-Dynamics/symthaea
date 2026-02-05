# Core.rs Decomposition Plan

## Current State
- `core.rs`: 19,852 lines (too large for maintainability)
- `types.rs`: Extracted shared types (ComponentId, ImprovementType, etc.)
- `performance_monitor.rs`: Partially extracted (needs import fixes)
- `routers/`: Already modularized with 12+ router implementations

## Proposed Module Structure

### Phase A: Core Types & Monitoring (Lines 46-612)
1. **`types.rs`** (existing) - Keep ComponentId, ImprovementType, AccuracyMetric, BottleneckType
2. **`performance_monitor.rs`** (existing) - Complete extraction of:
   - PhiMeasurement, LatencyMeasurement, AccuracyMeasurement
   - Bottleneck struct
   - PerformanceStats, MonitorConfig
   - PerformanceMonitor impl

### Phase B: Architecture & Experimentation (Lines 613-2060)
3. **`architectural_graph.rs`** - ArchitecturalCausalGraph and component interaction modeling
4. **`safe_experiment.rs`** - SafeExperiment framework for sandboxed testing
5. **`improvement_generator.rs`** - ImprovementGenerator for proposing optimizations

### Phase C: Optimization Loop (Lines 2061-3881)
6. **`recursive_optimizer.rs`** - RecursiveOptimizer coordination loop
7. **`consciousness_gradient.rs`** - Consciousness Gradient Optimization (#54)

### Phase D: Advanced Features (Lines 3882-9091+)
8. **`intrinsic_motivation.rs`** - Intrinsic Motivation & Goal Formation (#55)
9. **`self_model.rs`** - Self-Modeling Consciousness (#56)
10. **`world_model.rs`** - Consciousness World Models (#57)
11. **`meta_cognitive.rs`** - Meta-Cognitive Architecture (#58)
12. **`predictive_routing.rs`** - Predictive Meta-Cognitive Routing (#59)
13. **`oscillatory_routing.rs`** - Oscillatory Phase-Locked Routing (#60)

## Dependencies to Watch
- `types.rs` is foundational - all modules depend on it
- `performance_monitor.rs` depends on types
- `recursive_optimizer.rs` orchestrates most other modules
- Router implementations in `routers/` are already well-isolated

## Extraction Strategy
1. Move types first (already done)
2. Create new module files with `mod` declarations
3. Move structs/impls one section at a time
4. Update imports in `mod.rs`
5. Run `cargo check` after each extraction
6. Update tests to use new module paths

## Estimated Effort
- Phase A: 2-3 hours
- Phase B: 4-6 hours
- Phase C: 3-4 hours
- Phase D: 6-8 hours
- Total: 15-21 hours

## Priority
Medium - The module works correctly as-is, but maintainability suffers.
Focus on consciousness API facade (Phase 2.1) first for higher architectural impact.
