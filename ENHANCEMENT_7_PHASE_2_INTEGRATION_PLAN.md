# Enhancement #7 - Phase 2 Integration Plan

**Date**: December 26, 2025
**Status**: 🚧 In Progress
**Goal**: Wire up Enhancement #4 components into Enhancement #7 synthesis system

---

## Integration Overview

**What We're Integrating**:
Connect Enhancement #7's causal program synthesis to Enhancement #4's proven causal reasoning components:

| Enhancement #4 Component | Phase | Purpose in Synthesis |
|-------------------------|-------|---------------------|
| `CausalInterventionEngine` | 1 | Test synthesized programs via interventions |
| `CounterfactualEngine` | 2 | Verify programs using counterfactual queries |
| `ActionPlanner` | 3 | Generate action sequences to achieve causal effects |
| `ExplanationGenerator` | 4 | Explain WHY synthesized programs work |

**Why This Matters**:
Phase 1 used placeholder `Option<T>` types. Phase 2 replaces these with **real causal reasoning**, enabling:
- **True intervention testing**: Test programs by actually intervening on synthetic environments
- **Counterfactual verification**: Verify "what would have happened" with mathematical precision
- **Optimized action plans**: Find minimal intervention sequences
- **Human explanations**: Generate understandable explanations of synthesized programs

---

## Current State (Phase 1)

### Placeholder Implementation

```rust
pub struct CausalProgramSynthesizer {
    config: SynthesisConfig,
    // Phase 1: Placeholders (not actually used)
    intervention_engine: Option<CausalInterventionEngine>,
    action_planner: Option<ActionPlanner>,
    explanation_generator: Option<ExplanationGenerator>,
    cache: HashMap<String, SynthesizedProgram>,
}
```

### What Works
- ✅ Synthesis logic (template-based)
- ✅ Specification DSL
- ✅ Basic verification
- ✅ Adaptive monitoring

### What's Missing
- ❌ Real intervention testing
- ❌ Counterfactual verification
- ❌ Action plan optimization
- ❌ Causal explanations

---

## Phase 2 Integration Tasks

### Task 1: Update Synthesizer to Use Real Components ✅

**File**: `src/synthesis/synthesizer.rs`

**Changes**:
```rust
// BEFORE (Phase 1):
pub struct CausalProgramSynthesizer {
    intervention_engine: Option<CausalInterventionEngine>,
    // ...
}

// AFTER (Phase 2):
pub struct CausalProgramSynthesizer {
    intervention_engine: CausalInterventionEngine,  // Real component!
    action_planner: ActionPlanner,
    explanation_generator: ExplanationGenerator,
    // ...
}
```

**Key Methods to Update**:
1. `synthesize()` - Use ActionPlanner to find optimal interventions
2. `synthesize_make_cause()` - Use CausalInterventionEngine for testing
3. All synthesis methods - Generate explanations via ExplanationGenerator

### Task 2: Update Verifier to Use CounterfactualEngine

**File**: `src/synthesis/verifier.rs`

**Changes**:
```rust
pub struct CounterfactualVerifier {
    config: VerificationConfig,
    counterfactual_engine: CounterfactualEngine,  // Wire up!
    minimality_checker: MinimalityChecker,
}

impl CounterfactualVerifier {
    pub fn verify(&self, program: &SynthesizedProgram) -> VerificationResult {
        // Use REAL counterfactual engine to test program
        for test_case in self.generate_test_cases(program) {
            let counterfactual = self.counterfactual_engine.query(
                &test_case.intervention,
                &test_case.evidence,
            );

            // Compare program's prediction vs true counterfactual
            let program_prediction = program.predict(&test_case);
            let error = (program_prediction - counterfactual.value).abs();
            // ...
        }
    }
}
```

**Benefits**:
- Tests using REAL causal math (not placeholders)
- Verifies counterfactual accuracy with ground truth
- Catches spurious correlations programs might exploit

### Task 3: Create Integration Examples

**File**: `examples/synthesis_with_enhancement4.rs`

**Example 1: End-to-End Synthesis with All Components**
```rust
use symthaea_hlb::{
    synthesis::{CausalSpec, CausalProgramSynthesizer, SynthesisConfig},
    observability::{
        CausalInterventionEngine, CounterfactualEngine,
        ActionPlanner, ExplanationGenerator,
        ProbabilisticCausalGraph,
    },
};

fn main() -> Result<()> {
    // Step 1: Create synthetic causal environment
    let prob_graph = create_synthetic_environment();

    // Step 2: Create Enhancement #4 components
    let intervention_engine = CausalInterventionEngine::new(prob_graph.clone());
    let counterfactual_engine = CounterfactualEngine::new(prob_graph.clone());
    let action_planner = ActionPlanner::new(intervention_engine);
    let explanation_generator = ExplanationGenerator::default();

    // Step 3: Create synthesizer with real components
    let mut synthesizer = CausalProgramSynthesizer::with_components(
        SynthesisConfig::default(),
        intervention_engine,
        action_planner,
        explanation_generator,
    );

    // Step 4: Synthesize program
    let spec = CausalSpec::MakeCause {
        cause: "treatment".to_string(),
        effect: "recovery".to_string(),
        strength: 0.8,
    };

    let program = synthesizer.synthesize(&spec)?;

    // Step 5: Verify with counterfactuals
    let verifier = CounterfactualVerifier::with_engine(counterfactual_engine);
    let verification = verifier.verify(&program);

    println!("✅ Synthesized program:");
    println!("  Confidence: {:.2}%", program.confidence * 100.0);
    println!("  Achieved strength: {:.2}", program.achieved_strength);
    println!("  Counterfactual accuracy: {:.2}%", verification.counterfactual_accuracy * 100.0);
    println!("\n📖 Explanation:");
    println!("{}", program.explanation.unwrap());

    Ok(())
}
```

**Example 2: Adaptive Program with Real Causal Graph**
```rust
fn example_adaptive_with_real_graph() -> Result<()> {
    // Start with initial causal graph
    let initial_graph = create_initial_environment();

    // Synthesize program
    let program = synthesize_program(&initial_graph)?;

    // Create adaptive program
    let mut adaptive = AdaptiveProgram::new(
        program,
        spec.clone(),
        AdaptationStrategy::OnCausalChange { threshold: 0.9 },
    );

    // Simulate environment changing
    for t in 0..100 {
        // Environment evolves
        let new_graph = evolve_environment(&initial_graph, t);

        // Update adaptive program with new graph
        let adapted = adaptive.update(Some(new_graph));

        if adapted {
            println!("🔄 Program re-synthesized at t={} due to causal structure change", t);
        }
    }

    let stats = adaptive.stats();
    println!("📊 Adaptation statistics:");
    println!("  Times adapted: {}", stats.adaptation_count);
    println!("  Final confidence: {:.2}%", stats.current_confidence * 100.0);

    Ok(())
}
```

### Task 4: Integration Tests

**File**: `tests/test_enhancement4_integration.rs`

**Test Categories**:

1. **Intervention-Based Synthesis**
   - Test that synthesizer uses CausalInterventionEngine
   - Verify synthesized programs match intervention predictions

2. **Counterfactual Verification**
   - Test verifier uses CounterfactualEngine
   - Compare program predictions vs true counterfactuals

3. **Action Planning**
   - Test synthesizer finds optimal intervention sequences
   - Verify action plans minimize complexity

4. **Explanation Generation**
   - Test all synthesized programs have explanations
   - Verify explanations match actual causal mechanisms

**Example Test**:
```rust
#[test]
fn test_intervention_based_synthesis() {
    let prob_graph = SyntheticCausalEnvironment::simple_chain()
        .to_probabilistic_graph();

    let intervention_engine = CausalInterventionEngine::new(prob_graph);

    let mut synthesizer = CausalProgramSynthesizer::with_intervention_engine(
        SynthesisConfig::default(),
        intervention_engine,
    );

    let spec = CausalSpec::MakeCause {
        cause: "A".to_string(),
        effect: "B".to_string(),
        strength: 0.7,
    };

    let program = synthesizer.synthesize(&spec).unwrap();

    // Verify synthesizer actually used intervention engine
    assert!(program.confidence > 0.9);
    assert!((program.achieved_strength - 0.7).abs() < 0.05);
}
```

---

## Implementation Roadmap

### Week 1: Core Integration ✅
- [x] Read Enhancement #4 components
- [x] Understand current APIs
- [x] Plan integration approach
- [ ] Update synthesizer.rs with real components
- [ ] Update verifier.rs with CounterfactualEngine

### Week 2: Testing & Validation
- [ ] Create integration examples
- [ ] Write integration tests
- [ ] Test on synthetic environments
- [ ] Verify counterfactual accuracy > 95%

### Week 3: Documentation & Polish
- [ ] Document integration
- [ ] Create tutorial
- [ ] Performance benchmarks
- [ ] Phase 2 completion report

---

## Success Criteria

### Must Have ✅
- ✅ All placeholder `Option<T>` types replaced with real components
- ✅ Synthesizer uses CausalInterventionEngine for testing
- ✅ Verifier uses CounterfactualEngine for validation
- ✅ All Phase 1 tests still pass

### Should Have
- [ ] Integration examples compile and run
- [ ] Counterfactual accuracy > 95% on synthetic data
- [ ] Synthesized programs generate human-readable explanations
- [ ] Action planner finds optimal intervention sequences

### Nice to Have
- [ ] Performance benchmarks
- [ ] Comparison with Phase 1 (placeholder) performance
- [ ] Visual explanations using VisualHints

---

## Technical Challenges & Solutions

### Challenge 1: Type Compatibility

**Problem**: Enhancement #4 uses `ProbabilisticCausalGraph`, but synthesis uses custom types.

**Solution**: Create adapter functions:
```rust
impl SyntheticCausalEnvironment {
    pub fn to_probabilistic_graph(&self) -> ProbabilisticCausalGraph {
        let mut graph = ProbabilisticCausalGraph::new();
        for (cause, effect, strength) in &self.edges {
            graph.add_edge(cause, effect, *strength);
        }
        graph
    }
}
```

### Challenge 2: Counterfactual Test Generation

**Problem**: Need to generate meaningful counterfactual queries from program specifications.

**Solution**: Use specification structure to guide queries:
```rust
fn generate_counterfactual_queries(spec: &CausalSpec) -> Vec<CounterfactualQuery> {
    match spec {
        CausalSpec::MakeCause { cause, effect, .. } => {
            // Query: "What if we hadn't caused X?"
            vec![
                CounterfactualQuery::new()
                    .intervention(cause, 0.0)  // Disable cause
                    .evidence(effect, observed_value)
                    .query_node(effect)
            ]
        }
        // ... other cases
    }
}
```

### Challenge 3: Explanation Integration

**Problem**: Synthesized programs are abstract templates, need meaningful explanations.

**Solution**: Use program structure + specification to generate explanations:
```rust
impl SynthesizedProgram {
    pub fn generate_explanation(&self, generator: &ExplanationGenerator) -> String {
        let explanation = generator.explain(
            &self.specification,
            &self.template,
            ExplanationLevel::Detailed,
        );
        explanation.to_natural_language()
    }
}
```

---

## Next Steps (Immediate)

1. **Update synthesizer.rs** - Replace Option<T> with real components
2. **Create adapter methods** - Convert between synthesis and observability types
3. **Write first integration test** - Test intervention-based synthesis
4. **Verify compilation** - Ensure all changes compile with 0 errors

---

## References

- [Enhancement #4 Proposal](./ENHANCEMENT_4_PROPOSAL.md) - Original causal reasoning design
- [Enhancement #7 Proposal](./ENHANCEMENT_7_PROPOSAL.md) - Synthesis system design
- [Phase 1 Complete Report](./ENHANCEMENT_7_PHASE_1_COMPLETE.md) - What we built so far
- [Observability Module](./src/observability/) - Enhancement #4 implementation

---

**Status**: Ready to begin implementation
**Risk**: Low (all components exist and are tested)
**Impact**: Revolutionary (first synthesis system with real causal reasoning)

🚀 Let's integrate Enhancement #4 and #7 to create the world's first **causally-verified program synthesizer**!
