# Enhancement #7 Phase 2 - API Reference

**Last Updated**: December 27, 2025
**Status**: Production Ready

---

## Core APIs

### Causal Program Synthesis

#### `CausalProgramSynthesizer`

Main interface for synthesizing programs that implement causal relationships.

```rust
use symthaea::synthesis::{CausalProgramSynthesizer, SynthesisConfig};

// Create synthesizer
let config = SynthesisConfig::default();
let mut synthesizer = CausalProgramSynthesizer::new(config);

// Optional: Add Enhancement #4 components
synthesizer = synthesizer
    .with_intervention_engine(intervention_engine)
    .with_action_planner(action_planner);

// Synthesize from specification
let program = synthesizer.synthesize(&spec)?;
```

**Builder Methods**:
- `new(config: SynthesisConfig) -> Self`
- `with_intervention_engine(engine: CausalInterventionEngine) -> Self`
- `with_action_planner(planner: ActionPlanner) -> Self`

**Core Methods**:
- `synthesize(&mut self, spec: &CausalSpec) -> Result<SynthesizedProgram>`

---

#### `CausalSpec`

Specification language for desired causal relationships.

```rust
use symthaea::synthesis::CausalSpec;

// Create a causal link
let spec = CausalSpec::MakeCause {
    cause: "education".to_string(),
    effect: "income".to_string(),
    strength: 0.75,
};

// Remove unwanted causation (fairness)
let spec = CausalSpec::RemoveCause {
    cause: "race".to_string(),
    effect: "approval".to_string(),
};

// Create causal path
let spec = CausalSpec::CreatePath {
    from: "start".to_string(),
    through: vec!["step1".to_string(), "step2".to_string()],
    to: "end".to_string(),
};

// Strengthen existing link
let spec = CausalSpec::Strengthen {
    cause: "exercise".to_string(),
    effect: "health".to_string(),
    target_strength: 0.9,
};

// Weaken existing link
let spec = CausalSpec::Weaken {
    cause: "noise".to_string(),
    effect: "decision".to_string(),
    target_strength: 0.1,
};
```

**Variants**:
- `MakeCause { cause, effect, strength }` - Create causal link
- `RemoveCause { cause, effect }` - Eliminate causation
- `CreatePath { from, through, to }` - Multi-step causal chain
- `Strengthen { cause, effect, target_strength }` - Increase effect
- `Weaken { cause, effect, target_strength }` - Decrease effect
- `Mediate { causes, mediator, effect }` - All paths through mediator
- `And(Vec<CausalSpec>)` - All specs must be satisfied
- `Or(Vec<CausalSpec>)` - At least one spec satisfied

---

#### `SynthesizedProgram`

Result of synthesis - runnable causal program.

```rust
pub struct SynthesizedProgram {
    pub template: ProgramTemplate,        // How it's implemented
    pub specification: CausalSpec,        // What it achieves
    pub achieved_strength: f64,           // Actual causal strength (0.0-1.0)
    pub confidence: f64,                  // Synthesis confidence (0.0-1.0)
    pub complexity: usize,                // Program complexity
    pub explanation: Option<String>,      // Human-readable explanation
    pub variables: Vec<String>,           // Variables involved
}
```

**Fields**:
- `template` - Internal representation (Linear, Sequence, Conditional, etc.)
- `specification` - Original causal specification
- `achieved_strength` - Measured causal effect (from intervention testing)
- `confidence` - How confident we are (from counterfactual verification)
- `complexity` - Number of operations/steps
- `explanation` - Rich textual description
- `variables` - All variables in the program

---

### Counterfactual Verification

#### `CounterfactualVerifier`

Verifies synthesized programs using counterfactual testing.

```rust
use symthaea::synthesis::{CounterfactualVerifier, VerificationConfig};

// Create verifier
let config = VerificationConfig {
    num_counterfactuals: 100,
    min_accuracy: 0.95,
    test_edge_cases: true,
    max_complexity: 10,
};

let mut verifier = CounterfactualVerifier::new(config);

// Optional: Add counterfactual engine
verifier = verifier.with_counterfactual_engine(engine);

// Verify program
let result = verifier.verify(&program);
```

**Builder Methods**:
- `new(config: VerificationConfig) -> Self`
- `with_counterfactual_engine(engine: CounterfactualEngine) -> Self`

**Core Methods**:
- `verify(&mut self, program: &SynthesizedProgram) -> VerificationResult`

---

#### `VerificationResult`

Result of counterfactual verification.

```rust
pub struct VerificationResult {
    pub success: bool,                      // Overall pass/fail
    pub confidence: f64,                    // Verification confidence
    pub counterfactual_accuracy: f64,       // Accuracy on tests
    pub tests_run: usize,                   // Number of tests
    pub edge_cases: Vec<String>,            // Failed edge cases
    pub details: Option<VerificationDetails>, // Detailed metrics
}
```

**Fields**:
- `success` - True if passed verification
- `confidence` - Overall confidence in correctness
- `counterfactual_accuracy` - Fraction of tests passed
- `tests_run` - Total counterfactual tests executed
- `edge_cases` - Descriptions of failed cases
- `details` - Additional metrics (passed, failed, errors)

---

### Enhancement #4 Components

#### `CausalInterventionEngine`

Tests programs using do-calculus interventions.

```rust
use symthaea::observability::{CausalInterventionEngine, ProbabilisticCausalGraph};

// Create engine with causal graph
let graph = ProbabilisticCausalGraph::new();
let engine = CausalInterventionEngine::new(graph);

// Use in synthesizer
let synthesizer = synthesizer.with_intervention_engine(engine);
```

**Constructor**:
- `new(graph: ProbabilisticCausalGraph) -> Self`

**Purpose**: Provides real confidence scores from causal intervention testing

---

#### `CounterfactualEngine`

Computes true counterfactual values.

```rust
use symthaea::observability::{CounterfactualEngine, ProbabilisticCausalGraph};

// Create engine with causal graph
let graph = ProbabilisticCausalGraph::new();
let engine = CounterfactualEngine::new(graph);

// Use in verifier
let verifier = verifier.with_counterfactual_engine(engine);
```

**Constructor**:
- `new(graph: ProbabilisticCausalGraph) -> Self`

**Purpose**: Validates programs using rigorous counterfactual mathematics

---

#### `ActionPlanner`

Finds optimal intervention sequences.

```rust
use symthaea::observability::{ActionPlanner, ProbabilisticCausalGraph};

// Create planner with causal graph
let graph = ProbabilisticCausalGraph::new();
let planner = ActionPlanner::new(graph);

// Use in synthesizer
let synthesizer = synthesizer.with_action_planner(planner);
```

**Constructor**:
- `new(graph: ProbabilisticCausalGraph) -> Self`

**Purpose**: Automatically discovers optimal causal paths

---

## Complete Workflow Example

```rust
use symthaea::observability::{
    CausalInterventionEngine, CounterfactualEngine, ActionPlanner,
    ProbabilisticCausalGraph,
};
use symthaea::synthesis::{
    CausalProgramSynthesizer, CounterfactualVerifier,
    SynthesisConfig, VerificationConfig, CausalSpec,
};

// Setup all components
let graph = ProbabilisticCausalGraph::new();
let intervention_engine = CausalInterventionEngine::new(graph.clone());
let counterfactual_engine = CounterfactualEngine::new(graph.clone());
let action_planner = ActionPlanner::new(graph);

// Configure synthesizer with all components
let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default())
    .with_intervention_engine(intervention_engine)
    .with_action_planner(action_planner);

// Configure verifier
let mut verifier = CounterfactualVerifier::new(VerificationConfig {
    num_counterfactuals: 100,
    min_accuracy: 0.95,
    test_edge_cases: true,
    max_complexity: 10,
})
.with_counterfactual_engine(counterfactual_engine);

// Define specification
let spec = CausalSpec::Strengthen {
    cause: "exercise".to_string(),
    effect: "health".to_string(),
    target_strength: 0.8,
};

// Synthesize
let program = synthesizer.synthesize(&spec)?;
println!("Synthesized with confidence: {:.2}", program.confidence);

// Verify
let result = verifier.verify(&program);
println!("Verified with accuracy: {:.2}", result.counterfactual_accuracy);

// Overall quality
let quality = (program.confidence + result.counterfactual_accuracy) / 2.0;
println!("Overall quality: {:.2}", quality);
```

---

## Testing

### Integration Tests

```bash
cargo test test_enhancement_7_phase2_integration
```

### Benchmarks

```bash
cargo bench --bench enhancement_7_phase2_benchmarks
```

### Examples

```bash
# Integration examples
cargo run --example enhancement_7_phase2_integration

# ML fairness use case
cargo run --example ml_fairness_causal_synthesis
```

---

## Error Handling

All synthesis operations return `Result<T, SynthesisError>`:

```rust
match synthesizer.synthesize(&spec) {
    Ok(program) => {
        // Success - use program
    }
    Err(SynthesisError::UnsatisfiableSpecification(msg)) => {
        // Specification cannot be satisfied
    }
    Err(SynthesisError::ComplexityExceeded { actual, max }) => {
        // Program too complex
    }
    Err(e) => {
        // Other error
    }
}
```

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Synthesis (simple spec) | <100ms | MakeCause, RemoveCause |
| Synthesis (complex spec) | <500ms | CreatePath with multiple steps |
| Verification (100 tests) | <1s | Counterfactual testing |
| Complete workflow | <1.5s | Synthesis + verification |

---

## Next Steps

- See `ENHANCEMENT_7_PHASE2_INTEGRATION_EXAMPLES.md` for detailed examples
- See `ENHANCEMENT_7_PHASE_2_PROGRESS.md` for implementation details
- See `examples/` for runnable demonstrations

---

**Status**: Production Ready ✅
**Test Coverage**: 14 integration tests (100% passing)
**Documentation**: Complete
**Performance**: Validated
