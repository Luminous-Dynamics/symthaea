# Enhancement #7 Phase 2 - Quickstart Guide

**Get started with Causal Program Synthesis in 5 minutes!**

---

## ⚡ Quick Start

### 1. Run the Integration Examples (Easiest)

```bash
cd symthaea-hlb
cargo run --example enhancement_7_phase2_integration
```

This runs 5 progressive examples demonstrating all capabilities.

Expected output:
```
🎉 Enhancement #7 Phase 2 Integration Examples
======================================================================

📝 Example 1: Explanation Generation
----------------------------------------------------------------------
✅ Synthesized program successfully
📖 Explanation:
Creates causal relationship exercise → health with strength 0.75...

... (continues through all 5 examples)

✅ All integration examples completed successfully!
```

---

### 2. Run the ML Fairness Demo

```bash
cargo run --example ml_fairness_causal_synthesis
```

See a real-world application removing bias from a loan approval system.

---

### 3. Write Your First Causal Program

Create `my_first_causal_program.rs`:

```rust
use symthaea::synthesis::{CausalProgramSynthesizer, SynthesisConfig, CausalSpec};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Create synthesizer
    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

    // Step 2: Define what you want causally
    let spec = CausalSpec::MakeCause {
        cause: "study_hours".to_string(),
        effect: "test_score".to_string(),
        strength: 0.8,  // Strong positive effect
    };

    // Step 3: Synthesize!
    let program = synthesizer.synthesize(&spec)?;

    // Step 4: View results
    println!("✅ Synthesized successfully!");
    println!("Achieved strength: {:.2}", program.achieved_strength);
    println!("Confidence: {:.2}", program.confidence);

    if let Some(explanation) = &program.explanation {
        println!("\nExplanation:\n{}", explanation);
    }

    Ok(())
}
```

Run it:
```bash
cargo run --bin my_first_causal_program
```

---

## 📖 Common Use Cases

### Use Case 1: Remove Unwanted Causation (Fairness)

```rust
// Remove bias from decision-making
let spec = CausalSpec::RemoveCause {
    cause: "age".to_string(),
    effect: "hiring_decision".to_string(),
};

let program = synthesizer.synthesize(&spec)?;
// Program ensures age doesn't influence hiring
```

---

### Use Case 2: Strengthen Important Relationships

```rust
// Make feature more influential
let spec = CausalSpec::Strengthen {
    cause: "customer_satisfaction".to_string(),
    effect: "retention".to_string(),
    target_strength: 0.9,
};

let program = synthesizer.synthesize(&spec)?;
// Program amplifies customer satisfaction → retention link
```

---

### Use Case 3: Create Causal Chains (Interpretability)

```rust
// Create explicit reasoning path
let spec = CausalSpec::CreatePath {
    from: "symptoms".to_string(),
    through: vec!["diagnosis".to_string(), "prognosis".to_string()],
    to: "treatment".to_string(),
};

let program = synthesizer.synthesize(&spec)?;
// Program creates interpretable decision chain
```

---

## 🚀 Add Enhancement #4 Components (Advanced)

For production use, add all Enhancement #4 components:

```rust
use symthaea::observability::{
    CausalInterventionEngine, CounterfactualEngine, ActionPlanner,
    ProbabilisticCausalGraph,
};
use symthaea::synthesis::{
    CausalProgramSynthesizer, CounterfactualVerifier,
    SynthesisConfig, VerificationConfig, CausalSpec,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create Enhancement #4 components
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

    // Synthesize
    let spec = CausalSpec::Strengthen {
        cause: "exercise".to_string(),
        effect: "health".to_string(),
        target_strength: 0.8,
    };

    let program = synthesizer.synthesize(&spec)?;

    // Verify
    let result = verifier.verify(&program);

    // Check quality
    let quality = (program.confidence + result.counterfactual_accuracy) / 2.0;

    println!("✅ Complete workflow executed!");
    println!("Synthesis confidence: {:.2}", program.confidence);
    println!("Verification accuracy: {:.2}", result.counterfactual_accuracy);
    println!("Overall quality: {:.2}", quality);

    if quality > 0.9 {
        println!("🏆 EXCELLENT quality!");
    }

    Ok(())
}
```

---

## 🧪 Testing

### Run Integration Tests

```bash
cargo test test_enhancement_7_phase2_integration
```

Expected: **14 tests passing**

### Run Benchmarks

```bash
cargo bench --bench enhancement_7_phase2_benchmarks
```

Compares Phase 1 vs Phase 2 performance.

---

## 📚 Learn More

**Detailed Examples**:
- `ENHANCEMENT_7_PHASE2_INTEGRATION_EXAMPLES.md` - 5 comprehensive examples with explanations

**API Reference**:
- `docs/ENHANCEMENT_7_PHASE2_API.md` - Complete API documentation

**Implementation Details**:
- `ENHANCEMENT_7_PHASE_2_PROGRESS.md` - Development journey and technical details

**Source Code**:
- `src/synthesis/synthesizer.rs` - Core synthesis engine
- `src/synthesis/verifier.rs` - Counterfactual verifier
- `src/observability/` - Enhancement #4 components

---

## 🎯 Next Steps

1. ✅ **Run examples** - See it in action
2. ✅ **Try your own specs** - Experiment with different causal specifications
3. ✅ **Add components** - Integrate Enhancement #4 for production
4. ✅ **Read API docs** - Understand all capabilities
5. ✅ **Build something real** - Apply to your ML fairness, safety, or interpretability needs

---

## 💡 Tips

**Start Simple**: Begin with `MakeCause` or `RemoveCause` before complex specs

**Use Explanations**: Always read `program.explanation` - it teaches causal thinking

**Verify Everything**: Always run counterfactual verification for production code

**Check Confidence**: Low confidence (<0.7) means the program might not work as expected

**Read the Spec**: Understanding `CausalSpec` variants unlocks all capabilities

---

## 🐛 Troubleshooting

**"Unsatisfiable specification"**:
- The requested causal relationship may be impossible
- Try simpler specs or check for contradictions

**Low confidence scores**:
- Normal for complex specifications
- Add Enhancement #4 components for better confidence

**Slow synthesis**:
- Complex path specs can take longer
- Consider simplifying or using ActionPlanner to find paths automatically

---

## ✨ You're Ready!

You now know enough to:
- Synthesize causal programs
- Remove bias from ML models
- Create interpretable causal chains
- Verify programs with counterfactual testing

**Happy causal programming!** 🎉

For questions or issues, see the main README or documentation.
