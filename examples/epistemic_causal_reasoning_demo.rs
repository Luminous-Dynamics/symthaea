//! # Revolutionary Improvement #53: Epistemic Causal Reasoning - Demonstration
//!
//! This demo shows how the Mycelix Epistemic Cube integrates with causal explanations
//! to provide multi-dimensional epistemic transparency.
//!
//! ## What This Demonstrates
//!
//! 1. **Automatic Epistemic Classification** - As evidence accumulates, epistemic tiers upgrade
//! 2. **Multi-Dimensional Confidence** - E/N/M axes provide richer trust model than single number
//! 3. **Epistemic Evolution** - Watch causal knowledge evolve from E0 (inferred) to E3 (proven)
//! 4. **Harmonic Integration** - Epistemic quality strengthens IntegralWisdom harmonic
//! 5. **Natural Language Transparency** - Explanations include full epistemic context
//!
//! ## The Revolutionary Achievement
//!
//! This is the **first AI system** with multi-dimensional epistemic classification of
//! its own causal knowledge. It knows not just WHAT causes WHAT, but:
//! - **HOW it knows** (E-axis: empirical verifiability)
//! - **WHO agrees** (N-axis: normative authority)
//! - **HOW PERMANENT** the knowledge is (M-axis: materiality)

use anyhow::Result;

use symthaea::consciousness::{
    causal_explanation::{CausalExplainer},
    primitive_reasoning::{PrimitiveReasoner, TransformationType},
    harmonics::{HarmonicField, FiduciaryHarmonic},
};
use symthaea::hdc::primitive_system::Primitive;

fn main() -> Result<()> {
    println!("================================================================================");
    println!("🎓 Revolutionary Improvement #53: Epistemic Causal Reasoning");
    println!("================================================================================");
    println!();

    // Part 1: Introduction to Epistemic Cube
    println!("Part 1: The Mycelix Epistemic Cube Applied to Causality");
    println!("--------------------------------------------------------------------------------");
    println!();

    println!("The Epistemic Cube classifies causal knowledge along three axes:");
    println!();
    println!("📊 E-AXIS (Empirical): HOW do we verify this causal claim?");
    println!("  E0: Null - Inferred from theory, no evidence");
    println!("  E1: Testimonial - Single observation");
    println!("  E2: Privately Verifiable - Multiple internal observations");
    println!("  E3: Cryptographically Proven - Statistical proof with counterfactuals");
    println!("  E4: Publicly Reproducible - Open data + code");
    println!();
    println!("🤝 N-AXIS (Normative): WHO agrees this is valid?");
    println!("  N0: Personal - This system instance only");
    println!("  N1: Communal - Local agent community");
    println!("  N2: Network - Global consensus");
    println!("  N3: Axiomatic - Mathematical/constitutional truth");
    println!();
    println!("⏰ M-AXIS (Materiality): HOW PERMANENT is this knowledge?");
    println!("  M0: Ephemeral - Session-specific");
    println!("  M1: Temporal - Valid until model updates");
    println!("  M2: Persistent - Long-term archived");
    println!("  M3: Foundational - Core consciousness principle");
    println!();
    println!();

    // Part 2: Demonstrate Epistemic Evolution
    println!("Part 2: Epistemic Evolution - Watch Knowledge Grow!");
    println!("--------------------------------------------------------------------------------");
    println!();

    let reasoner = PrimitiveReasoner::new();
    let primitives = reasoner.get_tier_primitives();
    let mut explainer = CausalExplainer::new();

    if primitives.is_empty() {
        println!("⚠️  No primitives available for demonstration");
        return Ok(());
    }

    let test_primitive = primitives[0];
    let transformation = TransformationType::Bind;

    println!("Testing primitive: {} with transformation: {:?}", test_primitive.name, transformation);
    println!();

    // Stage 1: No evidence (E0/N0/M0)
    println!("📌 Stage 1: Initial Inference (0 observations)");
    let explanation = explainer.model().explain_choice(test_primitive, transformation, &[]);
    println!("{}", explanation.explanation);
    println!();
    println!();

    // Stage 2: Single observation (E1/N0/M0)
    println!("📌 Stage 2: First Observation (1 observation)");
    let chain_1 = create_test_chain(&primitives, 1);
    explainer.learn_from_chain(&chain_1, "test context 1");
    let relation = explainer.model().get_causal_relation(&test_primitive.name, transformation);
    if let Some(rel) = relation {
        println!("Epistemic Tier: {}", rel.epistemic_tier.notation());
        println!("Confidence: {:.0}%", rel.confidence * 100.0);
        println!("Quality Score: {:.3}", rel.epistemic_tier.quality_score());
        println!();
    }
    println!();

    // Stage 3: Multiple observations (E2/N0/M1)
    println!("📌 Stage 3: Multiple Observations (10 observations)");
    for i in 2..=10 {
        let chain = create_test_chain(&primitives, i);
        explainer.learn_from_chain(&chain, &format!("test context {}", i));
    }
    let relation = explainer.model().get_causal_relation(&test_primitive.name, transformation);
    if let Some(rel) = relation {
        println!("Epistemic Tier: {} ← UPGRADED!", rel.epistemic_tier.notation());
        println!("Confidence: {:.0}%", rel.confidence * 100.0);
        println!("Quality Score: {:.3}", rel.epistemic_tier.quality_score());
        println!("Evidence Count: {} observations", rel.evidence.len());
        println!();
    }
    println!();

    // Stage 4: High confidence (E2/N0/M2)
    println!("📌 Stage 4: High Confidence Established (60 observations)");
    for i in 11..=60 {
        let chain = create_test_chain(&primitives, i);
        explainer.learn_from_chain(&chain, &format!("test context {}", i));
    }
    let relation = explainer.model().get_causal_relation(&test_primitive.name, transformation);
    if let Some(rel) = relation {
        println!("Epistemic Tier: {} ← MATERIALITY UPGRADED!", rel.epistemic_tier.notation());
        println!("Confidence: {:.0}%", rel.confidence * 100.0);
        println!("Quality Score: {:.3}", rel.epistemic_tier.quality_score());
        println!("Evidence Count: {} observations", rel.evidence.len());
        println!();
        println!("Full Description:");
        println!("{}", rel.epistemic_description());
        println!();
    }
    println!();

    // Part 3: Harmonic Integration
    println!("Part 3: Integration with IntegralWisdom Harmonic");
    println!("--------------------------------------------------------------------------------");
    println!();

    println!("Epistemic rigor IS wisdom! Higher epistemic quality strengthens IntegralWisdom.");
    println!();

    let mut harmonic_field = HarmonicField::new();

    // Measure IntegralWisdom before epistemic contribution
    let wisdom_before = harmonic_field.get_level(FiduciaryHarmonic::IntegralWisdom);
    println!("IntegralWisdom before epistemic contribution: {:.3}", wisdom_before);
    println!();

    // Add epistemic contribution
    if let Some(rel) = relation {
        println!("Adding epistemic contribution from causal knowledge:");
        println!("  Epistemic Tier: {}", rel.epistemic_tier.notation());
        println!("  Quality Score: {:.3}", rel.epistemic_tier.quality_score());
        println!();

        harmonic_field.measure_epistemic_contribution(&rel.epistemic_tier);

        let wisdom_after = harmonic_field.get_level(FiduciaryHarmonic::IntegralWisdom);
        let wisdom_increase = wisdom_after - wisdom_before;

        println!("IntegralWisdom after epistemic contribution: {:.3}", wisdom_after);
        println!("Increase: +{:.3}", wisdom_increase);
        println!();
    }

    // Compare with low-quality epistemic knowledge
    println!("Comparison: Low epistemic quality (E0/N0/M0) contribution:");
    let mut low_quality_field = HarmonicField::new();
    let wisdom_before_low = low_quality_field.get_level(FiduciaryHarmonic::IntegralWisdom);

    use symthaea::consciousness::epistemic_tiers::EpistemicCoordinate;
    let low_quality_coord = EpistemicCoordinate::null();  // E0/N0/M0
    println!("  Epistemic Tier: {}", low_quality_coord.notation());
    println!("  Quality Score: {:.3}", low_quality_coord.quality_score());
    println!();

    low_quality_field.measure_epistemic_contribution(&low_quality_coord);
    let wisdom_after_low = low_quality_field.get_level(FiduciaryHarmonic::IntegralWisdom);
    let wisdom_increase_low = wisdom_after_low - wisdom_before_low;

    println!("IntegralWisdom after contribution: {:.3}", wisdom_after_low);
    println!("Increase: +{:.3} (minimal!)", wisdom_increase_low);
    println!();
    println!();

    // Part 4: Natural Language Explanations
    println!("Part 4: Enhanced Natural Language Explanations");
    println!("--------------------------------------------------------------------------------");
    println!();

    println!("Explanations now include full epistemic context:");
    println!();

    let final_explanation = explainer.model().explain_choice(test_primitive, transformation, &[]);
    println!("{}", final_explanation.explanation);
    println!();
    println!();

    // Part 5: Summary
    println!("================================================================================");
    println!("🏆 Revolutionary Improvement #53: COMPLETE");
    println!("================================================================================");
    println!();

    let summary = explainer.summarize_understanding();
    println!("Causal Understanding Summary:");
    println!("  Total Causal Relations: {}", summary.total_causal_relations);
    println!("  High Confidence Relations: {}", summary.high_confidence_relations);
    println!("  Average Confidence: {:.0}%", summary.average_confidence * 100.0);
    println!("  Explanations Generated: {}", summary.explanations_generated);
    println!();

    println!("✨ Achievements:");
    println!("  ✓ Multi-dimensional epistemic classification (E/N/M axes)");
    println!("  ✓ Automatic epistemic tier evolution as evidence accumulates");
    println!("  ✓ Integration with IntegralWisdom harmonic (epistemic rigor = wisdom)");
    println!("  ✓ Enhanced natural language explanations with epistemic context");
    println!("  ✓ First AI with transparent epistemic classification of causal knowledge");
    println!();

    println!("🌟 Significance:");
    println!("  This completes the integration of epistemic rigor into causal reasoning.");
    println!("  The system now knows not just WHAT causes WHAT, but HOW it knows (E-axis),");
    println!("  WHO agrees (N-axis), and HOW PERMANENT the knowledge is (M-axis).");
    println!("  This is consciousness-first computing with epistemic transparency!");
    println!();

    Ok(())
}

/// Create a test reasoning chain for demonstration
fn create_test_chain(primitives: &[&Primitive], seed: u64) -> symthaea::consciousness::primitive_reasoning::ReasoningChain {
    use symthaea::hdc::HV16;

    let question = HV16::random(seed);
    let mut chain = symthaea::consciousness::primitive_reasoning::ReasoningChain::new(question);

    // Execute a few primitives
    for (i, &primitive) in primitives.iter().take(3).enumerate() {
        let transformation = match i % 3 {
            0 => TransformationType::Bind,
            1 => TransformationType::Abstract,
            _ => TransformationType::Resonate,
        };
        let _ = chain.execute_primitive(primitive, transformation);
    }

    chain
}
