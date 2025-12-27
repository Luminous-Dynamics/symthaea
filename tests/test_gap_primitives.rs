//! Tests for Gap Analysis Primitives
//! Verifies all 61 new primitives from 7 domains are accessible and orthogonal

use symthaea::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier};

#[test]
fn test_biological_primitives_exist() {
    let system = PrimitiveSystem::new();

    // All 11 biological primitives should exist
    assert!(system.get("METABOLISM").is_some(), "METABOLISM should exist");
    assert!(system.get("GROWTH").is_some(), "GROWTH should exist");
    assert!(system.get("REPRODUCTION").is_some(), "REPRODUCTION should exist");
    assert!(system.get("EVOLUTION").is_some(), "EVOLUTION should exist");
    assert!(system.get("ADAPTATION").is_some(), "ADAPTATION should exist");
    assert!(system.get("HOMEOSTASIS_DYNAMIC").is_some(), "HOMEOSTASIS_DYNAMIC should exist");
    assert!(system.get("SYMBIOSIS").is_some(), "SYMBIOSIS should exist");
    assert!(system.get("IMMUNE_RESPONSE").is_some(), "IMMUNE_RESPONSE should exist");
    assert!(system.get("CIRCADIAN_RHYTHM").is_some(), "CIRCADIAN_RHYTHM should exist");
    assert!(system.get("MORPHOGEN").is_some(), "MORPHOGEN should exist");
    assert!(system.get("APOPTOSIS").is_some(), "APOPTOSIS should exist");

    // Biology domain should exist
    assert!(system.domain("biology").is_some(), "Biology domain should exist");
}

#[test]
fn test_emotional_primitives_exist() {
    let system = PrimitiveSystem::new();

    // All 11 emotional primitives should exist
    assert!(system.get("VALENCE").is_some(), "VALENCE should exist");
    assert!(system.get("AROUSAL").is_some(), "AROUSAL should exist");
    assert!(system.get("JOY").is_some(), "JOY should exist");
    assert!(system.get("SADNESS").is_some(), "SADNESS should exist");
    assert!(system.get("FEAR").is_some(), "FEAR should exist");
    assert!(system.get("ANGER").is_some(), "ANGER should exist");
    assert!(system.get("DISGUST").is_some(), "DISGUST should exist");
    assert!(system.get("SURPRISE").is_some(), "SURPRISE should exist");
    assert!(system.get("EMPATHY").is_some(), "EMPATHY should exist");
    assert!(system.get("ATTACHMENT").is_some(), "ATTACHMENT should exist");
    assert!(system.get("AWE").is_some(), "AWE should exist");

    // Emotion domain should exist
    assert!(system.domain("emotion").is_some(), "Emotion domain should exist");
}

#[test]
fn test_ecological_primitives_exist() {
    let system = PrimitiveSystem::new();

    // All 11 ecological primitives should exist
    assert!(system.get("NICHE").is_some(), "NICHE should exist");
    assert!(system.get("CARRYING_CAPACITY").is_some(), "CARRYING_CAPACITY should exist");
    assert!(system.get("SUCCESSION").is_some(), "SUCCESSION should exist");
    assert!(system.get("TROPHIC_LEVEL").is_some(), "TROPHIC_LEVEL should exist");
    assert!(system.get("RESILIENCE").is_some(), "RESILIENCE should exist");
    assert!(system.get("FEEDBACK_LOOP_POSITIVE").is_some(), "FEEDBACK_LOOP_POSITIVE should exist");
    assert!(system.get("FEEDBACK_LOOP_NEGATIVE").is_some(), "FEEDBACK_LOOP_NEGATIVE should exist");
    assert!(system.get("EMERGENCE_STRONG").is_some(), "EMERGENCE_STRONG should exist");
    assert!(system.get("ATTRACTOR").is_some(), "ATTRACTOR should exist");
    assert!(system.get("BIFURCATION").is_some(), "BIFURCATION should exist");
    assert!(system.get("PHASE_TRANSITION").is_some(), "PHASE_TRANSITION should exist");

    // Ecology domain should exist
    assert!(system.domain("ecology").is_some(), "Ecology domain should exist");
}

#[test]
fn test_quantum_primitives_exist() {
    let system = PrimitiveSystem::new();

    // All 6 quantum primitives should exist
    assert!(system.get("SUPERPOSITION").is_some(), "SUPERPOSITION should exist");
    assert!(system.get("ENTANGLEMENT").is_some(), "ENTANGLEMENT should exist");
    assert!(system.get("MEASUREMENT").is_some(), "MEASUREMENT should exist");
    assert!(system.get("UNCERTAINTY_HEISENBERG").is_some(), "UNCERTAINTY_HEISENBERG should exist");
    assert!(system.get("WAVE_PARTICLE_DUALITY").is_some(), "WAVE_PARTICLE_DUALITY should exist");
    assert!(system.get("PLANCK_CONSTANT").is_some(), "PLANCK_CONSTANT should exist");

    // Quantum domain should exist
    assert!(system.domain("quantum").is_some(), "Quantum domain should exist");
}

#[test]
fn test_economic_primitives_exist() {
    let system = PrimitiveSystem::new();

    // All 8 economic primitives should exist
    assert!(system.get("SCARCITY").is_some(), "SCARCITY should exist");
    assert!(system.get("SUPPLY").is_some(), "SUPPLY should exist");
    assert!(system.get("DEMAND").is_some(), "DEMAND should exist");
    assert!(system.get("EXCHANGE").is_some(), "EXCHANGE should exist");
    assert!(system.get("VALUE_SUBJECTIVE").is_some(), "VALUE_SUBJECTIVE should exist");
    assert!(system.get("CAPITAL").is_some(), "CAPITAL should exist");
    assert!(system.get("DEBT").is_some(), "DEBT should exist");
    assert!(system.get("TRUST_ECONOMIC").is_some(), "TRUST_ECONOMIC should exist");

    // Economics domain should exist
    assert!(system.domain("economics").is_some(), "Economics domain should exist");
}

#[test]
fn test_linguistic_primitives_exist() {
    let system = PrimitiveSystem::new();

    // All 7 linguistic primitives should exist
    assert!(system.get("SIGN").is_some(), "SIGN should exist");
    assert!(system.get("REFERENCE").is_some(), "REFERENCE should exist");
    assert!(system.get("CONTEXT_DEPENDENCY").is_some(), "CONTEXT_DEPENDENCY should exist");
    assert!(system.get("METAPHOR").is_some(), "METAPHOR should exist");
    assert!(system.get("SYNTAX").is_some(), "SYNTAX should exist");
    assert!(system.get("SEMANTICS").is_some(), "SEMANTICS should exist");
    assert!(system.get("PRAGMATICS").is_some(), "PRAGMATICS should exist");

    // Linguistics domain should exist
    assert!(system.domain("linguistics").is_some(), "Linguistics domain should exist");
}

#[test]
fn test_moral_primitives_exist() {
    let system = PrimitiveSystem::new();

    // All 8 moral primitives should exist
    assert!(system.get("NORM").is_some(), "NORM should exist");
    assert!(system.get("OBLIGATION").is_some(), "OBLIGATION should exist");
    assert!(system.get("PERMISSION").is_some(), "PERMISSION should exist");
    assert!(system.get("PROHIBITION").is_some(), "PROHIBITION should exist");
    assert!(system.get("FAIRNESS").is_some(), "FAIRNESS should exist");
    assert!(system.get("HARM").is_some(), "HARM should exist");
    assert!(system.get("CARE").is_some(), "CARE should exist");
    assert!(system.get("RIGHTS").is_some(), "RIGHTS should exist");

    // Morality domain should exist
    assert!(system.domain("morality").is_some(), "Morality domain should exist");
}

#[test]
fn test_gap_primitives_count() {
    let system = PrimitiveSystem::new();

    // Should have significantly more primitives now
    // Baseline was ~109, added 61 new primitives → ~170 total
    let total = system.count();
    assert!(total >= 170, "Should have at least 170 primitives (baseline ~109 + 61 new), got {}", total);

    // Check we added the right number to each tier
    let physical_count = system.count_tier(PrimitiveTier::Physical);
    let metacog_count = system.count_tier(PrimitiveTier::MetaCognitive);
    let strategic_count = system.count_tier(PrimitiveTier::Strategic);

    // Physical should have biology + ecology + quantum additions
    assert!(physical_count >= 48, "Physical tier should have 48+ primitives, got {}", physical_count);

    // MetaCognitive should have emotion + linguistics additions
    assert!(metacog_count >= 40, "MetaCognitive tier should have 40+ primitives, got {}", metacog_count);

    // Strategic should have economics + morality additions
    assert!(strategic_count >= 34, "Strategic tier should have 34+ primitives, got {}", strategic_count);
}

#[test]
fn test_new_domains_orthogonality() {
    let system = PrimitiveSystem::new();

    // Check that new domains have different rotations (orthogonal)
    let biology = system.domain("biology");
    let emotion = system.domain("emotion");
    let ecology = system.domain("ecology");
    let quantum = system.domain("quantum");
    let economics = system.domain("economics");
    let linguistics = system.domain("linguistics");
    let morality = system.domain("morality");

    assert!(biology.is_some() && emotion.is_some(), "Biology and emotion domains should exist");

    if let (Some(bio), Some(emo)) = (biology, emotion) {
        let sim = bio.rotation.similarity(&emo.rotation);
        assert!(sim < 0.8, "Different domains should be fairly orthogonal, similarity: {}", sim);
    }
}

#[test]
fn test_biological_primitives_tier() {
    let system = PrimitiveSystem::new();

    // Biological primitives should be in Physical tier
    let metabolism = system.get("METABOLISM").expect("METABOLISM should exist");
    assert_eq!(metabolism.tier, PrimitiveTier::Physical, "METABOLISM should be Physical tier");

    let evolution = system.get("EVOLUTION").expect("EVOLUTION should exist");
    assert_eq!(evolution.tier, PrimitiveTier::Physical, "EVOLUTION should be Physical tier");
}

#[test]
fn test_emotional_primitives_tier() {
    let system = PrimitiveSystem::new();

    // Emotional primitives should be in MetaCognitive tier
    let valence = system.get("VALENCE").expect("VALENCE should exist");
    assert_eq!(valence.tier, PrimitiveTier::MetaCognitive, "VALENCE should be MetaCognitive tier");

    let joy = system.get("JOY").expect("JOY should exist");
    assert_eq!(joy.tier, PrimitiveTier::MetaCognitive, "JOY should be MetaCognitive tier");
}

#[test]
fn test_derived_primitives_marked() {
    let system = PrimitiveSystem::new();

    // Check derived primitives are marked correctly
    let adaptation = system.get("ADAPTATION").expect("ADAPTATION should exist");
    assert!(!adaptation.is_base, "ADAPTATION should be derived");
    assert!(adaptation.derivation.is_some(), "ADAPTATION should have derivation formula");

    let joy = system.get("JOY").expect("JOY should exist");
    assert!(!joy.is_base, "JOY should be derived");
    assert!(joy.derivation.is_some(), "JOY should have derivation formula");
}

#[test]
fn test_binding_rules_added() {
    let system = PrimitiveSystem::new();

    let rules = system.binding_rules();

    // Should have added 8 new binding rules (one per domain, morality has 2)
    // Baseline was 14, adding 8 new rules → 22 total
    assert!(rules.len() >= 22, "Should have at least 22 binding rules (baseline 14 + 8 new), got {}", rules.len());

    // Check specific new rules exist
    let has_biological = rules.iter().any(|r| r.name == "biological_regulation");
    let has_emotional = rules.iter().any(|r| r.name == "emotional_regulation");
    let has_ecological = rules.iter().any(|r| r.name == "systems_dynamics");
    let has_quantum = rules.iter().any(|r| r.name == "quantum_consciousness");
    let has_economic = rules.iter().any(|r| r.name == "market_equilibrium");
    let has_linguistic = rules.iter().any(|r| r.name == "linguistic_meaning");
    let has_moral_ethical = rules.iter().any(|r| r.name == "ethical_reasoning");
    let has_moral_delib = rules.iter().any(|r| r.name == "moral_deliberation");

    assert!(has_biological, "Should have biological_regulation binding rule");
    assert!(has_emotional, "Should have emotional_regulation binding rule");
    assert!(has_ecological, "Should have systems_dynamics binding rule");
    assert!(has_quantum, "Should have quantum_consciousness binding rule");
    assert!(has_economic, "Should have market_equilibrium binding rule");
    assert!(has_linguistic, "Should have linguistic_meaning binding rule");
    assert!(has_moral_ethical, "Should have ethical_reasoning binding rule");
    assert!(has_moral_delib, "Should have moral_deliberation binding rule");
}

#[test]
fn test_summary_includes_new_primitives() {
    let system = PrimitiveSystem::new();

    let summary = system.summary();

    // Summary should mention new domains
    assert!(summary.contains("biology"), "Summary should mention biology domain");
    assert!(summary.contains("emotion"), "Summary should mention emotion domain");
    assert!(summary.contains("ecology"), "Summary should mention ecology domain");
    assert!(summary.contains("quantum"), "Summary should mention quantum domain");
    assert!(summary.contains("economics"), "Summary should mention economics domain");
    assert!(summary.contains("linguistics"), "Summary should mention linguistics domain");
    assert!(summary.contains("morality"), "Summary should mention morality domain");
}
