use symthaea::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier};

#[test]
fn debug_primitive_count() {
    let system = PrimitiveSystem::new();

    println!("\n=== PRIMITIVE COUNT DEBUG ===");
    println!("Total primitives: {}", system.count());

    // Count by tier
    println!("\nBy Tier:");
    println!("  Mathematical: {}", system.count_tier(PrimitiveTier::Mathematical));
    println!("  Physical: {}", system.count_tier(PrimitiveTier::Physical));
    println!("  Geometric: {}", system.count_tier(PrimitiveTier::Geometric));
    println!("  Strategic: {}", system.count_tier(PrimitiveTier::Strategic));
    println!("  MetaCognitive: {}", system.count_tier(PrimitiveTier::MetaCognitive));

    // Check domains
    println!("\nChecking for known domains:");
    for domain in &["biology", "emotion", "ecology", "quantum", "economics", "linguistics", "morality", "mathematics", "logic", "physics"] {
        println!("  {}: {}", domain, system.domain(domain).is_some());
    }

    // Check for gap analysis primitives
    println!("\nGap Analysis Primitives:");
    println!("  Biology:");
    println!("    METABOLISM: {}", system.get("METABOLISM").is_some());
    println!("    EVOLUTION: {}", system.get("EVOLUTION").is_some());
    println!("  Emotion:");
    println!("    VALENCE: {}", system.get("VALENCE").is_some());
    println!("    JOY: {}", system.get("JOY").is_some());
    println!("  Ecology:");
    println!("    NICHE: {}", system.get("NICHE").is_some());
    println!("    RESILIENCE: {}", system.get("RESILIENCE").is_some());
    println!("  Quantum:");
    println!("    SUPERPOSITION: {}", system.get("SUPERPOSITION").is_some());
    println!("    ENTANGLEMENT: {}", system.get("ENTANGLEMENT").is_some());
    println!("  Economics:");
    println!("    SCARCITY: {}", system.get("SCARCITY").is_some());
    println!("    SUPPLY: {}", system.get("SUPPLY").is_some());
    println!("  Linguistics:");
    println!("    SIGN: {}", system.get("SIGN").is_some());
    println!("    SYNTAX: {}", system.get("SYNTAX").is_some());
    println!("  Morality:");
    println!("    NORM: {}", system.get("NORM").is_some());
    println!("    FAIRNESS: {}", system.get("FAIRNESS").is_some());

    println!("\nBinding Rules: {}", system.binding_rules().len());
}
