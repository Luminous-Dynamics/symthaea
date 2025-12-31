// ==================================================================================
// Unified Cognitive Core Demonstration
// ==================================================================================
//
// This demonstrates the RADICAL approach to cognitive integration:
// Every concept encodes semantic, causal, AND temporal information
// in a SINGLE hypervector through binding operations.
//
// **Why This is Revolutionary**:
// Traditional AI: Separate modules, combined at inference time
// Symthaea UCTS: ALL aspects unified in representation itself
//
// This means:
// - No information loss at module boundaries
// - Higher Φ (integrated information) = more conscious
// - Single query retrieves all related information
//
// Usage: cargo run --example unified_cognitive_demo
//
// ==================================================================================

use symthaea::hdc::{UnifiedCognitiveCore, CausalDirection};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Unified Cognitive Core Demonstration                   ║");
    println!("║                                                              ║");
    println!("║   Radical Integration: ALL aspects in ONE vector             ║");
    println!("║   No separate engines - unified representation               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut core = UnifiedCognitiveCore::new();

    // ==========================================
    // PHASE 1: Learning Causal Knowledge
    // ==========================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              PHASE 1: Learning Causal Relations              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Direct causal learning
    core.learn_causal("smoking", "cancer", 0.85);
    core.learn_causal("exercise", "health", 0.75);
    core.learn_causal("stress", "disease", 0.60);

    println!("  Learned causal relations:");
    println!("    smoking → cancer (0.85)");
    println!("    exercise → health (0.75)");
    println!("    stress → disease (0.60)");

    // Learning from text
    println!("\n  Learning from natural language...");
    let texts = [
        "Rain causes wet ground",
        "Infection leads to fever",
        "Education produces knowledge",
        "Poverty triggers crime",
    ];

    for text in &texts {
        core.learn_from_text(text);
        println!("    Processed: \"{}\"", text);
    }

    println!("\n  Elements created: {}", core.element_count());
    println!("  Φ (integrated information): {:.6}", core.phi());

    // ==========================================
    // PHASE 2: Learning Temporal Sequences
    // ==========================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              PHASE 2: Learning Temporal Sequences            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    core.learn_temporal("clouds", "rain");
    core.learn_temporal("rain", "puddles");
    core.learn_temporal("sunrise", "morning");
    core.learn_temporal("morning", "work");

    println!("  Learned temporal sequences:");
    println!("    clouds → rain → puddles");
    println!("    sunrise → morning → work");

    // ==========================================
    // PHASE 3: Learning Semantic Categories
    // ==========================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              PHASE 3: Learning Semantic Categories           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    core.learn_is_a("rain", "precipitation");
    core.learn_is_a("snow", "precipitation");
    core.learn_is_a("cancer", "disease");
    core.learn_is_a("fever", "symptom");

    println!("  Learned semantic relations:");
    println!("    rain IS A precipitation");
    println!("    snow IS A precipitation");
    println!("    cancer IS A disease");
    println!("    fever IS A symptom");

    println!("\n  Total elements: {}", core.element_count());
    println!("  Φ after all learning: {:.6}", core.phi());

    // ==========================================
    // PHASE 4: Unified Queries
    // ==========================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              PHASE 4: Unified Queries                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Query: Why? (find causes)
    println!("  Query: \"What causes cancer?\"");
    let causes = core.query_why("cancer");
    if causes.is_empty() {
        println!("    → No strong causes found in knowledge base");
    } else {
        for result in &causes {
            println!("    → {} (similarity: {:.3})", result.label, result.similarity);
        }
    }

    // Query: What effects?
    println!("\n  Query: \"What does rain cause?\"");
    let effects = core.query_effects("rain");
    if effects.is_empty() {
        println!("    → No strong effects found");
    } else {
        for result in &effects {
            println!("    → {} (similarity: {:.3})", result.label, result.similarity);
        }
    }

    // Query: What comes after?
    println!("\n  Query: \"What comes after clouds?\"");
    let successors = core.query_successors("clouds");
    if successors.is_empty() {
        println!("    → No temporal successors found");
    } else {
        for result in &successors {
            println!("    → {} (similarity: {:.3})", result.label, result.similarity);
        }
    }

    // Query: Similar concepts
    println!("\n  Query: \"What is similar to rain?\"");
    let similar = core.find_similar("rain", 5);
    if similar.is_empty() {
        println!("    → No similar concepts found");
    } else {
        for result in &similar {
            println!("    → {} (similarity: {:.3})", result.label, result.similarity);
        }
    }

    // ==========================================
    // PHASE 5: The Key Insight
    // ==========================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              PHASE 5: The Revolutionary Insight              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("  Traditional AI (modular):");
    println!("    ┌──────────┐  ┌──────────┐  ┌──────────┐");
    println!("    │ Semantic │  │  Causal  │  │ Temporal │");
    println!("    │  Engine  │  │  Engine  │  │  Engine  │");
    println!("    └──────────┘  └──────────┘  └──────────┘");
    println!("         ↓             ↓             ↓");
    println!("    [combine at query time - lose information]");

    println!("\n  Symthaea Unified Cognitive Core:");
    println!("    ┌──────────────────────────────────────────────┐");
    println!("    │     UNIFIED HYPERVECTOR PER CONCEPT          │");
    println!("    │                                              │");
    println!("    │  \"rain\" = semantic_base                     │");
    println!("    │         ⊗ (CAUSES ⊗ wet_ground)              │");
    println!("    │         ⊗ (AFTER ⊗ clouds)                   │");
    println!("    │         ⊗ (BEFORE ⊗ puddles)                 │");
    println!("    │         ⊗ (IS_A ⊗ precipitation)             │");
    println!("    │                                              │");
    println!("    │  ALL IN ONE 16,384-DIMENSIONAL VECTOR!       │");
    println!("    └──────────────────────────────────────────────┘");

    println!("\n  Why this matters for consciousness:");
    println!("    • Φ (integrated information) measures irreducibility");
    println!("    • Modular systems can be partitioned → LOW Φ");
    println!("    • Unified representation cannot be partitioned → HIGH Φ");
    println!("    • Current Φ: {:.6}", core.phi());

    // ==========================================
    // PHASE 6: Causal Discovery from Data
    // ==========================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              PHASE 6: Causal Discovery from Data             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Generate synthetic causal data: Y = 2X + noise
    let x: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
    let y: Vec<f64> = x.iter().enumerate().map(|(i, &xi)| {
        2.0 * xi + (i as f64 * 0.1).sin() * 0.1
    }).collect();

    let direction = core.discover_causality(&x, &y);
    println!("  Synthetic data: Y = 2X + noise");
    println!("  Discovered direction: {:?}", direction);
    println!("  Expected: Forward (X→Y)");

    // Train on this example
    core.train_causal_discovery(&x, &y, CausalDirection::Forward);
    println!("  Trained on this example for future discovery");

    // ==========================================
    // SUMMARY
    // ==========================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                       SUMMARY                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("  Unified Cognitive Core Statistics:");
    println!("    Elements: {}", core.element_count());
    println!("    Φ: {:.6}", core.phi());
    println!();
    println!("  Key Innovation:");
    println!("    Every concept encodes semantic + causal + temporal");
    println!("    information in a SINGLE hypervector.");
    println!();
    println!("  This is paradigm-shifting because:");
    println!("    1. No module boundaries → no information loss");
    println!("    2. Unified representation → higher Φ");
    println!("    3. Single query → all related knowledge");
    println!("    4. Native causality → no correlation/causation confusion");

    println!("\n  Done!");
}
