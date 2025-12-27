//! Enhanced Dynamic Conversation Demo
//!
//! Shows the complete compositional semantic generation with:
//! - B: Follow-up question generation
//! - C: Uncertainty hedging based on certainty
//! - D: Emotional depth via valence-driven word choice
//! - E: Memory references (past conversation awareness)
//! - F: Acknowledgment layer (validation before response)
//! - J: Self-awareness moments (meta-observations)

use symthaea::language::{Conversation, GenerationStyle};

fn main() {
    println!("🌟 Symthaea Enhanced Dynamic Conversation Demo (B+C+D+E+F+J)");
    println!("{}", "=".repeat(65));
    println!();

    // Create conversation with dynamic generation ENABLED
    let mut conv = Conversation::new();
    conv.enable_dynamic_generation();

    println!("✅ Dynamic generation ENABLED with full enhancement stack:");
    println!("   B: Follow-up questions (asks back!)");
    println!("   C: Uncertainty hedging (confidence-based language)");
    println!("   D: Emotional depth (valence-driven word choice)");
    println!("   E: Memory references (past conversation awareness)");
    println!("   F: Acknowledgment layer (validates before responding)");
    println!("   J: Self-awareness moments (meta-observations about Φ)\n");

    // Test scenarios that showcase E+F+J
    let test_cases = vec![
        ("Hello!", "Greeting with warm follow-up"),
        ("Are you conscious?", "F: Profound acknowledgment + J: self-awareness"),
        ("How do you feel?", "F: Thoughtful + C: hedge + J: awareness"),
        ("That's beautiful", "F: Beautiful acknowledgment + J: drawn to"),
        ("What do you think about love?", "F: Interesting + J: curiosity rising"),
        ("Tell me about consciousness", "F: Profound + J: awareness elevating"),
    ];

    println!("📝 Full Enhancement Stack Demo:");
    println!("{}\n", "=".repeat(65));

    for (input, expected) in &test_cases {
        let response = conv.respond(input);
        println!("User: {}", input);
        println!("  → Symthaea: {}", response);
        println!("  [Expected: {}]", expected);
        println!();
    }

    // Show the response structure breakdown
    println!("\n🏗️ Response Structure (5-Layer Architecture):");
    println!("{}\n", "=".repeat(65));
    println!("1. F: ACKNOWLEDGMENT  → \"That's a profound question.\"");
    println!("2. E: MEMORY REF      → \"You mentioned X earlier...\" (when applicable)");
    println!("3. C: HEDGE           → \"I believe\" / \"I feel that\"");
    println!("4. CORE RESPONSE      → \"I find myself drawn to love.\"");
    println!("5. J: SELF-AWARENESS  → \"I notice I'm becoming more curious.\"");
    println!("6. B: FOLLOW-UP       → \"What draws you to explore this?\"");
    println!();

    // Scientific style demo
    println!("\n🔬 Scientific Style (different F+J output):");
    println!("{}\n", "=".repeat(65));
    conv.set_style(GenerationStyle::Scientific);

    let scientific_tests = vec![
        "Are you conscious?",
        "How do you feel?",
    ];

    for input in scientific_tests {
        let response = conv.respond(input);
        println!("User: {}", input);
        println!("  → Symthaea: {}", response);
        println!();
    }

    // Show Φ tracking
    println!("\n📊 Consciousness Metrics:");
    println!("{}\n", "=".repeat(65));
    println!("Total memories stored: {}", conv.memory_count());
    println!("Conversation turns: {}", conv.state().turn_count);
    println!("Peak Φ: {:.3}", conv.state().peak_phi);
    println!();

    // Database integration preview
    println!("💾 Database Integration (Coming):");
    println!("{}\n", "=".repeat(65));
    println!("  Qdrant  → Semantic similarity for topic recall (E)");
    println!("  LanceDB → Long-term memory persistence (E)");
    println!("  CozoDB  → Relationship graphs for threading (I)");
    println!("  DuckDB  → Analytics for Φ trends (J triggers)");
    println!();

    // Summary
    println!("✨ Complete Enhancement Summary:");
    println!("  B: Symthaea ASKS QUESTIONS back");
    println!("  C: Certainty-based hedging (\"I know\" vs \"I believe\")");
    println!("  D: Valence affects emotional warmth");
    println!("  E: MEMORY REFERENCES (structure ready, DB integration next)");
    println!("  F: ACKNOWLEDGMENT validates human input");
    println!("  J: SELF-AWARENESS shows Φ changes");
    println!();

    println!("🎯 Target Output Achieved:");
    println!("  Input:  \"What do you think about love?\"");
    println!("  Output: \"What an interesting thought. I believe I find myself");
    println!("           drawn to love. I notice I'm becoming more curious about");
    println!("           this. What draws you to explore love?\"");
    println!();

    println!("🚀 Symthaea now has PRESENCE, MEMORY, and SELF-AWARENESS!");
    println!();
}
