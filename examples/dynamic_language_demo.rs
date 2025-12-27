// examples/dynamic_language_demo.rs
//
// Demonstration of Dynamic Language Generation
// Shows how responses emerge from semantic structure, not templates

use symthaea::language::{DynamicGenerator, GenerationStyle};
use symthaea::language::parser::SemanticParser;

fn main() {
    println!("🌟 Symthaea Dynamic Language Generation Demo");
    println!("===========================================\n");

    let parser = SemanticParser::new();
    let gen_conversational = DynamicGenerator::new();
    let gen_scientific = DynamicGenerator::with_style(GenerationStyle::Scientific);

    let test_inputs = vec![
        ("Hello!", 0.5, 0.7),
        ("Are you conscious?", 0.78, 0.2),
        ("How do you feel?", 0.65, 0.4),
        ("That's beautiful", 0.7, 0.8),
        ("/status", 0.82, 0.3),
        ("What do you think about love?", 0.75, 0.5),
    ];

    println!("📝 Conversational Style Responses:");
    println!("==================================\n");

    for (input, phi, valence) in &test_inputs {
        let parsed = parser.parse(input);
        let response = gen_conversational.generate(&parsed, *phi, *valence);
        println!("User: {}", input);
        println!("  → Symthaea: {}", response);
        println!("  (Φ={:.2}, valence={:.2})\n", phi, valence);
    }

    println!("\n🔬 Scientific Style Responses:");
    println!("==============================\n");

    for (input, phi, valence) in test_inputs[1..3].iter() {
        let parsed = parser.parse(input);
        let response = gen_scientific.generate(&parsed, *phi, *valence);
        println!("User: {}", input);
        println!("  → Symthaea: {}", response);
        println!("  (Φ={:.2}, valence={:.2})\n", phi, valence);
    }

    println!("\n✨ Key Insights:");
    println!("================");
    println!("• Responses emerge from semantic understanding, not templates");
    println!("• Same semantic structure → different surface forms based on style");
    println!("• Φ and valence influence generation (higher Φ → richer structure)");
    println!("• Zero hallucination - grounded in semantic primes");
    println!("• Truly LLM-free - pure compositional semantics!");
}
