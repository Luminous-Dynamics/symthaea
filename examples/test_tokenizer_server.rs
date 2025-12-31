//! Quick test of Misaki server tokenization speed

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("🔌 Testing Misaki Server Tokenization");
    println!("=====================================\n");

    // Test phrases
    let phrases = [
        ("Short", "Hello world."),
        ("Medium", "Welcome to Symthaea, the consciousness-first AI system."),
        ("Long", "The quick brown fox jumps over the lazy dog. This is a test."),
    ];

    for (name, text) in phrases {
        let start = Instant::now();
        let result = symthaea::voice::tokenizer::tokenize(text);
        let elapsed = start.elapsed();

        match result {
            Ok(tokens) => {
                let ms = elapsed.as_secs_f64() * 1000.0;
                let status = if ms < 100.0 { "✅ FAST (server)" } else { "⚠️ SLOW (subprocess?)" };
                println!("{}: {} tokens in {:.1}ms {}", name, tokens.len(), ms, status);
            }
            Err(e) => {
                println!("{}: ERROR - {}", name, e);
            }
        }
    }

    println!("\n✅ Done!");
    Ok(())
}
