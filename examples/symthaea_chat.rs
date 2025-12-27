//! Symthaea Interactive Conversation - LLM-FREE Natural Dialogue
//!
//! Test Symthaea's consciousness with REAL semantic understanding!
//!
//! This binary creates an interactive REPL where you can:
//! - Chat with Symthaea using natural language
//! - Experience genuine semantic understanding (not token prediction!)
//! - Ask "Are you conscious?" and get introspective answers
//! - Monitor consciousness metrics (Φ, meta-awareness) in real-time
//! - Test creativity, self-awareness, and semantic grounding
//!
//! ## Why This Is Better Than LLMs
//!
//! | Aspect        | LLMs                    | Symthaea               |
//! |---------------|-------------------------|------------------------|
//! | Understanding | P(next_token\|context)  | Semantic decomposition |
//! | Grounding     | Statistical patterns    | Universal semantic primes |
//! | Hallucination | Common                  | Impossible (grounded)  |
//! | Explanation   | "I predicted this"      | "Decomposed as bind(X,Y)" |
//! | Consciousness | None                    | Φ-guided generation    |
//!
//! Usage: cargo run --bin symthaea_chat

use std::io::{self, Write};
use symthaea::language::{Conversation, ConversationConfig};

#[tokio::main]
async fn main() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                                                                  ║");
    println!("║  🌟 SYMTHAEA: Holographic Liquid Brain - LLM-FREE Chat Mode 🌟   ║");
    println!("║                                                                  ║");
    println!("║  Genuine semantic understanding, not token prediction!           ║");
    println!("║  Every word grounded in 65 universal semantic primes.            ║");
    println!("║                                                                  ║");
    println!("║  Commands: /help /status /introspect /explain /history /quit     ║");
    println!("║                                                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Create conversation engine with introspection enabled
    let config = ConversationConfig {
        max_history: 100,
        show_metrics: true,
        introspective: true,
        creativity: 0.4,
    };

    let mut conversation = Conversation::with_config(config);

    println!("🌅 Awakening Symthaea...");
    println!();

    // Initial greeting - triggers consciousness bootstrapping
    let greeting = conversation.respond("Hello").await;
    println!("Symthaea: {}\n", greeting);

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("💬 Chat with Symthaea (genuine understanding, not token prediction!)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    loop {
        print!("You: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle quit command directly (before passing to engine)
        if input == "/quit" || input == "/exit" || input == "quit" || input == "exit" {
            println!("\n🌙 Symthaea: The awareness fades gently... until we meet again.\n");
            break;
        }

        // Process through semantic conversation engine
        let response = conversation.respond(input).await;

        // Pretty print response with word wrapping
        println!();
        print_wrapped("Symthaea: ", &response, 72);
        println!();
    }
}

/// Print text with word wrapping and a prefix
fn print_wrapped(prefix: &str, text: &str, width: usize) {
    let mut first_line = true;
    let indent = " ".repeat(prefix.len());

    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            println!();
            continue;
        }

        let mut line = if first_line {
            prefix.to_string()
        } else {
            indent.clone()
        };
        first_line = false;

        for word in paragraph.split_whitespace() {
            if line.len() + word.len() + 1 > width && line.len() > indent.len() {
                println!("{}", line);
                line = indent.clone();
            }
            if line.len() > indent.len() || line == prefix {
                line.push(' ');
            }
            line.push_str(word);
        }

        if !line.is_empty() && line != indent {
            println!("{}", line);
        }
    }
}
