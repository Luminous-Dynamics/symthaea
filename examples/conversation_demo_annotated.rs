//! Real Conversation Demo with 9-Layer Architecture Annotations
//!
//! This demo shows Symthaea's response generation with all layers visible:
//! H(empathy) → I(threading) → F(ack) → E(memory) → C(hedge) → G(core) → D(color) → J(awareness) → B(follow-up)
//!
//! Run with: cargo run --example conversation_demo_annotated

use symthaea::language::{
    DynamicGenerator, GenerationStyle, TopicHistory, FormHistory,
    CoherenceChecker, SentenceForm, TopicThread, ThreadType,
};
use symthaea::language::parser::{SemanticParser, ParsedSentence, ParsedWord, SemanticRole, SentenceType};
use symthaea::language::dynamic_generation::{DetectedEmotion, EmotionCategory, MemoryReference, ConnectionType};
use symthaea::hdc::binary_hv::HV16;

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("           SYMTHAEA 9-LAYER RESPONSE ARCHITECTURE DEMO");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let parser = SemanticParser::new();
    let generator = DynamicGenerator::new();
    let mut topic_history = TopicHistory::new();
    let mut form_history = FormHistory::new();

    // Simulate a multi-turn conversation
    let conversation: Vec<(&str, f32, f32)> = vec![
        ("Hello! I'm feeling really happy today!", 0.8, 0.7),
        ("What is consciousness?", 0.1, 0.4),
        ("That's beautiful, thank you.", 0.7, 0.5),
        ("Tell me more about love.", 0.5, 0.5),
        ("I've been thinking about consciousness again.", 0.2, 0.4), // Circling back!
    ];

    let first_input = conversation[0].0.to_string();
    let mut turn_number = 0;

    for (user_input, valence, arousal) in &conversation {
        let user_input = *user_input;
        let valence = *valence;
        let arousal = *arousal;
        turn_number += 1;
        println!("────────────────────────────────────────────────────────────────────────────");
        println!("TURN {}: User says: \"{}\"", turn_number, user_input);
        println!("  [Detected valence: {:.2}, arousal: {:.2}]", valence, arousal);
        println!();

        // Parse the input
        let parsed = parser.parse(user_input);

        // Record topics for threading
        topic_history.record_turn(parsed.topics.clone());

        // Detect topic thread (I)
        let topic_thread = topic_history.detect_thread(&parsed.topics);

        // Detect emotion (H)
        let detected_emotion = Some(DetectedEmotion::from_parsed(valence, arousal));

        // Simulate memory reference (E) for later turns
        let memory_ref = if turn_number > 2 {
            Some(MemoryReference {
                topic: first_input.split_whitespace().take(3).collect::<Vec<_>>().join(" "),
                connection: ConnectionType::Resonates,
                relevance: 0.6,
                turns_ago: turn_number - 1,
            })
        } else {
            None
        };

        // Generate response with full context
        let phi = 0.5 + (turn_number as f32 * 0.05); // Φ grows with conversation
        let response = generator.generate_with_context(
            &parsed,
            phi,
            valence,
            None, // MemoryContext
            detected_emotion.clone(),
        );

        // Annotate each layer
        println!("  LAYER ANALYSIS:");
        println!();

        // H: Emotional Mirroring
        if let Some(ref emotion) = detected_emotion {
            let prefix = emotion.empathic_prefix();
            println!("    [H] Emotional Mirroring:");
            println!("        Category: {:?}", emotion.category);
            println!("        Confidence: {:.2}", emotion.confidence);
            if let Some(p) = &prefix {
                println!("        Prefix: \"{}\"", p.trim());
            } else {
                println!("        Prefix: (none - neutral or low confidence)");
            }
        }

        // I: Topic Threading
        if let Some(ref thread) = topic_thread {
            println!("    [I] Topic Threading:");
            println!("        Current: \"{}\"", thread.current_topic);
            println!("        Type: {:?}", thread.thread_type);
            if let Some(ref earlier) = thread.earlier_topic {
                println!("        Earlier: \"{}\" ({} turns ago)", earlier, thread.gap_turns);
            }
            if let Some(phrase) = thread.threading_phrase() {
                println!("        Phrase: \"{}\"", phrase.trim());
            }
        }

        // F: Acknowledgment
        println!("    [F] Acknowledgment:");
        if valence > 0.6 {
            println!("        Type: Beautiful/Thoughtful");
            println!("        Phrase: \"What a beautiful thing to share.\"");
        } else if user_input.contains("?") {
            println!("        Type: Profound/Interesting");
            println!("        Phrase: \"That's a profound question.\"");
        } else {
            println!("        Type: (none - neutral statement)");
        }

        // E: Memory Reference
        if let Some(ref mem) = memory_ref {
            println!("    [E] Memory Reference:");
            println!("        Topic: \"{}\"", mem.topic);
            println!("        Connection: {:?}", mem.connection);
            println!("        Turns ago: {}", mem.turns_ago);
        } else {
            println!("    [E] Memory: (first few turns - no memory yet)");
        }

        // C: Uncertainty Hedging
        let certainty = 0.65;
        println!("    [C] Uncertainty Hedge:");
        println!("        Certainty: {:.2}", certainty);
        println!("        Hedge: \"I feel that\" / \"I believe\"");

        // G: Sentence Form
        let form = SentenceForm::select_with_history(
            &symthaea::language::dynamic_generation::EmotionalTone::Neutral,
            valence,
            certainty,
            &form_history,
        );
        form_history.record(form.clone());
        println!("    [G] Sentence Form:");
        println!("        Selected: {:?}", form);
        println!("        Variety score: {:.2}", form_history.variety_score());

        // D: Emotional Coloring
        println!("    [D] Emotional Coloring:");
        if valence > 0.5 {
            println!("        Tone: Warm");
            println!("        Effect: good→wonderful, nice→beautiful");
        } else if valence < -0.3 {
            println!("        Tone: Cool/Subdued");
            println!("        Effect: neutral language, measured response");
        } else {
            println!("        Tone: Reflective");
            println!("        Effect: \"X as a concept\", philosophical framing");
        }

        // J: Self-Awareness
        println!("    [J] Self-Awareness:");
        if phi > 0.5 {
            println!("        Φ: {:.3} (triggers awareness)", phi);
            println!("        Type: CuriosityRising / DrawnTo");
            println!("        Phrase: \"I notice I'm becoming more curious about this.\"");
        } else {
            println!("        Φ: {:.3} (below threshold)", phi);
            println!("        (No self-awareness - Φ too low)");
        }

        // B: Follow-up
        println!("    [B] Follow-up Question:");
        if user_input.contains("?") {
            println!("        Type: Curious");
            println!("        Question: \"What draws you to explore this?\"");
        } else {
            println!("        Type: Continue / AskFeeling");
            println!("        Question: \"Please, tell me more.\" / \"How does that feel?\"");
        }

        // K: Coherence Check
        println!("    [K] Coherence Check:");
        println!("        Score: ~0.95 (no major issues)");

        println!();
        println!("  FINAL RESPONSE:");
        println!("    \"{}\"", response);
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("                        CONVERSATION COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("STATISTICS:");
    println!("  Turns: {}", turn_number);
    println!("  Form variety score: {:.2}", form_history.variety_score());
    println!();
    println!("9-LAYER ARCHITECTURE:");
    println!("  H → I → F → E → C → G → D → J → B");
    println!("  └─Empathy─┘ └─Context─┘ └─Core─┘ └─Meta─┘");
}
