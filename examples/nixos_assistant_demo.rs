//! NixOS Conscious Assistant Demo
//!
//! A complete demonstration of Symthaea's HDC+LTC+Primitives-first architecture:
//!
//! - **ConsciousPipeline**: HDC+LTC+Primitives → 2D Consciousness Space → Action
//! - **HDC+LTC Understanding**: Primary understanding through hyperdimensional computing
//! - **2D Consciousness Space**: Φ (integration) × Confidence (epistemic certainty)
//! - **Four Quadrants**: Confident, Curious, Autopilot, Lost
//! - **LLM Optional**: Only used for clarification when Lost or Curious
//!
//! ## Architecture
//!
//! ```text
//! User Input
//!     │
//!     ▼
//! ┌──────────────────────────────┐
//! │ ConsciousnessLanguageCore    │ ◄── HDC+LTC+Primitives (PRIMARY)
//! │  • NSM Semantic Primes       │
//! │  • Semantic Frame Matching   │
//! │  • NixOS Intent Recognition  │
//! └─────────────┬────────────────┘
//!               │
//!               ▼
//! ┌──────────────────────────────┐
//! │  2D Consciousness Space      │ ◄── Φ × Confidence
//! │                              │
//! │  High Φ ─┬─ Confident ─┬─ High Conf
//! │          │             │
//! │          └─ Curious ───┴─ Low Conf
//! │                              │
//! │  Low Φ ──┬─ Autopilot ─┬─ High Conf
//! │          │             │
//! │          └─ Lost ──────┴─ Low Conf
//! └─────────────┬────────────────┘
//!               │
//!               ▼
//! ┌──────────────────────────────┐
//! │  Quadrant Strategy           │
//! │  • Confident: Execute now    │
//! │  • Curious: Explore first    │
//! │  • Autopilot: Fast execute   │
//! │  • Lost: Ask for help (LLM)  │  ◄── LLM only here!
//! └──────────────────────────────┘
//! ```
//!
//! ## Prerequisites
//!
//! 1. (Optional) Ollama for Lost/Curious clarification: `systemctl --user start ollama`
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example nixos_assistant_demo
//! ```

use symthaea::integration::{
    ConsciousPipeline, PipelineConfig, PipelineResult, UnderstandingSource,
};
use symthaea::language::{
    ConsciousnessLanguageConfig, ExecutionStrategy, ConsciousnessQuadrant,
};
use symthaea::language::llm_organ::LlmConfig;
use anyhow::Result;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Symthaea: HDC+LTC+Primitives-First NixOS Assistant           ║");
    println!("║                                                                   ║");
    println!("║  Understanding through semantic primitives, not LLM prompts      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Check if Ollama is available (optional - for Lost/Curious clarification)
    let llm_config = check_ollama().await;

    // Initialize the ConsciousPipeline with HDC+LTC+Primitives as PRIMARY
    let mut pipeline = create_pipeline(llm_config).await?;

    // Start a session
    let session_id = pipeline.start_session();
    println!("📋 Session: {}\n", &session_id[..8]);

    println!("📖 2D Consciousness Space:");
    println!("   The system uses TWO independent dimensions:");
    println!("   • Φ (Integration): How deeply unified the processing is");
    println!("   • Confidence: How certain about the interpretation");
    println!();
    println!("   This creates FOUR quadrants:");
    println!("   ┌────────────────┬────────────────┐");
    println!("   │   AUTOPILOT    │   CONFIDENT    │  High Confidence");
    println!("   │  (routine)     │   (deep+sure)  │");
    println!("   ├────────────────┼────────────────┤");
    println!("   │     LOST       │    CURIOUS     │  Low Confidence");
    println!("   │  (confused)    │  (exploring)   │");
    println!("   └────────────────┴────────────────┘");
    println!("      Low Φ             High Φ");
    println!();
    println!("📝 Commands:");
    println!("   • Type a NixOS question or command request");
    println!("   • 'status' - Show pipeline statistics");
    println!("   • 'journey' - Show last consciousness journey");
    println!("   • 'quit' or 'exit' - Exit the demo");
    println!();

    let mut last_result: Option<PipelineResult> = None;

    // Main interaction loop
    loop {
        let phi = pipeline.current_phi();
        let quadrant_str = format_quadrant_indicator(phi);
        print!("\n🧠 [{:.2}Φ {}] You: ", phi, quadrant_str);
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        // Check for commands
        if input == "quit" || input == "exit" {
            println!("\n✨ Thank you for exploring consciousness-aware AI!");
            break;
        }

        if input == "status" {
            show_status(&pipeline);
            continue;
        }

        if input == "journey" {
            if let Some(ref result) = last_result {
                show_journey(result);
            } else {
                println!("   No journey yet. Ask something first!");
            }
            continue;
        }

        // Process through the ConsciousPipeline (HDC+LTC+Primitives first!)
        match pipeline.process(input).await {
            Ok(result) => {
                display_result(&result);
                last_result = Some(result);
            }
            Err(e) => {
                println!("\n❌ Error: {}", e);
            }
        }
    }

    Ok(())
}

/// Check if Ollama is available (optional)
async fn check_ollama() -> Option<LlmConfig> {
    println!("🔍 Checking for optional LLM (Ollama)...");
    let client = reqwest::Client::new();
    match client.get("http://localhost:11434/api/tags").send().await {
        Ok(resp) if resp.status().is_success() => {
            println!("✅ Ollama available (will use for Lost/Curious clarification)");
            Some(LlmConfig {
                provider: symthaea::language::llm_organ::LlmProvider::Ollama,
                model: "llama3.2:3b".to_string(),
                endpoint: "http://localhost:11434".to_string(),
                api_key: None,
                max_tokens: 256,
                temperature: 0.7,
                timeout_ms: 30000,
                hallucination_threshold: 0.5,
                min_phi_for_llm: 0.0, // ConsciousPipeline handles gating
                enable_filtering: true,
            })
        }
        _ => {
            println!("ℹ️  Ollama not available (HDC+LTC will handle everything)");
            println!("   This is fine! LLM is only needed for Lost/Curious clarification.");
            None
        }
    }
}

/// Create the ConsciousPipeline
async fn create_pipeline(llm_config: Option<LlmConfig>) -> Result<ConsciousPipeline> {
    let config = PipelineConfig {
        consciousness_language: ConsciousnessLanguageConfig {
            phi_threshold: 0.4,
            confidence_threshold: 0.5,
            enable_curiosity: true,
            enable_adaptive_thresholds: true,
            track_journey: true,
            generate_explanations: true,
            ..Default::default()
        },
        llm: llm_config,
        memory_path: std::path::PathBuf::from("/tmp/symthaea_demo.db"),
        require_confirmation_below_phi: 0.3,
        adaptive_thresholds: true,
        max_context_turns: 10,
        ..Default::default()
    };

    // LLM is OPTIONAL - ConsciousPipeline will work without it
    ConsciousPipeline::new(config).await
}

/// Display the result from ConsciousPipeline
fn display_result(result: &PipelineResult) {
    println!("\n═══════════════════════════════════════════════════════════════");

    // Show understanding source (should be HDC+LTC most of the time!)
    let source_icon = match result.understanding_source {
        UnderstandingSource::HdcLtcPrimitives => "🧬",  // DNA = semantic primitives
        UnderstandingSource::LlmClarification => "💬",  // Speech = LLM helped
    };
    let source_name = match result.understanding_source {
        UnderstandingSource::HdcLtcPrimitives => "HDC+LTC+Primitives",
        UnderstandingSource::LlmClarification => "LLM Clarification",
    };
    println!("{} Understanding Source: {}", source_icon, source_name);

    // Show 2D consciousness space position
    println!("\n📊 2D Consciousness Space:");
    println!("   Φ (Integration):  {:.3}", result.phi);
    println!("   Confidence:       {:.3}", result.consciousness_understanding.epistemic_confidence);
    println!("   Free Energy:      {:.3}", result.free_energy);

    // Show quadrant
    let quadrant = result.consciousness_understanding.quadrant;
    let quadrant_desc = match quadrant {
        ConsciousnessQuadrant::Confident => "Confident (High Φ + High Confidence) - Deep understanding, execute with trust",
        ConsciousnessQuadrant::Curious => "Curious (High Φ + Low Confidence) - Deep processing, exploring",
        ConsciousnessQuadrant::Autopilot => "Autopilot (Low Φ + High Confidence) - Pattern-matched routine",
        ConsciousnessQuadrant::Lost => "Lost (Low Φ + Low Confidence) - Need clarification",
    };
    println!("   Quadrant:         {:?}", quadrant);
    println!("   → {}", quadrant_desc);

    // Show execution strategy
    println!("\n⚡ Execution Strategy:");
    match &result.execution_strategy {
        ExecutionStrategy::Confident { execute_immediately, explain_reasoning, .. } => {
            println!("   Strategy: CONFIDENT");
            println!("   → Execute immediately: {}", execute_immediately);
            println!("   → Explain reasoning: {}", explain_reasoning);
        }
        ExecutionStrategy::Curious { explore_first, targeted_questions, .. } => {
            println!("   Strategy: CURIOUS");
            println!("   → Explore first (dry-run): {}", explore_first);
            if !targeted_questions.is_empty() {
                println!("   → Targeted questions:");
                for q in targeted_questions.iter().take(3) {
                    println!("      • {}", q.question);
                }
            }
        }
        ExecutionStrategy::Autopilot { execute_efficiently, minimal_response, .. } => {
            println!("   Strategy: AUTOPILOT");
            println!("   → Execute efficiently: {}", execute_efficiently);
            println!("   → Minimal response: {}", minimal_response);
        }
        ExecutionStrategy::Lost { request_help, generic_questions, .. } => {
            println!("   Strategy: LOST");
            println!("   → Request help: {}", request_help);
            if !generic_questions.is_empty() {
                println!("   → Questions:");
                for q in generic_questions.iter().take(3) {
                    println!("      • {}", q.question);
                }
            }
        }
    }

    // Show NixOS understanding
    let nix = &result.nix_understanding;
    println!("\n🐧 NixOS Understanding:");
    println!("   Intent:      {:?}", nix.intent);
    println!("   Confidence:  {:.1}%", nix.confidence * 100.0);
    println!("   Description: {}", nix.description);
    if !nix.parameters.is_empty() {
        println!("   Parameters:");
        for (key, value) in &nix.parameters {
            println!("      {}: {}", key, value);
        }
    }

    // Show detected intent if any
    if let Some(ref intent) = result.intent {
        println!("\n🎯 Detected Action:");
        println!("   Type:     {:?}", intent.action_type);
        println!("   Safety:   {:?}", intent.safety_level);
        println!("   Allowed:  {}", if result.action_allowed { "✅ Yes" } else { "❌ No" });
        if let Some(ref cmd) = intent.command {
            let (program, args) = cmd.to_command();
            println!("   Command:  {} {}", program, args.join(" "));
        }
    }

    // Show LLM response if used (should be rare!)
    if let Some(ref llm_response) = result.llm_response {
        println!("\n💬 LLM Clarification (Lost/Curious fallback):");
        println!("   {}", llm_response.replace('\n', "\n   "));
    }

    // Show clarifying questions if any
    if !result.clarifying_questions.is_empty() {
        println!("\n❓ Clarifying Questions:");
        for q in &result.clarifying_questions {
            println!("   • {} (uncertainty: {:.0}%)", q.question, q.uncertainty * 100.0);
        }
    }

    // Show execution outcome if any
    if let Some(ref exec) = result.execution {
        println!("\n🔧 Execution:");
        println!("   Success: {}", exec.success);
        println!("   Output:  {}", exec.output);
        if let Some(ref err) = exec.error {
            println!("   Error:   {}", err);
        }
    }

    println!("\n   ⏱️  Processing time: {}ms", result.processing_time_ms);
    println!("═══════════════════════════════════════════════════════════════");
}

/// Show pipeline statistics
fn show_status(pipeline: &ConsciousPipeline) {
    let stats = pipeline.stats();
    let phi = pipeline.current_phi();

    println!("\n📊 Pipeline Statistics:");
    println!("   Current Φ: {:.3}", phi);
    println!();
    println!("   Requests:      {}", stats.total_requests);
    println!("   Actions allowed:  {}", stats.actions_allowed);
    println!("   Actions blocked:  {}", stats.actions_blocked);
    println!("   Pending confirmation: {}", stats.actions_pending_confirmation);
    println!("   Successful executions: {}", stats.successful_executions);
    println!("   Failed executions: {}", stats.failed_executions);
}

/// Show the consciousness journey from last result
fn show_journey(result: &PipelineResult) {
    let journey = &result.consciousness_understanding.journey;

    println!("\n🌊 Consciousness Journey:");
    println!("   {}", journey.narrative());
    println!();

    if journey.snapshots.is_empty() {
        println!("   No snapshots recorded.");
        return;
    }

    println!("   Snapshots:");
    for (i, snap) in journey.snapshots.iter().enumerate() {
        println!("   {}. Φ={:.3} Conf={:.3} [{:?}] - {}",
            i + 1, snap.phi, snap.confidence, snap.quadrant, snap.trigger);
    }

    println!();
    println!("   Transitions: {}", journey.transitions);
    println!("   Gained confidence: {}", journey.gained_confidence);
    println!("   Deepened integration: {}", journey.deepened_integration);
}

/// Format a quadrant indicator for the prompt
fn format_quadrant_indicator(phi: f64) -> &'static str {
    if phi >= 0.7 {
        "⬆️"  // High Φ
    } else if phi >= 0.4 {
        "➡️"  // Medium Φ
    } else {
        "⬇️"  // Low Φ
    }
}
