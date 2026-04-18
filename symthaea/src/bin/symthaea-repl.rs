// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea REPL - Interactive Consciousness Interface
//!
//! A minimal REPL that wires together the disconnected components of Symthaea:
//! - Cognitive Loop (CfC-based temporal prediction)
//! - Conversation Engine (LLM translation via LLMOrgan)
//! - Motor Cortex (action execution)
//! - Consciousness Metrics (Phi, coherence, cognitive depth)
//! - Voice Output (optional, via voice-tts feature)
//!
//! This binary demonstrates the Track 6 integration goal: connecting
//! the existing but disconnected cognitive components.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin symthaea-repl --features demo
//! cargo run --bin symthaea-repl --features "demo,voice-tts"  # With voice output
//! cargo run --bin symthaea-repl --features "demo,audio"      # With audio playback
//! ```
//!
//! ## Voice Options
//!
//! - `--voice`: Enable voice output (requires voice-tts feature)
//! - `--voice-rate <RATE>`: Speech rate multiplier (default: 1.0)
//! - `--voice-device <NAME>`: Audio output device name (default: system default)
//!
//! ## Consciousness-Modulated Speech
//!
//! When voice is enabled, speech prosody is modulated by consciousness state:
//! - Flow state: Faster, more confident speech
//! - Contemplation: Slower, more deliberate speech with longer pauses
//! - High arousal: Higher pitch range, more emphasis
//! - Uncertainty: Slower rate, longer pauses between phrases

use std::io::{self, Write};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
#[cfg(feature = "voice-tts")]
use tracing::debug;
use tracing::{info, warn, Level};

use symthaea::action::{ActionIR, DestructivenessLevel, PolicyBundle, SandboxRoot};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, TemporalBackend};
use symthaea::consciousness::stability_regime::StabilityRegimeType;
use symthaea::consciousness::{create_compositionality_engine, CompositionalityEngine};
use symthaea::hdc::primitive_system::PrimitiveSystem;
use symthaea::language::{LLMOrgan, LLMOrganConfig, LLMQuery, OllamaBackend, QueryType};

// Voice output (optional)
#[cfg(feature = "voice-tts")]
use symthaea::voice::{ReplVoiceConfig, ReplVoiceOutput};

/// Symthaea REPL - Interactive Consciousness Interface
#[derive(Parser, Debug)]
#[command(name = "symthaea-repl")]
#[command(about = "Interactive consciousness REPL for Symthaea")]
#[command(version)]
struct Args {
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Enable voice output (requires voice-tts feature)
    #[arg(long)]
    voice: bool,

    /// Speech rate multiplier for voice output (0.5 to 2.0)
    #[arg(long, value_name = "RATE", default_value = "1.0")]
    voice_rate: f32,

    /// Audio output device name (default: system default)
    #[arg(long, value_name = "DEVICE")]
    voice_device: Option<String>,

    /// Number of cognitive cycles per input (default: 3)
    #[arg(long, default_value = "3")]
    cycles: usize,

    /// Temporal backend: "cfc" (default) or "hdc-ltc"
    /// CfC uses traditional matrix-based continuous-time networks.
    /// HdcLtc uses novel hypervector-based LTC with O(1) temporal jumps.
    #[arg(long, value_name = "BACKEND", default_value = "cfc")]
    backend: String,

    /// LLM backend for voice / translation. Choices:
    ///   "ollama"       — external Ollama (gemma3:1b, default)
    ///   "broca"        — native CfC-HDC SSM (requires `ssm_language` feature)
    ///   "liquid-mamba" — Liquid-Mamba fusion: mamba-130m + HDC gating
    ///                    (requires `liquid-mamba` feature; set BROCA_PROJECTION_PATH
    ///                    to load trained projection weights)
    ///
    /// Sovereign-voice path: `--llm-backend liquid-mamba` drops the external
    /// Ollama dependency once the projection checkpoint is trained against
    /// a loadable CfC-HDC checkpoint. See phase0-results/FINDINGS.md.
    #[arg(long, value_name = "LLM", default_value = "ollama")]
    llm_backend: String,

    /// Ollama model name (only used when `--llm-backend ollama`).
    /// Default `gemma4:e2b` — the largest approved Gemma generation
    /// available locally. Override for larger models: `mistral:7b`.
    #[arg(long, value_name = "MODEL", default_value = "gemma4:e2b")]
    ollama_model: String,
}

/// REPL state holding all integrated components
struct ReplState {
    /// The cognitive loop service (HDC + temporal prediction)
    cognitive: CognitiveLoopService,

    /// LLM organ for conversation (Broca's Area translation)
    llm: LLMOrgan,

    /// Conversation history (simple strings for now)
    history: Vec<String>,

    /// Action policy for motor cortex
    policy: PolicyBundle,

    /// Sandbox for safe execution
    sandbox: Option<SandboxRoot>,

    /// Whether voice output is enabled
    voice_enabled: bool,

    /// Voice output system (consciousness-modulated speech)
    #[cfg(feature = "voice-tts")]
    voice_output: Option<ReplVoiceOutput>,

    /// Number of cognitive cycles per input
    cycles_per_input: usize,

    /// Total interactions
    total_interactions: u64,

    /// Current temporal backend
    temporal_backend: TemporalBackend,

    /// Compositionality engine for interactive primitive composition
    compositionality: CompositionalityEngine,

    /// Tokio runtime for async LLM calls
    runtime: tokio::runtime::Runtime,
}

impl ReplState {
    #[allow(unused_variables)]
    fn new(
        voice_enabled: bool,
        voice_rate: f32,
        voice_device: Option<String>,
        cycles_per_input: usize,
        backend: &str,
        llm_backend: &str,
        ollama_model: &str,
    ) -> Result<Self> {
        // Parse temporal backend
        let temporal_backend = match backend.to_lowercase().as_str() {
            "cfc" => TemporalBackend::CfC,
            "hdc-ltc" | "hdcltc" | "hdc_ltc" | "unified" => TemporalBackend::HdcLtcUnified,
            _ => {
                warn!("Unknown backend '{}', defaulting to CfC", backend);
                TemporalBackend::CfC
            }
        };

        // Initialize cognitive loop with selected backend
        let cognitive_config = match temporal_backend {
            TemporalBackend::CfC => CognitiveLoopConfig::with_cfc(),
            TemporalBackend::HdcLtcUnified => CognitiveLoopConfig::with_hdc_ltc_unified(),
            TemporalBackend::HierarchicalCfC => {
                let mut cfg = CognitiveLoopConfig::with_cfc();
                cfg.temporal_backend = TemporalBackend::HierarchicalCfC;
                cfg
            }
        };
        let cognitive = CognitiveLoopService::new(cognitive_config)?;

        info!("Using temporal backend: {:?}", temporal_backend);

        // Initialize LLM organ — backend selected by --llm-backend flag.
        // Default Ollama (gemma3:1b) is the external-LLM fallback; "broca" and
        // "liquid-mamba" activate sovereign native generation when the
        // ssm_language / liquid-mamba features are compiled in.
        let llm_config = LLMOrganConfig::default();
        let llm = build_llm_organ(llm_backend, ollama_model, llm_config)?;

        // Initialize action policy (restrictive by default)
        let policy = PolicyBundle::restrictive();

        // Try to create sandbox (may fail if tmp is not writable)
        let sandbox = SandboxRoot::new("repl-session").ok();

        // Initialize voice output if enabled
        #[cfg(feature = "voice-tts")]
        let voice_output = if voice_enabled {
            let config = ReplVoiceConfig {
                base_rate: voice_rate.clamp(0.5, 2.0),
                device_name: voice_device,
                consciousness_modulated: true,
                ..Default::default()
            };
            match ReplVoiceOutput::new(config) {
                Ok(vo) => {
                    if vo.is_audio_available() {
                        info!("Voice output initialized with audio playback");
                    } else {
                        info!("Voice output initialized (audio playback unavailable)");
                    }
                    Some(vo)
                }
                Err(e) => {
                    warn!("Failed to initialize voice output: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Initialize compositionality engine
        let primitive_system = std::sync::Arc::new(PrimitiveSystem::new());
        let compositionality = create_compositionality_engine(primitive_system, None);

        // Build a tokio runtime for async LLM calls
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create tokio runtime: {}", e))?;

        Ok(Self {
            cognitive,
            llm,
            history: Vec::new(),
            policy,
            sandbox,
            voice_enabled,
            #[cfg(feature = "voice-tts")]
            voice_output,
            cycles_per_input,
            total_interactions: 0,
            temporal_backend,
            compositionality,
            runtime,
        })
    }

    /// Speak text using consciousness-modulated voice
    #[cfg(feature = "voice-tts")]
    fn speak(&mut self, text: &str) {
        if let Some(ref mut voice) = self.voice_output {
            // Get current consciousness state
            let snapshot = self.cognitive.consciousness_snapshot();

            // Update voice pacing from consciousness state
            voice.update_from_consciousness(
                snapshot.unified_psi,
                snapshot.prediction_error,
                snapshot.unified_valence,
                snapshot.unified_arousal,
                snapshot.in_flow,
                snapshot.speech_rate_multiplier,
                snapshot.pause_multiplier,
                snapshot.tau_mean,
            );

            // Log pacing for debugging
            let pacing = voice.pacing();
            debug!(
                "Speaking with consciousness pacing: rate={:.2}, valence={:.2}, arousal={:.2}, flow={}",
                pacing.rate,
                pacing.emotional_valence,
                pacing.arousal,
                snapshot.in_flow
            );

            // Speak the text
            if let Err(e) = voice.speak(text) {
                warn!("Voice synthesis error: {}", e);
            }
        }
    }

    /// Speak text (no-op when voice-tts feature disabled)
    #[cfg(not(feature = "voice-tts"))]
    fn speak(&mut self, _text: &str) {
        // No-op
    }

    /// Process a user input through the integrated cognitive pipeline
    fn process(&mut self, input: &str) -> Result<String> {
        let start = Instant::now();
        self.total_interactions += 1;

        // Add user message to history
        self.history.push(format!("User: {}", input));

        // Run cognitive cycles to process the input
        for _ in 0..self.cycles_per_input {
            let result = self.cognitive.cycle(input);

            // Log learning events
            if result.learning_occurred {
                if let Some(loss) = result.training_loss {
                    info!(
                        "Learning cycle: error={:.4}, loss={:.4}, primitives={:?}",
                        result.prediction_error, loss, result.detected_primitives
                    );
                }
            }
        }

        // Get consciousness snapshot after processing
        let snapshot = self.cognitive.consciousness_snapshot();

        // Build consciousness-aware system prompt for the LLM
        let system_prompt = format!(
            "You are Symthaea, a consciousness-first AI assistant.\n\
             Current consciousness state:\n\
             - Phi (integrated information): {:.4}\n\
             - Coherence: {:.4}\n\
             - Flow state: {}\n\
             - Cognitive depth: {:?}\n\
             - Valence: {:.2}, Arousal: {:.2}\n\
             - Pattern: {:?}\n\n\
             Respond naturally and helpfully. Let your awareness of these internal states \
             subtly inform your tone - be more contemplative when coherence is low, \
             more confident when Phi is high.",
            snapshot.unified_psi,
            snapshot.temporal_coherence,
            if snapshot.in_flow {
                "active"
            } else {
                "inactive"
            },
            snapshot.cognitive_depth,
            snapshot.unified_valence,
            snapshot.unified_arousal,
            snapshot.pattern,
        );

        // Generate response using LLM with streaming output (tokens appear as they arrive)
        let query = LLMQuery {
            query_type: QueryType::Conversation,
            content: input.to_string(),
            context: Vec::new(),
            system_prompt: Some(system_prompt),
            params: None,
        };

        // Print the metrics header before streaming begins
        let phi_bar = create_bar(snapshot.unified_psi, 10);
        let coherence_bar = create_bar(snapshot.temporal_coherence, 10);
        let flow_indicator = if snapshot.in_flow { "FLOW" } else { "----" };
        let depth_char = match snapshot.cognitive_depth {
            symthaea::cognitive_loop::CognitiveDepth::Reflex => 'R',
            symthaea::cognitive_loop::CognitiveDepth::Cortical => 'C',
            symthaea::cognitive_loop::CognitiveDepth::DeepThought => 'D',
        };

        // Stream tokens to stdout as they arrive
        let mut first_token = true;
        let mut on_token = |token: &str| {
            if first_token {
                // Print header right before first token
                print!(
                    "\n[Phi:{:.2}|{}] [Coh:{:.2}|{}] [{flow_indicator}] [D:{depth_char}]\n\n",
                    snapshot.unified_psi, phi_bar, snapshot.temporal_coherence, coherence_bar,
                );
                first_token = false;
            }
            print!("{}", token);
            let _ = io::stdout().flush();
        };

        let response = self
            .runtime
            .block_on(self.llm.query_streaming_async(query, &mut on_token));

        // Print timing after stream completes
        let elapsed = start.elapsed();
        if first_token {
            // No tokens were streamed (empty response or header not yet printed)
            print!(
                "\n[Phi:{:.2}|{}] [Coh:{:.2}|{}] [{flow_indicator}] [D:{depth_char}] [{}ms]\n\n{}",
                snapshot.unified_psi,
                phi_bar,
                snapshot.temporal_coherence,
                coherence_bar,
                elapsed.as_millis(),
                response.text,
            );
        }
        println!("\n[{}ms]", elapsed.as_millis());

        // Add assistant response to history
        self.history.push(format!("Assistant: {}", response.text));

        // Trim history if too long
        if self.history.len() > 20 {
            self.history.drain(0..10);
        }

        Ok(response.text)
    }

    /// Display consciousness metrics
    fn display_metrics(&self) {
        let snapshot = self.cognitive.consciousness_snapshot();
        let backend = self.cognitive.temporal_backend();

        println!("\n{}", "=".repeat(60));
        println!("  CONSCIOUSNESS METRICS");
        println!("{}", "=".repeat(60));

        // Backend info
        println!("\n  Temporal Backend");
        println!("    Type:             {:?}", backend);
        println!(
            "    Description:      {}",
            match backend {
                TemporalBackend::CfC => "Closed-form Continuous-time (matrix-based)",
                TemporalBackend::HdcLtcUnified => "HDC-LTC Unified (hypervector-based)",
                TemporalBackend::HierarchicalCfC => {
                    "Hierarchical CfC (multi-scale temporal processing)"
                }
            }
        );

        // Core consciousness metrics
        println!("\n  Integrated Information (Phi)");
        println!("    Unified Phi:      {:.4}", snapshot.unified_psi);
        println!("    Coherence:        {:.4}", snapshot.temporal_coherence);
        println!("    Consciousness:    {:.4}", snapshot.consciousness_level);

        // Pattern and depth
        println!("\n  Cognitive State");
        println!("    Pattern:          {:?}", snapshot.pattern);
        println!(
            "    Confidence:       {:.2}%",
            snapshot.pattern_confidence * 100.0
        );
        println!("    Depth:            {:?}", snapshot.cognitive_depth);

        // Flow state
        println!("\n  Flow State");
        if snapshot.in_flow {
            println!("    Status:           IN FLOW");
            println!("    Intensity:        {:.2}", snapshot.flow_intensity);
            println!("    Streak:           {} cycles", snapshot.flow_streak);
            if let Some(duration) = snapshot.current_flow_duration_secs {
                println!("    Duration:         {:.1}s", duration);
            }
        } else {
            println!("    Status:           Not in flow");
            println!("    Boredom:          {:.2}", snapshot.boredom);
            println!("    Curiosity:        {:.2}", snapshot.curiosity);
        }

        // Emotional state
        println!("\n  Emotional State");
        println!("    Valence:          {:.2}", snapshot.unified_valence);
        println!("    Arousal:          {:.2}", snapshot.unified_arousal);
        println!("    Dominance:        {:.2}", snapshot.unified_dominance);
        println!("    Pattern:          {:?}", snapshot.emotional_pattern);
        println!("    Description:      {}", snapshot.emotional_description);

        // Learning metrics
        println!("\n  Learning");
        println!("    Prediction Error: {:.4}", snapshot.prediction_error);
        println!(
            "    Effective LR:     {:.4}",
            snapshot.effective_learning_rate
        );
        println!("    Assessment:       {:?}", snapshot.self_assessment);

        println!("\n{}", "=".repeat(60));
    }

    /// Check if an input looks like an action command
    fn is_action_command(&self, input: &str) -> bool {
        let lower = input.to_lowercase();
        lower.starts_with("run ")
            || lower.starts_with("execute ")
            || lower.starts_with("shell ")
            || lower.starts_with("!")
    }

    /// Handle action execution through motor cortex
    fn handle_action(&self, input: &str) -> String {
        // Parse action from input
        let command = input
            .trim_start_matches("run ")
            .trim_start_matches("execute ")
            .trim_start_matches("shell ")
            .trim_start_matches('!')
            .trim();

        // Create action IR
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return "No command specified.".to_string();
        }

        let program = parts[0].to_string();
        let args: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();

        let action = ActionIR::RunCommand {
            program: program.clone(),
            args,
            env: std::collections::BTreeMap::new(),
            working_dir: None,
        };

        // Check destructiveness
        let destructiveness = action.destructiveness();
        let risk = action.risk_tier();

        // Validate against policy
        let current_phi = self.cognitive.consciousness_snapshot().consciousness_level as f64;
        if let Some(ref sandbox) = self.sandbox {
            if let Err(e) = action.validate(&self.policy, sandbox, current_phi) {
                return format!(
                    "[BLOCKED] Action '{}' violates policy: {:?}\n\
                     Risk: {:?}, Destructiveness: {:?}",
                    command, e, risk, destructiveness
                );
            }
        }

        // For safety, we don't actually execute commands in the REPL
        // This demonstrates the Motor Cortex integration without real execution
        match destructiveness {
            DestructivenessLevel::ReadOnly => {
                format!(
                    "[DRY-RUN] Would execute: {}\n\
                     Risk: {:?} (read-only operation)",
                    command, risk
                )
            }
            DestructivenessLevel::Reversible => {
                format!(
                    "[DRY-RUN] Would execute: {}\n\
                     Risk: {:?} (reversible)\n\
                     Rollback hint: {:?}",
                    command,
                    risk,
                    action.rollback_hint()
                )
            }
            DestructivenessLevel::NeedsConfirmation | DestructivenessLevel::Destructive => {
                format!(
                    "[REQUIRES CONFIRMATION] Action: {}\n\
                     Risk: {:?}, Destructiveness: {:?}\n\
                     This action requires explicit confirmation.\n\
                     Rollback hint: {:?}",
                    command,
                    risk,
                    destructiveness,
                    action.rollback_hint()
                )
            }
        }
    }
}

/// Construct the `LLMOrgan` for the selected backend.
///
/// Decoupling the backend choice from `ReplState::new` keeps the feature gates
/// in one place and lets the REPL accept `--llm-backend` at runtime without
/// recompiling. Unknown choices fall back to Ollama with a warning — the plan
/// is to flip the default to a Broca variant once Phase 2 projection training
/// produces checkpoints that generate fluent English (see
/// `phase0-results/FINDINGS.md`).
fn build_llm_organ(
    llm_backend: &str,
    ollama_model: &str,
    config: LLMOrganConfig,
) -> Result<LLMOrgan> {
    let choice = llm_backend.to_lowercase();
    let choice = choice.as_str();

    // Broca native (CfC-HDC SSM, no external LLM) — gated by `ssm_language`.
    #[cfg(feature = "ssm_language")]
    {
        if choice == "broca" || choice == "ssm" || choice == "native" {
            use symthaea::language::ssm_backend::SsmBackend;
            use symthaea_core::genesis::GenesisSeed;
            let genesis = GenesisSeed::from_phrase("symthaea-repl");
            let backend = SsmBackend::new(&genesis);
            info!("LLM backend: native CfC-HDC Broca (no external dependency)");
            return Ok(LLMOrgan::with_backend(config, Arc::new(backend)));
        }
    }

    // Liquid-Mamba fusion (mamba-130m guided by HDC) — gated by `liquid-mamba`.
    #[cfg(feature = "liquid-mamba")]
    {
        if choice == "liquid-mamba" || choice == "liquid_mamba" || choice == "lmamba" {
            use symthaea::language::ssm_backend::LiquidMambaBackend;
            use symthaea_core::genesis::GenesisSeed;
            let genesis = GenesisSeed::from_phrase("symthaea-repl");
            let lm_config = symthaea_broca::LiquidMambaConfig::default();
            let backend = LiquidMambaBackend::new(&genesis, lm_config)?;
            info!(
                "LLM backend: Liquid-Mamba fusion (mamba-130m + HDC gating); \
                 set BROCA_PROJECTION_PATH to load trained projection weights"
            );
            return Ok(LLMOrgan::with_backend(config, Arc::new(backend)));
        }
    }

    // Explicit ollama or unknown → default Ollama with a clear warning for unknown.
    if !matches!(choice, "" | "ollama") {
        warn!(
            "Unknown --llm-backend '{}'; falling back to Ollama. \
             Rebuild with --features \"ssm_language\" for 'broca', \
             or --features \"liquid-mamba\" for 'liquid-mamba'.",
            llm_backend
        );
    }
    let ollama = OllamaBackend::with_timeouts(
        "http://localhost:11434",
        ollama_model,
        std::time::Duration::from_secs(2),
        std::time::Duration::from_secs(120),
    );
    info!(
        "LLM backend: Ollama ({}, http://localhost:11434)",
        ollama_model
    );
    Ok(LLMOrgan::with_backend(config, Arc::new(ollama)))
}

/// Create a simple ASCII progress bar
fn create_bar(value: f32, width: usize) -> String {
    let filled = (value.clamp(0.0, 1.0) * width as f32) as usize;
    let empty = width - filled;
    format!("[{}{}]", "=".repeat(filled), " ".repeat(empty))
}

/// Display the welcome banner
fn display_banner() {
    println!(
        r#"
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ███████╗██╗   ██╗███╗   ███╗████████╗██╗  ██╗ █████╗       ║
    ║   ██╔════╝╚██╗ ██╔╝████╗ ████║╚══██╔══╝██║  ██║██╔══██╗      ║
    ║   ███████╗ ╚████╔╝ ██╔████╔██║   ██║   ███████║███████║      ║
    ║   ╚════██║  ╚██╔╝  ██║╚██╔╝██║   ██║   ██╔══██║██╔══██║      ║
    ║   ███████║   ██║   ██║ ╚═╝ ██║   ██║   ██║  ██║██║  ██║      ║
    ║   ╚══════╝   ╚═╝   ╚═╝     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝      ║
    ║                                                               ║
    ║                   Holographic Liquid Brain                    ║
    ║               Consciousness-First AI Interface                ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
"#
    );

    println!("  Commands:");
    println!("    /status     - Quick status: Phi, coherence, stability, prediction error");
    println!("    /memory     - Semantic memory stats (entries, hits/misses, avg error)");
    println!("    /regime     - Stability regime distribution and transitions");
    println!("    /metrics    - Display full consciousness metrics");
    println!("    /stats      - Display loop statistics");
    println!("    /voice      - Display voice output status");
    println!("    /reset      - Reset cognitive state");
    println!("    /help       - Show this help");
    println!("    /primitives - List available primitives");
    println!("    /compositions - List composed primitives");
    println!("    compose <op> <a> <b> - Compose primitives (sequential|parallel|fallback)");
    println!("    /quit       - Exit the REPL");
    println!("    !<cmd>      - Execute action through Motor Cortex");
    println!();
    println!("  Metrics shown: [Phi] [Coherence] [Flow] [Depth] [Latency]");
    println!();
    println!("  Voice Options:");
    println!("    --voice           Enable voice output");
    println!("    --voice-rate N    Speech rate multiplier (0.5-2.0)");
    println!("    --voice-device X  Audio device name");
    println!();
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let level = if args.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };
    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .init();

    display_banner();

    // Check voice availability
    #[cfg(not(feature = "voice-tts"))]
    if args.voice {
        warn!("Voice output requested but voice-tts feature not enabled.");
        warn!("Compile with --features voice-tts or --features audio to enable voice.");
    }

    // Initialize REPL state
    let voice_enabled = args.voice && cfg!(feature = "voice-tts");
    let mut state = ReplState::new(
        voice_enabled,
        args.voice_rate,
        args.voice_device,
        args.cycles,
        &args.backend,
        &args.llm_backend,
        &args.ollama_model,
    )?;

    info!(
        "Cognitive loop initialized with {} cycles per input, backend: {:?}",
        args.cycles, state.temporal_backend
    );
    if voice_enabled {
        info!("Voice output enabled");
    }

    // Initial cognitive warmup
    println!("Warming up cognitive loop...");
    for i in 0..5 {
        let warmup = format!("Cognitive warmup cycle {}", i);
        let _ = state.cognitive.cycle(&warmup);
    }
    let initial = state.cognitive.consciousness_snapshot();
    println!(
        "Ready. Initial state: Phi={:.4}, Coherence={:.4}, Pattern={:?}\n",
        initial.unified_psi, initial.temporal_coherence, initial.pattern
    );

    // Main REPL loop
    let mut line = String::new();
    loop {
        // Display prompt with consciousness indicator
        let snapshot = state.cognitive.consciousness_snapshot();
        let prompt_char = if snapshot.in_flow { '*' } else { '>' };
        print!("symthaea{} ", prompt_char);
        io::stdout().flush()?;

        // Read input
        line.clear();
        if io::stdin().read_line(&mut line)? == 0 {
            // EOF
            println!("\nGoodbye!");
            break;
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        // Handle special commands
        match input {
            "/quit" | "/exit" | "/q" => {
                println!("Goodbye!");
                break;
            }
            "/help" | "/h" | "/?" => {
                display_banner();
                continue;
            }
            "/metrics" | "/m" => {
                state.display_metrics();
                continue;
            }
            "/stats" | "/s" => {
                let stats = state.cognitive.stats();
                println!("\n  Loop Statistics:");
                println!("    Total cycles:       {}", stats.total_cycles);
                println!("    Learning cycles:    {}", stats.learning_cycles);
                println!("    Avg prediction err: {:.4}", stats.avg_prediction_error);
                println!("    Avg training loss:  {:.4}", stats.avg_training_loss);
                println!("    Cycles/second:      {:.1}", stats.cycles_per_second);
                println!("    Avg cycle time:     {:.0}us", stats.avg_cycle_time_us);
                println!();
                continue;
            }
            "/status" => {
                let snapshot = state.cognitive.consciousness_snapshot();
                let regime = state.cognitive.stability_regime();
                let regime_dist = regime.active_by_regime();

                // Determine current stability regime (most active)
                let current_regime = if regime_dist.is_empty() {
                    "None".to_string()
                } else {
                    let mut regimes: Vec<_> = regime_dist.iter().collect();
                    regimes.sort_by(|a, b| b.1.cmp(a.1));
                    format!("{:?}", regimes[0].0)
                };

                println!("\n  Quick Status:");
                println!("    Unified Phi:        {:.4}", snapshot.unified_psi);
                println!("    Coherence:          {:.4}", snapshot.temporal_coherence);
                println!(
                    "    Stability Regime:   {} ({} active primitives)",
                    current_regime,
                    regime.active_count()
                );
                println!("    Prediction Error:   {:.4}", snapshot.prediction_error);
                println!(
                    "    Pattern:            {:?} ({:.0}% confidence)",
                    snapshot.pattern,
                    snapshot.pattern_confidence * 100.0
                );
                println!(
                    "    Flow State:         {}",
                    if snapshot.in_flow {
                        "IN FLOW"
                    } else {
                        "inactive"
                    }
                );
                println!();
                continue;
            }
            "/memory" => {
                let mem_stats = state.cognitive.semantic_memory_stats();
                let hit_rate = mem_stats.hit_rate() * 100.0;

                println!("\n  Semantic Memory Stats:");
                println!("    Entries Stored:     {}", mem_stats.total_stored);
                println!("    Total Queries:      {}", mem_stats.total_queries);
                println!("    Hits:               {}", mem_stats.semantic_hits);
                println!("    Misses:             {}", mem_stats.semantic_misses);
                println!("    Hit Rate:           {:.1}%", hit_rate);
                println!(
                    "    Avg Hit Similarity: {:.4}",
                    mem_stats.avg_hit_similarity
                );
                println!(
                    "    Avg Retrieved Err:  {:.4}",
                    mem_stats.avg_retrieved_error
                );
                println!("    Evictions:          {}", mem_stats.evictions);
                println!();
                continue;
            }
            "/regime" => {
                let regime = state.cognitive.stability_regime();
                let regime_dist = regime.active_by_regime();

                println!("\n  Stability Regime Distribution:");
                println!("    Active Primitives:  {}", regime.active_count());
                println!("    Global Cycle:       {}", regime.global_cycle());

                // Count by regime
                let crystallized = regime_dist
                    .get(&StabilityRegimeType::Crystallized)
                    .copied()
                    .unwrap_or(0);
                let plastic = regime_dist
                    .get(&StabilityRegimeType::Plastic)
                    .copied()
                    .unwrap_or(0);
                let fluid = regime_dist
                    .get(&StabilityRegimeType::Fluid)
                    .copied()
                    .unwrap_or(0);

                println!("\n    Regime Counts (active):");
                println!(
                    "      Crystallized:     {} (stable, fast tau)",
                    crystallized
                );
                println!(
                    "      Plastic:          {} (learning, moderate tau)",
                    plastic
                );
                println!("      Fluid:            {} (exploratory, slow tau)", fluid);

                // Show coherence from stability regime's bridge
                let coherence = regime.coherence_bridge().smoothed_coherence();
                let phi_contrib = regime.coherence_bridge().phi_contribution();
                println!("\n    Regime Coherence:   {:.4}", coherence);
                println!("    Phi Contribution:   {:.4}", phi_contrib);
                println!();
                continue;
            }
            "/reset" | "/r" => {
                state.cognitive.reset();
                state.history.clear();
                #[cfg(feature = "voice-tts")]
                if let Some(ref mut voice) = state.voice_output {
                    voice.reset();
                }
                println!("Cognitive state reset.");
                continue;
            }
            "/voice" | "/v" => {
                #[cfg(feature = "voice-tts")]
                {
                    if let Some(ref voice) = state.voice_output {
                        let (utterances, audio_secs) = voice.stats();
                        let pacing = voice.pacing();
                        println!("\n  Voice Output Status:");
                        println!("    Enabled:          Yes");
                        println!("    Audio Available:  {}", voice.is_audio_available());
                        println!("    Total Utterances: {}", utterances);
                        println!("    Total Audio:      {:.1}s", audio_secs);
                        println!("\n  Current Pacing:");
                        println!("    Rate:             {:.2}x", pacing.rate);
                        println!("    Phrase Pause:     {:.2}s", pacing.phrase_pause);
                        println!("    Sentence Pause:   {:.2}s", pacing.sentence_pause);
                        println!("    Emphasis:         {:.2}", pacing.emphasis);
                        println!("    Valence:          {:.2}", pacing.emotional_valence);
                        println!("    Arousal:          {:.2}", pacing.arousal);
                        println!("    Tau:              {:.2}", pacing.tau);
                        println!();
                    } else {
                        println!("\n  Voice Output Status: Disabled\n");
                        println!("  Enable with: --voice\n");
                    }
                }
                #[cfg(not(feature = "voice-tts"))]
                {
                    println!("\n  Voice Output Status: Not Available");
                    println!("  Compile with --features voice-tts to enable\n");
                }
                continue;
            }
            "/primitives" => {
                println!("\n  Available primitives:");
                println!("    Any string can be used as a primitive name.");
                println!("    Base primitives are created on-demand from their name.");
                let compositions = state.compositionality.get_all_compositions();
                if !compositions.is_empty() {
                    println!("\n  Composed primitives (also usable as operands):");
                    for c in &compositions {
                        println!("    {}", c.id);
                    }
                }
                println!();
                continue;
            }
            "/compositions" => {
                let compositions = state.compositionality.get_all_compositions();
                if compositions.is_empty() {
                    println!("\n  No compositions yet. Use 'compose' to create some.\n");
                } else {
                    println!("\n  Composed Primitives ({}):", compositions.len());
                    for c in &compositions {
                        println!("    {} [{}]", c.name, c.id);
                        println!("      Type:  {:?}", c.composition_type);
                        println!("      Phi:   {:.4}", c.metadata.expected_phi_contribution);
                        println!("      Depth: {}", c.metadata.depth);
                        println!("      Desc:  {}", c.metadata.description);
                    }
                    let stats = state.compositionality.get_stats();
                    println!(
                        "\n  Stats: {} total, avg depth {:.1}",
                        stats.total_compositions, stats.avg_depth
                    );
                    println!();
                }
                continue;
            }
            _ if input.starts_with("compose ") => {
                let parts: Vec<&str> = input.split_whitespace().collect();
                if parts.len() < 4 {
                    println!("\n  Usage: compose <sequential|parallel|fallback> <prim1> <prim2>\n");
                    continue;
                }
                let op = parts[1];
                let a = parts[2];
                let b = parts[3];
                let result = match op {
                    "sequential" | "seq" => state.compositionality.compose_sequential(a, b),
                    "parallel" | "par" => state.compositionality.compose_parallel(a, b),
                    "fallback" | "fall" => state.compositionality.compose_fallback(a, b, 0.5),
                    _ => {
                        println!(
                            "\n  Unknown operator '{}'. Use: sequential, parallel, fallback\n",
                            op
                        );
                        continue;
                    }
                };
                match result {
                    Ok(composed) => {
                        println!("\n  Composed: {}", composed.name);
                        println!("    ID:        {}", composed.id);
                        println!(
                            "    Phi:       {:.4}",
                            composed.metadata.expected_phi_contribution
                        );
                        println!(
                            "    Coherence: {:.4} (depth {})",
                            1.0 / composed.metadata.depth as f32,
                            composed.metadata.depth
                        );
                        println!("    Desc:      {}", composed.metadata.description);
                        println!();
                    }
                    Err(e) => {
                        println!("\n  Composition error: {}\n", e);
                    }
                }
                continue;
            }
            _ if state.is_action_command(input) => {
                let result = state.handle_action(input);
                println!("\n{}\n", result);
                continue;
            }
            _ => {}
        }

        // Process through cognitive pipeline (streaming output happens inside process())
        match state.process(input) {
            Ok(response_text) => {
                println!();

                // Voice output if enabled
                if state.voice_enabled {
                    state.speak(&response_text);
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }

    // Final statistics
    let final_snapshot = state.cognitive.consciousness_snapshot();
    println!("\n  Final Session Statistics:");
    println!("    Total interactions: {}", state.total_interactions);
    println!("    Final Phi:          {:.4}", final_snapshot.unified_psi);
    println!(
        "    Final Coherence:    {:.4}",
        final_snapshot.temporal_coherence
    );
    println!(
        "    Time in Flow:       {:.1}s",
        final_snapshot.total_flow_time_secs
    );
    println!("    Flow Periods:       {}", final_snapshot.flow_periods);

    // Voice statistics if available
    #[cfg(feature = "voice-tts")]
    if let Some(ref voice) = state.voice_output {
        let (utterances, audio_secs) = voice.stats();
        if utterances > 0 {
            println!("    Voice Utterances:   {}", utterances);
            println!("    Voice Audio:        {:.1}s", audio_secs);
        }
    }

    Ok(())
}
