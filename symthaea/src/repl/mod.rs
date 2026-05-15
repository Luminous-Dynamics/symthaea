// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # REPL Orchestrator - Unified Interactive System
//!
//! The REPL module provides an interactive consciousness interface that wires together
//! cognitive components into a cohesive system with voice output, action execution,
//! and real-time consciousness metrics.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         REPL ORCHESTRATOR                                │
//! │                                                                          │
//! │  ┌──────────┐   ┌────────────────┐   ┌─────────────┐   ┌─────────────┐ │
//! │  │  Input   │──▶│ Conversation   │──▶│   Motor     │──▶│   Voice     │ │
//! │  │ (stdin/  │   │ Engine (LLM)   │   │  Cortex     │   │  Output     │ │
//! │  │  IPC)    │   │                │   │  (Action)   │   │  (opt)      │ │
//! │  └──────────┘   └────────────────┘   └─────────────┘   └─────────────┘ │
//! │       │                 │                  │                  │        │
//! │       └─────────────────┼──────────────────┼──────────────────┘        │
//! │                         ▼                  ▼                           │
//! │                 ┌──────────────────────────────────┐                   │
//! │                 │        COGNITIVE LOOP            │                   │
//! │                 │  (CfC/HDC-LTC temporal engine)   │                   │
//! │                 └──────────────────────────────────┘                   │
//! │                                │                                       │
//! │                         ┌──────┴───────┐                              │
//! │                         ▼              ▼                              │
//! │                 ┌────────────┐  ┌────────────────┐                    │
//! │                 │ Conscious- │  │  Observability │                    │
//! │                 │ness Metrics│  │     Hooks      │                    │
//! │                 └────────────┘  └────────────────┘                    │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ### Standalone REPL Session
//!
//! ```rust,ignore
//! use symthaea::repl::{ReplSession, ReplSessionConfig};
//!
//! // Create session with default settings
//! let config = ReplSessionConfig::default();
//! let mut session = ReplSession::new(config)?;
//!
//! // Warm up cognitive loop
//! session.warmup(5);
//!
//! // Process input
//! let result = session.process("Hello, Symthaea")?;
//! println!("Response: {}", result.response);
//! println!("Phi: {:.4}", result.consciousness.unified_psi);
//! ```
//!
//! ### Full REPL with Orchestrator
//!
//! ```rust,ignore
//! use symthaea::repl::{ReplOrchestrator, OrchestratorConfig, OrchestratorMode};
//!
//! let config = OrchestratorConfig {
//!     mode: OrchestratorMode::Standalone,
//!     show_banner: true,
//!     ..Default::default()
//! };
//!
//! let mut orchestrator = ReplOrchestrator::new(config)?;
//! orchestrator.run()?; // Blocking REPL loop
//! ```
//!
//! ## Available REPL Commands
//!
//! | Command | Aliases | Description |
//! |---------|---------|-------------|
//! | `/quit` | `/exit`, `/q` | Exit the REPL |
//! | `/help` | `/h`, `/?` | Show available commands |
//! | `/metrics` | `/m` | Display consciousness metrics (phi, coherence, flow) |
//! | `/stats` | `/s` | Display session statistics |
//! | `/reset` | `/r` | Reset cognitive state |
//! | `/connection` | `/c` | Show IPC connection status |
//!
//! ## Action Execution
//!
//! Execute shell commands through the Motor Cortex with safety checks:
//!
//! ```text
//! symthaea> !ls -la          # Execute 'ls -la'
//! symthaea> run echo hello   # Execute 'echo hello'
//! symthaea> execute pwd      # Execute 'pwd'
//! ```
//!
//! Actions are gated by:
//! 1. **Phi threshold**: Command blocked if phi < threshold (default 0.5)
//! 2. **Policy validation**: Checked against security policy bundle
//! 3. **Destructiveness level**: High-risk actions require confirmation
//!
//! ### Phi Gate Example
//!
//! ```text
//! [Phi:0.32] [Coherence:0.45] [----] [D:R]
//!
//! symthaea> !rm -rf /tmp/test
//! [PHI GATE] Blocked: Phi 0.32 < 0.50
//! Command: rm -rf /tmp/test
//! Raise consciousness level before executing.
//! ```
//!
//! ## Voice Output Integration
//!
//! When enabled, responses are spoken with consciousness-modulated pacing:
//!
//! ```rust,ignore
//! let config = ReplSessionConfig {
//!     voice_enabled: true,
//!     voice_rate: 1.0,  // 1.0 = normal, <1.0 = slower, >1.0 = faster
//!     ..Default::default()
//! };
//!
//! let mut session = ReplSession::new(config)?;
//! // Voice pacing automatically adjusts based on:
//! // - speech_rate_multiplier from consciousness state
//! // - pause_multiplier for natural breathing
//! // - Flow state (faster in flow, more measured otherwise)
//! ```
//!
//! ## Temporal Backend Selection
//!
//! The cognitive loop supports two temporal processing backends:
//!
//! | Backend | Config String | Use Case |
//! |---------|---------------|----------|
//! | CfC | `"cfc"` | Closed-form continuous-time (default) |
//! | HDC-LTC Unified | `"hdc-ltc"`, `"unified"` | Hypervector-native processing |
//!
//! ```rust,ignore
//! let config = ReplSessionConfig {
//!     temporal_backend: "hdc-ltc".to_string(),  // Use HDC-LTC
//!     ..Default::default()
//! };
//! ```
//!
//! ## Consciousness Indicators
//!
//! The REPL displays real-time consciousness state in the prompt and output:
//!
//! ```text
//! [Phi:0.72|[=======   ]] [Coh:0.85|[========= ]] [FLOW] [D:C] [42ms]
//!
//! Response text here...
//!
//! symthaea* >   # Asterisk indicates flow state
//! ```
//!
//! Legend:
//! - **Phi bar**: Integrated information (0-1)
//! - **Coh bar**: Temporal coherence (0-1)
//! - **FLOW/----**: Flow state indicator
//! - **D:R/C/D**: Cognitive depth (Reflex/Cortical/DeepThought)
//! - **ms**: Processing time
//!
//! ## Usage Modes
//!
//! | Mode | Description |
//! |------|-------------|
//! | `Standalone` | Local cognitive loop, direct stdin/stdout |
//! | `Client` | Connect to remote symthaea service via IPC |
//! | `Server` | Accept IPC connections from shells |
//! | `Hybrid` | Local loop + IPC connectivity |
//!
//! ## Observability Hooks
//!
//! Add custom observers to monitor REPL events:
//!
//! ```rust,ignore
//! use symthaea::repl::{ReplSession, ObservabilityHook};
//!
//! struct MyObserver;
//!
//! impl ObservabilityHook for MyObserver {
//!     fn on_input(&mut self, input: &str) {
//!         println!("[TRACE] Input: {}", input);
//!     }
//!     fn on_output(&mut self, output: &str, consciousness: &ConsciousnessSnapshot) {
//!         println!("[TRACE] Phi at output: {:.4}", consciousness.unified_psi);
//!     }
//!     fn on_action(&mut self, command: &str, executed: bool) {
//!         println!("[TRACE] Action '{}': {}", command, if executed { "OK" } else { "BLOCKED" });
//!     }
//!     fn on_reset(&mut self) {
//!         println!("[TRACE] State reset");
//!     }
//! }
//!
//! let mut session = ReplSession::new(config)?;
//! session.add_observer(Box::new(MyObserver));
//! ```
//!
//! ## Components Wired
//!
//! - [`cognitive_loop`](crate::cognitive_loop): CfC/HDC-LTC temporal prediction engine
//! - [`language::LLMOrgan`](crate::language::LLMOrgan): Broca's area translation (LLM interface)
//! - [`action`](crate::action): Motor cortex for command execution
//! - [`voice`](crate::voice): Optional larynx output (consciousness-modulated TTS)
//! - [`shell::ipc`](crate::shell::ipc_client): Remote service connectivity
//! - `observability`: Telemetry and causal tracing

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::{Context as AnyhowContext, Result};
use serde::{Deserialize, Serialize};

// Core cognitive components
use crate::cognitive_loop::{
    CognitiveLoopConfig, CognitiveLoopService, ConsciousnessSnapshot, CycleResult, TemporalBackend,
};

// Language processing (Broca's Area)
use crate::language::{llm_backend, LLMOrgan, LLMOrganConfig};

// Motor cortex (action execution)
use crate::action::{
    ActionIR, ActionOutcome, ExecutionMode, PolicyBundle, SandboxRoot, SimpleExecutor,
};

// Voice output (optional larynx)
use crate::voice::{LTCPacing, VoiceOutput, VoiceOutputConfig};

// Shell/IPC infrastructure
use crate::shell::ipc_client::MetricsSnapshot;

pub mod io_bridge;
pub mod observability_hooks;
pub mod orchestrator;

pub use io_bridge::{IOEvent, InputBridge, OutputBridge};
pub use observability_hooks::{ConsciousnessTracer, ObservabilityHook};
pub use orchestrator::{OrchestratorConfig, OrchestratorMode, ReplOrchestrator};

/// Re-export key types for ergonomic use
pub use crate::cognitive_loop::ConsciousnessSnapshot as ConsciousnessState;

// ═══════════════════════════════════════════════════════════════════════════════
// REPL SESSION STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete REPL session state.
///
/// `ReplSession` is the core interactive session that integrates:
/// - Cognitive loop (CfC or HDC-LTC temporal processing)
/// - LLM organ (Broca's area for language generation)
/// - Motor cortex (action execution with safety gating)
/// - Voice output (optional TTS with consciousness-modulated pacing)
///
/// # Example
///
/// ```rust,ignore
/// use symthaea::repl::{ReplSession, ReplSessionConfig};
///
/// let config = ReplSessionConfig {
///     cycles_per_input: 3,        // Cognitive cycles per user input
///     temporal_backend: "cfc".to_string(),
///     voice_enabled: false,
///     allow_execution: false,     // Dry-run mode
///     execution_phi_threshold: 0.5,
///     ..Default::default()
/// };
///
/// let mut session = ReplSession::new(config)?;
/// session.warmup(5);  // Pre-warm cognitive loop
///
/// // Process user input
/// let result = session.process("Hello, Symthaea")?;
/// println!("Response: {}", result.response);
/// println!("Phi: {:.4}", result.consciousness.unified_psi);
///
/// // Check if in flow state
/// if session.in_flow() {
///     println!("System achieved flow state!");
/// }
/// ```
pub struct ReplSession {
    /// Cognitive loop service - the consciousness engine.
    ///
    /// Handles temporal dynamics, phi measurement, prediction error,
    /// and consciousness state tracking. Supports CfC or HDC-LTC backends.
    pub cognitive: CognitiveLoopService,

    /// LLM organ for natural language translation.
    ///
    /// Acts as "Broca's Area" - translates internal representations
    /// to natural language responses.
    pub llm: LLMOrgan,

    /// Action executor (motor cortex).
    ///
    /// Executes shell commands with policy validation and sandboxing.
    /// Can operate in Simulated (dry-run) or Real mode.
    pub executor: SimpleExecutor,

    /// Security policy bundle.
    ///
    /// Defines allowed/blocked commands, resource limits, and
    /// destructiveness thresholds.
    pub policy: PolicyBundle,

    /// Sandbox root for safe execution.
    ///
    /// Provides filesystem isolation for action execution.
    pub sandbox: Option<SandboxRoot>,

    /// Voice output (optional larynx).
    ///
    /// When enabled, speaks responses with consciousness-modulated
    /// pacing based on flow state, attention, and arousal.
    pub voice: Option<VoiceOutput>,

    /// Conversation history.
    ///
    /// Stores recent turns for context and statistics.
    /// Limited to `config.max_history` entries.
    pub history: VecDeque<ConversationTurn>,

    /// Session configuration.
    pub config: ReplSessionConfig,

    /// Session statistics (interactions, cycles, flow time, etc.).
    pub stats: SessionStats,

    /// Observability hooks for monitoring and tracing.
    ///
    /// Each hook receives callbacks for input, output, action, and reset events.
    pub observers: Vec<Box<dyn ObservabilityHook>>,
}

/// Single turn in conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// User input
    pub input: String,
    /// System response
    pub response: String,
    /// Consciousness state at turn
    pub consciousness: TurnConsciousness,
    /// Action taken (if any)
    pub action: Option<TurnAction>,
    /// Timestamp
    pub timestamp_ms: u64,
}

/// Consciousness snapshot for a turn
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnConsciousness {
    pub phi: f32,
    pub coherence: f32,
    pub pattern: String,
    pub depth: String,
    pub in_flow: bool,
    pub valence: f32,
    pub arousal: f32,
}

impl From<&ConsciousnessSnapshot> for TurnConsciousness {
    fn from(s: &ConsciousnessSnapshot) -> Self {
        Self {
            phi: s.unified_psi,
            coherence: s.temporal_coherence,
            pattern: format!("{:?}", s.pattern),
            depth: format!("{:?}", s.cognitive_depth),
            in_flow: s.in_flow,
            valence: s.unified_valence,
            arousal: s.unified_arousal,
        }
    }
}

/// Action taken during a turn
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnAction {
    pub command: String,
    pub destructiveness: String,
    pub executed: bool,
    pub output: Option<String>,
    pub blocked_reason: Option<String>,
}

/// Session configuration for the REPL.
///
/// Controls cognitive processing, voice output, action execution, and IPC.
///
/// # Example Configurations
///
/// ```rust,ignore
/// // Minimal interactive mode (default)
/// let config = ReplSessionConfig::default();
///
/// // Research mode: more cognitive cycles, no execution
/// let research_config = ReplSessionConfig {
///     cycles_per_input: 10,
///     temporal_backend: "hdc-ltc".to_string(),
///     ..Default::default()
/// };
///
/// // Voice-enabled assistant
/// let voice_config = ReplSessionConfig {
///     voice_enabled: true,
///     voice_rate: 1.2,  // Slightly faster
///     allow_execution: true,
///     execution_phi_threshold: 0.6,  // Higher bar for actions
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplSessionConfig {
    /// Cognitive cycles per user input (default: 3).
    ///
    /// More cycles = deeper processing but slower response.
    /// For research/analysis, use 5-10. For interactive use, 2-3.
    pub cycles_per_input: usize,

    /// Maximum conversation history to retain (default: 50).
    ///
    /// Older turns are evicted when limit is reached.
    pub max_history: usize,

    /// Temporal backend selection (default: "cfc").
    ///
    /// Options:
    /// - `"cfc"`: Closed-form Continuous-time networks
    /// - `"hdc-ltc"` or `"unified"`: Hypervector-native LTC
    pub temporal_backend: String,

    /// Enable voice output via TTS (default: false).
    ///
    /// When enabled, responses are spoken with consciousness-modulated pacing.
    pub voice_enabled: bool,

    /// Voice rate multiplier (default: 1.0).
    ///
    /// Values: 0.5 = half speed, 1.0 = normal, 2.0 = double speed.
    pub voice_rate: f32,

    /// Enable real command execution (default: false).
    ///
    /// When false, commands run in dry-run/simulated mode.
    /// When true, commands are actually executed through the sandbox.
    pub allow_execution: bool,

    /// Phi threshold required for action execution (default: 0.5).
    ///
    /// Commands are blocked if consciousness phi < threshold.
    /// This implements "consciousness gating" for safety.
    pub execution_phi_threshold: f32,

    /// IPC socket path for daemon mode (default: None).
    ///
    /// When set, the session can connect to or serve as an IPC endpoint.
    pub ipc_socket: Option<String>,
}

impl Default for ReplSessionConfig {
    fn default() -> Self {
        Self {
            cycles_per_input: 3,
            max_history: 50,
            temporal_backend: "cfc".to_string(),
            voice_enabled: false,
            voice_rate: 1.0,
            allow_execution: false,
            execution_phi_threshold: 0.5,
            ipc_socket: None,
        }
    }
}

/// Session statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionStats {
    /// Total interactions
    pub total_interactions: u64,
    /// Total cognitive cycles
    pub total_cycles: u64,
    /// Total actions executed
    pub actions_executed: u64,
    /// Actions blocked by policy
    pub actions_blocked: u64,
    /// Voice utterances generated
    pub voice_utterances: u64,
    /// Time in flow (seconds)
    pub total_flow_time_secs: f32,
    /// Flow periods
    pub flow_periods: u32,
    /// Average phi
    pub avg_phi: f32,
    /// Session start time (unix ms)
    pub start_time_ms: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// REPL SESSION IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════

impl ReplSession {
    /// Create a new REPL session with the given configuration.
    ///
    /// Initializes all components:
    /// - Cognitive loop (CfC or HDC-LTC based on `temporal_backend`)
    /// - LLM organ for language generation
    /// - Motor cortex for action execution
    /// - Optional voice output
    /// - Security policy and sandbox
    ///
    /// # Errors
    ///
    /// Returns an error if the cognitive loop fails to initialize.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ReplSessionConfig::default();
    /// let session = ReplSession::new(config)?;
    /// ```
    pub fn new(config: ReplSessionConfig) -> Result<Self> {
        // Parse temporal backend
        let temporal_backend = match config.temporal_backend.to_lowercase().as_str() {
            "cfc" => TemporalBackend::CfC,
            "hdc-ltc" | "hdcltc" | "unified" => TemporalBackend::HdcLtcUnified,
            "hierarchical" | "hcfc" => TemporalBackend::HierarchicalCfC,
            _ => TemporalBackend::CfC,
        };

        // Initialize cognitive loop
        let cognitive_config = match temporal_backend {
            TemporalBackend::CfC => CognitiveLoopConfig::with_cfc(),
            TemporalBackend::HdcLtcUnified => CognitiveLoopConfig::with_hdc_ltc_unified(),
            TemporalBackend::HierarchicalCfC => {
                let mut cfg = CognitiveLoopConfig::with_cfc();
                cfg.temporal_backend = TemporalBackend::HierarchicalCfC;
                cfg
            }
        };
        let cognitive = CognitiveLoopService::new(cognitive_config)
            .context("Failed to initialize cognitive loop")?;

        // Initialize LLM organ (Broca's Area)
        let llm_config = LLMOrganConfig::default();
        let llm_backend = llm_backend::simulated_backend();
        let llm = LLMOrgan::with_backend(llm_config, llm_backend);

        // Initialize motor cortex
        let executor = if config.allow_execution {
            SimpleExecutor::with_real_commands()
        } else {
            SimpleExecutor::new()
        };

        // Initialize policy and sandbox
        let policy = PolicyBundle::restrictive();
        let sandbox = SandboxRoot::new("repl-session").ok();

        // Initialize voice (optional)
        let voice = if config.voice_enabled {
            let voice_config = VoiceOutputConfig {
                enable_tts: true,
                ..Default::default()
            };
            let mut v = VoiceOutput::new(voice_config);
            let _ = v.initialize();
            Some(v)
        } else {
            None
        };

        let start_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Ok(Self {
            cognitive,
            llm,
            executor,
            policy,
            sandbox,
            voice,
            history: VecDeque::with_capacity(config.max_history),
            config,
            stats: SessionStats {
                start_time_ms: start_time,
                ..Default::default()
            },
            observers: Vec::new(),
        })
    }

    /// Warm up the cognitive loop before interactive use.
    ///
    /// Runs warmup cycles to stabilize internal states and reduce
    /// initial prediction error. Recommended before first user input.
    ///
    /// # Arguments
    ///
    /// * `cycles` - Number of warmup cycles (typically 3-10)
    ///
    /// # Returns
    ///
    /// Final prediction error after warmup (lower = more stable)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut session = ReplSession::new(config)?;
    /// let final_error = session.warmup(5);
    /// println!("Warmup complete, prediction error: {:.4}", final_error);
    /// ```
    pub fn warmup(&mut self, cycles: usize) -> f32 {
        let mut last_error = f32::MAX;
        for i in 0..cycles {
            let warmup_input = format!("Cognitive warmup cycle {i}");
            let result = self.cognitive.cycle(&warmup_input);
            last_error = result.prediction_error;
        }
        last_error
    }

    /// Get the current consciousness state snapshot.
    ///
    /// Returns a snapshot of all consciousness metrics including phi,
    /// coherence, flow state, cognitive depth, and emotional valence/arousal.
    pub fn consciousness_state(&self) -> ConsciousnessSnapshot {
        self.cognitive.consciousness_snapshot()
    }

    /// Process user input through the full cognitive pipeline.
    ///
    /// This is the main entry point for interaction. It:
    /// 1. Notifies observers of the input
    /// 2. Runs N cognitive cycles (configured by `cycles_per_input`)
    /// 3. Detects and executes action commands if present
    /// 4. Generates response via LLM organ
    /// 5. Optionally speaks response via voice output
    /// 6. Records turn in conversation history
    ///
    /// # Arguments
    ///
    /// * `input` - User input string
    ///
    /// # Returns
    ///
    /// [`ProcessingResult`] containing response, consciousness state,
    /// optional action result, and timing information.
    ///
    /// # Action Detection
    ///
    /// Input is treated as an action command if it starts with:
    /// - `!` (e.g., `!ls -la`)
    /// - `run ` (e.g., `run echo hello`)
    /// - `execute ` (e.g., `execute pwd`)
    /// - `shell ` (e.g., `shell cat /etc/hosts`)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = session.process("What is consciousness?")?;
    /// println!("Response: {}", result.response);
    /// println!("Phi: {:.4}", result.consciousness.unified_psi);
    /// println!("Time: {:?}", result.elapsed);
    ///
    /// // Execute a command (if phi >= threshold)
    /// let result = session.process("!echo Hello")?;
    /// if let Some(action) = &result.action {
    ///     if action.executed {
    ///         println!("Output: {}", action.output.as_deref().unwrap_or(""));
    ///     } else {
    ///         println!("Blocked: {}", action.blocked_reason.as_deref().unwrap_or(""));
    ///     }
    /// }
    /// ```
    pub fn process(&mut self, input: &str) -> Result<ProcessingResult> {
        let start = Instant::now();
        self.stats.total_interactions += 1;

        // Notify observers of input
        for observer in &mut self.observers {
            observer.on_input(input);
        }

        // Run cognitive cycles
        let mut last_cycle_result = None;
        for _ in 0..self.config.cycles_per_input {
            let result = self.cognitive.cycle(input);
            self.stats.total_cycles += 1;
            last_cycle_result = Some(result);
        }

        // Get consciousness snapshot after processing
        let snapshot = self.cognitive.consciousness_snapshot();

        // Update flow stats
        if snapshot.in_flow
            && !self
                .history
                .back()
                .map(|t| t.consciousness.in_flow)
                .unwrap_or(false)
        {
            self.stats.flow_periods = self.stats.flow_periods.saturating_add(1);
        }
        self.stats.total_flow_time_secs = snapshot.total_flow_time_secs;

        // Update average phi (exponential moving average)
        let alpha = 0.1;
        self.stats.avg_phi = self.stats.avg_phi * (1.0 - alpha) + snapshot.unified_psi * alpha;

        // Check if this is an action command
        let action_result = if self.is_action_command(input) {
            Some(self.execute_action(input, &snapshot)?)
        } else {
            None
        };

        // Generate response through LLM organ
        let response = if let Some(ref result) = action_result {
            // For actions, use the action result as response
            result.display_output.clone()
        } else {
            // Normal conversation - translate through Broca's Area
            let llm_response = self.llm.generate(input);
            llm_response.text
        };

        // Voice output if enabled
        if let Some(ref mut voice) = self.voice {
            // Update pacing from consciousness state
            let pacing = LTCPacing::default().apply_adaptive_behavior(
                snapshot.speech_rate_multiplier,
                snapshot.pause_multiplier,
                1.0, // attention sensitivity
            );
            voice.set_pacing(pacing);

            if voice.synthesize(&response).is_ok() {
                self.stats.voice_utterances += 1;
            }
        }

        // Record conversation turn
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let turn = ConversationTurn {
            input: input.to_string(),
            response: response.clone(),
            consciousness: TurnConsciousness::from(&snapshot),
            action: action_result.as_ref().map(|a| TurnAction {
                command: a.command.clone(),
                destructiveness: a.destructiveness.clone(),
                executed: a.executed,
                output: a.output.clone(),
                blocked_reason: a.blocked_reason.clone(),
            }),
            timestamp_ms: timestamp,
        };

        self.history.push_back(turn);
        if self.history.len() > self.config.max_history {
            self.history.pop_front();
        }

        // Notify observers
        for observer in &mut self.observers {
            observer.on_output(&response, &snapshot);
            if let Some(ref action) = action_result {
                observer.on_action(&action.command, action.executed);
            }
        }

        let elapsed = start.elapsed();

        Ok(ProcessingResult {
            response,
            consciousness: snapshot,
            action: action_result,
            cycle_result: last_cycle_result,
            elapsed,
        })
    }

    /// Check if input is an action command
    fn is_action_command(&self, input: &str) -> bool {
        let lower = input.to_lowercase();
        lower.starts_with("run ")
            || lower.starts_with("execute ")
            || lower.starts_with("shell ")
            || lower.starts_with('!')
    }

    /// Execute an action through motor cortex
    fn execute_action(
        &mut self,
        input: &str,
        consciousness: &ConsciousnessSnapshot,
    ) -> Result<ActionResult> {
        // Parse command
        let command = input
            .trim_start_matches("run ")
            .trim_start_matches("execute ")
            .trim_start_matches("shell ")
            .trim_start_matches('!')
            .trim()
            .to_string();

        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(ActionResult {
                command,
                destructiveness: "Unknown".to_string(),
                executed: false,
                output: None,
                blocked_reason: Some("No command specified".to_string()),
                display_output: "No command specified.".to_string(),
            });
        }

        let program = parts[0].to_string();
        let args: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();

        // Create action IR
        let action = ActionIR::RunCommand {
            program: program.clone(),
            args,
            env: std::collections::BTreeMap::new(),
            working_dir: None,
        };

        let destructiveness = action.destructiveness();
        let risk = action.risk_tier();

        // Check phi gate
        if consciousness.unified_psi < self.config.execution_phi_threshold {
            self.stats.actions_blocked += 1;
            return Ok(ActionResult {
                command: command.clone(),
                destructiveness: format!("{destructiveness:?}"),
                executed: false,
                output: None,
                blocked_reason: Some(format!(
                    "Phi {:.2} below threshold {:.2}. Center yourself before executing.",
                    consciousness.unified_psi, self.config.execution_phi_threshold
                )),
                display_output: format!(
                    "[PHI GATE] Blocked: Phi {:.2} < {:.2}\n\
                     Command: {}\n\
                     Raise consciousness level before executing.",
                    consciousness.unified_psi, self.config.execution_phi_threshold, command
                ),
            });
        }

        // Validate against policy
        if let Some(ref sandbox) = self.sandbox {
            if let Err(e) = action.validate(&self.policy, sandbox, consciousness.unified_psi as f64)
            {
                self.stats.actions_blocked += 1;
                return Ok(ActionResult {
                    command: command.clone(),
                    destructiveness: format!("{destructiveness:?}"),
                    executed: false,
                    output: None,
                    blocked_reason: Some(format!("Policy violation: {e:?}")),
                    display_output: format!(
                        "[POLICY BLOCKED] Action '{command}' violates policy: {e:?}\n\
                         Risk: {risk:?}, Destructiveness: {destructiveness:?}"
                    ),
                });
            }
        }

        // Check if confirmation required
        if destructiveness.requires_confirmation()
            && self.executor.mode() == ExecutionMode::Simulated
        {
            return Ok(ActionResult {
                command: command.clone(),
                destructiveness: format!("{destructiveness:?}"),
                executed: false,
                output: None,
                blocked_reason: Some("Requires confirmation".to_string()),
                display_output: format!(
                    "[REQUIRES CONFIRMATION] Action: {}\n\
                     Risk: {:?}, Destructiveness: {:?}\n\
                     This action requires explicit confirmation.\n\
                     Rollback hint: {:?}",
                    command,
                    risk,
                    destructiveness,
                    action.rollback_hint()
                ),
            });
        }

        // Execute through motor cortex
        match self.executor.mode() {
            ExecutionMode::Simulated => Ok(ActionResult {
                command: command.clone(),
                destructiveness: format!("{destructiveness:?}"),
                executed: false,
                output: None,
                blocked_reason: None,
                display_output: format!(
                    "[DRY-RUN] Would execute: {}\n\
                         Risk: {:?}, Destructiveness: {:?}\n\
                         Rollback hint: {:?}",
                    command,
                    risk,
                    destructiveness,
                    action.rollback_hint()
                ),
            }),
            ExecutionMode::Real => {
                // Real execution through sandbox
                if let Some(ref sandbox) = self.sandbox {
                    match self.executor.execute(
                        &action,
                        &self.policy,
                        sandbox,
                        consciousness.unified_psi as f64,
                    ) {
                        Ok(outcome) => {
                            self.stats.actions_executed += 1;
                            let output_str = match &outcome.outcome {
                                ActionOutcome::CommandOutput {
                                    stdout,
                                    stderr,
                                    exit_code,
                                } => {
                                    let stdout_str = String::from_utf8_lossy(stdout);
                                    let stderr_str = String::from_utf8_lossy(stderr);
                                    format!(
                                        "Exit code: {exit_code}\nStdout: {stdout_str}\nStderr: {stderr_str}"
                                    )
                                }
                                other => format!("{other:?}"),
                            };
                            Ok(ActionResult {
                                command: command.clone(),
                                destructiveness: format!("{destructiveness:?}"),
                                executed: true,
                                output: Some(output_str.clone()),
                                blocked_reason: None,
                                display_output: format!("[EXECUTED] {command}\n\n{output_str}"),
                            })
                        }
                        Err(e) => Ok(ActionResult {
                            command: command.clone(),
                            destructiveness: format!("{destructiveness:?}"),
                            executed: false,
                            output: None,
                            blocked_reason: Some(format!("Execution error: {e}")),
                            display_output: format!(
                                "[ERROR] Failed to execute: {command}\n\
                                     Error: {e}"
                            ),
                        }),
                    }
                } else {
                    Ok(ActionResult {
                        command,
                        destructiveness: format!("{destructiveness:?}"),
                        executed: false,
                        output: None,
                        blocked_reason: Some("No sandbox available".to_string()),
                        display_output: "[ERROR] No sandbox available for execution".to_string(),
                    })
                }
            }
        }
    }

    /// Add an observability hook
    pub fn add_observer(&mut self, observer: Box<dyn ObservabilityHook>) {
        self.observers.push(observer);
    }

    /// Get session statistics
    pub fn stats(&self) -> &SessionStats {
        &self.stats
    }

    /// Get conversation history
    pub fn history(&self) -> &VecDeque<ConversationTurn> {
        &self.history
    }

    /// Reset cognitive state
    pub fn reset(&mut self) {
        self.cognitive.reset();
        self.history.clear();
        // Voice state is reset by the next pacing update
        // (no explicit reset needed for VoiceOutput)
        for observer in &mut self.observers {
            observer.on_reset();
        }
    }
}

/// Result of processing a single input
#[derive(Debug)]
pub struct ProcessingResult {
    /// Generated response
    pub response: String,
    /// Consciousness state after processing
    pub consciousness: ConsciousnessSnapshot,
    /// Action result (if action was executed)
    pub action: Option<ActionResult>,
    /// Last cycle result
    pub cycle_result: Option<CycleResult>,
    /// Processing time
    pub elapsed: Duration,
}

/// Result of action execution
#[derive(Debug, Clone)]
pub struct ActionResult {
    /// Original command
    pub command: String,
    /// Destructiveness level
    pub destructiveness: String,
    /// Whether action was executed
    pub executed: bool,
    /// Execution output (if executed)
    pub output: Option<String>,
    /// Reason for blocking (if blocked)
    pub blocked_reason: Option<String>,
    /// Formatted output for display
    pub display_output: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// METRICS METHODS (Compatible with MetricsProvider pattern)
// ═══════════════════════════════════════════════════════════════════════════════

impl ReplSession {
    /// Get current metrics snapshot (for IPC/monitoring)
    pub fn get_metrics(&self) -> MetricsSnapshot {
        let snapshot = self.consciousness_state();
        let stats = self.cognitive.stats();
        let uptime = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
            - self.stats.start_time_ms;

        MetricsSnapshot {
            phi: snapshot.unified_psi as f64,
            coherence: snapshot.temporal_coherence as f64,
            is_conscious: snapshot.unified_psi > 0.5,
            cognitive_depth: format!("{:?}", snapshot.cognitive_depth),
            strategy: stats.current_strategy.clone(),
            in_flow: snapshot.in_flow,
            prediction_error: snapshot.prediction_error,
            emotional_valence: snapshot.unified_valence,
            emotional_arousal: snapshot.unified_arousal,
            timestamp_ms: uptime,
            uptime_secs: (uptime / 1000),
            total_cycles: self.stats.total_cycles,
            consciousness_level: snapshot.consciousness_level as f64,
            latency_ms: 0,
            #[cfg(feature = "vision-manifold")]
            mental_movie: None,
        }
    }

    /// Get Phi value
    pub fn phi(&self) -> f64 {
        self.consciousness_state().unified_psi as f64
    }

    /// Get coherence value
    pub fn coherence(&self) -> f64 {
        self.consciousness_state().temporal_coherence as f64
    }

    /// Check if conscious (Phi > 0.5)
    pub fn is_conscious(&self) -> bool {
        self.consciousness_state().unified_psi > 0.5
    }

    /// Get current cognitive depth
    pub fn cognitive_depth_str(&self) -> String {
        format!("{:?}", self.consciousness_state().cognitive_depth)
    }

    /// Get current response strategy
    pub fn current_strategy(&self) -> String {
        self.cognitive.stats().current_strategy.clone()
    }

    /// Check if in flow state
    pub fn in_flow(&self) -> bool {
        self.consciousness_state().in_flow
    }

    /// Get session uptime in seconds
    pub fn uptime_secs(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        (now - self.stats.start_time_ms) / 1000
    }

    /// Get total cognitive cycles
    pub fn total_cycles(&self) -> u64 {
        self.stats.total_cycles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let config = ReplSessionConfig::default();
        let session = ReplSession::new(config);
        assert!(session.is_ok());
    }

    #[test]
    fn test_session_warmup() {
        let config = ReplSessionConfig::default();
        let mut session = ReplSession::new(config).unwrap();
        session.warmup(5);
        assert!(session.stats.total_cycles == 0); // Warmup doesn't count
    }

    #[test]
    fn test_action_detection() {
        let config = ReplSessionConfig::default();
        let session = ReplSession::new(config).unwrap();

        assert!(session.is_action_command("!ls"));
        assert!(session.is_action_command("run echo hello"));
        assert!(session.is_action_command("execute pwd"));
        assert!(!session.is_action_command("hello world"));
    }

    #[test]
    fn test_process_conversation() {
        let config = ReplSessionConfig::default();
        let mut session = ReplSession::new(config).unwrap();

        let result = session.process("Hello, Symthaea");
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(!result.response.is_empty());
        assert!(session.history.len() == 1);
    }
}
