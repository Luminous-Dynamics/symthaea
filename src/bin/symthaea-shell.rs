// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea AI-Native Sidecar Shell
//!
//! A consciousness-aware terminal interface with:
//! - IntelliSense completions powered by HDC semantic similarity
//! - Phi-gated execution with destructiveness classification
//! - Real-time consciousness metrics display
//! - Command preview and confirmation dialogs
//!
//! ## Layout
//!
//! ```text
//! ┌─────────────────────────────┬────────────────────────────┐
//! │  Command Input              │  Consciousness Metrics     │
//! │  [> install nginx_]         │  Phi: 0.87 [========>  ]   │
//! ├─────────────────────────────┤  Coherence: 92%            │
//! │  IntelliSense Buffer        │  Safety: GREEN             │
//! │  ┌───────────────────────┐  ├────────────────────────────┤
//! │  │ 1. install nginx      │  │  Command Preview           │
//! │  │ 2. install nginx-full │  │  Step 1: Add to config     │
//! │  │ 3. install nginx-proxy│  │  Step 2: nixos-rebuild     │
//! │  └───────────────────────┘  │  Step 3: Enable service    │
//! ├─────────────────────────────┴────────────────────────────┤
//! │  Output / History                                         │
//! │  [10:30:15] $ install firefox                            │
//! │  [Phi: 0.85] Added firefox to environment.systemPackages │
//! └──────────────────────────────────────────────────────────┘
//! ```

use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Gauge, List, ListItem, Paragraph, Wrap},
};

use std::path::PathBuf;
use std::sync::Arc;
use symthaea::action::DestructivenessLevel;
use symthaea::shell::{
    CommandContext,
    Completion,
    CompletionKind,
    EpistemicOverlay,
    // B5: Epistemic Overlays
    EpistemicOverlayEngine,
    EpistemicStyle,
    // B8: Error Explanation
    ErrorExplainer,
    ErrorExplanation,
    // B7: Live Flake Context
    FlakeContext,
    GateDecision,
    IntelliSenseEngine,
    KnowledgeSource,
    OverlayPosition,
    OverlayType,
    PhiGate,
    ShellContext,
    // B6: Session Persistence
    StateManager,
    SuggestionSource,
    WhatIfResult,
    // B9: What-If Simulation
    WhatIfSimulator,
    classify_command_destructiveness,
    ipc_client::{ConnectionState, MetricsSnapshot, ShellIpcClient, discover_socket},
};
use tokio::runtime::Runtime;
use tokio::sync::watch;

/// Application state
struct App {
    /// Current input buffer
    input: String,

    /// Cursor position in input
    cursor: usize,

    /// Command history
    history: Vec<HistoryEntry>,

    /// Selected completion index
    selected_completion: usize,

    /// Current completions
    completions: Vec<Completion>,

    /// Shell context
    context: ShellContext,

    /// IntelliSense engine (local fallback)
    intellisense: IntelliSenseEngine,

    /// Phi gate (local fallback)
    phi_gate: PhiGate,

    /// Whether to show completions popup
    show_completions: bool,

    /// Output buffer
    output_lines: Vec<OutputLine>,

    /// Current mode
    mode: ShellMode,

    /// Last update time
    last_update: Instant,

    /// Should quit
    should_quit: bool,

    /// IPC client for symthaea service
    ipc_client: Option<ShellIpcClient>,

    /// Tokio runtime for async IPC
    runtime: Arc<Runtime>,

    /// Socket path for IPC (discovered or configured)
    socket_path: Option<PathBuf>,

    /// Connection state for visual indicators
    connection_state: ConnectionState,

    /// Streaming metrics receiver (push-based)
    metrics_rx: Option<watch::Receiver<MetricsSnapshot>>,

    /// Last metrics snapshot
    last_metrics: MetricsSnapshot,

    /// Service metrics (from IPC or simulated)
    service_phi: f64,
    service_coherence: f64,
    service_conscious: bool,

    // === B1: Tab Completion Enhancement ===
    /// Whether Tab was just pressed (for cycling)
    tab_pressed: bool,

    // === B2: History Search ===
    /// Current history search query
    history_search_query: String,
    /// Filtered history matches
    history_search_matches: Vec<usize>,
    /// Selected match index
    history_search_selected: usize,

    // === B3: Command Preview ===
    /// Current command preview (what the command will do)
    command_preview: Option<CommandPreview>,

    // === B4: Confirmation Dialog ===
    /// Command pending confirmation (for destructive commands)
    pending_command: Option<PendingCommand>,

    // === B5: Epistemic Overlays ===
    /// Epistemic overlay engine for contextual knowledge markup
    epistemic_engine: EpistemicOverlayEngine,
    /// Currently active epistemic overlays
    active_overlays: Vec<EpistemicOverlay>,
    /// Whether to show epistemic overlays
    show_epistemic_overlays: bool,

    // === B6: Session Persistence ===
    /// State manager for persisting history and consciousness state
    state_manager: Option<StateManager>,
    /// Last save timestamp
    last_save: Instant,
    /// Auto-save interval (60 seconds)
    save_interval: Duration,

    // === B7: Live Flake Context ===
    /// Parsed flake.nix context for contextual completions
    flake_context: FlakeContext,
    /// Last flake reload check
    last_flake_check: Instant,
    /// Flake reload interval (5 seconds)
    flake_check_interval: Duration,

    // === B8: Error Explanation ===
    /// Error explainer for inline diagnosis (used by process_output_for_errors)
    error_explainer: ErrorExplainer,
    /// Last error explanation (for /explain command)
    last_error: Option<ErrorExplanation>,

    // === B9: What-If Simulation ===
    /// What-if simulator for dry-run preview
    whatif_simulator: WhatIfSimulator,
    /// Last what-if result
    last_whatif: Option<WhatIfResult>,

    // === B10: Genesis-seeded RNG for deterministic viz drift ===
    /// Optional seeded RNG for deterministic Phi drift in local/fallback mode
    viz_rng: Option<symthaea_core::genesis::ShakeRng>,
}

/// Command awaiting confirmation
#[derive(Debug, Clone)]
struct PendingCommand {
    /// The command to execute
    command: String,
    /// Why confirmation is needed
    reason: String,
    /// Risk level (for UI coloring)
    risk_level: RiskLevel,
    /// Rollback hint if available
    rollback_hint: Option<String>,
}

/// Risk level for pending commands
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RiskLevel {
    Low,    // Reversible operations
    Medium, // Needs confirmation but recoverable
    High,   // Destructive, non-reversible
}

/// B3: Command preview showing what a command will do
#[derive(Debug, Clone)]
struct CommandPreview {
    /// Brief description
    description: String,
    /// Steps that will be executed
    steps: Vec<String>,
    /// Affected files/paths
    affected: Vec<String>,
    /// Whether this requires confirmation
    needs_confirmation: bool,
    /// Estimated time (if known)
    estimated_time: Option<String>,
}

/// Command history entry
#[derive(Debug, Clone)]
struct HistoryEntry {
    command: String,
    #[allow(dead_code)]
    timestamp: chrono::DateTime<chrono::Local>,
    phi_at_execution: f64,
    success: bool,
}

/// Output line with styling
#[derive(Debug, Clone)]
struct OutputLine {
    content: String,
    style: OutputStyle,
    #[allow(dead_code)]
    timestamp: chrono::DateTime<chrono::Local>,
}

/// Output styling
#[derive(Debug, Clone, Copy)]
enum OutputStyle {
    Normal,
    Success,
    Warning,
    Error,
    Info,
    Phi,
}

/// Shell modes
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum ShellMode {
    Normal,
    Completing,
    Confirming,
    Help,
    /// B2: Searching through command history with HDC semantic matching
    HistorySearch,
}

// ============================================================================
// B4: Nix Syntax Highlighting
// ============================================================================

/// Syntax token type for highlighting
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum NixToken {
    Keyword,    // let, in, if, then, else, with, inherit, rec
    Builtin,    // builtins, import, fetchurl, derivation
    String,     // "..." or ''...''
    Path,       // ./path or /absolute/path
    Comment,    // # comment
    Operator,   // = : ; { } [ ] @ ? //
    Number,     // 123, 1.5
    Identifier, // variable names
    Attribute,  // attr.path
    Boolean,    // true, false, null
    Normal,     // default
}

#[allow(dead_code)]
impl NixToken {
    fn color(&self) -> Color {
        match self {
            Self::Keyword => Color::Magenta,
            Self::Builtin => Color::Cyan,
            Self::String => Color::Green,
            Self::Path => Color::Yellow,
            Self::Comment => Color::DarkGray,
            Self::Operator => Color::White,
            Self::Number => Color::LightBlue,
            Self::Identifier => Color::White,
            Self::Attribute => Color::LightCyan,
            Self::Boolean => Color::LightMagenta,
            Self::Normal => Color::White,
        }
    }
}

/// Simple Nix syntax highlighter
#[allow(dead_code)]
struct NixHighlighter;

#[allow(dead_code)]
impl NixHighlighter {
    const KEYWORDS: &'static [&'static str] = &[
        "let", "in", "if", "then", "else", "with", "inherit", "rec", "assert", "or", "and",
        "import", "throw", "abort",
    ];

    const BUILTINS: &'static [&'static str] = &[
        "builtins",
        "fetchurl",
        "fetchTarball",
        "fetchGit",
        "derivation",
        "toString",
        "toJSON",
        "fromJSON",
        "map",
        "filter",
        "foldl'",
        "pkgs",
        "lib",
        "config",
        "options",
        "nixpkgs",
        "mkOption",
        "mkIf",
        "mkMerge",
        "mkDefault",
        "mkForce",
        "mkOverride",
        "services",
        "programs",
        "environment",
        "systemPackages",
    ];

    fn highlight(input: &str) -> Vec<(String, NixToken)> {
        let mut tokens = Vec::new();
        let mut chars = input.chars().peekable();
        let mut current = String::new();

        while let Some(c) = chars.next() {
            match c {
                // String start
                '"' => {
                    if !current.is_empty() {
                        tokens.push((current.clone(), Self::classify_word(&current)));
                        current.clear();
                    }
                    let mut s = String::from(c);
                    while let Some(&nc) = chars.peek() {
                        s.push(
                            chars
                                .next()
                                .expect("peek() returned Some so next() must succeed"),
                        );
                        if nc == '"' && !s.ends_with("\\\"") {
                            break;
                        }
                    }
                    tokens.push((s, NixToken::String));
                }
                // Comment
                '#' => {
                    if !current.is_empty() {
                        tokens.push((current.clone(), Self::classify_word(&current)));
                        current.clear();
                    }
                    let mut s = String::from(c);
                    while let Some(&nc) = chars.peek() {
                        if nc == '\n' {
                            break;
                        }
                        s.push(
                            chars
                                .next()
                                .expect("peek() returned Some so next() must succeed"),
                        );
                    }
                    tokens.push((s, NixToken::Comment));
                }
                // Operators and punctuation
                '=' | ':' | ';' | '{' | '}' | '[' | ']' | '(' | ')' | '@' | '?' | ',' | '.' => {
                    if !current.is_empty() {
                        tokens.push((current.clone(), Self::classify_word(&current)));
                        current.clear();
                    }
                    tokens.push((c.to_string(), NixToken::Operator));
                }
                // Path detection
                '/' | '~' if current.is_empty() || current.starts_with('.') => {
                    current.push(c);
                }
                // Whitespace
                ' ' | '\t' | '\n' => {
                    if !current.is_empty() {
                        tokens.push((current.clone(), Self::classify_word(&current)));
                        current.clear();
                    }
                    tokens.push((c.to_string(), NixToken::Normal));
                }
                // Regular characters
                _ => {
                    current.push(c);
                }
            }
        }

        if !current.is_empty() {
            tokens.push((current.clone(), Self::classify_word(&current)));
        }

        tokens
    }

    fn classify_word(word: &str) -> NixToken {
        // Check for path
        if word.starts_with("./") || word.starts_with("/") || word.starts_with("~/") {
            return NixToken::Path;
        }

        // Check for number
        if word.parse::<f64>().is_ok() {
            return NixToken::Number;
        }

        // Check for boolean/null
        if matches!(word, "true" | "false" | "null") {
            return NixToken::Boolean;
        }

        // Check for keyword
        if Self::KEYWORDS.contains(&word) {
            return NixToken::Keyword;
        }

        // Check for builtin
        if Self::BUILTINS.contains(&word) {
            return NixToken::Builtin;
        }

        // Attribute path detection
        if word.contains('.') {
            return NixToken::Attribute;
        }

        NixToken::Identifier
    }

    /// Convert tokens to ratatui Spans for rendering
    fn to_spans(input: &str) -> Vec<Span<'static>> {
        Self::highlight(input)
            .into_iter()
            .map(|(text, token)| Span::styled(text, Style::default().fg(token.color())))
            .collect()
    }
}

impl App {
    fn new() -> Self {
        // Create tokio runtime for async IPC
        let runtime = Arc::new(Runtime::new().expect("Failed to create tokio runtime"));

        let mut context = ShellContext::new();
        // Initial consciousness state (will be updated from service if connected)
        context.update_metrics(0.75, 0.85, true);

        let mut phi_gate = PhiGate::new();
        phi_gate.update_metrics(0.75, 0.85, true);

        let mut intellisense = IntelliSenseEngine::new();
        intellisense.set_phi(0.75);

        // Discover symthaea service socket automatically
        let socket_path = discover_socket();

        // Default metrics snapshot
        let default_metrics = MetricsSnapshot {
            phi: 0.75,
            coherence: 0.85,
            consciousness_level: 0.5,
            is_conscious: true,
            timestamp_ms: 0,
            latency_ms: 0,
            ..Default::default()
        };

        let (
            ipc_client,
            connection_state,
            metrics_rx,
            initial_phi,
            initial_coherence,
            initial_conscious,
        ) = if let Some(ref path) = socket_path {
            // Try to connect to the service
            let mut client = ShellIpcClient::with_socket_path(path);
            let connected = runtime.block_on(async { client.connect().await.is_ok() });

            if connected {
                // Try to set up streaming metrics
                let metrics_receiver =
                    runtime.block_on(async { client.subscribe_metrics_watch(500).await.ok() });

                // Get initial status
                let (phi, coherence, conscious) = if let Some(ref rx) = metrics_receiver {
                    let m = rx.borrow();
                    (m.phi, m.coherence, m.is_conscious)
                } else {
                    // Fallback to polling if streaming not available
                    runtime
                        .block_on(async { client.get_status().await.unwrap_or((0.75, 0.85, true)) })
                };

                (
                    Some(client),
                    ConnectionState::Connected,
                    metrics_receiver,
                    phi,
                    coherence,
                    conscious,
                )
            } else {
                (None, ConnectionState::Disconnected, None, 0.75, 0.85, true)
            }
        } else {
            (None, ConnectionState::Disconnected, None, 0.75, 0.85, true)
        };

        // Update context and gates with service metrics
        context.update_metrics(initial_phi, initial_coherence, initial_conscious);
        phi_gate.update_metrics(initial_phi, initial_coherence, initial_conscious);
        intellisense.set_phi(initial_phi);

        let startup_msg = match connection_state {
            ConnectionState::Connected => format!(
                "{} Connected to symthaea service (Phi: {:.2})",
                connection_state.indicator(),
                initial_phi
            ),
            _ => match &socket_path {
                Some(p) => format!(
                    "{} Socket found at {} but connection failed",
                    connection_state.indicator(),
                    p.display()
                ),
                None => format!(
                    "{} No service found - using standalone mode",
                    connection_state.indicator()
                ),
            },
        };

        Self {
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            selected_completion: 0,
            completions: Vec::new(),
            context,
            intellisense,
            phi_gate,
            show_completions: false,
            output_lines: vec![
                OutputLine {
                    content: "Symthaea Shell - AI-Native Terminal".to_string(),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                },
                OutputLine {
                    content: startup_msg,
                    style: if matches!(connection_state, ConnectionState::Connected) {
                        OutputStyle::Success
                    } else {
                        OutputStyle::Warning
                    },
                    timestamp: chrono::Local::now(),
                },
                OutputLine {
                    content: "Type /help for commands, or start typing naturally".to_string(),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                },
            ],
            mode: ShellMode::Normal,
            last_update: Instant::now(),
            should_quit: false,
            ipc_client,
            runtime,
            socket_path,
            connection_state,
            metrics_rx,
            last_metrics: default_metrics,
            service_phi: initial_phi,
            service_coherence: initial_coherence,
            service_conscious: initial_conscious,
            // B1: Tab cycling
            tab_pressed: false,
            // B2: History search
            history_search_query: String::new(),
            history_search_matches: Vec::new(),
            history_search_selected: 0,
            // B3: Command preview
            command_preview: None,
            // B4: Confirmation dialog
            pending_command: None,
            // B5: Epistemic overlays
            epistemic_engine: EpistemicOverlayEngine::new(),
            active_overlays: Vec::new(),
            show_epistemic_overlays: true,
            // B6: Session persistence
            state_manager: Self::init_state_manager(),
            last_save: Instant::now(),
            save_interval: Duration::from_secs(60),
            // B7: Live flake context
            flake_context: FlakeContext::discover_and_load(),
            last_flake_check: Instant::now(),
            flake_check_interval: Duration::from_secs(5),
            // B8: Error explanation
            error_explainer: ErrorExplainer::new(),
            last_error: None,
            // B9: What-if simulation
            whatif_simulator: WhatIfSimulator::new(),
            last_whatif: None,
            // B10: Genesis-seeded viz RNG (None in default; set via from_genesis)
            viz_rng: None,
        }
    }

    /// Create an App with deterministic RNG from a genesis seed.
    #[allow(dead_code)]
    fn from_genesis(genesis: &symthaea_core::genesis::GenesisSeed, label: &str) -> Self {
        let mut app = Self::new();
        app.viz_rng = Some(genesis.domain(&format!("{label}::shell_viz")));
        app
    }

    /// Initialize state manager with XDG state directory
    fn init_state_manager() -> Option<StateManager> {
        // Use XDG state directory: ~/.local/state/symthaea
        let state_dir = dirs::state_dir()
            .or_else(dirs::data_local_dir)
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("symthaea");

        match StateManager::new(&state_dir) {
            Ok(manager) => {
                tracing::info!(path = %state_dir.display(), "Session state loaded");
                Some(manager)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to initialize state manager");
                None
            }
        }
    }

    // =========================================================================
    // B2: Semantic History Search
    // =========================================================================

    /// Start history search mode
    fn start_history_search(&mut self) {
        self.mode = ShellMode::HistorySearch;
        self.history_search_query.clear();
        self.history_search_matches.clear();
        self.history_search_selected = 0;
        self.update_history_search();
    }

    /// Update history search results based on query
    fn update_history_search(&mut self) {
        if self.history_search_query.is_empty() {
            // Show all history (most recent first)
            self.history_search_matches = (0..self.history.len()).rev().collect();
        } else {
            // Filter by fuzzy match
            let query_lower = self.history_search_query.to_lowercase();
            self.history_search_matches = self
                .history
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.command.to_lowercase().contains(&query_lower))
                .map(|(i, _)| i)
                .rev()
                .collect();
        }
        self.history_search_selected = 0;
    }

    /// Accept the selected history search result
    fn accept_history_search(&mut self) {
        if let Some(&idx) = self
            .history_search_matches
            .get(self.history_search_selected)
            && let Some(entry) = self.history.get(idx)
        {
            self.input = entry.command.clone();
            self.cursor = self.input.len();
        }
        self.mode = ShellMode::Normal;
        self.update_completions();
    }

    // =========================================================================
    // B3: Command Preview Generation
    // =========================================================================

    /// Generate preview for current command
    fn update_command_preview(&mut self) {
        let input = self.input.trim();
        if input.is_empty() {
            self.command_preview = None;
            return;
        }

        // Parse command and generate preview
        self.command_preview = Some(self.generate_preview(input));
    }

    /// Generate command preview based on input
    fn generate_preview(&self, input: &str) -> CommandPreview {
        let parts: Vec<&str> = input.split_whitespace().collect();
        let cmd = parts.first().copied().unwrap_or("");

        match cmd {
            "install" | "add" => {
                let packages: Vec<_> = parts.iter().skip(1).copied().collect();
                CommandPreview {
                    description: format!("Install {} package(s)", packages.len()),
                    steps: vec![
                        format!("Add to environment.systemPackages: {}", packages.join(", ")),
                        "Run: nixos-rebuild switch".to_string(),
                    ],
                    affected: vec!["/etc/nixos/configuration.nix".to_string()],
                    needs_confirmation: true,
                    estimated_time: Some("~2-5 min".to_string()),
                }
            }
            "remove" | "uninstall" => {
                let packages: Vec<_> = parts.iter().skip(1).copied().collect();
                CommandPreview {
                    description: format!("Remove {} package(s)", packages.len()),
                    steps: vec![
                        format!("Remove from systemPackages: {}", packages.join(", ")),
                        "Run: nixos-rebuild switch".to_string(),
                    ],
                    affected: vec!["/etc/nixos/configuration.nix".to_string()],
                    needs_confirmation: true,
                    estimated_time: Some("~1-3 min".to_string()),
                }
            }
            "enable" => {
                let service = parts.get(1).copied().unwrap_or("unknown");
                CommandPreview {
                    description: format!("Enable service: {}", service),
                    steps: vec![
                        format!("Set services.{}.enable = true", service),
                        "Run: nixos-rebuild switch".to_string(),
                        format!("Start: systemctl start {}", service),
                    ],
                    affected: vec![
                        "/etc/nixos/configuration.nix".to_string(),
                        format!("/etc/systemd/system/{}.service", service),
                    ],
                    needs_confirmation: true,
                    estimated_time: Some("~1-2 min".to_string()),
                }
            }
            "search" | "nix-search" => {
                let query = parts.get(1).copied().unwrap_or("");
                CommandPreview {
                    description: format!("Search packages: {}", query),
                    steps: vec!["Query nixpkgs index".to_string()],
                    affected: vec![],
                    needs_confirmation: false,
                    estimated_time: Some("~1-5 sec".to_string()),
                }
            }
            "rebuild" | "switch" => CommandPreview {
                description: "Rebuild NixOS configuration".to_string(),
                steps: vec![
                    "Evaluate configuration".to_string(),
                    "Build derivations".to_string(),
                    "Activate new generation".to_string(),
                ],
                affected: vec![
                    "/nix/store/...".to_string(),
                    "/run/current-system".to_string(),
                ],
                needs_confirmation: true,
                estimated_time: Some("~2-10 min".to_string()),
            },
            _ => CommandPreview {
                description: format!("Execute: {}", input),
                steps: vec!["Run command".to_string()],
                affected: vec![],
                needs_confirmation: false,
                estimated_time: None,
            },
        }
    }

    fn on_tick(&mut self) {
        let elapsed = self.last_update.elapsed().as_secs_f64();
        if elapsed < 0.1 {
            return; // Check every 100ms for smoother updates
        }

        // Get metrics from streaming channel if available
        let (new_phi, new_coherence, is_conscious) = if let Some(ref rx) = self.metrics_rx {
            // Check if there's a new value (non-blocking)
            if rx.has_changed().unwrap_or(false) {
                let snapshot = rx.borrow().clone();
                self.last_metrics = snapshot.clone();
                self.service_phi = snapshot.phi;
                self.service_coherence = snapshot.coherence;
                self.service_conscious = snapshot.is_conscious;
                self.connection_state = ConnectionState::Connected;
                (snapshot.phi, snapshot.coherence, snapshot.is_conscious)
            } else {
                // Use last known values
                (
                    self.service_phi,
                    self.service_coherence,
                    self.service_conscious,
                )
            }
        } else if matches!(self.connection_state, ConnectionState::Connected) {
            // Fallback: Poll if streaming not available but connected
            if self.last_update.elapsed().as_secs() >= 2 {
                if let Some(ref mut client) = self.ipc_client {
                    let rt = Arc::clone(&self.runtime);
                    match rt.block_on(async { client.get_status().await }) {
                        Ok((phi, coherence, conscious)) => {
                            self.service_phi = phi;
                            self.service_coherence = coherence;
                            self.service_conscious = conscious;
                            (phi, coherence, conscious)
                        }
                        Err(_) => {
                            // Connection lost
                            self.connection_state = ConnectionState::Disconnected;
                            self.output_lines.push(OutputLine {
                                content: format!(
                                    "{} Lost connection to symthaea service",
                                    self.connection_state.indicator()
                                ),
                                style: OutputStyle::Warning,
                                timestamp: chrono::Local::now(),
                            });
                            // Fall back to local simulation
                            let drift_raw: f64 = if let Some(ref mut rng) = self.viz_rng {
                                rand::Rng::r#gen(rng)
                            } else {
                                rand::random::<f64>()
                            };
                            let drift = (drift_raw - 0.5) * 0.02;
                            let new_phi = (self.context.current_phi + drift).clamp(0.3, 0.95);
                            (new_phi, self.context.current_coherence, new_phi > 0.5)
                        }
                    }
                } else {
                    (
                        self.service_phi,
                        self.service_coherence,
                        self.service_conscious,
                    )
                }
            } else {
                (
                    self.service_phi,
                    self.service_coherence,
                    self.service_conscious,
                )
            }
        } else {
            // Local mode: Natural Phi drift simulation
            let drift_raw: f64 = if let Some(ref mut rng) = self.viz_rng {
                rand::Rng::r#gen(rng)
            } else {
                rand::random::<f64>()
            };
            let drift = (drift_raw - 0.5) * 0.02;
            let new_phi = (self.context.current_phi + drift).clamp(0.3, 0.95);
            (new_phi, self.context.current_coherence, new_phi > 0.5)
        };

        self.context
            .update_metrics(new_phi, new_coherence, is_conscious);
        self.phi_gate
            .update_metrics(new_phi, new_coherence, is_conscious);
        self.intellisense.set_phi(new_phi);

        self.last_update = Instant::now();
    }

    fn update_completions(&mut self) {
        if self.input.is_empty() {
            self.completions.clear();
            self.show_completions = false;
            self.active_overlays.clear();
            return;
        }

        // B7: Check for flake.nix changes periodically
        self.maybe_reload_flake();

        // Get completions based on input and cursor position
        self.completions = self.intellisense.complete(&self.input, self.cursor);

        // B7: Add contextual suggestions from flake.nix
        let flake_suggestions = self.flake_context.get_contextual_suggestions(&self.input);
        for suggestion in flake_suggestions {
            // Convert to Completion and insert at appropriate position
            let completion = Completion {
                text: suggestion.text.clone(),
                display: suggestion.text.clone(),
                description: suggestion.description.clone(),
                kind: match suggestion.source {
                    SuggestionSource::InstalledPackage => CompletionKind::NixCommand,
                    SuggestionSource::EnabledService => CompletionKind::NixCommand,
                    SuggestionSource::KnownService => CompletionKind::NixCommand,
                    SuggestionSource::OptionPath => CompletionKind::AttrPath,
                    SuggestionSource::Import => CompletionKind::Path,
                    SuggestionSource::UserConfig => CompletionKind::Variable,
                },
                confidence: suggestion.confidence as f32,
                hdc_distance: 0.0,
                ..Default::default()
            };

            // Insert based on confidence (higher confidence = earlier position)
            let pos = self
                .completions
                .iter()
                .position(|c| c.confidence < completion.confidence)
                .unwrap_or(self.completions.len());
            self.completions.insert(pos, completion);
        }

        self.show_completions = !self.completions.is_empty();
        self.selected_completion = 0;

        // B5: Update epistemic overlays based on current context
        if self.show_epistemic_overlays {
            self.update_epistemic_overlays();
        }
    }

    /// B7: Check if flake.nix needs reloading
    fn maybe_reload_flake(&mut self) {
        if self.last_flake_check.elapsed() >= self.flake_check_interval {
            if self.flake_context.reload_if_changed() {
                // Notify user of flake reload
                self.output_lines.push(OutputLine {
                    content: format!(
                        "Flake context reloaded: {}",
                        self.flake_context.status_summary()
                    ),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                });
            }
            self.last_flake_check = Instant::now();
        }
    }

    /// B5: Update epistemic overlays based on current input and context
    fn update_epistemic_overlays(&mut self) {
        // Build context from current state
        let context = if let Some(completion) = self.completions.first() {
            // Use first completion's confidence as basis
            let source = if self
                .history
                .iter()
                .any(|h| h.command.starts_with(&completion.text))
            {
                KnowledgeSource::UserHistory
            } else if completion.text.contains("nix") {
                KnowledgeSource::NixosDocs
            } else {
                KnowledgeSource::SemanticInference
            };

            CommandContext {
                source,
                confidence: completion.confidence as f64,
                k_index: completion.confidence as f64 * 0.9,
                epistemic_uncertainty: 1.0 - completion.confidence as f64,
                aleatoric_uncertainty: 0.1,
                theory_weights: vec![
                    ("HDC Semantic".to_string(), 0.4),
                    ("Pattern Match".to_string(), 0.3),
                    ("History".to_string(), 0.3),
                ],
                phi_required: self.context.current_phi,
                current_phi: self.context.current_phi,
            }
        } else {
            // No completions - low confidence context
            CommandContext {
                source: KnowledgeSource::Unknown,
                confidence: 0.2,
                k_index: 0.2,
                epistemic_uncertainty: 0.8,
                aleatoric_uncertainty: 0.1,
                theory_weights: Vec::new(),
                phi_required: 0.5,
                current_phi: self.context.current_phi,
            }
        };

        self.active_overlays = self
            .epistemic_engine
            .generate_for_command(&self.input, &context);
    }

    fn execute_command(&mut self) {
        let command = self.input.trim().to_string();
        if command.is_empty() {
            return;
        }

        // Handle special commands
        if command.starts_with('/') {
            self.handle_slash_command(&command);
            self.input.clear();
            self.cursor = 0;
            self.update_completions();
            return;
        }

        // Create execution request and evaluate through Phi gate
        let destructiveness = classify_command_destructiveness(&command);
        let decision = self.phi_gate.evaluate(&command, destructiveness);

        match decision {
            GateDecision::Allowed { phi, confidence } => {
                // Execute the command
                self.output_lines.push(OutputLine {
                    content: format!("$ {}", command),
                    style: OutputStyle::Normal,
                    timestamp: chrono::Local::now(),
                });

                self.output_lines.push(OutputLine {
                    content: format!(
                        "[Phi: {:.2}] Command executed (confidence: {:.0}%)",
                        phi,
                        confidence * 100.0
                    ),
                    style: OutputStyle::Phi,
                    timestamp: chrono::Local::now(),
                });

                // Add to history
                self.history.push(HistoryEntry {
                    command: command.clone(),
                    timestamp: chrono::Local::now(),
                    phi_at_execution: phi,
                    success: true,
                });

                self.context.add_to_history(command.clone());

                // B6: Persist to state manager
                self.persist_command(&command, phi, true);
            }

            GateDecision::NeedsConfirmation {
                reason,
                phi,
                prompt: _,
            } => {
                // Classify risk level and get rollback hint
                let (risk_level, rollback_hint) = Self::classify_command_risk(&command);

                self.output_lines.push(OutputLine {
                    content: format!("$ {}", command),
                    style: OutputStyle::Warning,
                    timestamp: chrono::Local::now(),
                });

                // Show risk-appropriate confirmation dialog
                let risk_symbol = match risk_level {
                    RiskLevel::High => "⚠️  NON-REVERSIBLE COMMAND",
                    RiskLevel::Medium => "⚡ CONFIRMATION REQUIRED",
                    RiskLevel::Low => "📋 CONFIRMATION NEEDED",
                };

                self.output_lines.push(OutputLine {
                    content: format!("{} [Phi: {:.2}]", risk_symbol, phi),
                    style: OutputStyle::Warning,
                    timestamp: chrono::Local::now(),
                });

                self.output_lines.push(OutputLine {
                    content: format!("Reason: {}", reason),
                    style: OutputStyle::Warning,
                    timestamp: chrono::Local::now(),
                });

                // Show rollback hint if available
                if let Some(ref hint) = rollback_hint {
                    self.output_lines.push(OutputLine {
                        content: format!("Rollback: {}", hint),
                        style: OutputStyle::Info,
                        timestamp: chrono::Local::now(),
                    });
                }

                self.output_lines.push(OutputLine {
                    content: "Press 'y' to confirm, 'd' for dry-run, 'n' to cancel".to_string(),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                });

                // Store the pending command
                self.pending_command = Some(PendingCommand {
                    command: command.clone(),
                    reason: reason.clone(),
                    risk_level,
                    rollback_hint,
                });

                self.mode = ShellMode::Confirming;
            }

            GateDecision::Vetoed { reason: _, message } => {
                self.output_lines.push(OutputLine {
                    content: format!("$ {}", command),
                    style: OutputStyle::Error,
                    timestamp: chrono::Local::now(),
                });

                self.output_lines.push(OutputLine {
                    content: "VETOED by safety system".to_string(),
                    style: OutputStyle::Error,
                    timestamp: chrono::Local::now(),
                });

                self.output_lines.push(OutputLine {
                    content: message,
                    style: OutputStyle::Error,
                    timestamp: chrono::Local::now(),
                });

                // B6: Record veto in session stats
                self.record_veto();
            }

            GateDecision::InsufficientPhi {
                current_phi,
                required_phi,
                centering_time_secs,
            } => {
                self.output_lines.push(OutputLine {
                    content: format!("$ {}", command),
                    style: OutputStyle::Warning,
                    timestamp: chrono::Local::now(),
                });

                self.output_lines.push(OutputLine {
                    content: format!(
                        "Insufficient Phi: {:.2} / {:.2} required",
                        current_phi, required_phi
                    ),
                    style: OutputStyle::Warning,
                    timestamp: chrono::Local::now(),
                });

                self.output_lines.push(OutputLine {
                    content: format!(
                        "Wait ~{:.0}s for centering, or use /center",
                        centering_time_secs
                    ),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                });
            }

            GateDecision::Pending { .. } => {
                // Should not happen in synchronous mode
            }
        }

        self.input.clear();
        self.cursor = 0;
        self.update_completions();
    }

    /// Classify command risk level and provide rollback hints
    fn classify_command_risk(command: &str) -> (RiskLevel, Option<String>) {
        let cmd_lower = command.to_lowercase();
        let parts: Vec<&str> = cmd_lower.split_whitespace().collect();
        let first_word = parts.first().copied().unwrap_or("");

        // High risk: destructive, non-reversible commands
        if cmd_lower.contains("--delete")
            || cmd_lower.contains("-d ") && first_word.contains("nix-collect-garbage")
            || cmd_lower.contains("nix-collect-garbage")
            || cmd_lower.contains("rm -rf")
            || cmd_lower.contains("nixos-rebuild switch")
            || cmd_lower.contains("format")
            || cmd_lower.contains("mkfs")
        {
            let rollback = if cmd_lower.contains("nixos-rebuild switch") {
                Some("nixos-rebuild switch --rollback".to_string())
            } else if cmd_lower.contains("nix-collect-garbage") {
                Some("Cannot recover deleted generations".to_string())
            } else {
                None
            };
            return (RiskLevel::High, rollback);
        }

        // Medium risk: system changes that can be reverted
        if cmd_lower.contains("nixos-rebuild")
            || first_word == "nix-env" && cmd_lower.contains("-e")
            || cmd_lower.contains("systemctl")
                && (cmd_lower.contains("stop") || cmd_lower.contains("restart"))
            || cmd_lower.contains("service") && cmd_lower.contains("restart")
        {
            let rollback = if cmd_lower.contains("nixos-rebuild") {
                Some("nixos-rebuild switch --rollback".to_string())
            } else if cmd_lower.contains("nix-env") && cmd_lower.contains("-e") {
                Some("nix-env -i <package>".to_string())
            } else if cmd_lower.contains("systemctl") || cmd_lower.contains("service") {
                Some("systemctl start <service>".to_string())
            } else {
                None
            };
            return (RiskLevel::Medium, rollback);
        }

        // Low risk: installs, enables (generally reversible)
        if first_word == "nix-env" && cmd_lower.contains("-i")
            || cmd_lower.contains("systemctl enable")
            || cmd_lower.contains("nix profile install")
        {
            let rollback = if cmd_lower.contains("nix-env") && cmd_lower.contains("-i") {
                Some("nix-env -e <package>".to_string())
            } else if cmd_lower.contains("nix profile install") {
                Some("nix profile remove <package>".to_string())
            } else {
                Some("systemctl disable <service>".to_string())
            };
            return (RiskLevel::Low, rollback);
        }

        // Default to medium risk for unknown commands requiring confirmation
        (RiskLevel::Medium, None)
    }

    /// Execute a command that has already been confirmed by user
    fn execute_confirmed_command(&mut self) {
        let command = self.input.trim().to_string();
        if command.is_empty() {
            return;
        }

        // Get current Phi for logging (command is already confirmed, so we skip gate check)
        let current_phi = self.context.current_phi;

        self.output_lines.push(OutputLine {
            content: format!("$ {}", command),
            style: OutputStyle::Success,
            timestamp: chrono::Local::now(),
        });

        self.output_lines.push(OutputLine {
            content: format!("[Phi: {:.2}] Confirmed command executed", current_phi),
            style: OutputStyle::Phi,
            timestamp: chrono::Local::now(),
        });

        // Add to history with confirmation flag
        self.history.push(HistoryEntry {
            command: command.clone(),
            timestamp: chrono::Local::now(),
            phi_at_execution: current_phi,
            success: true,
        });

        self.context.add_to_history(command.clone());

        // B6: Persist to state manager (confirmed commands also tracked)
        if let Some(ref mut manager) = self.state_manager {
            manager.record_confirmation();
        }
        self.persist_command(&command, current_phi, true);

        // Clear input
        self.input.clear();
        self.cursor = 0;
        self.update_completions();
    }

    // =========================================================================
    // B6: Session Persistence Helpers
    // =========================================================================

    /// Persist a command to the state manager
    fn persist_command(&mut self, command: &str, phi: f64, success: bool) {
        if let Some(ref mut manager) = self.state_manager {
            manager.add_to_history(command.to_string(), phi, success);

            // Update consciousness metrics
            manager.update_consciousness(
                self.context.current_phi,
                self.context.current_coherence,
                self.context.is_conscious,
            );
        }
    }

    /// Record a vetoed command
    fn record_veto(&mut self) {
        if let Some(ref mut manager) = self.state_manager {
            manager.record_veto();
        }
    }

    /// Auto-save if interval has elapsed
    fn maybe_auto_save(&mut self) {
        if self.last_save.elapsed() >= self.save_interval {
            self.save_state();
            self.last_save = Instant::now();
        }
    }

    /// Save current state to disk
    fn save_state(&mut self) {
        if let Some(ref mut manager) = self.state_manager
            && let Err(e) = manager.save()
        {
            tracing::warn!(error = %e, "Failed to save session state");
        }
    }

    /// Load history from persistent state into local history
    fn load_history_from_state(&mut self) {
        if let Some(ref manager) = self.state_manager {
            let state = manager.state();
            // Convert persistent history to local history entries
            for cmd in state.command_history.iter().rev().take(100) {
                self.history.push(HistoryEntry {
                    command: cmd.command.clone(),
                    timestamp: cmd.timestamp.with_timezone(&chrono::Local),
                    phi_at_execution: cmd.phi_at_execution,
                    success: cmd.success,
                });
            }

            // Restore consciousness metrics if available
            self.context.update_metrics(
                state.consciousness.phi,
                state.consciousness.coherence,
                state.consciousness.is_conscious,
            );
        }
    }

    // =========================================================================
    // B8: Error Explanation Helpers
    // =========================================================================

    /// Process output and detect/explain errors
    #[allow(dead_code)]
    fn process_output_for_errors(&mut self, output: &str) {
        // Check if this looks like an error
        if ErrorExplainer::is_error_output(output) {
            self.explain_error(output);
        }
    }

    /// Explain an error and display inline
    #[allow(dead_code)]
    fn explain_error(&mut self, error_output: &str) {
        let explanation = self.error_explainer.explain(error_output);

        // Store for /explain command
        self.last_error = Some(explanation.clone());

        // Display inline explanation
        self.output_lines.push(OutputLine {
            content: format!(
                "{} {} ({}% confidence)",
                explanation.icon, explanation.summary, explanation.confidence
            ),
            style: OutputStyle::Warning,
            timestamp: chrono::Local::now(),
        });

        // Show brief explanation
        if !explanation.explanation.is_empty() {
            self.output_lines.push(OutputLine {
                content: format!("   {}", explanation.explanation),
                style: OutputStyle::Info,
                timestamp: chrono::Local::now(),
            });
        }

        // Show primary fix if available
        if let Some(fix) = explanation
            .fixes
            .iter()
            .find(|f| f.primary)
            .or_else(|| explanation.fixes.first())
        {
            self.output_lines.push(OutputLine {
                content: format!(
                    "   {} Fix: {} [{}]",
                    fix.risk.icon(),
                    fix.description,
                    fix.risk.label()
                ),
                style: OutputStyle::Info,
                timestamp: chrono::Local::now(),
            });

            if let Some(ref cmd) = fix.command {
                self.output_lines.push(OutputLine {
                    content: format!("      → Try: {}", cmd),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                });
            }
        }

        self.output_lines.push(OutputLine {
            content: "   Use /explain for full diagnosis".to_string(),
            style: OutputStyle::Info,
            timestamp: chrono::Local::now(),
        });
    }

    // =========================================================================
    // B9: What-If Simulation Helpers
    // =========================================================================

    /// Run a what-if simulation and display results
    fn run_whatif_simulation(&mut self, command: &str) {
        // Update simulator with current flake context
        self.whatif_simulator
            .set_known_packages(self.flake_context.installed_packages.clone());
        self.whatif_simulator
            .set_known_services(self.flake_context.enabled_services.clone());

        // Run simulation
        let result = self.whatif_simulator.simulate(command);

        // Store for /whatif command
        self.last_whatif = Some(result.clone());

        // Display results
        self.output_lines.push(OutputLine {
            content: format!("=== What-If: {} ===", result.summary),
            style: OutputStyle::Info,
            timestamp: chrono::Local::now(),
        });

        self.output_lines.push(OutputLine {
            content: format!("Confidence: {:.0}%", result.confidence * 100.0),
            style: OutputStyle::Phi,
            timestamp: chrono::Local::now(),
        });

        // Packages
        if !result.packages_added.is_empty() {
            self.output_lines.push(OutputLine {
                content: format!("+ Add: {}", result.packages_added.join(", ")),
                style: OutputStyle::Success,
                timestamp: chrono::Local::now(),
            });
        }
        if !result.packages_removed.is_empty() {
            self.output_lines.push(OutputLine {
                content: format!("- Remove: {}", result.packages_removed.join(", ")),
                style: OutputStyle::Warning,
                timestamp: chrono::Local::now(),
            });
        }

        // Services
        if !result.services_enabled.is_empty() {
            self.output_lines.push(OutputLine {
                content: format!("▶ Enable: {}", result.services_enabled.join(", ")),
                style: OutputStyle::Success,
                timestamp: chrono::Local::now(),
            });
        }
        if !result.services_disabled.is_empty() {
            self.output_lines.push(OutputLine {
                content: format!("■ Disable: {}", result.services_disabled.join(", ")),
                style: OutputStyle::Warning,
                timestamp: chrono::Local::now(),
            });
        }

        // Time and reboot
        if let Some(ref time) = result.estimated_time {
            self.output_lines.push(OutputLine {
                content: format!("⏱ Time: {}", time),
                style: OutputStyle::Info,
                timestamp: chrono::Local::now(),
            });
        }

        if result.reboot_required {
            self.output_lines.push(OutputLine {
                content: "🔄 Reboot required".to_string(),
                style: OutputStyle::Warning,
                timestamp: chrono::Local::now(),
            });
        }

        // Reversibility
        if result.reversible {
            if let Some(ref cmd) = result.rollback_command {
                self.output_lines.push(OutputLine {
                    content: format!("↩ Rollback: {}", cmd),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                });
            }
        } else {
            self.output_lines.push(OutputLine {
                content: "⚠ NOT REVERSIBLE".to_string(),
                style: OutputStyle::Error,
                timestamp: chrono::Local::now(),
            });
        }

        // Warnings
        for warning in &result.warnings {
            self.output_lines.push(OutputLine {
                content: format!("⚠ {}", warning),
                style: OutputStyle::Warning,
                timestamp: chrono::Local::now(),
            });
        }

        // Dry-run command hint
        if let Some(ref cmd) = result.dry_run_command {
            self.output_lines.push(OutputLine {
                content: format!("🔍 Try dry-run: {}", cmd),
                style: OutputStyle::Info,
                timestamp: chrono::Local::now(),
            });
        }
    }

    /// Display full error explanation
    fn display_full_explanation(&mut self) {
        if let Some(ref explanation) = self.last_error {
            // Header
            self.output_lines.push(OutputLine {
                content: format!("=== {} {} ===", explanation.icon, explanation.summary),
                style: OutputStyle::Warning,
                timestamp: chrono::Local::now(),
            });

            // Category and confidence
            self.output_lines.push(OutputLine {
                content: format!(
                    "Category: {} | Type: {} | Confidence: {}%",
                    explanation.category, explanation.error_type, explanation.confidence
                ),
                style: OutputStyle::Info,
                timestamp: chrono::Local::now(),
            });

            // Explanation
            if !explanation.explanation.is_empty() {
                self.output_lines.push(OutputLine {
                    content: explanation.explanation.clone(),
                    style: OutputStyle::Normal,
                    timestamp: chrono::Local::now(),
                });
            }

            // Causes
            if !explanation.causes.is_empty() {
                self.output_lines.push(OutputLine {
                    content: "Likely Causes:".to_string(),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                });
                for (i, cause) in explanation.causes.iter().enumerate() {
                    self.output_lines.push(OutputLine {
                        content: format!("  {}. {}", i + 1, cause),
                        style: OutputStyle::Normal,
                        timestamp: chrono::Local::now(),
                    });
                }
            }

            // Fixes
            if !explanation.fixes.is_empty() {
                self.output_lines.push(OutputLine {
                    content: "Suggested Fixes:".to_string(),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                });
                for fix in &explanation.fixes {
                    let primary = if fix.primary { "★ " } else { "  " };
                    self.output_lines.push(OutputLine {
                        content: format!(
                            "{}{} {} [{}]",
                            primary,
                            fix.risk.icon(),
                            fix.description,
                            fix.risk.label()
                        ),
                        style: OutputStyle::Normal,
                        timestamp: chrono::Local::now(),
                    });

                    if let Some(ref cmd) = fix.command {
                        self.output_lines.push(OutputLine {
                            content: format!("      → {}", cmd),
                            style: OutputStyle::Info,
                            timestamp: chrono::Local::now(),
                        });
                    }
                }
            }

            // Location and affected paths
            if let Some(ref loc) = explanation.location {
                self.output_lines.push(OutputLine {
                    content: format!("Location: {}", loc),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                });
            }

            if !explanation.affected_paths.is_empty() {
                self.output_lines.push(OutputLine {
                    content: format!("Affected: {}", explanation.affected_paths.join(", ")),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                });
            }
        } else {
            self.output_lines.push(OutputLine {
                content: "No recent error to explain. Errors are captured when Nix commands fail."
                    .to_string(),
                style: OutputStyle::Info,
                timestamp: chrono::Local::now(),
            });
        }
    }

    fn handle_slash_command(&mut self, command: &str) {
        match command {
            "/quit" | "/exit" | "/q" => {
                // B6: Save state before quitting
                self.save_state();
                self.should_quit = true;
            }

            "/help" | "/?" => {
                self.output_lines.push(OutputLine {
                    content: "=== Symthaea Shell Commands ===".to_string(),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                });
                for line in [
                    "/help      - Show this help",
                    "/status    - Show consciousness status",
                    "/reconnect - Reconnect to symthaea service",
                    "/center    - Initiate centering (boost Phi)",
                    "/history   - Show command history",
                    "/overlays  - Toggle epistemic overlays (K)",
                    "/flake     - Show flake.nix context",
                    "/explain   - Show full error diagnosis",
                    "/whatif <cmd> - Simulate command without executing",
                    "/clear     - Clear output",
                    "/quit      - Exit shell",
                    "",
                    "Tab        - Accept completion",
                    "Up/Down    - Navigate completions",
                    "Ctrl+R     - History search",
                    "Ctrl+C     - Cancel current input",
                ] {
                    self.output_lines.push(OutputLine {
                        content: line.to_string(),
                        style: OutputStyle::Info,
                        timestamp: chrono::Local::now(),
                    });
                }
            }

            "/status" => {
                self.output_lines.push(OutputLine {
                    content: format!(
                        "{} {} | Phi: {:.2} | Coherence: {:.0}% | Conscious: {}",
                        self.connection_state.indicator(),
                        self.connection_state.label(),
                        self.context.current_phi,
                        self.context.current_coherence * 100.0,
                        if self.context.is_conscious {
                            "YES"
                        } else {
                            "NO"
                        }
                    ),
                    style: OutputStyle::Phi,
                    timestamp: chrono::Local::now(),
                });
                if let Some(ref path) = self.socket_path {
                    self.output_lines.push(OutputLine {
                        content: format!("Socket: {}", path.display()),
                        style: OutputStyle::Info,
                        timestamp: chrono::Local::now(),
                    });
                }
                if self.metrics_rx.is_some() {
                    self.output_lines.push(OutputLine {
                        content: "Metrics: STREAMING (push-based)".to_string(),
                        style: OutputStyle::Info,
                        timestamp: chrono::Local::now(),
                    });
                }
            }

            "/reconnect" => {
                if matches!(self.connection_state, ConnectionState::Connected) {
                    self.output_lines.push(OutputLine {
                        content: format!(
                            "{} Already connected to symthaea service",
                            self.connection_state.indicator()
                        ),
                        style: OutputStyle::Info,
                        timestamp: chrono::Local::now(),
                    });
                } else {
                    self.connection_state = ConnectionState::Connecting;
                    self.output_lines.push(OutputLine {
                        content: format!(
                            "{} Attempting to reconnect with auto-discovery...",
                            self.connection_state.indicator()
                        ),
                        style: OutputStyle::Info,
                        timestamp: chrono::Local::now(),
                    });

                    // Re-discover socket and connect with retry
                    let discovered = discover_socket();
                    if let Some(ref _path) = discovered {
                        let mut client = ShellIpcClient::new(); // Uses auto-discovery
                        let rt = Arc::clone(&self.runtime);

                        // Try connect with retry (exponential backoff)
                        let connected =
                            rt.block_on(async { client.connect_with_retry(3).await.is_ok() });

                        if connected {
                            self.socket_path = discovered;
                            self.connection_state = ConnectionState::Connected;

                            // Try to set up streaming metrics
                            let metrics_receiver = rt
                                .block_on(async { client.subscribe_metrics_watch(500).await.ok() });

                            if let Some(ref rx) = metrics_receiver {
                                let m = rx.borrow();
                                self.service_phi = m.phi;
                                self.service_coherence = m.coherence;
                                self.service_conscious = m.is_conscious;
                                self.context
                                    .update_metrics(m.phi, m.coherence, m.is_conscious);
                                self.phi_gate
                                    .update_metrics(m.phi, m.coherence, m.is_conscious);
                            }

                            self.ipc_client = Some(client);
                            self.metrics_rx = metrics_receiver;

                            self.output_lines.push(OutputLine {
                                content: format!(
                                    "{} Connected! Phi: {:.2} | Streaming: {}",
                                    self.connection_state.indicator(),
                                    self.service_phi,
                                    if self.metrics_rx.is_some() {
                                        "YES"
                                    } else {
                                        "NO"
                                    }
                                ),
                                style: OutputStyle::Success,
                                timestamp: chrono::Local::now(),
                            });
                        } else {
                            self.connection_state = ConnectionState::Disconnected;
                            self.output_lines.push(OutputLine {
                                content: format!(
                                    "{} Connection failed after retries",
                                    self.connection_state.indicator()
                                ),
                                style: OutputStyle::Error,
                                timestamp: chrono::Local::now(),
                            });
                        }
                    } else {
                        self.connection_state = ConnectionState::Disconnected;
                        self.output_lines.push(OutputLine {
                            content: format!(
                                "{} No service socket found",
                                self.connection_state.indicator()
                            ),
                            style: OutputStyle::Error,
                            timestamp: chrono::Local::now(),
                        });
                    }
                }
            }

            "/center" => {
                self.output_lines.push(OutputLine {
                    content: "Initiating centering...".to_string(),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                });
                // Boost Phi
                let new_phi = (self.context.current_phi + 0.2).min(0.95);
                self.context.update_metrics(new_phi, 0.9, true);
                self.phi_gate.update_metrics(new_phi, 0.9, true);

                self.output_lines.push(OutputLine {
                    content: format!("Centered. Phi now: {:.2}", new_phi),
                    style: OutputStyle::Success,
                    timestamp: chrono::Local::now(),
                });
            }

            "/history" => {
                if self.history.is_empty() {
                    self.output_lines.push(OutputLine {
                        content: "No command history".to_string(),
                        style: OutputStyle::Info,
                        timestamp: chrono::Local::now(),
                    });
                } else {
                    for (i, entry) in self.history.iter().rev().take(10).enumerate() {
                        self.output_lines.push(OutputLine {
                            content: format!(
                                "{:2}. [Phi:{:.2}] {}",
                                i + 1,
                                entry.phi_at_execution,
                                entry.command
                            ),
                            style: if entry.success {
                                OutputStyle::Normal
                            } else {
                                OutputStyle::Error
                            },
                            timestamp: chrono::Local::now(),
                        });
                    }
                }
            }

            "/clear" => {
                self.output_lines.clear();
            }

            "/overlays" | "/epistemic" => {
                self.show_epistemic_overlays = !self.show_epistemic_overlays;
                self.output_lines.push(OutputLine {
                    content: format!(
                        "Epistemic overlays: {}",
                        if self.show_epistemic_overlays {
                            "ON"
                        } else {
                            "OFF"
                        }
                    ),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                });

                if self.show_epistemic_overlays {
                    self.update_epistemic_overlays();
                } else {
                    self.active_overlays.clear();
                }
            }

            // B7: Flake context commands
            "/flake" => {
                self.output_lines.push(OutputLine {
                    content: self.flake_context.status_summary(),
                    style: OutputStyle::Info,
                    timestamp: chrono::Local::now(),
                });

                // Show installed packages
                if !self.flake_context.installed_packages.is_empty() {
                    let pkgs: Vec<&str> = self
                        .flake_context
                        .installed_packages
                        .iter()
                        .take(10)
                        .map(|s| s.as_str())
                        .collect();
                    self.output_lines.push(OutputLine {
                        content: format!("Packages: {} ...", pkgs.join(", ")),
                        style: OutputStyle::Info,
                        timestamp: chrono::Local::now(),
                    });
                }

                // Show enabled services
                if !self.flake_context.enabled_services.is_empty() {
                    let svcs: Vec<&str> = self
                        .flake_context
                        .enabled_services
                        .iter()
                        .take(10)
                        .map(|s| s.as_str())
                        .collect();
                    self.output_lines.push(OutputLine {
                        content: format!("Services: {} ...", svcs.join(", ")),
                        style: OutputStyle::Info,
                        timestamp: chrono::Local::now(),
                    });
                }
            }

            "/flake reload" => {
                // Force reload flake context
                self.flake_context = FlakeContext::discover_and_load();
                self.output_lines.push(OutputLine {
                    content: format!("Reloaded: {}", self.flake_context.status_summary()),
                    style: OutputStyle::Success,
                    timestamp: chrono::Local::now(),
                });
            }

            // B8: Error explanation command
            "/explain" | "/error" => {
                self.display_full_explanation();
            }

            // B9: What-if simulation (with argument)
            cmd if cmd.starts_with("/whatif ") => {
                let sim_cmd = cmd.trim_start_matches("/whatif ").trim();
                self.run_whatif_simulation(sim_cmd);
            }

            // Show last what-if result
            "/whatif" => {
                if let Some(ref result) = self.last_whatif {
                    for line in self.whatif_simulator.format_for_terminal(result) {
                        self.output_lines.push(OutputLine {
                            content: line,
                            style: OutputStyle::Info,
                            timestamp: chrono::Local::now(),
                        });
                    }
                } else {
                    self.output_lines.push(OutputLine {
                        content: "Usage: /whatif <command> - Simulate a command".to_string(),
                        style: OutputStyle::Info,
                        timestamp: chrono::Local::now(),
                    });
                    self.output_lines.push(OutputLine {
                        content: "Example: /whatif install firefox".to_string(),
                        style: OutputStyle::Info,
                        timestamp: chrono::Local::now(),
                    });
                }
            }

            _ => {
                self.output_lines.push(OutputLine {
                    content: format!("Unknown command: {}", command),
                    style: OutputStyle::Error,
                    timestamp: chrono::Local::now(),
                });
            }
        }
    }

    fn accept_completion(&mut self) {
        if let Some(completion) = self.completions.get(self.selected_completion) {
            self.input = completion.text.clone();
            self.cursor = self.input.len();
            self.completions.clear();
            self.show_completions = false;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and load persistent history
    let mut app = App::new();
    app.load_history_from_state();
    let tick_rate = Duration::from_millis(100);

    // Main loop
    loop {
        // B6: Periodic auto-save check
        app.maybe_auto_save();
        terminal.draw(|f| ui(f, &app))?;

        let timeout = tick_rate;
        if event::poll(timeout)?
            && let Event::Key(key) = event::read()?
        {
            match app.mode {
                ShellMode::Normal | ShellMode::Completing => {
                    // Reset tab_pressed on any non-Tab key
                    if key.code != KeyCode::Tab {
                        app.tab_pressed = false;
                    }

                    match key.code {
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            app.input.clear();
                            app.cursor = 0;
                            app.update_completions();
                            app.update_command_preview();
                        }
                        KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            // B6: Save state before quitting
                            app.save_state();
                            app.should_quit = true;
                        }
                        // B2: Ctrl+R for reverse history search
                        KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            app.start_history_search();
                        }
                        KeyCode::Char(c) => {
                            app.input.insert(app.cursor, c);
                            app.cursor += 1;
                            app.update_completions();
                            app.update_command_preview();
                        }
                        KeyCode::Backspace if app.cursor > 0 => {
                            app.cursor -= 1;
                            app.input.remove(app.cursor);
                            app.update_completions();
                        }
                        KeyCode::Delete if app.cursor < app.input.len() => {
                            app.input.remove(app.cursor);
                            app.update_completions();
                        }
                        KeyCode::Left if app.cursor > 0 => {
                            app.cursor -= 1;
                        }
                        KeyCode::Right if app.cursor < app.input.len() => {
                            app.cursor += 1;
                        }
                        KeyCode::Home => {
                            app.cursor = 0;
                        }
                        KeyCode::End => {
                            app.cursor = app.input.len();
                        }
                        KeyCode::Up if app.show_completions && app.selected_completion > 0 => {
                            app.selected_completion -= 1;
                        }
                        KeyCode::Down
                            if app.show_completions
                                && app.selected_completion
                                    < app.completions.len().saturating_sub(1) =>
                        {
                            app.selected_completion += 1;
                        }
                        // B1: Enhanced Tab completion with cycling
                        KeyCode::Tab if app.show_completions && !app.completions.is_empty() => {
                            if app.tab_pressed {
                                // Second Tab: cycle to next completion
                                app.selected_completion =
                                    (app.selected_completion + 1) % app.completions.len();
                                // Update input to show selected completion
                                if let Some(completion) =
                                    app.completions.get(app.selected_completion)
                                {
                                    app.input = completion.text.clone();
                                    app.cursor = app.input.len();
                                }
                            } else {
                                // First Tab: show first completion
                                app.tab_pressed = true;
                                if let Some(completion) =
                                    app.completions.get(app.selected_completion)
                                {
                                    app.input = completion.text.clone();
                                    app.cursor = app.input.len();
                                }
                            }
                            app.update_command_preview();
                        }
                        // Shift+Tab: cycle backwards
                        KeyCode::BackTab if app.show_completions && !app.completions.is_empty() => {
                            if app.selected_completion > 0 {
                                app.selected_completion -= 1;
                            } else {
                                app.selected_completion = app.completions.len() - 1;
                            }
                            if let Some(completion) = app.completions.get(app.selected_completion) {
                                app.input = completion.text.clone();
                                app.cursor = app.input.len();
                            }
                            app.update_command_preview();
                        }
                        KeyCode::Enter => {
                            if app.show_completions {
                                app.accept_completion();
                            }
                            app.execute_command();
                        }
                        KeyCode::Esc => {
                            app.show_completions = false;
                            app.mode = ShellMode::Normal;
                        }
                        _ => {}
                    }
                }
                ShellMode::Confirming => {
                    match key.code {
                        KeyCode::Char('y') | KeyCode::Char('Y') => {
                            // Execute the pending command
                            if let Some(pending) = app.pending_command.take() {
                                app.output_lines.push(OutputLine {
                                    content: format!("Confirmed - executing: {}", pending.command),
                                    style: OutputStyle::Success,
                                    timestamp: chrono::Local::now(),
                                });

                                // Execute command (simplified - real impl would use ProcessRunner)
                                app.input = pending.command;
                                app.execute_confirmed_command();
                            } else {
                                app.output_lines.push(OutputLine {
                                    content: "Confirmed (no pending command)".to_string(),
                                    style: OutputStyle::Warning,
                                    timestamp: chrono::Local::now(),
                                });
                            }
                            app.mode = ShellMode::Normal;
                        }
                        KeyCode::Char('d') | KeyCode::Char('D') => {
                            // Dry-run mode
                            if let Some(ref pending) = app.pending_command {
                                app.output_lines.push(OutputLine {
                                    content: format!(
                                        "[DRY-RUN] Would execute: {}",
                                        pending.command
                                    ),
                                    style: OutputStyle::Info,
                                    timestamp: chrono::Local::now(),
                                });
                                app.output_lines.push(OutputLine {
                                    content: "No changes made. Press 'y' to execute for real."
                                        .to_string(),
                                    style: OutputStyle::Info,
                                    timestamp: chrono::Local::now(),
                                });
                            }
                        }
                        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                            if let Some(ref pending) = app.pending_command {
                                app.output_lines.push(OutputLine {
                                    content: format!("Cancelled: {}", pending.command),
                                    style: OutputStyle::Warning,
                                    timestamp: chrono::Local::now(),
                                });
                            }
                            app.pending_command = None;
                            app.mode = ShellMode::Normal;
                        }
                        _ => {}
                    }
                }
                ShellMode::Help => {
                    if key.code == KeyCode::Esc || key.code == KeyCode::Char('q') {
                        app.mode = ShellMode::Normal;
                    }
                }

                // B2: History search mode
                ShellMode::HistorySearch => {
                    match key.code {
                        KeyCode::Esc | KeyCode::Char('g')
                            if key.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            // Cancel search
                            app.mode = ShellMode::Normal;
                        }
                        KeyCode::Enter => {
                            // Accept selected
                            app.accept_history_search();
                        }
                        KeyCode::Up | KeyCode::Char('p')
                            if key.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            // Previous match
                            if app.history_search_selected > 0 {
                                app.history_search_selected -= 1;
                            }
                        }
                        KeyCode::Down | KeyCode::Char('n')
                            if key.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            // Next match
                            if app.history_search_selected
                                < app.history_search_matches.len().saturating_sub(1)
                            {
                                app.history_search_selected += 1;
                            }
                        }
                        KeyCode::Backspace => {
                            app.history_search_query.pop();
                            app.update_history_search();
                        }
                        KeyCode::Char(c) => {
                            app.history_search_query.push(c);
                            app.update_history_search();
                        }
                        _ => {}
                    }
                }
            }
        }
        app.on_tick();

        if app.should_quit {
            break;
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}

fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Input
            Constraint::Min(10),   // Main area
            Constraint::Length(3), // Status bar
        ])
        .split(f.area());

    // Input area with connection state indicator
    let connection_color = match app.connection_state {
        ConnectionState::Connected => Color::Green,
        ConnectionState::Connecting | ConnectionState::Reconnecting => Color::Yellow,
        ConnectionState::Degraded => Color::Rgb(255, 165, 0), // Orange
        ConnectionState::Disconnected => Color::Gray,
    };

    let input_block = Block::default()
        .borders(Borders::ALL)
        .title(format!(
            " {} {} Symthaea Shell [Phi: {:.2}] ",
            app.connection_state.indicator(),
            app.context.consciousness_indicator(),
            app.context.current_phi
        ))
        .border_style(Style::default().fg(connection_color));

    let input_text = Paragraph::new(format!("> {}", app.input)).block(input_block);
    f.render_widget(input_text, chunks[0]);

    // Main area split
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(chunks[1]);

    // Left side: Output + Completions
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),
            Constraint::Length(if app.show_completions { 8 } else { 0 }),
        ])
        .split(main_chunks[0]);

    // Output area
    let output_items: Vec<ListItem> = app
        .output_lines
        .iter()
        .rev()
        .take(20)
        .rev()
        .map(|line| {
            let style = match line.style {
                OutputStyle::Normal => Style::default(),
                OutputStyle::Success => Style::default().fg(Color::Green),
                OutputStyle::Warning => Style::default().fg(Color::Yellow),
                OutputStyle::Error => Style::default().fg(Color::Red),
                OutputStyle::Info => Style::default().fg(Color::Cyan),
                OutputStyle::Phi => Style::default().fg(Color::Magenta),
            };
            ListItem::new(Line::from(Span::styled(&line.content, style)))
        })
        .collect();

    let output_list =
        List::new(output_items).block(Block::default().borders(Borders::ALL).title(" Output "));
    f.render_widget(output_list, left_chunks[0]);

    // Completions popup
    if app.show_completions && !app.completions.is_empty() {
        let completion_items: Vec<ListItem> = app
            .completions
            .iter()
            .enumerate()
            .take(6)
            .map(|(i, c)| {
                let selected = i == app.selected_completion;
                let style = if selected {
                    Style::default()
                        .bg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };

                let destructiveness_color = match c.destructiveness {
                    DestructivenessLevel::ReadOnly => Color::Green,
                    DestructivenessLevel::Reversible => Color::Blue,
                    DestructivenessLevel::NeedsConfirmation => Color::Yellow,
                    DestructivenessLevel::Destructive => Color::Red,
                };

                ListItem::new(Line::from(vec![
                    Span::styled(c.kind.icon(), Style::default().fg(Color::Cyan)),
                    Span::raw(" "),
                    Span::styled(&c.text, style),
                    Span::raw(" "),
                    Span::styled(
                        format!("[{:.0}%]", c.confidence * 100.0),
                        Style::default().fg(destructiveness_color),
                    ),
                ]))
            })
            .collect();

        let completions_list = List::new(completion_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Completions (Tab to accept) "),
        );
        f.render_widget(completions_list, left_chunks[1]);
    }

    // Right side: Metrics + Preview
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12), // Expanded for more metrics
            Constraint::Min(5),
        ])
        .split(main_chunks[1]);

    // Enhanced metrics panel with streaming indicators
    let metrics_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Phi gauge
            Constraint::Length(3), // Coherence gauge
            Constraint::Length(3), // Status line
        ])
        .split(right_chunks[0]);

    // Phi gauge with streaming indicator
    let stream_indicator = if app.metrics_rx.is_some() {
        " [LIVE]"
    } else {
        ""
    };
    let phi_gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Phi{} ", stream_indicator)),
        )
        .gauge_style(Style::default().fg(if app.context.current_phi >= 0.7 {
            Color::Green
        } else if app.context.current_phi >= 0.4 {
            Color::Yellow
        } else {
            Color::Red
        }))
        .ratio(app.context.current_phi.clamp(0.0, 1.0))
        .label(format!("{:.2}", app.context.current_phi));
    f.render_widget(phi_gauge, metrics_chunks[0]);

    // Coherence gauge
    let coherence_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Coherence "))
        .gauge_style(
            Style::default().fg(if app.context.current_coherence >= 0.8 {
                Color::Cyan
            } else if app.context.current_coherence >= 0.5 {
                Color::Blue
            } else {
                Color::Magenta
            }),
        )
        .ratio(app.context.current_coherence.clamp(0.0, 1.0))
        .label(format!("{:.0}%", app.context.current_coherence * 100.0));
    f.render_widget(coherence_gauge, metrics_chunks[1]);

    // Status line with threat level and connection
    let threat_level = app.phi_gate.threat_level_value();
    let threat_color = if threat_level >= 0.8 {
        Color::Red
    } else if threat_level >= 0.5 {
        Color::Yellow
    } else {
        Color::Green
    };
    let conscious_icon = if app.context.is_conscious {
        "●"
    } else {
        "○"
    };
    let connection = if app.ipc_client.is_some() {
        "Connected"
    } else {
        "Local"
    };

    let status_spans = vec![
        Span::styled(
            format!("{} ", conscious_icon),
            Style::default().fg(if app.context.is_conscious {
                Color::Green
            } else {
                Color::DarkGray
            }),
        ),
        Span::styled(
            format!("Threat: {:.0}% ", threat_level * 100.0),
            Style::default().fg(threat_color),
        ),
        Span::styled(
            format!("| {} ", connection),
            Style::default().fg(Color::DarkGray),
        ),
    ];
    let status_para = Paragraph::new(Line::from(status_spans))
        .block(Block::default().borders(Borders::ALL).title(" Status "));
    f.render_widget(status_para, metrics_chunks[2]);

    // B3: Preview panel with enhanced command preview
    let preview_text = if app.mode == ShellMode::HistorySearch {
        // B2: Show history search results
        let mut lines: Vec<Line> = vec![
            Line::from(Span::styled(
                format!("Search: {}_", app.history_search_query),
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];

        for (i, &idx) in app.history_search_matches.iter().take(5).enumerate() {
            if let Some(entry) = app.history.get(idx) {
                let style = if i == app.history_search_selected {
                    Style::default().bg(Color::DarkGray).fg(Color::White)
                } else {
                    Style::default()
                };
                lines.push(Line::from(Span::styled(&entry.command, style)));
            }
        }

        if app.history_search_matches.is_empty() && !app.history_search_query.is_empty() {
            lines.push(Line::from(Span::styled(
                "No matches",
                Style::default().fg(Color::DarkGray),
            )));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "↑↓ Navigate | Enter Accept | Esc Cancel",
            Style::default().fg(Color::DarkGray),
        )));
        lines
    } else if let Some(ref preview) = app.command_preview {
        // B3: Show command preview from our generate_preview
        let mut lines: Vec<Line> = vec![
            Line::from(Span::styled(
                &preview.description,
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .fg(if preview.needs_confirmation {
                        Color::Yellow
                    } else {
                        Color::Green
                    }),
            )),
            Line::from(""),
        ];

        lines.push(Line::from(Span::styled(
            "Steps:",
            Style::default().fg(Color::Cyan),
        )));
        for (i, step) in preview.steps.iter().enumerate() {
            lines.push(Line::from(format!("  {}. {}", i + 1, step)));
        }

        if !preview.affected.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "Affected:",
                Style::default().fg(Color::Yellow),
            )));
            for path in &preview.affected {
                lines.push(Line::from(format!("  • {}", path)));
            }
        }

        if let Some(ref time) = preview.estimated_time {
            lines.push(Line::from(""));
            lines.push(Line::from(format!("Est. time: {}", time)));
        }

        if preview.needs_confirmation {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "⚠ Requires confirmation",
                Style::default().fg(Color::Yellow),
            )));
        }

        lines
    } else if let Some(completion) = app.completions.get(app.selected_completion) {
        // Fallback to completion preview
        if let Some(ref preview) = completion.preview {
            let mut lines: Vec<Line> = vec![
                Line::from(Span::styled(
                    format!("Command: {}", completion.text),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
            ];

            for step in &preview.steps {
                lines.push(Line::from(format!("{}. {}", step.number, step.description)));
            }

            if let Some(ref time) = preview.estimated_time {
                lines.push(Line::from(""));
                lines.push(Line::from(format!("Est. time: {}", time)));
            }

            lines
        } else {
            vec![Line::from("Select a completion to see preview")]
        }
    } else {
        vec![
            Line::from("Type to see completions"),
            Line::from(""),
            Line::from(Span::styled("Shortcuts:", Style::default().fg(Color::Cyan))),
            Line::from("  Tab     Cycle completions"),
            Line::from("  Ctrl+R  Search history"),
            Line::from("  /help   Show commands"),
        ]
    };

    let preview = Paragraph::new(preview_text)
        .block(Block::default().borders(Borders::ALL).title(
            if app.mode == ShellMode::HistorySearch {
                " History Search (Ctrl+R) "
            } else {
                " Preview "
            },
        ))
        .wrap(Wrap { trim: true });
    f.render_widget(preview, right_chunks[1]);

    // Status bar with connection indicator
    let status = Paragraph::new(format!(
        " {} {} | Mode: {:?} | History: {} | Coherence: {:.0}% | /help ",
        app.connection_state.indicator(),
        app.connection_state.label(),
        app.mode,
        app.history.len(),
        app.context.current_coherence * 100.0
    ))
    .style(Style::default().bg(Color::DarkGray).fg(connection_color));
    f.render_widget(status, chunks[2]);

    // B4: Confirmation dialog overlay
    if app.mode == ShellMode::Confirming
        && let Some(ref pending) = app.pending_command
    {
        // Calculate centered popup area
        let popup_width = 60u16.min(f.area().width.saturating_sub(4));
        let popup_height = 14u16.min(f.area().height.saturating_sub(4));
        let popup_x = (f.area().width.saturating_sub(popup_width)) / 2;
        let popup_y = (f.area().height.saturating_sub(popup_height)) / 2;
        let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

        // Clear the area first
        f.render_widget(Clear, popup_area);

        // Risk-based styling
        let (border_color, title_color, icon) = match pending.risk_level {
            RiskLevel::High => (Color::Red, Color::Red, "⚠️ "),
            RiskLevel::Medium => (Color::Yellow, Color::Yellow, "⚡ "),
            RiskLevel::Low => (Color::Cyan, Color::Cyan, "📋 "),
        };

        let risk_label = match pending.risk_level {
            RiskLevel::High => "NON-REVERSIBLE COMMAND",
            RiskLevel::Medium => "CONFIRMATION REQUIRED",
            RiskLevel::Low => "CONFIRMATION NEEDED",
        };

        // Build dialog content
        let mut lines: Vec<Line> = vec![
            Line::from(Span::styled(
                format!("{}{}", icon, risk_label),
                Style::default()
                    .fg(title_color)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("Command: ", Style::default().fg(Color::Cyan)),
                Span::raw(&pending.command),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Reason: ", Style::default().fg(Color::Yellow)),
                Span::raw(&pending.reason),
            ]),
        ];

        // Add rollback hint if available
        if let Some(ref hint) = pending.rollback_hint {
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("Rollback: ", Style::default().fg(Color::Green)),
                Span::raw(hint),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "─".repeat(popup_width.saturating_sub(4) as usize),
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(vec![
            Span::styled(
                " [Y] ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("Confirm  "),
            Span::styled(
                " [D] ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("Dry-run  "),
            Span::styled(
                " [N] ",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::raw("Cancel"),
        ]));

        let dialog = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(border_color))
                    .title(" Confirm Execution ")
                    .title_style(
                        Style::default()
                            .fg(title_color)
                            .add_modifier(Modifier::BOLD),
                    ),
            )
            .wrap(Wrap { trim: true });
        f.render_widget(dialog, popup_area);
    }

    // B5: Epistemic overlays - floating contextual knowledge markup
    if app.show_epistemic_overlays && !app.active_overlays.is_empty() {
        // Only show when not in confirming mode
        if app.mode != ShellMode::Confirming {
            // Render each overlay based on its position
            for (i, overlay) in app.active_overlays.iter().take(2).enumerate() {
                let (overlay_x, overlay_y, overlay_width, overlay_height) = match overlay.position {
                    OverlayPosition::InputCursor => {
                        // Near input cursor, offset by overlay index
                        let x = 5 + app.cursor as u16 + (i as u16 * 3);
                        let y = 4 + (i as u16 * 2);
                        (x.min(f.area().width.saturating_sub(35)), y, 32u16, 5u16)
                    }
                    OverlayPosition::AboveCompletion => {
                        // Above completions list
                        let x = 2;
                        let y = chunks[1].y.saturating_sub(6).max(chunks[0].bottom());
                        (x, y, 40u16, 5u16)
                    }
                    OverlayPosition::MetricsArea => {
                        // In the metrics/right panel area
                        let x = main_chunks[1].x + 1;
                        let y = right_chunks[0].bottom() + (i as u16 * 6);
                        (x, y.min(f.area().height.saturating_sub(6)), 28u16, 5u16)
                    }
                    OverlayPosition::Centered => {
                        // Centered on screen
                        let w = 45u16;
                        let h = 8u16;
                        let x = (f.area().width.saturating_sub(w)) / 2;
                        let y = (f.area().height.saturating_sub(h)) / 2;
                        (x, y, w, h)
                    }
                };

                // Don't render if off-screen
                if overlay_x >= f.area().width || overlay_y >= f.area().height {
                    continue;
                }

                let overlay_area = Rect::new(
                    overlay_x,
                    overlay_y,
                    overlay_width.min(f.area().width.saturating_sub(overlay_x)),
                    overlay_height.min(f.area().height.saturating_sub(overlay_y)),
                );

                // Epistemic style colors
                let style_color = match overlay.style {
                    EpistemicStyle::HighConfidence => Color::Green,
                    EpistemicStyle::MediumConfidence => Color::Yellow,
                    EpistemicStyle::LowConfidence => Color::Red,
                    EpistemicStyle::Unknown => Color::DarkGray,
                    EpistemicStyle::Warning => Color::Rgb(255, 165, 0), // Orange
                };

                // Build overlay content
                let mut lines: Vec<Line> = vec![Line::from(Span::styled(
                    &overlay.header,
                    Style::default()
                        .fg(style_color)
                        .add_modifier(Modifier::BOLD),
                ))];

                for body_line in &overlay.body {
                    lines.push(Line::from(Span::raw(body_line)));
                }

                if let Some(ref footer) = overlay.footer {
                    lines.push(Line::from(Span::styled(
                        footer,
                        Style::default().fg(Color::DarkGray),
                    )));
                }

                // Clear area and render
                f.render_widget(Clear, overlay_area);
                let overlay_widget = Paragraph::new(lines)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(style_color))
                            .title(format!(
                                " {} ",
                                match overlay.overlay_type {
                                    OverlayType::KnowledgeSource => "K",
                                    OverlayType::UncertaintyWarning => "?",
                                    OverlayType::SafetyHint => "!",
                                    OverlayType::ConfidenceLevel => "%",
                                    OverlayType::TheoryInsight => "T",
                                }
                            )),
                    )
                    .wrap(Wrap { trim: true });
                f.render_widget(overlay_widget, overlay_area);
            }
        }
    }
}
