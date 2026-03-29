// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shell Context - State and Configuration for Shell Sessions
//!
//! Manages shell state including:
//! - Current working directory and flake context
//! - Command history and completion state
//! - Phi-gated execution policies
//! - HDC-based semantic context for intelligent completions

use crate::action::DestructivenessLevel;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

// ═══════════════════════════════════════════════════════════════════════════════
// SHELL CONTEXT
// ═══════════════════════════════════════════════════════════════════════════════

/// Shell session context
#[derive(Debug, Clone)]
pub struct ShellContext {
    /// Current working directory
    pub cwd: PathBuf,
    /// Current flake path (if in a flake project)
    pub flake_path: Option<PathBuf>,
    /// Environment variables
    pub env: HashMap<String, String>,
    /// Current user
    pub user: String,
    /// Hostname
    pub hostname: String,
    /// Whether running as root
    pub is_root: bool,
    /// Current Nix profile
    pub nix_profile: Option<String>,
    /// Shell history (most recent last)
    pub history: VecDeque<String>,
    /// Maximum history size
    pub max_history: usize,
    /// Session ID
    pub session_id: String,
    /// Current Phi level (consciousness integration)
    pub current_phi: f64,
    /// Current coherence level
    pub current_coherence: f64,
    /// Whether system is conscious
    pub is_conscious: bool,
}

impl ShellContext {
    /// Create a new shell context
    pub fn new() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        let user = std::env::var("USER").unwrap_or_else(|_| "unknown".to_string());

        let hostname = std::env::var("HOSTNAME").unwrap_or_else(|_| {
            std::fs::read_to_string("/etc/hostname")
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|_| "localhost".to_string())
        });

        Self {
            cwd,
            flake_path: None,
            env: std::env::vars().collect(),
            user: user.clone(),
            hostname,
            is_root: user == "root",
            nix_profile: std::env::var("NIX_PROFILES").ok(),
            history: VecDeque::new(),
            max_history: 1000,
            session_id: uuid::Uuid::new_v4().to_string(),
            current_phi: 0.5,
            current_coherence: 0.5,
            is_conscious: false,
        }
    }

    /// Detect flake context from current directory
    pub fn detect_flake(&mut self) {
        let mut path = self.cwd.clone();
        loop {
            let flake_nix = path.join("flake.nix");
            if flake_nix.exists() {
                self.flake_path = Some(path);
                return;
            }
            if !path.pop() {
                break;
            }
        }
        self.flake_path = None;
    }

    /// Add command to history
    pub fn add_history(&mut self, command: String) {
        // Don't add duplicates (check if already exists anywhere in history)
        if !self.history.contains(&command) {
            self.history.push_back(command);
        }
        // Trim history
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Add command to history (alias for add_history)
    pub fn add_to_history(&mut self, command: String) {
        self.add_history(command);
    }

    /// Get environment variable
    pub fn get_env(&self, key: &str) -> Option<&str> {
        self.env.get(key).map(|s| s.as_str())
    }

    /// Set environment variable
    pub fn set_env(&mut self, key: String, value: String) {
        self.env.insert(key, value);
    }

    /// Change directory
    pub fn cd(&mut self, path: PathBuf) {
        self.cwd = path;
        self.detect_flake();
    }

    /// Update context from metrics values
    pub fn update_metrics(&mut self, phi: f64, coherence: f64, is_conscious: bool) {
        self.current_phi = phi;
        self.current_coherence = coherence;
        self.is_conscious = is_conscious;
    }

    /// Update context from metrics snapshot
    pub fn update_from_snapshot(&mut self, metrics: &super::ipc_client::MetricsSnapshot) {
        self.current_phi = metrics.phi;
        self.current_coherence = metrics.coherence;
        self.is_conscious = metrics.is_conscious;
    }

    /// Get consciousness indicator character
    pub fn consciousness_indicator(&self) -> &'static str {
        if self.is_conscious {
            "●" // Conscious (filled circle)
        } else {
            "○" // Not conscious (empty circle)
        }
    }

    /// Get prompt string
    pub fn prompt(&self) -> String {
        let path_display = if let Some(ref flake) = self.flake_path {
            if self.cwd.starts_with(flake) {
                let relative = self.cwd.strip_prefix(flake).unwrap_or(&self.cwd);
                format!("❄ {}", relative.display())
            } else {
                self.cwd.display().to_string()
            }
        } else {
            self.cwd.display().to_string()
        };

        let user_host = format!("{}@{}", self.user, self.hostname);
        let symbol = if self.is_root { "#" } else { "λ" };

        format!("{user_host} {path_display} {symbol} ")
    }

    /// Get ANSI color code based on current Phi level
    /// Returns escape sequence like "\x1b[32m" for green
    pub fn status_color(&self) -> String {
        if self.current_phi >= 0.7 {
            "\x1b[32m".to_string() // Green - high consciousness
        } else if self.current_phi >= 0.4 {
            "\x1b[33m".to_string() // Yellow - medium consciousness
        } else {
            "\x1b[31m".to_string() // Red - low consciousness
        }
    }
}

impl Default for ShellContext {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPLETIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Kind of completion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CompletionKind {
    /// Package name
    #[default]
    Package,
    /// Command
    Command,
    /// File path
    Path,
    /// Flake reference
    Flake,
    /// Option/attribute
    Option,
    /// History entry
    History,
    /// Alias
    Alias,
    /// NixOS module
    Module,
    /// Service name
    Service,
    /// Nix command (nix build, nix run, etc.)
    NixCommand,
    /// Attribute path (pkgs.foo.bar)
    AttrPath,
    /// Variable or environment setting
    Variable,
}

impl CompletionKind {
    /// Get icon for this completion kind
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Package => "📦",
            Self::Command => "⚡",
            Self::Path => "📁",
            Self::Flake => "❄",
            Self::Option => "⚙",
            Self::History => "🕒",
            Self::Alias => "→",
            Self::Module => "🧩",
            Self::Service => "🔧",
            Self::NixCommand => "λ",
            Self::AttrPath => "•",
            Self::Variable => "$",
        }
    }
}

/// A step in a command preview
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreviewStep {
    /// Step number
    pub number: u32,
    /// Step description
    pub description: String,
}

/// Preview of what a command will do
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompletionPreview {
    /// Steps that will be executed
    pub steps: Vec<PreviewStep>,
    /// Estimated time (if known)
    pub estimated_time: Option<String>,
}

/// A completion suggestion
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Completion {
    /// The completion text to insert
    pub text: String,
    /// Display label (may differ from text)
    pub label: String,
    /// Display string for UI (with icon)
    pub display: String,
    /// Description of the completion
    pub description: String,
    /// Kind of completion
    pub kind: CompletionKind,
    /// Similarity score from HDC (0.0 to 1.0)
    pub similarity: f32,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// HDC distance (lower = more similar)
    pub hdc_distance: f32,
    /// Optional documentation
    pub documentation: Option<String>,
    /// Whether this is a partial completion (more to come)
    pub partial: bool,
    /// Destructiveness level for UI display
    pub destructiveness: DestructivenessLevel,
    /// Preview of what the command will do
    pub preview: Option<CompletionPreview>,
}

impl Completion {
    /// Create a new completion
    pub fn new(text: impl Into<String>, kind: CompletionKind) -> Self {
        let text = text.into();
        let display = format!("{} {}", kind.icon(), text);
        Self {
            label: text.clone(),
            display,
            description: String::new(),
            text,
            kind,
            similarity: 1.0,
            confidence: 1.0,
            hdc_distance: 0.0,
            documentation: None,
            partial: false,
            destructiveness: DestructivenessLevel::ReadOnly,
            preview: None,
        }
    }

    /// With description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// With confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// With HDC distance
    pub fn with_hdc_distance(mut self, distance: f32) -> Self {
        self.hdc_distance = distance;
        self
    }

    /// With documentation
    pub fn with_docs(mut self, docs: impl Into<String>) -> Self {
        self.documentation = Some(docs.into());
        self
    }

    /// With similarity score
    pub fn with_similarity(mut self, score: f32) -> Self {
        self.similarity = score;
        self
    }

    /// With custom label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTELLISENSE ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

/// HDC-based IntelliSense engine for semantic completions
pub struct IntelliSenseEngine {
    /// Known packages (lazy loaded)
    packages: Vec<String>,
    /// Known commands
    commands: Vec<String>,
    /// Known flake outputs (reserved for future use)
    _flake_outputs: Vec<String>,
    /// Context for semantic matching
    context: ShellContext,
    /// Whether packages are loaded
    packages_loaded: bool,
}

impl IntelliSenseEngine {
    /// Create a new IntelliSense engine with default context
    pub fn new() -> Self {
        Self::with_context(ShellContext::new())
    }

    /// Create a new IntelliSense engine with specific context
    pub fn with_context(context: ShellContext) -> Self {
        // Common NixOS commands - both short forms and full command names
        let commands = vec![
            // Short command names
            "install",
            "remove",
            "search",
            "info",
            "list",
            "rebuild",
            "switch",
            "test",
            "rollback",
            "gc",
            "flake",
            "update",
            "build",
            "run",
            "develop",
            "profile",
            "channel",
            "doctor",
            "store",
            // Full nix commands
            "nix",
            "nix-env",
            "nix-shell",
            "nix-build",
            "nix-store",
            "nix-channel",
            "nix-collect-garbage",
            "nix-instantiate",
            "nixos-rebuild",
            "nixos-option",
            "nixos-generate-config",
            "nix search",
            "nix build",
            "nix run",
            "nix develop",
            "nix shell",
            "nix flake",
            "nix profile",
            "nix-env -i",
            "nix-env -e",
            "nix-env -q",
            "nix profile install",
            "nix profile remove",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        Self {
            packages: Vec::new(),
            commands,
            _flake_outputs: Vec::new(),
            context,
            packages_loaded: false,
        }
    }

    /// Get completions for input
    pub fn complete(&self, input: &str, cursor_pos: usize) -> Vec<Completion> {
        let prefix = &input[..cursor_pos.min(input.len())];
        let mut completions = Vec::new();

        // Split into parts
        let parts: Vec<&str> = prefix.split_whitespace().collect();

        if parts.is_empty() || (parts.len() == 1 && !prefix.ends_with(' ')) {
            // Complete commands
            let cmd_prefix = parts.first().unwrap_or(&"");

            // Direct prefix matching
            for cmd in &self.commands {
                if cmd.starts_with(cmd_prefix) {
                    completions.push(
                        Completion::new(cmd.clone(), CompletionKind::Command)
                            .with_confidence(self.prefix_similarity(cmd, cmd_prefix))
                            .with_similarity(self.prefix_similarity(cmd, cmd_prefix)),
                    );
                }
            }

            // Semantic similarity matching for common terms
            let semantic_matches = self.get_semantic_matches(cmd_prefix);
            for (cmd, similarity) in semantic_matches {
                // Don't add if already present
                if !completions.iter().any(|c| c.text == cmd) {
                    completions.push(
                        Completion::new(cmd, CompletionKind::NixCommand)
                            .with_confidence(similarity)
                            .with_similarity(similarity),
                    );
                }
            }
        } else {
            // Complete arguments based on command
            let cmd = parts[0];
            let arg_prefix = if prefix.ends_with(' ') {
                ""
            } else {
                parts.last().unwrap_or(&"")
            };

            match cmd {
                "install" | "remove" | "search" | "info" => {
                    // Package completions
                    for pkg in &self.packages {
                        if pkg.starts_with(arg_prefix) {
                            completions.push(
                                Completion::new(pkg.clone(), CompletionKind::Package)
                                    .with_confidence(self.prefix_similarity(pkg, arg_prefix))
                                    .with_similarity(self.prefix_similarity(pkg, arg_prefix)),
                            );
                        }
                    }
                }
                "flake" => {
                    // Flake subcommands
                    let flake_cmds = ["update", "lock", "check", "show", "metadata", "archive"];
                    for subcmd in flake_cmds {
                        if subcmd.starts_with(arg_prefix) {
                            completions.push(
                                Completion::new(subcmd, CompletionKind::Command)
                                    .with_confidence(0.9),
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        // Add history completions
        for hist in self.context.history.iter().rev().take(5) {
            if hist.starts_with(prefix) && hist != prefix {
                completions.push(
                    Completion::new(hist.clone(), CompletionKind::History)
                        .with_confidence(0.5)
                        .with_similarity(0.5),
                );
            }
        }

        // Sort by confidence (descending)
        completions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        completions.truncate(10);

        completions
    }

    /// Get semantically related commands for a given input
    fn get_semantic_matches(&self, input: &str) -> Vec<(String, f32)> {
        let input_lower = input.to_lowercase();
        let mut matches = Vec::new();

        // Semantic mappings: input term -> related nix commands
        let semantic_relations: &[(&str, &[&str])] = &[
            ("install", &["nix-env -i", "nix profile install"]),
            ("remove", &["nix-env -e", "nix profile remove"]),
            ("uninstall", &["nix-env -e", "nix profile remove"]),
            ("search", &["nix search"]),
            ("build", &["nix build", "nix-build"]),
            ("shell", &["nix shell", "nix-shell"]),
            ("run", &["nix run"]),
            ("update", &["nix flake update", "nix-channel --update"]),
            ("garbage", &["nix-collect-garbage"]),
            ("gc", &["nix-collect-garbage"]),
            ("rebuild", &["nixos-rebuild"]),
        ];

        for (term, related_cmds) in semantic_relations {
            if term.contains(&input_lower) || input_lower.contains(term) {
                for cmd in *related_cmds {
                    matches.push((cmd.to_string(), 0.7));
                }
            }
        }

        matches
    }

    /// Simple prefix similarity (placeholder for HDC similarity)
    fn prefix_similarity(&self, text: &str, prefix: &str) -> f32 {
        if text == prefix {
            1.0
        } else if text.starts_with(prefix) {
            0.8 + 0.2 * (prefix.len() as f32 / text.len() as f32)
        } else {
            0.0
        }
    }

    /// Load package list (async, lazy)
    pub fn load_packages(&mut self, packages: Vec<String>) {
        self.packages = packages;
        self.packages_loaded = true;
    }

    /// Set current Phi level (affects completion ranking)
    pub fn set_phi(&mut self, _phi: f64) {
        // Could be used to adjust completion confidence thresholds
        // Reserved for future HDC-Phi integration
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHI GATE - Consciousness-Aware Command Execution
// ═══════════════════════════════════════════════════════════════════════════════

/// Gate decision result
#[derive(Debug, Clone)]
pub enum GateDecision {
    /// Execute immediately
    Allowed { phi: f64, confidence: f64 },
    /// Require confirmation
    NeedsConfirmation {
        phi: f64,
        reason: String,
        prompt: String,
    },
    /// Block execution
    Vetoed { reason: String, message: String },
    /// Insufficient Phi for execution
    InsufficientPhi {
        current_phi: f64,
        required_phi: f64,
        centering_time_secs: u64,
    },
    /// Pending user input
    Pending { phi: f64, waiting_for: String },
}

impl GateDecision {
    /// Create an Allowed decision
    pub fn new_allowed(phi: f64, confidence: f64) -> Self {
        Self::Allowed { phi, confidence }
    }

    /// Create a NeedsConfirmation decision
    pub fn new_needs_confirmation(phi: f64, reason: String, prompt: String) -> Self {
        Self::NeedsConfirmation {
            phi,
            reason,
            prompt,
        }
    }

    /// Create a Vetoed decision
    pub fn new_vetoed(reason: String, message: String) -> Self {
        Self::Vetoed { reason, message }
    }
}

impl GateDecision {
    /// Check if execution is allowed
    pub fn is_allowed(&self) -> bool {
        matches!(self, GateDecision::Allowed { .. })
    }

    /// Check if confirmation is needed
    pub fn needs_confirmation(&self) -> bool {
        matches!(self, GateDecision::NeedsConfirmation { .. })
    }

    /// Check if vetoed
    pub fn is_vetoed(&self) -> bool {
        matches!(self, GateDecision::Vetoed { .. })
    }
}

/// Phi gate for consciousness-aware command execution
pub struct PhiGate {
    /// Minimum Phi for any execution
    min_phi: f64,
    /// Phi threshold for auto-confirmation
    auto_confirm_phi: f64,
    /// Always-confirm destructiveness levels (NeedsConfirmation and Destructive)
    always_confirm_destructive: bool,
    /// Current Phi value
    current_phi: f64,
}

impl PhiGate {
    /// Create a new Phi gate
    pub fn new() -> Self {
        Self {
            min_phi: 0.3,
            auto_confirm_phi: 0.7,
            always_confirm_destructive: true,
            current_phi: 0.5,
        }
    }

    /// Update current Phi from service
    pub fn update_phi(&mut self, phi: f64) {
        self.current_phi = phi.clamp(0.0, 1.0);
    }

    /// Update metrics (phi, coherence, is_conscious) - for binary compatibility
    pub fn update_metrics(&mut self, phi: f64, _coherence: f64, _is_conscious: bool) {
        self.current_phi = phi.clamp(0.0, 1.0);
    }

    /// Get current Phi
    pub fn phi(&self) -> f64 {
        self.current_phi
    }

    /// Check if a command should be allowed
    pub fn check(&self, command: &str, destructiveness: DestructivenessLevel) -> GateDecision {
        self.evaluate(command, destructiveness)
    }

    /// Evaluate if a command should be allowed (alias for check)
    pub fn evaluate(&self, command: &str, destructiveness: DestructivenessLevel) -> GateDecision {
        // Always veto if below minimum
        if self.current_phi < self.min_phi {
            return GateDecision::Vetoed {
                reason: format!(
                    "Consciousness level ({:.2}) too low for safe execution (minimum: {:.2})",
                    self.current_phi, self.min_phi
                ),
                message: "Wait for consciousness to stabilize or reduce system load".to_string(),
            };
        }

        // Check destructiveness
        if self.always_confirm_destructive
            && matches!(
                destructiveness,
                DestructivenessLevel::Destructive | DestructivenessLevel::NeedsConfirmation
            )
        {
            return GateDecision::NeedsConfirmation {
                phi: self.current_phi,
                reason: format!(
                    "Command classified as {destructiveness:?} - confirmation required"
                ),
                prompt: format!("About to execute: {command}"),
            };
        }

        // Auto-confirm if high Phi
        if self.current_phi >= self.auto_confirm_phi {
            return GateDecision::Allowed {
                phi: self.current_phi,
                confidence: self.current_phi, // Use phi as confidence
            };
        }

        // Medium Phi - require confirmation for anything beyond read-only
        if destructiveness > DestructivenessLevel::ReadOnly {
            return GateDecision::NeedsConfirmation {
                phi: self.current_phi,
                reason: format!(
                    "Medium consciousness ({:.2}) - confirming {:?} action",
                    self.current_phi, destructiveness
                ),
                prompt: format!("Execute: {command}"),
            };
        }

        // Allow read-only at any valid Phi
        GateDecision::Allowed {
            phi: self.current_phi,
            confidence: 1.0, // High confidence for read-only
        }
    }

    /// Set minimum Phi threshold
    pub fn set_min_phi(&mut self, phi: f64) {
        self.min_phi = phi.clamp(0.1, 0.9);
    }

    /// Set auto-confirm threshold
    pub fn set_auto_confirm_phi(&mut self, phi: f64) {
        self.auto_confirm_phi = phi.clamp(0.5, 1.0);
    }

    /// Update from metrics snapshot
    pub fn update_from_snapshot(&mut self, metrics: &super::ipc_client::MetricsSnapshot) {
        self.current_phi = metrics.phi;
    }

    /// Get threat level value (inverse of Phi - lower consciousness = higher threat)
    /// Returns 0.0-1.0 where 1.0 means maximum threat (very low consciousness)
    pub fn threat_level_value(&self) -> f64 {
        // Threat is inverse of consciousness: low Phi = high threat
        // Below min_phi is maximum threat
        if self.current_phi < self.min_phi {
            1.0
        } else {
            // Scale from 0.0 (at auto_confirm_phi) to ~0.7 (at min_phi)
            let range = self.auto_confirm_phi - self.min_phi;
            if range > 0.0 {
                let normalized = (self.auto_confirm_phi - self.current_phi) / range;
                (normalized * 0.7).clamp(0.0, 1.0)
            } else {
                0.0
            }
        }
    }

    /// Evaluate request with destructiveness classification
    pub fn evaluate_request(&self, request: &super::context::ExecutionRequest) -> GateDecision {
        let destructiveness = classify_command_destructiveness(&request.command);
        self.evaluate(&request.command, destructiveness)
    }
}

/// Classify command destructiveness based on common patterns
pub fn classify_command_destructiveness(command: &str) -> DestructivenessLevel {
    let cmd = command.trim().to_lowercase();

    // Destructive operations
    if cmd.contains("--purge")
        || cmd.contains("gc")
        || cmd.contains("delete")
        || cmd.contains("rm ")
        || cmd.contains("remove")
        || cmd.starts_with("nix-collect-garbage")
        || cmd.contains("wipe")
        || cmd.contains("format")
    {
        return DestructivenessLevel::Destructive;
    }

    // Needs confirmation
    if cmd.starts_with("nixos-rebuild")
        || cmd.contains("switch")
        || cmd.starts_with("nix profile install")
        || cmd.starts_with("nix-env -i")
        || cmd.contains("upgrade")
        || cmd.contains("update")
    {
        return DestructivenessLevel::NeedsConfirmation;
    }

    // Reversible
    if cmd.starts_with("nix build")
        || cmd.starts_with("nix develop")
        || cmd.starts_with("nix shell")
        || cmd.contains("install")
    {
        return DestructivenessLevel::Reversible;
    }

    // Read-only
    if cmd.starts_with("nix search")
        || cmd.starts_with("nix-env -q")
        || cmd.starts_with("nixos-option")
        || cmd.contains("list")
        || cmd.contains("info")
        || cmd.contains("show")
        || cmd.contains("search")
        || cmd.starts_with("nix flake show")
        || cmd.starts_with("nix flake metadata")
    {
        return DestructivenessLevel::ReadOnly;
    }

    // Default to needing confirmation for unknown commands
    DestructivenessLevel::NeedsConfirmation
}

/// Full classification of a command for safety analysis
#[derive(Debug, Clone)]
pub struct CommandClassification {
    /// The classified command
    pub command: String,
    /// Destructiveness level
    pub destructiveness: DestructivenessLevel,
    /// Whether confirmation is needed
    pub needs_confirmation: bool,
    /// Hint for how to rollback if applicable
    pub rollback_hint: Option<String>,
    /// Required Phi level for execution
    pub required_phi: f64,
}

impl CommandClassification {
    /// Create classification from a command string
    pub fn from_command(command: &str) -> Self {
        let destructiveness = classify_command_destructiveness(command);
        let needs_confirmation = destructiveness.requires_confirmation();
        let rollback_hint = Self::get_rollback_hint(command);
        let required_phi = match destructiveness {
            DestructivenessLevel::ReadOnly => 0.3,
            DestructivenessLevel::Reversible => 0.4,
            DestructivenessLevel::NeedsConfirmation => 0.5,
            DestructivenessLevel::Destructive => 0.7,
        };

        Self {
            command: command.to_string(),
            destructiveness,
            needs_confirmation,
            rollback_hint,
            required_phi,
        }
    }

    fn get_rollback_hint(command: &str) -> Option<String> {
        let cmd = command.trim().to_lowercase();
        if cmd.starts_with("nixos-rebuild") {
            Some("nixos-rebuild switch --rollback".to_string())
        } else if cmd.contains("nix-env -i") || cmd.contains("nix profile install") {
            Some("nix-env -e <package> or nix profile remove".to_string())
        } else if cmd.contains("nix-collect-garbage") {
            None // No rollback for GC
        } else {
            None
        }
    }
}

impl Default for PhiGate {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXECUTION REQUEST
// ═══════════════════════════════════════════════════════════════════════════════

/// Request to execute a command with Phi gating
#[derive(Debug, Clone)]
pub struct ExecutionRequest {
    /// The command to execute
    pub command: String,
    /// Required minimum Phi level (optional, uses gate default if None)
    pub require_phi: Option<f64>,
    /// Whether to perform dry-run
    pub dry_run: bool,
    /// Request ID for tracking
    pub request_id: String,
}

impl ExecutionRequest {
    /// Create from a command string
    pub fn from_command(command: &str) -> Self {
        Self {
            command: command.to_string(),
            require_phi: None,
            dry_run: false,
            request_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// With required Phi level
    pub fn with_phi(mut self, phi: f64) -> Self {
        self.require_phi = Some(phi);
        self
    }

    /// As dry-run
    pub fn dry_run(mut self) -> Self {
        self.dry_run = true;
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMMAND CONTEXT (for epistemic overlay)
// ═══════════════════════════════════════════════════════════════════════════════

/// Context about a command being edited/executed
#[derive(Debug, Clone, Default)]
pub struct CommandContext {
    /// The command text
    pub command: String,
    /// Detected command type
    pub command_type: Option<String>,
    /// Detected arguments
    pub arguments: Vec<String>,
    /// Detected target (package, service, etc.)
    pub target: Option<String>,
    /// Is this a dry-run/preview
    pub is_preview: bool,
}

impl CommandContext {
    /// Create from command string
    pub fn from_command(command: &str) -> Self {
        let parts: Vec<&str> = command.split_whitespace().collect();
        let (command_type, arguments) = if parts.is_empty() {
            (None, Vec::new())
        } else {
            (
                Some(parts[0].to_string()),
                parts[1..].iter().map(|s| s.to_string()).collect(),
            )
        };

        Self {
            command: command.to_string(),
            command_type,
            arguments: arguments.clone(),
            target: arguments.first().cloned(),
            is_preview: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_context_creation() {
        let ctx = ShellContext::new();
        assert!(!ctx.session_id.is_empty());
    }

    #[test]
    fn test_completion_creation() {
        let c = Completion::new("nginx", CompletionKind::Package)
            .with_docs("High-performance web server")
            .with_similarity(0.9);
        assert_eq!(c.text, "nginx");
        assert_eq!(c.kind, CompletionKind::Package);
        assert!((c.similarity - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_phi_gate_low_phi() {
        let gate = PhiGate {
            current_phi: 0.2,
            ..PhiGate::new()
        };
        let decision = gate.check("install nginx", DestructivenessLevel::Reversible);
        assert!(decision.is_vetoed());
    }

    #[test]
    fn test_phi_gate_high_phi() {
        let mut gate = PhiGate::new();
        gate.update_phi(0.85);
        let decision = gate.check("search nginx", DestructivenessLevel::ReadOnly);
        assert!(decision.is_allowed());
    }

    #[test]
    fn test_phi_gate_destructive() {
        let mut gate = PhiGate::new();
        gate.update_phi(0.9);
        let decision = gate.check("gc --delete-old", DestructivenessLevel::Destructive);
        assert!(decision.needs_confirmation());
    }

    #[test]
    fn test_intellisense_commands() {
        let ctx = ShellContext::new();
        let engine = IntelliSenseEngine::with_context(ctx);
        let completions = engine.complete("inst", 4);
        assert!(!completions.is_empty());
        assert!(completions.iter().any(|c| c.text == "install"));
    }

    #[test]
    fn test_command_context() {
        let ctx = CommandContext::from_command("install nginx firefox");
        assert_eq!(ctx.command_type, Some("install".to_string()));
        assert_eq!(ctx.target, Some("nginx".to_string()));
        assert_eq!(ctx.arguments.len(), 2);
    }
}
