//! IntelliSense Engine - HDC-Based Semantic Command Completion
//!
//! Provides intelligent command completion using hyperdimensional computing:
//! - Semantic similarity matching via HV16 cosine similarity
//! - Safety-filtered completions (Amygdala pre-filter)
//! - Phi-weighted confidence scoring
//! - Command preview generation
//!
//! ## Architecture
//!
//! ```text
//! Partial Input → Encode(HV16) → Similarity Search → Safety Filter → Rank by Φ
//!       │                              │                   │              │
//!       │                              ▼                   ▼              ▼
//!       │                       Command Library      Amygdala        Completions
//!       │                       (NixOS, shell)      (fast veto)     (with preview)
//!       └──────────────────────────────────────────────────────────────────┘
//! ```

use crate::hdc::binary_hv::HV16;
use crate::hdc::HDC_DIMENSION;
use crate::action::DestructivenessLevel;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Completion kind for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionKind {
    /// NixOS-specific command (nix, nixos-rebuild, etc.)
    NixCommand,
    /// Package name from nixpkgs
    Package,
    /// NixOS option path
    NixOption,
    /// Shell builtin or common utility
    ShellCommand,
    /// File or directory path
    Path,
    /// From command history
    History,
}

impl CompletionKind {
    /// Get display icon for completion kind
    pub fn icon(&self) -> &'static str {
        match self {
            Self::NixCommand => "❄",
            Self::Package => "📦",
            Self::NixOption => "⚙",
            Self::ShellCommand => "$",
            Self::Path => "📁",
            Self::History => "🕐",
        }
    }
}

/// A single completion suggestion
#[derive(Debug, Clone)]
pub struct Completion {
    /// The completion text
    pub text: String,

    /// Display label (may differ from text)
    pub label: String,

    /// Kind of completion
    pub kind: CompletionKind,

    /// Semantic similarity score (0.0 - 1.0)
    pub similarity: f32,

    /// Phi-weighted confidence (0.0 - 1.0)
    pub confidence: f32,

    /// Destructiveness level of this completion
    pub destructiveness: DestructivenessLevel,

    /// Short description
    pub description: Option<String>,

    /// Preview of what this command would do
    pub preview: Option<CommandPreview>,
}

impl Completion {
    /// Create a new completion
    pub fn new(text: impl Into<String>, kind: CompletionKind) -> Self {
        let text = text.into();
        Self {
            label: text.clone(),
            text,
            kind,
            similarity: 0.0,
            confidence: 0.0,
            destructiveness: DestructivenessLevel::ReadOnly,
            description: None,
            preview: None,
        }
    }

    /// Set similarity score
    pub fn with_similarity(mut self, sim: f32) -> Self {
        self.similarity = sim;
        self
    }

    /// Set confidence score
    pub fn with_confidence(mut self, conf: f32) -> Self {
        self.confidence = conf;
        self
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set destructiveness level
    pub fn with_destructiveness(mut self, level: DestructivenessLevel) -> Self {
        self.destructiveness = level;
        self
    }

    /// Set command preview
    pub fn with_preview(mut self, preview: CommandPreview) -> Self {
        self.preview = Some(preview);
        self
    }
}

/// Preview of command execution steps
#[derive(Debug, Clone)]
pub struct CommandPreview {
    /// Steps that would be executed
    pub steps: Vec<PreviewStep>,

    /// Estimated execution time (human readable)
    pub estimated_time: Option<String>,

    /// Whether this requires sudo/root
    pub requires_root: bool,

    /// Files that would be modified
    pub affected_files: Vec<String>,
}

/// A single step in command preview
#[derive(Debug, Clone)]
pub struct PreviewStep {
    /// Step number (1-indexed)
    pub number: usize,

    /// Description of this step
    pub description: String,

    /// The actual command/action
    pub action: String,

    /// Whether this step is reversible
    pub reversible: bool,
}

/// IntelliSense engine configuration
#[derive(Debug, Clone)]
pub struct IntelliSenseConfig {
    /// Maximum number of completions to return
    pub max_completions: usize,

    /// Minimum similarity threshold (0.0 - 1.0)
    pub min_similarity: f32,

    /// Whether to include history completions
    pub include_history: bool,

    /// Whether to filter unsafe completions
    pub filter_unsafe: bool,

    /// Current Phi level (affects confidence weighting)
    pub current_phi: f64,
}

impl Default for IntelliSenseConfig {
    fn default() -> Self {
        Self {
            max_completions: 10,
            min_similarity: 0.3,
            include_history: true,
            filter_unsafe: true,
            current_phi: 0.5,
        }
    }
}

/// HDC-based IntelliSense engine
pub struct IntelliSenseEngine {
    /// Configuration
    config: IntelliSenseConfig,

    /// Command library: command -> (HV16 encoding, metadata)
    command_library: HashMap<String, CommandEntry>,

    /// HDC dimension
    dimension: usize,
}

/// Entry in command library
#[derive(Debug, Clone)]
struct CommandEntry {
    /// HDC encoding of command
    hv: HV16,

    /// Completion kind
    kind: CompletionKind,

    /// Description
    description: String,

    /// Destructiveness level
    destructiveness: DestructivenessLevel,

    /// Preview steps template
    preview_template: Option<Vec<String>>,
}

impl IntelliSenseEngine {
    /// Create new IntelliSense engine
    pub fn new() -> Self {
        Self::with_config(IntelliSenseConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: IntelliSenseConfig) -> Self {
        let mut engine = Self {
            config,
            command_library: HashMap::new(),
            dimension: HDC_DIMENSION,
        };

        // Initialize with NixOS command library
        engine.init_nix_commands();
        engine.init_shell_commands();

        engine
    }

    /// Initialize NixOS-specific commands
    fn init_nix_commands(&mut self) {
        // Nix search commands (ReadOnly)
        self.add_command(
            "nix search",
            CompletionKind::NixCommand,
            "Search for packages in nixpkgs",
            DestructivenessLevel::ReadOnly,
            None,
        );

        self.add_command(
            "nix flake show",
            CompletionKind::NixCommand,
            "Show flake outputs",
            DestructivenessLevel::ReadOnly,
            None,
        );

        self.add_command(
            "nix flake metadata",
            CompletionKind::NixCommand,
            "Show flake metadata and inputs",
            DestructivenessLevel::ReadOnly,
            None,
        );

        self.add_command(
            "nix eval",
            CompletionKind::NixCommand,
            "Evaluate a Nix expression",
            DestructivenessLevel::ReadOnly,
            None,
        );

        // Nix environment commands (Reversible)
        self.add_command(
            "nix profile install",
            CompletionKind::NixCommand,
            "Install package to profile",
            DestructivenessLevel::Reversible,
            Some(vec![
                "Download package closure".to_string(),
                "Add to user profile".to_string(),
                "Update profile manifest".to_string(),
            ]),
        );

        self.add_command(
            "nix profile remove",
            CompletionKind::NixCommand,
            "Remove package from profile",
            DestructivenessLevel::Reversible,
            Some(vec![
                "Remove from profile manifest".to_string(),
                "Update profile generation".to_string(),
            ]),
        );

        self.add_command(
            "nix-env -i",
            CompletionKind::NixCommand,
            "Install package (legacy)",
            DestructivenessLevel::Reversible,
            None,
        );

        // NixOS rebuild commands (NeedsConfirmation)
        self.add_command(
            "nixos-rebuild switch",
            CompletionKind::NixCommand,
            "Rebuild and activate NixOS configuration",
            DestructivenessLevel::NeedsConfirmation,
            Some(vec![
                "Evaluate configuration.nix".to_string(),
                "Build system derivation".to_string(),
                "Create new generation".to_string(),
                "Activate new configuration".to_string(),
                "Restart affected services".to_string(),
            ]),
        );

        self.add_command(
            "nixos-rebuild boot",
            CompletionKind::NixCommand,
            "Rebuild NixOS for next boot",
            DestructivenessLevel::NeedsConfirmation,
            Some(vec![
                "Evaluate configuration.nix".to_string(),
                "Build system derivation".to_string(),
                "Create new generation".to_string(),
                "Set as boot default".to_string(),
            ]),
        );

        self.add_command(
            "nixos-rebuild test",
            CompletionKind::NixCommand,
            "Test configuration without making it default",
            DestructivenessLevel::Reversible,
            Some(vec![
                "Evaluate configuration.nix".to_string(),
                "Build system derivation".to_string(),
                "Activate temporarily".to_string(),
            ]),
        );

        // Destructive commands
        self.add_command(
            "nix-collect-garbage -d",
            CompletionKind::NixCommand,
            "Delete old generations and garbage collect",
            DestructivenessLevel::Destructive,
            Some(vec![
                "Delete old profile generations".to_string(),
                "Run garbage collector".to_string(),
                "Free disk space (irreversible)".to_string(),
            ]),
        );

        self.add_command(
            "nix store gc",
            CompletionKind::NixCommand,
            "Garbage collect the Nix store",
            DestructivenessLevel::Destructive,
            Some(vec![
                "Find unreachable store paths".to_string(),
                "Delete unreachable paths".to_string(),
            ]),
        );
    }

    /// Initialize common shell commands
    fn init_shell_commands(&mut self) {
        // Safe shell commands
        self.add_command(
            "ls",
            CompletionKind::ShellCommand,
            "List directory contents",
            DestructivenessLevel::ReadOnly,
            None,
        );

        self.add_command(
            "cat",
            CompletionKind::ShellCommand,
            "Display file contents",
            DestructivenessLevel::ReadOnly,
            None,
        );

        self.add_command(
            "pwd",
            CompletionKind::ShellCommand,
            "Print working directory",
            DestructivenessLevel::ReadOnly,
            None,
        );

        self.add_command(
            "cd",
            CompletionKind::ShellCommand,
            "Change directory",
            DestructivenessLevel::ReadOnly,
            None,
        );

        self.add_command(
            "grep",
            CompletionKind::ShellCommand,
            "Search file contents",
            DestructivenessLevel::ReadOnly,
            None,
        );

        self.add_command(
            "find",
            CompletionKind::ShellCommand,
            "Find files",
            DestructivenessLevel::ReadOnly,
            None,
        );

        // Reversible shell commands
        self.add_command(
            "mkdir",
            CompletionKind::ShellCommand,
            "Create directory",
            DestructivenessLevel::Reversible,
            None,
        );

        self.add_command(
            "cp",
            CompletionKind::ShellCommand,
            "Copy files",
            DestructivenessLevel::Reversible,
            None,
        );

        self.add_command(
            "mv",
            CompletionKind::ShellCommand,
            "Move/rename files",
            DestructivenessLevel::Reversible,
            None,
        );

        // Needs confirmation
        self.add_command(
            "systemctl restart",
            CompletionKind::ShellCommand,
            "Restart a systemd service",
            DestructivenessLevel::NeedsConfirmation,
            None,
        );

        // Destructive shell commands
        self.add_command(
            "rm -rf",
            CompletionKind::ShellCommand,
            "Remove files recursively (DANGEROUS)",
            DestructivenessLevel::Destructive,
            None,
        );
    }

    /// Add a command to the library
    fn add_command(
        &mut self,
        command: &str,
        kind: CompletionKind,
        description: &str,
        destructiveness: DestructivenessLevel,
        preview_template: Option<Vec<String>>,
    ) {
        let hv = self.encode_command(command);

        self.command_library.insert(
            command.to_string(),
            CommandEntry {
                hv,
                kind,
                description: description.to_string(),
                destructiveness,
                preview_template,
            },
        );
    }

    /// Convert string to u64 seed for HV16::random
    fn string_to_seed(s: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    /// Encode command string to HV16
    fn encode_command(&self, command: &str) -> HV16 {
        // Simple character n-gram encoding
        let mut vectors: Vec<HV16> = Vec::new();
        let chars: Vec<char> = command.chars().collect();

        // Use 3-grams for encoding
        for window in chars.windows(3) {
            let trigram: String = window.iter().collect();
            let seed = Self::string_to_seed(&trigram);
            vectors.push(HV16::random(seed));
        }

        // Also encode individual words
        for word in command.split_whitespace() {
            let seed = Self::string_to_seed(word);
            vectors.push(HV16::random(seed));
        }

        // Bundle all vectors together
        if vectors.is_empty() {
            HV16::zero()
        } else {
            HV16::bundle(&vectors)
        }
    }

    /// Get completions for partial input
    pub fn complete(&self, partial: &str, history: &[String]) -> Vec<Completion> {
        if partial.is_empty() {
            return Vec::new();
        }

        let partial_hv = self.encode_command(partial);
        let mut completions = Vec::new();

        // Search command library
        for (cmd, entry) in &self.command_library {
            // Skip if doesn't start with partial (prefix match)
            let prefix_match = cmd.to_lowercase().starts_with(&partial.to_lowercase());

            // Also check semantic similarity (Hamming-based)
            let similarity = partial_hv.similarity(&entry.hv);

            // Include if prefix matches OR similarity is high
            if prefix_match || similarity > self.config.min_similarity {
                // Filter unsafe if configured
                if self.config.filter_unsafe
                    && entry.destructiveness == DestructivenessLevel::Destructive
                    && self.config.current_phi < 0.8
                {
                    continue;
                }

                // Calculate Phi-weighted confidence
                let confidence = self.calculate_confidence(similarity, entry.destructiveness);

                let mut completion = Completion::new(cmd.clone(), entry.kind)
                    .with_similarity(similarity)
                    .with_confidence(confidence)
                    .with_description(&entry.description)
                    .with_destructiveness(entry.destructiveness);

                // Add preview if available
                if let Some(ref template) = entry.preview_template {
                    completion = completion.with_preview(self.generate_preview(cmd, template));
                }

                completions.push(completion);
            }
        }

        // Add history completions if configured
        if self.config.include_history {
            for hist_cmd in history.iter().take(50) {
                if hist_cmd.to_lowercase().starts_with(&partial.to_lowercase()) {
                    let similarity = self.encode_command(hist_cmd).similarity(&partial_hv);

                    completions.push(
                        Completion::new(hist_cmd.clone(), CompletionKind::History)
                            .with_similarity(similarity)
                            .with_confidence(similarity * 0.8) // History gets slightly lower confidence
                            .with_description("From history"),
                    );
                }
            }
        }

        // Sort by confidence (descending)
        completions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        completions.truncate(self.config.max_completions);

        completions
    }

    /// Calculate Phi-weighted confidence
    fn calculate_confidence(&self, similarity: f32, destructiveness: DestructivenessLevel) -> f32 {
        let phi = self.config.current_phi as f32;

        // Base confidence from similarity
        let base = similarity;

        // Adjust by Phi and destructiveness
        let adjustment = match destructiveness {
            DestructivenessLevel::ReadOnly => 1.0,
            DestructivenessLevel::Reversible => 0.9,
            DestructivenessLevel::NeedsConfirmation => 0.7 * phi,
            DestructivenessLevel::Destructive => 0.5 * phi,
        };

        (base * adjustment).clamp(0.0, 1.0)
    }

    /// Generate command preview
    fn generate_preview(&self, command: &str, template: &[String]) -> CommandPreview {
        let steps: Vec<PreviewStep> = template
            .iter()
            .enumerate()
            .map(|(i, desc)| PreviewStep {
                number: i + 1,
                description: desc.clone(),
                action: format!("Step {} of {}", i + 1, template.len()),
                reversible: i < template.len() - 1, // Last step often irreversible
            })
            .collect();

        CommandPreview {
            steps,
            estimated_time: Some(self.estimate_time(command)),
            requires_root: command.contains("nixos-rebuild")
                || command.contains("systemctl")
                || command.starts_with("sudo"),
            affected_files: self.guess_affected_files(command),
        }
    }

    /// Estimate execution time (rough heuristic)
    fn estimate_time(&self, command: &str) -> String {
        if command.contains("nixos-rebuild") {
            "2-10 minutes".to_string()
        } else if command.contains("nix-collect-garbage") {
            "1-5 minutes".to_string()
        } else if command.contains("nix profile install") || command.contains("nix-env -i") {
            "30 seconds - 5 minutes".to_string()
        } else {
            "< 1 second".to_string()
        }
    }

    /// Guess affected files based on command
    fn guess_affected_files(&self, command: &str) -> Vec<String> {
        let mut files = Vec::new();

        if command.contains("nixos-rebuild") {
            files.push("/etc/nixos/configuration.nix".to_string());
            files.push("/nix/var/nix/profiles/system".to_string());
        }

        if command.contains("nix profile") {
            files.push("~/.nix-profile".to_string());
        }

        if command.contains("nix-collect-garbage") {
            files.push("/nix/store/*".to_string());
        }

        files
    }

    /// Update current Phi level
    pub fn set_phi(&mut self, phi: f64) {
        self.config.current_phi = phi;
    }

    /// Add custom command to library
    pub fn add_custom_command(
        &mut self,
        command: &str,
        kind: CompletionKind,
        description: &str,
        destructiveness: DestructivenessLevel,
    ) {
        self.add_command(command, kind, description, destructiveness, None);
    }
}

impl Default for IntelliSenseEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intellisense_creation() {
        let engine = IntelliSenseEngine::new();
        assert!(!engine.command_library.is_empty());
    }

    #[test]
    fn test_complete_nix() {
        let engine = IntelliSenseEngine::new();
        let completions = engine.complete("nix s", &[]);

        assert!(!completions.is_empty());
        // Should find "nix search"
        assert!(completions.iter().any(|c| c.text.contains("search")));
    }

    #[test]
    fn test_complete_nixos_rebuild() {
        let engine = IntelliSenseEngine::new();
        let completions = engine.complete("nixos-rebuild", &[]);

        assert!(!completions.is_empty(), "Should have completions for nixos-rebuild");

        // Should find at least one nixos-rebuild command
        // Note: NeedsConfirmation commands (switch, boot) have lower confidence than
        // Reversible commands (test) when phi=0.5, so we check for any match
        let has_nixos_rebuild = completions.iter().any(|c| c.text.starts_with("nixos-rebuild"));
        assert!(
            has_nixos_rebuild,
            "Should find nixos-rebuild command in completions: {:?}",
            completions.iter().map(|c| c.text.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_destructive_filtering() {
        let mut config = IntelliSenseConfig::default();
        config.filter_unsafe = true;
        config.current_phi = 0.3; // Low Phi

        let engine = IntelliSenseEngine::with_config(config);
        let completions = engine.complete("nix-collect", &[]);

        // Destructive commands should be filtered at low Phi
        let has_destructive = completions
            .iter()
            .any(|c| c.destructiveness == DestructivenessLevel::Destructive);
        assert!(!has_destructive);
    }

    #[test]
    fn test_history_completion() {
        let engine = IntelliSenseEngine::new();
        let history = vec!["nix search nixpkgs#firefox".to_string()];

        let completions = engine.complete("nix search", &history);

        assert!(completions.iter().any(|c| c.kind == CompletionKind::History));
    }

    #[test]
    fn test_completion_kind_icons() {
        assert_eq!(CompletionKind::NixCommand.icon(), "❄");
        assert_eq!(CompletionKind::Package.icon(), "📦");
        assert_eq!(CompletionKind::History.icon(), "🕐");
    }
}
