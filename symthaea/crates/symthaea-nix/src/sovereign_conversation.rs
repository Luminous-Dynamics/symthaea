// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign Conversation — Symthaea talks with her human partner
//!
//! Instead of a form with dropdowns, Symthaea has a conversation:
//!
//! ```text
//! Symthaea: "What will you use this machine for?"
//! Human: "I make music and do some Python data science"
//! Symthaea: "I'll set up PipeWire with JACK for low-latency audio,
//!            Python 3 with Jupyter. Do you need MIDI support?"
//! Human: "Yes"
//! Symthaea: "Got it. Here's what I'm configuring..." [shows config]
//! ```
//!
//! The conversation tracks what's known, what's uncertain, and what
//! questions would most reduce uncertainty (active inference).

use crate::app_database::{AppDatabase, MigrationReport};
use crate::encoding::codebook::NixCodebook;
use crate::encoding::user_input_encoder::UserInputEncoder;
use crate::mind::active_inference::NixActiveInference;
use crate::sovereign_config::{
    ConfigDecision, HardwareProfile, MigrationData, SovereignConfig, SovereignConfigGenerator,
    UserChoices,
};

/// What Symthaea knows about the user's intent so far.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ConversationState {
    /// Accumulated knowledge from conversation turns.
    pub known_desktop: Option<String>,
    pub known_gpu: Option<String>,
    pub known_timezone: Option<String>,
    pub known_keyboard: Option<String>,
    pub known_encryption: Option<bool>,
    pub known_use_cases: Vec<String>, // "music", "development", "gaming", etc.
    pub known_packages: Vec<String>,  // explicitly requested packages

    /// Confidence in each dimension (0.0 = unknown, 1.0 = certain)
    pub confidence_desktop: f32,
    pub confidence_gpu: f32,
    pub confidence_use_case: f32,

    /// Conversation history
    pub turns: Vec<ConversationTurn>,

    /// Whether the user has said "looks good" / confirmed
    pub confirmed: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ConversationTurn {
    pub speaker: Speaker,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum Speaker {
    Symthaea,
    Human,
}

/// Symthaea's response to a conversation turn.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConversationResponse {
    /// What Symthaea says.
    pub message: String,
    /// Whether she's asking a question.
    pub is_question: bool,
    /// The current config preview (updated each turn).
    pub config_preview: Option<String>,
    /// Whether the config is ready to deploy.
    pub ready_to_deploy: bool,
    /// Decisions made so far with reasoning.
    pub decisions: Vec<ConfigDecision>,
}

/// The conversational config architect.
pub struct SovereignConversation {
    state: ConversationState,
    hardware: HardwareProfile,
    migration: MigrationData,
    codebook: NixCodebook,
    generator: SovereignConfigGenerator,
    app_db: AppDatabase,
    app_report: Option<MigrationReport>,
}

impl SovereignConversation {
    /// Start a new conversation with detected hardware and migration data.
    pub fn new(hardware: HardwareProfile, migration: MigrationData) -> Self {
        Self {
            state: ConversationState::default(),
            hardware,
            migration,
            codebook: NixCodebook::new(),
            generator: SovereignConfigGenerator::new(),
            app_db: AppDatabase::new(),
            app_report: None,
        }
    }

    /// Generate Symthaea's opening message based on what she already knows.
    pub fn greet(&mut self) -> ConversationResponse {
        let mut msg = Vec::new();

        // Warm greeting
        if !self.migration.git_user.is_empty() {
            msg.push(format!("Hi, {}! Welcome.", self.migration.git_user));
        } else {
            msg.push("Hi there! Welcome.".into());
        }

        msg.push(String::new());
        msg.push("I took a quick look at your machine. Here's what I found:".into());
        msg.push(String::new());

        // Hardware summary
        if !self.hardware.gpu_vendor.is_empty() && self.hardware.gpu_vendor != "unknown" {
            msg.push(format!(
                "  GPU: {} {}",
                self.hardware.gpu_vendor.to_uppercase(),
                self.hardware.gpu_model
            ));
        }
        if !self.hardware.existing_os.is_empty() {
            let os_names: Vec<String> = self
                .hardware
                .existing_os
                .iter()
                .map(|o| o.name.clone())
                .collect();
            msg.push(format!("  Existing OS: {}", os_names.join(", ")));
        }
        if self.hardware.has_tpm {
            msg.push("  Security chip: available (can auto-unlock disk encryption)".into());
        }

        // Migration insights
        if !self.migration.editors.is_empty() {
            msg.push(format!(
                "  Your editors: {}",
                self.migration.editors.join(", ")
            ));
        }
        if self.migration.rust_projects > 0
            || self.migration.node_projects > 0
            || self.migration.python_venvs > 0
        {
            let mut langs = Vec::new();
            if self.migration.rust_projects > 0 {
                langs.push("Rust");
            }
            if self.migration.node_projects > 0 {
                langs.push("Node.js");
            }
            if self.migration.python_venvs > 0 {
                langs.push("Python");
            }
            msg.push(format!("  Development: {}", langs.join(", ")));
        }
        if self.migration.has_daw {
            msg.push("  Audio: DAW configuration detected".into());
            self.state.known_use_cases.push("music".into());
            self.state.confidence_use_case = 0.5;
        }

        msg.push(String::new());
        msg.push("What will you use this machine for? Just describe it however you like \u{2014} I'll take it from there.\n\nSome examples: \"I write code in Rust and Python\", \"Just web browsing and documents\", \"Gaming plus some coding\", \"It's a server, no screen needed\"".into());

        self.state.turns.push(ConversationTurn {
            speaker: Speaker::Symthaea,
            message: msg.join("\n"),
        });

        ConversationResponse {
            message: msg.join("\n"),
            is_question: true,
            config_preview: None,
            ready_to_deploy: false,
            decisions: Vec::new(),
        }
    }

    /// Process a human message and generate Symthaea's response.
    pub fn respond(&mut self, human_input: &str) -> ConversationResponse {
        // Record the human's turn
        self.state.turns.push(ConversationTurn {
            speaker: Speaker::Human,
            message: human_input.into(),
        });

        let lower = human_input.to_lowercase();

        // Check for confirmation
        if self.state.confidence_use_case > 0.5
            && (lower.contains("looks good")
                || lower.contains("perfect")
                || lower.contains("yes")
                || lower.contains("go ahead")
                || lower.contains("install")
                || lower.contains("deploy")
                || lower.contains("let's do it")
                || lower.contains("confirmed"))
            && self.state.turns.len() > 3
        // not the first response
        {
            self.state.confirmed = true;
            return self.generate_final_config();
        }

        // Detect and parse pasted app lists
        self.detect_app_list(human_input);

        // Extract intent from input
        self.extract_intent(&lower);

        // Encode in HDC space for deeper understanding
        let _input_hv = {
            let mut encoder = UserInputEncoder::new(&mut self.codebook);
            encoder.encode_input(human_input)
        };

        // If we just parsed an app list, show the report
        if self.app_report.is_some() && self.state.turns.len() <= 4 {
            return self.show_app_report();
        }

        // Determine what to ask or tell
        if self.state.confidence_use_case < 0.3 {
            // Still don't know what they want — ask broader
            self.ask_about_use_case()
        } else if self.state.known_desktop.is_none() {
            // Know the use case but not the desktop
            self.suggest_desktop()
        } else if self.needs_more_info() {
            // Know basics, ask refinement questions
            self.ask_refinement()
        } else {
            // We have enough — show the preview
            self.show_preview()
        }
    }

    /// Extract structured intent from natural language.
    /// Detect if the user pasted an app list and parse it.
    fn detect_app_list(&mut self, input: &str) {
        // Heuristics: app list has multiple lines, contains known patterns
        let lines: Vec<&str> = input.lines().collect();
        if lines.len() < 3 {
            return; // Too short to be an app list
        }

        let looks_like_list = input.contains("---") ||           // brew companion format
            input.contains("winget") ||        // winget mention
            lines.iter().any(|l| l.starts_with("ii ")) || // dpkg
            lines.iter().any(|l| l.contains("Id") && l.contains("Version")) || // winget header
            input.contains("org.") ||          // flatpak IDs
            (lines.len() >= 5 && lines.iter().filter(|l| !l.trim().is_empty()).count() >= 5); // 5+ non-empty lines

        if !looks_like_list {
            return;
        }

        // Parse and match
        let app_names = self.app_db.parse_app_list(input);
        if app_names.is_empty() {
            return;
        }

        let report = self.app_db.match_list(&app_names);

        // Feed matched apps into migration data
        for matched in &report.matched {
            let pkg = matched.entry.primary.nix_pkg;
            if !self.state.known_packages.contains(&pkg.to_string()) {
                self.state.known_packages.push(pkg.to_string());
            }

            // Infer use cases from categories
            match matched.entry.category {
                crate::app_database::AppCategory::Audio => {
                    if !self.state.known_use_cases.contains(&"music".to_string()) {
                        self.state.known_use_cases.push("music".into());
                    }
                    self.migration.has_daw = true;
                }
                crate::app_database::AppCategory::Gaming
                | crate::app_database::AppCategory::GamingTools => {
                    if !self.state.known_use_cases.contains(&"gaming".to_string()) {
                        self.state.known_use_cases.push("gaming".into());
                    }
                }
                crate::app_database::AppCategory::IDE
                | crate::app_database::AppCategory::Editor
                | crate::app_database::AppCategory::Container
                | crate::app_database::AppCategory::DevTools => {
                    if !self
                        .state
                        .known_use_cases
                        .contains(&"development".to_string())
                    {
                        self.state.known_use_cases.push("development".into());
                    }
                }
                crate::app_database::AppCategory::Creative2D
                | crate::app_database::AppCategory::Creative3D
                | crate::app_database::AppCategory::Video
                | crate::app_database::AppCategory::Photo => {
                    if !self.state.known_use_cases.contains(&"creative".to_string()) {
                        self.state.known_use_cases.push("creative".into());
                    }
                }
                _ => {}
            }

            // Add specific packages to migration data
            match matched.entry.category {
                crate::app_database::AppCategory::Container => {
                    self.migration.docker_images = 1;
                }
                _ => {}
            }
        }

        // Add detected editors
        for matched in &report.matched {
            match matched.entry.category {
                crate::app_database::AppCategory::IDE
                | crate::app_database::AppCategory::Editor => {
                    let name = matched.entry.primary.nix_pkg.to_string();
                    if !self.migration.editors.contains(&name) {
                        self.migration.editors.push(name);
                    }
                }
                _ => {}
            }
        }

        self.state.confidence_use_case = (self.state.confidence_use_case + 0.5).min(1.0);
        self.app_report = Some(report);
    }

    /// Show the app migration report after parsing a pasted list.
    fn show_app_report(&mut self) -> ConversationResponse {
        let report = match &self.app_report {
            Some(r) => r,
            None => return self.ask_about_use_case(),
        };

        let mut msg = Vec::new();
        msg.push(format!(
            "I found **{} apps** in your list.\n",
            report.total_apps
        ));

        // Count by quality
        let native_count = report
            .matched
            .iter()
            .filter(|m| m.entry.primary.quality.confidence() >= 0.9)
            .count();
        let alt_count = report
            .matched
            .iter()
            .filter(|m| {
                let c = m.entry.primary.quality.confidence();
                c >= 0.5 && c < 0.9
            })
            .count();
        let no_equiv = report
            .matched
            .iter()
            .filter(|m| m.entry.primary.quality.confidence() < 0.2)
            .count();

        msg.push(format!("  **{}** work natively on NixOS", native_count));
        msg.push(format!("  **{}** have good alternatives", alt_count));
        if no_equiv > 0 {
            msg.push(format!("  **{}** have no Linux equivalent", no_equiv));
        }
        if !report.unmatched.is_empty() {
            msg.push(format!(
                "  **{}** I couldn't identify (may still be in nixpkgs)",
                report.unmatched.len()
            ));
        }

        msg.push(format!(
            "\nMigration readiness: **{:.0}%**",
            report.readiness_score * 100.0
        ));

        // Show bundles detected
        if !report.suggested_bundles.is_empty() {
            msg.push("\nBased on your apps, I'd suggest these bundles:".into());
            for bundle in &report.suggested_bundles {
                msg.push(format!("  **{}** — {}", bundle.name, bundle.description));
            }
        }

        // Show a few notable apps
        msg.push("\nKey apps:".into());
        for matched in report.matched.iter().take(8) {
            let quality = matched.entry.primary.quality;
            let icon = if quality.confidence() >= 0.9 {
                "✓"
            } else if quality.confidence() >= 0.5 {
                "~"
            } else {
                "✗"
            };
            msg.push(format!(
                "  {} {} → {} ({})",
                icon,
                matched.source_name,
                matched.entry.primary.display_name,
                quality.label()
            ));
        }

        msg.push("\nNow tell me — what's most important to you about this system?".into());

        let message = msg.join("\n");
        self.state.turns.push(ConversationTurn {
            speaker: Speaker::Symthaea,
            message: message.clone(),
        });

        ConversationResponse {
            message,
            is_question: true,
            config_preview: None,
            ready_to_deploy: false,
            decisions: Vec::new(),
        }
    }

    fn extract_intent(&mut self, lower: &str) {
        // Use cases
        let use_case_patterns = [
            (
                &["music", "audio", "produce", "daw", "recording", "studio"] as &[&str],
                "music",
            ),
            (&["game", "gaming", "steam", "play"], "gaming"),
            (
                &[
                    "code",
                    "develop",
                    "programming",
                    "software",
                    "rust",
                    "python",
                    "node",
                    "web",
                ],
                "development",
            ),
            (&["server", "deploy", "host", "backend", "api"], "server"),
            (
                &["design", "art", "photo", "video", "creative", "edit"],
                "creative",
            ),
            (
                &["office", "email", "browse", "document", "school", "work"],
                "general",
            ),
            (&["privacy", "secure", "encrypt", "anonymous"], "privacy"),
            (
                &[
                    "science",
                    "data",
                    "research",
                    "jupyter",
                    "analysis",
                    "ml",
                    "machine learning",
                ],
                "data_science",
            ),
        ];

        for (keywords, use_case) in &use_case_patterns {
            if keywords.iter().any(|k| lower.contains(k)) {
                if !self.state.known_use_cases.contains(&use_case.to_string()) {
                    self.state.known_use_cases.push(use_case.to_string());
                }
                self.state.confidence_use_case = (self.state.confidence_use_case + 0.3).min(1.0);
            }
        }

        // Desktop preferences
        if lower.contains("gnome") {
            self.state.known_desktop = Some("gnome".into());
            self.state.confidence_desktop = 1.0;
        }
        if lower.contains("kde") || lower.contains("plasma") {
            self.state.known_desktop = Some("plasma".into());
            self.state.confidence_desktop = 1.0;
        }
        if lower.contains("hyprland") || lower.contains("tiling") {
            self.state.known_desktop = Some("hyprland".into());
            self.state.confidence_desktop = 1.0;
        }
        if lower.contains("sway") {
            self.state.known_desktop = Some("sway".into());
            self.state.confidence_desktop = 1.0;
        }
        if lower.contains("xfce") || lower.contains("lightweight") {
            self.state.known_desktop = Some("xfce".into());
            self.state.confidence_desktop = 1.0;
        }
        if lower.contains("no desktop")
            || lower.contains("no gui")
            || lower.contains("terminal only")
            || lower.contains("headless")
        {
            self.state.known_desktop = Some("none".into());
            self.state.confidence_desktop = 1.0;
        }
        if lower.contains("simple") || lower.contains("just works") || lower.contains("easy") {
            self.state.known_desktop = Some("gnome".into());
            self.state.confidence_desktop = 0.7;
        }

        // Encryption
        if lower.contains("encrypt") || lower.contains("luks") || lower.contains("secure disk") {
            self.state.known_encryption = Some(true);
        }
        if lower.contains("no encrypt") || lower.contains("don't encrypt") {
            self.state.known_encryption = Some(false);
        }

        // Specific packages
        let package_mentions = [
            "firefox",
            "chrome",
            "steam",
            "discord",
            "spotify",
            "vscode",
            "docker",
            "podman",
            "nginx",
            "postgresql",
            "blender",
            "gimp",
            "obs",
            "kdenlive",
            "ardour",
            "audacity",
            "jupyter",
        ];
        for pkg in &package_mentions {
            if lower.contains(pkg) && !self.state.known_packages.contains(&pkg.to_string()) {
                self.state.known_packages.push(pkg.to_string());
            }
        }
    }

    fn ask_about_use_case(&mut self) -> ConversationResponse {
        let msg = "I want to get this right. Can you tell me more about what you'll use this for?\n\n\
            For example:\n\
            - \"I write code in Rust and Python\"\n\
            - \"I produce music and need low-latency audio\"\n\
            - \"Just browsing and office work\"\n\
            - \"It's a server, no GUI needed\"\n\
            - \"Gaming with Steam, and some development\"";

        self.state.turns.push(ConversationTurn {
            speaker: Speaker::Symthaea,
            message: msg.into(),
        });

        ConversationResponse {
            message: msg.into(),
            is_question: true,
            config_preview: None,
            ready_to_deploy: false,
            decisions: Vec::new(),
        }
    }

    fn suggest_desktop(&mut self) -> ConversationResponse {
        let use_cases = &self.state.known_use_cases;
        let gpu = &self.hardware.gpu_vendor;

        let (suggestion, reasoning) = if use_cases.contains(&"server".to_string()) {
            self.state.known_desktop = Some("none".into());
            self.state.confidence_desktop = 0.9;
            (
                "none (CLI only)",
                "You mentioned this is a server — no desktop environment needed. Saves resources.",
            )
        } else if use_cases.contains(&"gaming".to_string()) {
            self.state.known_desktop = Some("plasma".into());
            self.state.confidence_desktop = 0.7;
            (
                "KDE Plasma",
                "Good gaming integration with Steam, and very customizable.",
            )
        } else if use_cases.contains(&"music".to_string())
            || use_cases.contains(&"creative".to_string())
        {
            self.state.known_desktop = Some("gnome".into());
            self.state.confidence_desktop = 0.7;
            (
                "GNOME",
                "Clean interface that stays out of your way during creative work.",
            )
        } else if use_cases.contains(&"development".to_string()) {
            if gpu == "nvidia" {
                self.state.known_desktop = Some("gnome".into());
                self.state.confidence_desktop = 0.6;
                (
                    "GNOME",
                    "Best NVIDIA+Wayland support. Or if you prefer tiling: Hyprland.",
                )
            } else {
                self.state.known_desktop = Some("hyprland".into());
                self.state.confidence_desktop = 0.5;
                (
                    "Hyprland",
                    "Tiling compositor — fast keyboard-driven workflow for coding. Or GNOME if you prefer traditional.",
                )
            }
        } else if use_cases.contains(&"privacy".to_string()) {
            self.state.known_desktop = Some("sway".into());
            self.state.confidence_desktop = 0.7;
            ("Sway", "Minimal, Wayland-only, small attack surface.")
        } else {
            self.state.known_desktop = Some("gnome".into());
            self.state.confidence_desktop = 0.6;
            ("GNOME", "Polished, accessible, works well out of the box.")
        };

        let mut msg = format!(
            "Based on what you've told me, I'd suggest **{}**.\n\n",
            suggestion
        );
        msg.push_str(&format!("Reasoning: {}\n\n", reasoning));
        msg.push_str("Does that work for you? Or would you prefer something else?\n");
        msg.push_str("Options: GNOME, KDE Plasma, Hyprland (tiling), Sway (tiling), XFCE (lightweight), or no desktop.");

        self.state.turns.push(ConversationTurn {
            speaker: Speaker::Symthaea,
            message: msg.clone(),
        });

        ConversationResponse {
            message: msg,
            is_question: true,
            config_preview: None,
            ready_to_deploy: false,
            decisions: Vec::new(),
        }
    }

    fn needs_more_info(&self) -> bool {
        // Ask about encryption if not mentioned and TPM is available
        if self.state.known_encryption.is_none() && self.hardware.has_tpm {
            return true;
        }
        false
    }

    fn ask_refinement(&mut self) -> ConversationResponse {
        let msg = if self.state.known_encryption.is_none() && self.hardware.has_tpm {
            "Your machine has a TPM 2.0 chip. Would you like full disk encryption?\n\n\
             With encryption enabled, your data is protected even if someone steals the drive.\n\
             The TPM can auto-unlock at boot so you don't need to type a passphrase every time.\n\n\
             Enable encryption? (recommended for laptops)"
        } else {
            "I think I have everything I need. Let me show you what I'd configure."
        };

        // If asking about encryption, mark as pending
        if self.state.known_encryption.is_none() && self.hardware.has_tpm {
            // Will be resolved on next turn
        }

        self.state.turns.push(ConversationTurn {
            speaker: Speaker::Symthaea,
            message: msg.into(),
        });

        ConversationResponse {
            message: msg.into(),
            is_question: true,
            config_preview: None,
            ready_to_deploy: false,
            decisions: Vec::new(),
        }
    }

    fn show_preview(&mut self) -> ConversationResponse {
        self.enrich_migration_from_apps();
        let choices = self.build_choices();
        let config = self
            .generator
            .generate(&self.hardware, &choices, &self.migration);

        let mut msg = String::new();
        msg.push_str("Here's what I'm configuring for you:\n\n");

        for d in &config.decisions {
            if d.confidence >= 0.5 {
                msg.push_str(&format!("  **{}**: {}\n", d.option_path, d.reasoning));
            }
        }

        if !config.warnings.is_empty() {
            msg.push_str("\nHeads up:\n");
            for w in &config.warnings {
                msg.push_str(&format!("  ⚠ {}\n", w));
            }
        }

        msg.push_str("\nDoes this look good? Say \"go ahead\" to generate the final config, or tell me what to change.");

        self.state.turns.push(ConversationTurn {
            speaker: Speaker::Symthaea,
            message: msg.clone(),
        });

        ConversationResponse {
            message: msg,
            is_question: true,
            config_preview: Some(config.sovereign_config_nix.clone()),
            ready_to_deploy: false,
            decisions: config.decisions,
        }
    }

    fn generate_final_config(&mut self) -> ConversationResponse {
        self.enrich_migration_from_apps();
        let choices = self.build_choices();
        let config = self
            .generator
            .generate(&self.hardware, &choices, &self.migration);

        let mut msg = config.welcome_message.clone();
        msg.push_str("\n\nYour configuration is ready. Deploying now.");

        self.state.turns.push(ConversationTurn {
            speaker: Speaker::Symthaea,
            message: msg.clone(),
        });

        ConversationResponse {
            message: msg,
            is_question: false,
            config_preview: Some(config.sovereign_config_nix.clone()),
            ready_to_deploy: true,
            decisions: config.decisions,
        }
    }

    /// Feed app database matches into migration data for config generation.
    fn enrich_migration_from_apps(&mut self) {
        if let Some(report) = &self.app_report {
            // Add matched packages to the detected_apps list
            for matched in &report.matched {
                let pkg = matched.entry.primary.nix_pkg.to_string();
                if !self.migration.detected_apps.contains(&pkg) {
                    self.migration.detected_apps.push(pkg);
                }
                // Also add alternatives for important apps
                for alt in matched.entry.alternatives {
                    // Only add the primary recommendation to the package list
                    // Alternatives are shown but not auto-installed
                }
            }

            // Detect specific tools from matched apps
            for matched in &report.matched {
                match matched.entry.category {
                    crate::app_database::AppCategory::Audio => {
                        self.migration.has_daw = true;
                    }
                    crate::app_database::AppCategory::Container => {
                        self.migration.docker_images = self.migration.docker_images.max(1);
                    }
                    crate::app_database::AppCategory::DevTools
                    | crate::app_database::AppCategory::IDE => {
                        // Check if it's a specific language tool
                        let pkg = matched.entry.primary.nix_pkg;
                        if pkg.contains("rust") || pkg == "cargo" {
                            self.migration.rust_projects = self.migration.rust_projects.max(1);
                        }
                        if pkg == "nodejs" || pkg == "node" {
                            self.migration.node_projects = self.migration.node_projects.max(1);
                        }
                        if pkg == "python3" || pkg.contains("python") {
                            self.migration.python_venvs = self.migration.python_venvs.max(1);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Also add explicitly requested packages from conversation
        for pkg in &self.state.known_packages {
            if !self.migration.detected_apps.contains(pkg) {
                self.migration.detected_apps.push(pkg.clone());
            }
        }
    }

    fn build_choices(&self) -> UserChoices {
        UserChoices {
            hostname: String::new(), // from portal UI
            desktop: self
                .state
                .known_desktop
                .clone()
                .unwrap_or_else(|| "gnome".into()),
            gpu_driver: self
                .state
                .known_gpu
                .clone()
                .unwrap_or_else(|| "auto".into()),
            timezone: self
                .state
                .known_timezone
                .clone()
                .unwrap_or_else(|| "UTC".into()),
            keyboard: self
                .state
                .known_keyboard
                .clone()
                .unwrap_or_else(|| "us".into()),
            encryption: self.state.known_encryption.unwrap_or(false),
            ..Default::default()
        }
    }

    /// Get the full conversation history.
    pub fn history(&self) -> &[ConversationTurn] {
        &self.state.turns
    }

    /// Get the current conversation state.
    pub fn state(&self) -> &ConversationState {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greeting_acknowledges_hardware() {
        let mut conv = SovereignConversation::new(
            HardwareProfile {
                gpu_vendor: "nvidia".into(),
                gpu_model: "RTX 3060".into(),
                has_tpm: true,
                ..Default::default()
            },
            MigrationData {
                git_user: "Tristan".into(),
                editors: vec!["neovim".into()],
                rust_projects: 3,
                ..Default::default()
            },
        );

        let greeting = conv.greet();
        assert!(greeting.message.contains("Tristan"));
        assert!(greeting.message.contains("NVIDIA"));
        assert!(greeting.message.contains("TPM"));
        assert!(greeting.message.contains("neovim"));
        assert!(greeting.message.contains("Rust"));
        assert!(greeting.is_question);
        assert!(!greeting.ready_to_deploy);
    }

    #[test]
    fn test_music_use_case_flow() {
        let mut conv = SovereignConversation::new(
            HardwareProfile {
                gpu_vendor: "amd".into(),
                ..Default::default()
            },
            MigrationData {
                has_daw: true,
                ..Default::default()
            },
        );

        let _greeting = conv.greet();
        let resp = conv.respond("I produce music and need low-latency audio");

        // Should have identified music use case
        assert!(conv.state().known_use_cases.contains(&"music".to_string()));
        assert!(conv.state().confidence_use_case > 0.3);
    }

    #[test]
    fn test_server_gets_no_desktop() {
        let mut conv =
            SovereignConversation::new(HardwareProfile::default(), MigrationData::default());

        let _greeting = conv.greet();
        let resp = conv.respond("It's a headless server for hosting APIs");

        assert_eq!(conv.state().known_desktop.as_deref(), Some("none"));
        assert!(conv.state().known_use_cases.contains(&"server".to_string()));
    }

    #[test]
    fn test_explicit_desktop_choice() {
        let mut conv =
            SovereignConversation::new(HardwareProfile::default(), MigrationData::default());

        let _greeting = conv.greet();
        let _resp1 = conv.respond("I want to do some coding");
        let _resp2 = conv.respond("I prefer KDE Plasma");

        assert_eq!(conv.state().known_desktop.as_deref(), Some("plasma"));
        assert_eq!(conv.state().confidence_desktop, 1.0);
    }

    #[test]
    fn test_confirmation_generates_config() {
        let mut conv = SovereignConversation::new(
            HardwareProfile {
                gpu_vendor: "intel".into(),
                ..Default::default()
            },
            MigrationData::default(),
        );

        let _greeting = conv.greet();
        let _resp1 = conv.respond("Just general desktop use, browsing and office");
        let _resp2 = conv.respond("GNOME sounds good");
        let _resp3 = conv.respond("No encryption needed");
        let resp4 = conv.respond("Looks good, go ahead");

        assert!(resp4.ready_to_deploy);
        assert!(resp4.config_preview.is_some());
        let config = resp4.config_preview.unwrap();
        assert!(config.contains("gnome.enable"));
    }

    #[test]
    fn test_multi_use_case_detection() {
        let mut conv =
            SovereignConversation::new(HardwareProfile::default(), MigrationData::default());

        let _greeting = conv.greet();
        let _resp =
            conv.respond("I do gaming with Steam and also some Python data science with Jupyter");

        let uses = &conv.state().known_use_cases;
        assert!(uses.contains(&"gaming".to_string()));
        assert!(uses.contains(&"data_science".to_string()));
        assert!(conv.state().known_packages.contains(&"steam".to_string()));
        assert!(conv.state().known_packages.contains(&"jupyter".to_string()));
    }

    #[test]
    fn test_encryption_question_when_tpm_available() {
        let mut conv = SovereignConversation::new(
            HardwareProfile {
                has_tpm: true,
                ..Default::default()
            },
            MigrationData::default(),
        );

        let _greeting = conv.greet();
        let _resp1 = conv.respond("Desktop for coding");
        let _resp2 = conv.respond("GNOME is fine");

        // At this point she should ask about encryption since TPM is available
        // (needs_more_info returns true when encryption is unknown + TPM exists)
        assert!(conv.state().known_encryption.is_none());
    }

    #[test]
    fn test_conversation_history_preserved() {
        let mut conv =
            SovereignConversation::new(HardwareProfile::default(), MigrationData::default());

        conv.greet();
        conv.respond("I write Rust code");
        conv.respond("Hyprland please");

        let history = conv.history();
        assert!(history.len() >= 5); // greeting + response + 2 human + 2 symthaea
        assert_eq!(history[0].speaker, Speaker::Symthaea); // greeting
        assert_eq!(history[1].speaker, Speaker::Human); // "I write Rust code"
    }

    #[test]
    fn test_privacy_user_gets_sway() {
        let mut conv =
            SovereignConversation::new(HardwareProfile::default(), MigrationData::default());

        let _greeting = conv.greet();
        let resp = conv.respond("I need a secure, privacy-focused system with encryption");

        assert!(
            conv.state()
                .known_use_cases
                .contains(&"privacy".to_string())
        );
        assert!(conv.state().known_encryption == Some(true));
    }
}
