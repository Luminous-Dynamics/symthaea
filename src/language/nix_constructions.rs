//! NixOS-Specific Constructions (Construction Grammar)
//!
//! This module implements construction patterns for NixOS understanding.
//! Constructions are form-meaning pairings that go beyond simple frames to capture
//! complex reasoning patterns specific to NixOS.
//!
//! ## NixOS-Specific Constructions
//!
//! 1. **ProvenanceExplanation** - "X is Y because Z defined it in W"
//! 2. **WhatIfSimulation** - "If you change X to Y, then Z would happen"
//! 3. **ConflictDetection** - "X conflicts with Y because they both set Z"
//! 4. **MinimalFixSuggestion** - "To fix X, minimally change Y to Z"
//! 5. **SecurityReview** - "X is insecure because Y; recommend Z"
//! 6. **DependencyExplanation** - "X requires Y which requires Z because..."
//! 7. **OverrideChain** - "X overrides Y with priority Z from W"
//! 8. **ModuleComposition** - "Module X imports Y, providing Z"

use super::nix_frames::{NixFrameType, NixFrameRole, NixFrameInstance, NixFrameFiller};
use super::nix_primitives::{NixPrimitive, NixPrimitiveTier};
use crate::hdc::HV16;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// CONSTRUCTION TYPES
// =============================================================================

/// Types of NixOS-specific constructions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NixConstructionType {
    /// "X is Y because Z defined it in W"
    ProvenanceExplanation,
    /// "If you change X to Y, then Z would happen"
    WhatIfSimulation,
    /// "X conflicts with Y because they both set Z"
    ConflictDetection,
    /// "To fix X, minimally change Y to Z"
    MinimalFixSuggestion,
    /// "X is insecure because Y; recommend Z"
    SecurityReview,
    /// "X requires Y which requires Z because..."
    DependencyExplanation,
    /// "X overrides Y with priority Z from W"
    OverrideChain,
    /// "Module X imports Y, providing Z"
    ModuleComposition,
    /// Simple command execution
    CommandExecution,
    /// "Search for X matching Y in Z"
    SearchQuery,
    /// Status/information query
    StatusQuery,
}

impl NixConstructionType {
    /// Get the name of this construction
    pub fn name(&self) -> &'static str {
        match self {
            Self::ProvenanceExplanation => "ProvenanceExplanation",
            Self::WhatIfSimulation => "WhatIfSimulation",
            Self::ConflictDetection => "ConflictDetection",
            Self::MinimalFixSuggestion => "MinimalFixSuggestion",
            Self::SecurityReview => "SecurityReview",
            Self::DependencyExplanation => "DependencyExplanation",
            Self::OverrideChain => "OverrideChain",
            Self::ModuleComposition => "ModuleComposition",
            Self::CommandExecution => "CommandExecution",
            Self::SearchQuery => "SearchQuery",
            Self::StatusQuery => "StatusQuery",
        }
    }

    /// Get description of this construction pattern
    pub fn description(&self) -> &'static str {
        match self {
            Self::ProvenanceExplanation =>
                "Explains where a value comes from and why",
            Self::WhatIfSimulation =>
                "Predicts consequences of configuration changes",
            Self::ConflictDetection =>
                "Identifies and explains option conflicts",
            Self::MinimalFixSuggestion =>
                "Suggests smallest change to fix an issue",
            Self::SecurityReview =>
                "Analyzes security implications and recommends improvements",
            Self::DependencyExplanation =>
                "Traces and explains dependency relationships",
            Self::OverrideChain =>
                "Shows how overrides stack and resolve",
            Self::ModuleComposition =>
                "Explains module import and composition",
            Self::CommandExecution =>
                "Simple imperative command",
            Self::SearchQuery =>
                "Search for packages or options",
            Self::StatusQuery =>
                "Query system status or information",
        }
    }

    /// Get the template pattern for this construction
    pub fn template(&self) -> &'static str {
        match self {
            Self::ProvenanceExplanation =>
                "{Option} has value {Value} because {Source} defined it in {Location}",
            Self::WhatIfSimulation =>
                "If you {Change} {Target} to {NewValue}, then {Consequence}",
            Self::ConflictDetection =>
                "{OptionA} conflicts with {OptionB} because {Reason}",
            Self::MinimalFixSuggestion =>
                "To fix {Error}, change {Target} from {OldValue} to {NewValue}",
            Self::SecurityReview =>
                "{Option} {SecurityIssue}; recommend {Recommendation}",
            Self::DependencyExplanation =>
                "{Package} requires {Dependency} which requires {TransitiveDep}",
            Self::OverrideChain =>
                "{Option} is overridden by {Override} with priority {Priority}",
            Self::ModuleComposition =>
                "Module {Module} imports {Import}, providing {Provides}",
            Self::CommandExecution =>
                "{Action} {Target}",
            Self::SearchQuery =>
                "Search {Repository} for {Query}",
            Self::StatusQuery =>
                "Show {InfoType} for {Target}",
        }
    }

    /// Get slots required by this construction
    pub fn required_slots(&self) -> Vec<ConstructionSlot> {
        match self {
            Self::ProvenanceExplanation => vec![
                ConstructionSlot::Option,
                ConstructionSlot::Value,
                ConstructionSlot::Source,
            ],
            Self::WhatIfSimulation => vec![
                ConstructionSlot::Target,
                ConstructionSlot::NewValue,
                ConstructionSlot::Consequence,
            ],
            Self::ConflictDetection => vec![
                ConstructionSlot::OptionA,
                ConstructionSlot::OptionB,
                ConstructionSlot::Reason,
            ],
            Self::MinimalFixSuggestion => vec![
                ConstructionSlot::Error,
                ConstructionSlot::Target,
                ConstructionSlot::NewValue,
            ],
            Self::SecurityReview => vec![
                ConstructionSlot::Option,
                ConstructionSlot::SecurityIssue,
                ConstructionSlot::Recommendation,
            ],
            Self::DependencyExplanation => vec![
                ConstructionSlot::Package,
                ConstructionSlot::Dependency,
            ],
            Self::OverrideChain => vec![
                ConstructionSlot::Option,
                ConstructionSlot::Override,
                ConstructionSlot::Priority,
            ],
            Self::ModuleComposition => vec![
                ConstructionSlot::Module,
                ConstructionSlot::Import,
            ],
            Self::CommandExecution => vec![
                ConstructionSlot::Action,
                ConstructionSlot::Target,
            ],
            Self::SearchQuery => vec![
                ConstructionSlot::Query,
            ],
            Self::StatusQuery => vec![
                ConstructionSlot::InfoType,
            ],
        }
    }

    /// Get relevant frame types for this construction
    pub fn related_frames(&self) -> Vec<NixFrameType> {
        match self {
            Self::ProvenanceExplanation => vec![
                NixFrameType::OverrideResolution,
                NixFrameType::ModuleImport,
            ],
            Self::WhatIfSimulation => vec![
                NixFrameType::EnableService,
                NixFrameType::PackageInstall,
                NixFrameType::PackageRemove,
            ],
            Self::ConflictDetection => vec![
                NixFrameType::FailureDiagnosis,
                NixFrameType::OverrideResolution,
            ],
            Self::MinimalFixSuggestion => vec![
                NixFrameType::FailureDiagnosis,
            ],
            Self::SecurityReview => vec![
                NixFrameType::SecurityPosture,
                NixFrameType::EnableService,
            ],
            Self::DependencyExplanation => vec![
                NixFrameType::DependencyChain,
            ],
            Self::OverrideChain => vec![
                NixFrameType::OverrideResolution,
            ],
            Self::ModuleComposition => vec![
                NixFrameType::ModuleImport,
            ],
            Self::CommandExecution => vec![
                NixFrameType::PackageInstall,
                NixFrameType::PackageRemove,
                NixFrameType::SystemUpgrade,
                NixFrameType::GenerationRollback,
                NixFrameType::GarbageCollect,
            ],
            Self::SearchQuery => vec![
                NixFrameType::PackageSearch,
            ],
            Self::StatusQuery => vec![
                NixFrameType::DependencyChain,
                NixFrameType::SecurityPosture,
            ],
        }
    }

    /// Get all construction types
    pub fn all() -> Vec<Self> {
        vec![
            Self::ProvenanceExplanation,
            Self::WhatIfSimulation,
            Self::ConflictDetection,
            Self::MinimalFixSuggestion,
            Self::SecurityReview,
            Self::DependencyExplanation,
            Self::OverrideChain,
            Self::ModuleComposition,
            Self::CommandExecution,
            Self::SearchQuery,
            Self::StatusQuery,
        ]
    }
}

// =============================================================================
// CONSTRUCTION SLOTS
// =============================================================================

/// Slots that can be filled in constructions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstructionSlot {
    // Entity slots
    /// A NixOS option path
    Option,
    /// A second option (for conflicts)
    OptionA,
    /// Another option
    OptionB,
    /// A value
    Value,
    /// A new value (for changes)
    NewValue,
    /// An old value
    OldValue,
    /// The target of an operation
    Target,

    // Source/provenance slots
    /// Source of a definition
    Source,
    /// File location
    Location,
    /// A module name
    Module,
    /// An import path
    Import,

    // Action slots
    /// An action verb
    Action,
    /// A change being made
    Change,
    /// A query string
    Query,
    /// Type of information requested
    InfoType,

    // Package/dependency slots
    /// A package name
    Package,
    /// A dependency
    Dependency,
    /// A transitive dependency
    TransitiveDep,
    /// Repository to search
    Repository,

    // Override slots
    /// An override definition
    Override,
    /// Priority level
    Priority,

    // Reasoning slots
    /// Reason for something
    Reason,
    /// Consequence of an action
    Consequence,
    /// What a module provides
    Provides,
    /// An error condition
    Error,

    // Security slots
    /// Security issue description
    SecurityIssue,
    /// Security recommendation
    Recommendation,
}

impl ConstructionSlot {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Option => "Option",
            Self::OptionA => "OptionA",
            Self::OptionB => "OptionB",
            Self::Value => "Value",
            Self::NewValue => "NewValue",
            Self::OldValue => "OldValue",
            Self::Target => "Target",
            Self::Source => "Source",
            Self::Location => "Location",
            Self::Module => "Module",
            Self::Import => "Import",
            Self::Action => "Action",
            Self::Change => "Change",
            Self::Query => "Query",
            Self::InfoType => "InfoType",
            Self::Package => "Package",
            Self::Dependency => "Dependency",
            Self::TransitiveDep => "TransitiveDep",
            Self::Repository => "Repository",
            Self::Override => "Override",
            Self::Priority => "Priority",
            Self::Reason => "Reason",
            Self::Consequence => "Consequence",
            Self::Provides => "Provides",
            Self::Error => "Error",
            Self::SecurityIssue => "SecurityIssue",
            Self::Recommendation => "Recommendation",
        }
    }

    /// Map to corresponding frame role if applicable
    pub fn to_frame_role(&self) -> Option<NixFrameRole> {
        match self {
            Self::Target | Self::Package => Some(NixFrameRole::Target),
            Self::Action => Some(NixFrameRole::Action),
            Self::Value | Self::NewValue | Self::OldValue => Some(NixFrameRole::Value),
            Self::Source | Self::Location => Some(NixFrameRole::Source),
            Self::Dependency | Self::TransitiveDep => Some(NixFrameRole::Requirement),
            Self::Priority => Some(NixFrameRole::Priority),
            Self::Reason => Some(NixFrameRole::Reason),
            Self::Consequence => Some(NixFrameRole::Effect),
            Self::Error => Some(NixFrameRole::Error),
            Self::Recommendation => Some(NixFrameRole::Fix),
            _ => None,
        }
    }
}

// =============================================================================
// CONSTRUCTION INSTANCE
// =============================================================================

/// A filled slot in a construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilledSlot {
    pub slot: ConstructionSlot,
    pub value: String,
    pub confidence: f32,
}

/// An instantiated construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixConstructionInstance {
    /// The construction type
    pub construction_type: NixConstructionType,
    /// Confidence in this construction
    pub confidence: f32,
    /// Filled slots
    pub slots: Vec<FilledSlot>,
    /// Related frame instances
    pub related_frames: Vec<NixFrameType>,
    /// Active primitives
    pub active_primitives: Vec<NixPrimitive>,
}

impl NixConstructionInstance {
    /// Create a new construction instance
    pub fn new(construction_type: NixConstructionType) -> Self {
        Self {
            construction_type,
            confidence: 0.0,
            slots: Vec::new(),
            related_frames: Vec::new(),
            active_primitives: Vec::new(),
        }
    }

    /// Add a filled slot
    pub fn fill_slot(&mut self, slot: ConstructionSlot, value: String, confidence: f32) {
        self.slots.push(FilledSlot { slot, value, confidence });
    }

    /// Get a slot's value
    pub fn get_slot(&self, slot: ConstructionSlot) -> Option<&str> {
        self.slots.iter()
            .find(|s| s.slot == slot)
            .map(|s| s.value.as_str())
    }

    /// Check if all required slots are filled
    pub fn is_complete(&self) -> bool {
        let required = self.construction_type.required_slots();
        required.iter().all(|slot| self.get_slot(*slot).is_some())
    }

    /// Get missing required slots
    pub fn missing_slots(&self) -> Vec<ConstructionSlot> {
        let required = self.construction_type.required_slots();
        required.into_iter()
            .filter(|slot| self.get_slot(*slot).is_none())
            .collect()
    }

    /// Generate output text from the construction template
    pub fn generate_text(&self) -> String {
        let mut text = self.construction_type.template().to_string();

        for filled in &self.slots {
            let placeholder = format!("{{{}}}", filled.slot.name());
            text = text.replace(&placeholder, &filled.value);
        }

        // Remove unfilled placeholders
        let re = regex::Regex::new(r"\{[^}]+\}").unwrap_or_else(|_| {
            // Fallback: simple replacement
            return regex::Regex::new(r".*").unwrap();
        });
        text = re.replace_all(&text, "[unknown]").to_string();

        text
    }
}

// =============================================================================
// CONSTRUCTION PARSER
// =============================================================================

/// Parser that identifies and instantiates constructions from input
pub struct NixConstructionParser {
    /// Patterns for matching construction types
    patterns: HashMap<NixConstructionType, Vec<&'static str>>,
}

impl NixConstructionParser {
    /// Create a new parser
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        // ProvenanceExplanation patterns
        patterns.insert(NixConstructionType::ProvenanceExplanation, vec![
            "where does", "where did", "why is", "who set", "which module",
            "defined in", "comes from", "originates from", "set by",
        ]);

        // WhatIfSimulation patterns
        patterns.insert(NixConstructionType::WhatIfSimulation, vec![
            "what if", "if i change", "if i set", "would happen if",
            "what would", "what happens when", "consequences of",
        ]);

        // ConflictDetection patterns
        patterns.insert(NixConstructionType::ConflictDetection, vec![
            "conflict", "conflicts with", "incompatible", "can't both",
            "contradicts", "clash", "mutual exclusion",
        ]);

        // MinimalFixSuggestion patterns
        patterns.insert(NixConstructionType::MinimalFixSuggestion, vec![
            "how to fix", "how do i fix", "fix this", "resolve this",
            "solve this", "minimal change", "smallest fix",
        ]);

        // SecurityReview patterns
        patterns.insert(NixConstructionType::SecurityReview, vec![
            "is it secure", "security", "insecure", "vulnerable",
            "safe", "dangerous", "risky", "hardening",
        ]);

        // DependencyExplanation patterns
        patterns.insert(NixConstructionType::DependencyExplanation, vec![
            "depends on", "dependency", "requires", "needed by",
            "what does * need", "why is * required",
        ]);

        // OverrideChain patterns
        patterns.insert(NixConstructionType::OverrideChain, vec![
            "override", "mkforce", "mkdefault", "priority",
            "who wins", "which value", "final value",
        ]);

        // ModuleComposition patterns
        patterns.insert(NixConstructionType::ModuleComposition, vec![
            "import", "imports", "module", "include",
            "split configuration", "separate file",
        ]);

        // CommandExecution patterns (simple imperatives)
        patterns.insert(NixConstructionType::CommandExecution, vec![
            "install", "remove", "uninstall", "add", "delete",
            "upgrade", "update", "rollback", "clean", "gc",
            "enable", "disable", "start", "stop",
        ]);

        // SearchQuery patterns
        patterns.insert(NixConstructionType::SearchQuery, vec![
            "search for", "find", "look for", "looking for",
            "discover", "locate", "where is",
        ]);

        // StatusQuery patterns
        patterns.insert(NixConstructionType::StatusQuery, vec![
            "show", "list", "display", "what is", "status",
            "info", "information", "details",
        ]);

        Self { patterns }
    }

    /// Parse input and identify the best construction match
    pub fn parse(&self, input: &str) -> NixConstructionInstance {
        let input_lower = input.to_lowercase();
        let mut best_match: Option<(NixConstructionType, f32)> = None;

        // Check each construction type's patterns
        for (construction_type, patterns) in &self.patterns {
            let match_count = patterns.iter()
                .filter(|p| input_lower.contains(*p))
                .count();

            if match_count > 0 {
                let score = match_count as f32 / patterns.len() as f32;

                // Boost score for exact matches
                let boost = if patterns.iter().any(|p| input_lower.starts_with(*p)) {
                    0.2
                } else {
                    0.0
                };

                let final_score = score + boost;

                if best_match.is_none() || final_score > best_match.unwrap().1 {
                    best_match = Some((*construction_type, final_score));
                }
            }
        }

        let (construction_type, confidence) = best_match
            .unwrap_or((NixConstructionType::CommandExecution, 0.1));

        let mut instance = NixConstructionInstance::new(construction_type);
        instance.confidence = confidence;

        // Extract slots based on construction type
        self.extract_slots(input, &mut instance);

        // Add related frames
        instance.related_frames = construction_type.related_frames();

        instance
    }

    /// Extract slot values from input
    fn extract_slots(&self, input: &str, instance: &mut NixConstructionInstance) {
        let tokens: Vec<&str> = input.split_whitespace().collect();
        let input_lower = input.to_lowercase();

        match instance.construction_type {
            NixConstructionType::CommandExecution => {
                // Extract action and target
                let action_keywords = ["install", "remove", "uninstall", "add", "delete",
                    "upgrade", "update", "rollback", "clean", "gc", "enable", "disable"];

                for (i, token) in tokens.iter().enumerate() {
                    let token_lower = token.to_lowercase();
                    if action_keywords.iter().any(|kw| token_lower.contains(kw)) {
                        instance.fill_slot(ConstructionSlot::Action, token.to_string(), 0.9);

                        // Next non-flag token is likely the target
                        if let Some(target) = tokens.get(i + 1) {
                            if !target.starts_with("-") {
                                instance.fill_slot(ConstructionSlot::Target, target.to_string(), 0.8);
                            }
                        }
                        break;
                    }
                }
            }

            NixConstructionType::SearchQuery => {
                // Extract query
                let search_keywords = ["search", "find", "look for", "looking for"];
                for kw in &search_keywords {
                    if let Some(pos) = input_lower.find(kw) {
                        let query = input[pos + kw.len()..].trim();
                        if !query.is_empty() {
                            // Remove common prepositions
                            let query = query.trim_start_matches(|c| " for".contains(c));
                            instance.fill_slot(ConstructionSlot::Query, query.to_string(), 0.8);
                            break;
                        }
                    }
                }
            }

            NixConstructionType::WhatIfSimulation => {
                // Extract target and potential change
                if let Some(pos) = input_lower.find("if") {
                    let rest = &input[pos + 2..].trim();

                    // Look for "change X to Y" pattern
                    if let Some(change_pos) = rest.to_lowercase().find("change") {
                        let change_text = &rest[change_pos..];
                        instance.fill_slot(ConstructionSlot::Change, change_text.to_string(), 0.7);
                    }

                    // Look for target after "i" or "we"
                    let subject_words = ["i ", "we "];
                    for subject in &subject_words {
                        if let Some(subj_pos) = rest.to_lowercase().find(subject) {
                            let after_subject = &rest[subj_pos + subject.len()..];
                            if let Some(first_word) = after_subject.split_whitespace().next() {
                                instance.fill_slot(ConstructionSlot::Action, first_word.to_string(), 0.6);
                            }
                        }
                    }
                }
            }

            NixConstructionType::ProvenanceExplanation => {
                // Look for option path patterns (a.b.c)
                let option_pattern = regex::Regex::new(r"[a-zA-Z]+\.[a-zA-Z]+(\.[a-zA-Z]+)*")
                    .ok();
                if let Some(re) = option_pattern {
                    if let Some(m) = re.find(input) {
                        instance.fill_slot(ConstructionSlot::Option, m.as_str().to_string(), 0.9);
                    }
                }
            }

            NixConstructionType::ConflictDetection => {
                // Look for "X conflicts with Y" pattern
                if let Some(pos) = input_lower.find("conflict") {
                    // Everything before might be option A
                    let before = input[..pos].trim();
                    if !before.is_empty() {
                        let last_word = before.split_whitespace().last().unwrap_or("");
                        instance.fill_slot(ConstructionSlot::OptionA, last_word.to_string(), 0.6);
                    }

                    // Everything after might contain option B
                    let after = &input[pos..];
                    if let Some(with_pos) = after.to_lowercase().find("with") {
                        let option_b = after[with_pos + 4..].trim();
                        if !option_b.is_empty() {
                            let first_word = option_b.split_whitespace().next().unwrap_or("");
                            instance.fill_slot(ConstructionSlot::OptionB, first_word.to_string(), 0.6);
                        }
                    }
                }
            }

            NixConstructionType::DependencyExplanation => {
                // Look for package names
                let dep_keywords = ["depends on", "requires", "needs"];
                for kw in &dep_keywords {
                    if let Some(pos) = input_lower.find(kw) {
                        // Before keyword is likely the package
                        let before = input[..pos].trim();
                        if !before.is_empty() {
                            let pkg = before.split_whitespace().last().unwrap_or("");
                            instance.fill_slot(ConstructionSlot::Package, pkg.to_string(), 0.7);
                        }

                        // After keyword is likely the dependency
                        let after = input[pos + kw.len()..].trim();
                        if !after.is_empty() {
                            let dep = after.split_whitespace().next().unwrap_or("");
                            instance.fill_slot(ConstructionSlot::Dependency, dep.to_string(), 0.7);
                        }
                        break;
                    }
                }
            }

            NixConstructionType::MinimalFixSuggestion => {
                // Look for error descriptions
                if input_lower.contains("error") || input_lower.contains("failed") {
                    instance.fill_slot(ConstructionSlot::Error, input.to_string(), 0.6);
                }
            }

            NixConstructionType::SecurityReview => {
                // Look for security-related option/service
                let security_keywords = ["firewall", "port", "ssh", "ssl", "tls", "password", "auth"];
                for kw in &security_keywords {
                    if input_lower.contains(kw) {
                        instance.fill_slot(ConstructionSlot::Option, kw.to_string(), 0.7);
                        break;
                    }
                }
            }

            _ => {
                // Generic extraction - first substantial token as target
                for token in &tokens {
                    if token.len() > 2 && !token.starts_with("-") {
                        instance.fill_slot(ConstructionSlot::Target, token.to_string(), 0.3);
                        break;
                    }
                }
            }
        }
    }
}

impl Default for NixConstructionParser {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CONSTRUCTION GENERATOR
// =============================================================================

/// Generates natural language outputs from constructions
pub struct NixConstructionGenerator;

impl NixConstructionGenerator {
    /// Generate explanation text from a construction instance
    pub fn generate(instance: &NixConstructionInstance) -> String {
        match instance.construction_type {
            NixConstructionType::ProvenanceExplanation => {
                let option = instance.get_slot(ConstructionSlot::Option).unwrap_or("this option");
                let value = instance.get_slot(ConstructionSlot::Value).unwrap_or("its value");
                let source = instance.get_slot(ConstructionSlot::Source).unwrap_or("an unknown source");

                format!("{} has value {} because {} defined it.", option, value, source)
            }

            NixConstructionType::WhatIfSimulation => {
                let change = instance.get_slot(ConstructionSlot::Change).unwrap_or("that change");
                let consequence = instance.get_slot(ConstructionSlot::Consequence)
                    .unwrap_or("the system would be affected");

                format!("If you {}, then {}.", change, consequence)
            }

            NixConstructionType::ConflictDetection => {
                let option_a = instance.get_slot(ConstructionSlot::OptionA).unwrap_or("Option A");
                let option_b = instance.get_slot(ConstructionSlot::OptionB).unwrap_or("Option B");
                let reason = instance.get_slot(ConstructionSlot::Reason)
                    .unwrap_or("they set conflicting values");

                format!("{} conflicts with {} because {}.", option_a, option_b, reason)
            }

            NixConstructionType::MinimalFixSuggestion => {
                let error = instance.get_slot(ConstructionSlot::Error).unwrap_or("this error");
                let target = instance.get_slot(ConstructionSlot::Target).unwrap_or("the configuration");
                let new_value = instance.get_slot(ConstructionSlot::NewValue).unwrap_or("a different value");

                format!("To fix {}, change {} to {}.", error, target, new_value)
            }

            NixConstructionType::SecurityReview => {
                let option = instance.get_slot(ConstructionSlot::Option).unwrap_or("this setting");
                let issue = instance.get_slot(ConstructionSlot::SecurityIssue)
                    .unwrap_or("may have security implications");
                let rec = instance.get_slot(ConstructionSlot::Recommendation)
                    .unwrap_or("review the configuration");

                format!("{} {}. Recommendation: {}.", option, issue, rec)
            }

            NixConstructionType::DependencyExplanation => {
                let package = instance.get_slot(ConstructionSlot::Package).unwrap_or("This package");
                let dep = instance.get_slot(ConstructionSlot::Dependency).unwrap_or("its dependencies");

                format!("{} requires {}.", package, dep)
            }

            NixConstructionType::OverrideChain => {
                let option = instance.get_slot(ConstructionSlot::Option).unwrap_or("The option");
                let override_val = instance.get_slot(ConstructionSlot::Override).unwrap_or("an override");
                let priority = instance.get_slot(ConstructionSlot::Priority).unwrap_or("default");

                format!("{} is set by {} with {} priority.", option, override_val, priority)
            }

            NixConstructionType::ModuleComposition => {
                let module = instance.get_slot(ConstructionSlot::Module).unwrap_or("The module");
                let import = instance.get_slot(ConstructionSlot::Import).unwrap_or("other modules");

                format!("{} imports {}.", module, import)
            }

            NixConstructionType::CommandExecution => {
                let action = instance.get_slot(ConstructionSlot::Action).unwrap_or("Execute");
                let target = instance.get_slot(ConstructionSlot::Target).unwrap_or("command");

                format!("{} {}.", action, target)
            }

            NixConstructionType::SearchQuery => {
                let query = instance.get_slot(ConstructionSlot::Query).unwrap_or("packages");

                format!("Searching for {}...", query)
            }

            NixConstructionType::StatusQuery => {
                let info_type = instance.get_slot(ConstructionSlot::InfoType).unwrap_or("status");
                let target = instance.get_slot(ConstructionSlot::Target).unwrap_or("system");

                format!("Showing {} for {}.", info_type, target)
            }
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction_types() {
        let all = NixConstructionType::all();
        assert!(all.len() >= 10);
    }

    #[test]
    fn test_construction_templates() {
        let template = NixConstructionType::ProvenanceExplanation.template();
        assert!(template.contains("{Option}"));
        assert!(template.contains("{Value}"));
    }

    #[test]
    fn test_construction_required_slots() {
        let slots = NixConstructionType::CommandExecution.required_slots();
        assert!(slots.contains(&ConstructionSlot::Action));
        assert!(slots.contains(&ConstructionSlot::Target));
    }

    #[test]
    fn test_parser_creation() {
        let parser = NixConstructionParser::new();
        let instance = parser.parse("install firefox");
        assert_eq!(instance.construction_type, NixConstructionType::CommandExecution);
    }

    #[test]
    fn test_parse_command() {
        let parser = NixConstructionParser::new();
        let instance = parser.parse("install firefox");

        assert_eq!(instance.construction_type, NixConstructionType::CommandExecution);
        assert!(instance.confidence > 0.0);
        assert_eq!(instance.get_slot(ConstructionSlot::Action), Some("install"));
        assert_eq!(instance.get_slot(ConstructionSlot::Target), Some("firefox"));
    }

    #[test]
    fn test_parse_search() {
        let parser = NixConstructionParser::new();
        let instance = parser.parse("search for text editors");

        assert_eq!(instance.construction_type, NixConstructionType::SearchQuery);
        assert!(instance.get_slot(ConstructionSlot::Query).is_some());
    }

    #[test]
    fn test_parse_what_if() {
        let parser = NixConstructionParser::new();
        let instance = parser.parse("what if I change services.nginx.enable to false");

        assert_eq!(instance.construction_type, NixConstructionType::WhatIfSimulation);
    }

    #[test]
    fn test_parse_provenance() {
        let parser = NixConstructionParser::new();
        let instance = parser.parse("where does services.nginx.port come from");

        assert_eq!(instance.construction_type, NixConstructionType::ProvenanceExplanation);
    }

    #[test]
    fn test_parse_security() {
        let parser = NixConstructionParser::new();
        // Use "security" keyword directly which triggers SecurityReview
        let instance = parser.parse("check security of my ssh settings");

        assert_eq!(instance.construction_type, NixConstructionType::SecurityReview);
    }

    #[test]
    fn test_parse_dependency() {
        let parser = NixConstructionParser::new();
        let instance = parser.parse("firefox depends on what packages");

        assert_eq!(instance.construction_type, NixConstructionType::DependencyExplanation);
    }

    #[test]
    fn test_generate_text() {
        let mut instance = NixConstructionInstance::new(NixConstructionType::CommandExecution);
        instance.fill_slot(ConstructionSlot::Action, "install".to_string(), 0.9);
        instance.fill_slot(ConstructionSlot::Target, "firefox".to_string(), 0.9);

        let text = NixConstructionGenerator::generate(&instance);
        assert!(text.contains("install"));
        assert!(text.contains("firefox"));
    }

    #[test]
    fn test_instance_completion() {
        let mut instance = NixConstructionInstance::new(NixConstructionType::CommandExecution);

        assert!(!instance.is_complete());

        instance.fill_slot(ConstructionSlot::Action, "install".to_string(), 0.9);
        instance.fill_slot(ConstructionSlot::Target, "firefox".to_string(), 0.9);

        assert!(instance.is_complete());
    }

    #[test]
    fn test_slot_to_frame_role() {
        assert_eq!(ConstructionSlot::Target.to_frame_role(), Some(NixFrameRole::Target));
        assert_eq!(ConstructionSlot::Action.to_frame_role(), Some(NixFrameRole::Action));
        assert_eq!(ConstructionSlot::Error.to_frame_role(), Some(NixFrameRole::Error));
    }

    #[test]
    fn test_related_frames() {
        let frames = NixConstructionType::CommandExecution.related_frames();
        assert!(frames.contains(&NixFrameType::PackageInstall));
    }
}
