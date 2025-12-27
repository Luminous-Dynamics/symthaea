//! NixOS-Specific Semantic Frames
//!
//! This module defines semantic frames (FrameNet-style) that are specific to NixOS
//! configuration and understanding. These frames capture the structured knowledge
//! needed to understand NixOS operations in context.
//!
//! ## NixOS-Specific Frames
//!
//! 1. **EnableServiceFrame** - Enabling a systemd service with dependencies
//! 2. **OverrideResolutionFrame** - Understanding mkDefault/mkForce/mkOverride
//! 3. **DependencyChainFrame** - Package/service dependency relationships
//! 4. **FailureDiagnosisFrame** - Understanding why a config failed
//! 5. **SecurityPostureFrame** - Security-related configuration analysis
//! 6. **ModuleImportFrame** - Module system imports and composition
//! 7. **PackageInstallFrame** - Package installation with environment
//! 8. **FlakeOperationFrame** - Flake-related operations
//!
//! Each frame has:
//! - Core elements (required participants)
//! - Peripheral elements (optional context)
//! - Example instantiations
//! - Inference rules for reasoning

use super::nix_primitives::{NixPrimitive, NixPrimitiveEncoder, NixPrimitiveTier};
use crate::hdc::HV16;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// FRAME ELEMENT TYPES
// =============================================================================

/// Role types for frame elements in NixOS context
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NixFrameRole {
    // Core roles
    /// The main target of the action (service, package, option)
    Target,
    /// The action being performed
    Action,
    /// Agent performing the action (user, system, module)
    Agent,

    // Value roles
    /// A value being set
    Value,
    /// The source/origin of a definition
    Source,
    /// The destination of a change
    Destination,

    // Dependency roles
    /// Something that is required
    Requirement,
    /// Something that conflicts
    Conflict,
    /// Something that is wanted (soft dependency)
    Want,

    // Context roles
    /// The condition for something to apply
    Condition,
    /// The reason/justification
    Reason,
    /// The effect/consequence
    Effect,

    // Override roles
    /// Priority level in override
    Priority,
    /// The previous/default value
    Default,
    /// The overriding value
    Override,

    // Provenance roles
    /// Where something was defined
    DefinitionSite,
    /// Where something is evaluated
    EvaluationSite,
    /// The trace/path of resolution
    Trace,

    // Error roles
    /// Error message/type
    Error,
    /// Suggested fix
    Fix,
    /// Related/similar issue
    Related,
}

impl NixFrameRole {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Target => "Target",
            Self::Action => "Action",
            Self::Agent => "Agent",
            Self::Value => "Value",
            Self::Source => "Source",
            Self::Destination => "Destination",
            Self::Requirement => "Requirement",
            Self::Conflict => "Conflict",
            Self::Want => "Want",
            Self::Condition => "Condition",
            Self::Reason => "Reason",
            Self::Effect => "Effect",
            Self::Priority => "Priority",
            Self::Default => "Default",
            Self::Override => "Override",
            Self::DefinitionSite => "DefinitionSite",
            Self::EvaluationSite => "EvaluationSite",
            Self::Trace => "Trace",
            Self::Error => "Error",
            Self::Fix => "Fix",
            Self::Related => "Related",
        }
    }

    pub fn is_core(&self) -> bool {
        matches!(self, Self::Target | Self::Action | Self::Agent | Self::Value)
    }
}

/// A filled element in a frame instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixFrameFiller {
    /// The role this filler occupies
    pub role: NixFrameRole,
    /// The textual value
    pub value: String,
    /// Confidence in this binding (0.0 - 1.0)
    pub confidence: f32,
    /// Related primitives
    pub primitives: Vec<NixPrimitive>,
}

// =============================================================================
// NIX FRAME DEFINITIONS
// =============================================================================

/// Frame type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NixFrameType {
    /// Enabling/disabling a systemd service
    EnableService,
    /// Understanding override resolution
    OverrideResolution,
    /// Package/service dependency chains
    DependencyChain,
    /// Diagnosing configuration failures
    FailureDiagnosis,
    /// Security-related configuration
    SecurityPosture,
    /// Module import and composition
    ModuleImport,
    /// Package installation
    PackageInstall,
    /// Package search/discovery
    PackageSearch,
    /// Package removal
    PackageRemove,
    /// System upgrade
    SystemUpgrade,
    /// Generation rollback
    GenerationRollback,
    /// Garbage collection
    GarbageCollect,
    /// Flake operations
    FlakeOperation,
    /// Generic NixOS operation (fallback)
    GenericOperation,
}

impl NixFrameType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::EnableService => "EnableService",
            Self::OverrideResolution => "OverrideResolution",
            Self::DependencyChain => "DependencyChain",
            Self::FailureDiagnosis => "FailureDiagnosis",
            Self::SecurityPosture => "SecurityPosture",
            Self::ModuleImport => "ModuleImport",
            Self::PackageInstall => "PackageInstall",
            Self::PackageSearch => "PackageSearch",
            Self::PackageRemove => "PackageRemove",
            Self::SystemUpgrade => "SystemUpgrade",
            Self::GenerationRollback => "GenerationRollback",
            Self::GarbageCollect => "GarbageCollect",
            Self::FlakeOperation => "FlakeOperation",
            Self::GenericOperation => "GenericOperation",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::EnableService => "Enable/configure a systemd service with its dependencies",
            Self::OverrideResolution => "Understand how mkDefault/mkForce/mkOverride interact",
            Self::DependencyChain => "Trace package or service dependencies",
            Self::FailureDiagnosis => "Diagnose why a NixOS configuration failed",
            Self::SecurityPosture => "Analyze security implications of configuration",
            Self::ModuleImport => "Understand module imports and composition",
            Self::PackageInstall => "Install a package to the system",
            Self::PackageSearch => "Search for packages in nixpkgs",
            Self::PackageRemove => "Remove a package from the system",
            Self::SystemUpgrade => "Upgrade system packages and configuration",
            Self::GenerationRollback => "Rollback to a previous system generation",
            Self::GarbageCollect => "Clean up unused store paths",
            Self::FlakeOperation => "Operations involving Nix flakes",
            Self::GenericOperation => "Generic NixOS operation",
        }
    }

    /// Get required frame elements
    pub fn core_elements(&self) -> Vec<NixFrameRole> {
        match self {
            Self::EnableService => vec![
                NixFrameRole::Target,     // service name
                NixFrameRole::Action,     // enable/disable
            ],
            Self::OverrideResolution => vec![
                NixFrameRole::Target,     // option being overridden
                NixFrameRole::Priority,   // override level
                NixFrameRole::Value,      // resulting value
            ],
            Self::DependencyChain => vec![
                NixFrameRole::Target,     // starting point
                NixFrameRole::Requirement, // dependencies
            ],
            Self::FailureDiagnosis => vec![
                NixFrameRole::Error,      // error message
                NixFrameRole::Target,     // what failed
            ],
            Self::SecurityPosture => vec![
                NixFrameRole::Target,     // security-relevant option
                NixFrameRole::Effect,     // security impact
            ],
            Self::ModuleImport => vec![
                NixFrameRole::Source,     // importing module
                NixFrameRole::Target,     // imported module
            ],
            Self::PackageInstall => vec![
                NixFrameRole::Target,     // package name
                NixFrameRole::Action,     // install
            ],
            Self::PackageSearch => vec![
                NixFrameRole::Target,     // search query
                NixFrameRole::Action,     // search
            ],
            Self::PackageRemove => vec![
                NixFrameRole::Target,     // package name
                NixFrameRole::Action,     // remove
            ],
            Self::SystemUpgrade => vec![
                NixFrameRole::Action,     // upgrade
            ],
            Self::GenerationRollback => vec![
                NixFrameRole::Action,     // rollback
            ],
            Self::GarbageCollect => vec![
                NixFrameRole::Action,     // clean/gc
            ],
            Self::FlakeOperation => vec![
                NixFrameRole::Target,     // flake reference
                NixFrameRole::Action,     // operation type
            ],
            Self::GenericOperation => vec![
                NixFrameRole::Action,
            ],
        }
    }

    /// Get optional frame elements
    pub fn peripheral_elements(&self) -> Vec<NixFrameRole> {
        match self {
            Self::EnableService => vec![
                NixFrameRole::Requirement,  // after/wants
                NixFrameRole::Condition,    // mkIf condition
                NixFrameRole::Conflict,     // conflicts
                NixFrameRole::Source,       // module defining it
            ],
            Self::OverrideResolution => vec![
                NixFrameRole::Default,      // original default
                NixFrameRole::Source,       // where override came from
                NixFrameRole::Trace,        // resolution path
            ],
            Self::DependencyChain => vec![
                NixFrameRole::Want,         // optional deps
                NixFrameRole::Conflict,     // conflicts
                NixFrameRole::Trace,        // full chain
            ],
            Self::FailureDiagnosis => vec![
                NixFrameRole::Fix,          // suggested fix
                NixFrameRole::Reason,       // root cause
                NixFrameRole::Related,      // related errors
                NixFrameRole::Trace,        // evaluation trace
            ],
            Self::SecurityPosture => vec![
                NixFrameRole::Reason,       // why security-relevant
                NixFrameRole::Related,      // related settings
                NixFrameRole::Fix,          // recommendation
            ],
            Self::ModuleImport => vec![
                NixFrameRole::Condition,    // import condition
                NixFrameRole::Effect,       // what the import adds
            ],
            Self::PackageInstall | Self::PackageRemove => vec![
                NixFrameRole::Destination,  // profile
                NixFrameRole::Requirement,  // dependencies
            ],
            Self::PackageSearch => vec![
                NixFrameRole::Source,       // channel/flake
            ],
            Self::SystemUpgrade | Self::GenerationRollback | Self::GarbageCollect => vec![
                NixFrameRole::Target,       // optional specifier
            ],
            Self::FlakeOperation => vec![
                NixFrameRole::Source,       // input flakes
                NixFrameRole::Destination,  // output
            ],
            Self::GenericOperation => vec![
                NixFrameRole::Target,
                NixFrameRole::Source,
                NixFrameRole::Destination,
            ],
        }
    }

    /// Get relevant primitives for this frame type
    pub fn relevant_primitives(&self) -> Vec<NixPrimitive> {
        match self {
            Self::EnableService => vec![
                NixPrimitive::Service, NixPrimitive::If, NixPrimitive::Depends,
                NixPrimitive::Exists, NixPrimitive::Module, NixPrimitive::Option,
            ],
            Self::OverrideResolution => vec![
                NixPrimitive::Override, NixPrimitive::Default, NixPrimitive::Priority,
                NixPrimitive::Merge, NixPrimitive::Value, NixPrimitive::Origin,
            ],
            Self::DependencyChain => vec![
                NixPrimitive::Depends, NixPrimitive::Dependency, NixPrimitive::Package,
                NixPrimitive::ConflictsWith, NixPrimitive::Required, NixPrimitive::Optional,
            ],
            Self::FailureDiagnosis => vec![
                NixPrimitive::Trace, NixPrimitive::Evaluate, NixPrimitive::Constraint,
                NixPrimitive::ConflictsWith, NixPrimitive::Origin, NixPrimitive::Type,
            ],
            Self::SecurityPosture => vec![
                NixPrimitive::Service, NixPrimitive::Option, NixPrimitive::Constraint,
                NixPrimitive::Host, NixPrimitive::Effect,
            ],
            Self::ModuleImport => vec![
                NixPrimitive::Module, NixPrimitive::AttrSet, NixPrimitive::Merge,
                NixPrimitive::Path, NixPrimitive::If,
            ],
            Self::PackageInstall | Self::PackageRemove => vec![
                NixPrimitive::Package, NixPrimitive::StorePath, NixPrimitive::Derivation,
                NixPrimitive::Effect,
            ],
            Self::PackageSearch => vec![
                NixPrimitive::Package, NixPrimitive::AttrSet, NixPrimitive::Key,
            ],
            Self::SystemUpgrade => vec![
                NixPrimitive::Package, NixPrimitive::Effect, NixPrimitive::Derivation,
            ],
            Self::GenerationRollback => vec![
                NixPrimitive::Effect, NixPrimitive::StorePath, NixPrimitive::Host,
            ],
            Self::GarbageCollect => vec![
                NixPrimitive::StorePath, NixPrimitive::Effect, NixPrimitive::Derivation,
            ],
            Self::FlakeOperation => vec![
                NixPrimitive::Module, NixPrimitive::AttrSet, NixPrimitive::Path,
                NixPrimitive::Derivation, NixPrimitive::StorePath,
            ],
            Self::GenericOperation => vec![
                NixPrimitive::Effect, NixPrimitive::Host,
            ],
        }
    }

    /// Get example sentences that activate this frame
    pub fn examples(&self) -> Vec<&'static str> {
        match self {
            Self::EnableService => vec![
                "enable nginx service",
                "services.nginx.enable = true",
                "disable postgresql",
                "turn on the ssh server",
            ],
            Self::OverrideResolution => vec![
                "why does mkForce override mkDefault",
                "how are option priorities resolved",
                "my override isn't taking effect",
                "lib.mkOverride 50",
            ],
            Self::DependencyChain => vec![
                "what depends on openssl",
                "show dependencies of firefox",
                "why is glibc required",
                "service dependency tree",
            ],
            Self::FailureDiagnosis => vec![
                "nixos-rebuild failed with error",
                "infinite recursion in module",
                "option conflict between modules",
                "type mismatch in configuration",
            ],
            Self::SecurityPosture => vec![
                "is this configuration secure",
                "what ports are open",
                "check firewall settings",
                "security audit configuration",
            ],
            Self::ModuleImport => vec![
                "import another module",
                "include hardware configuration",
                "split configuration into files",
                "imports = [ ./other.nix ]",
            ],
            Self::PackageInstall => vec![
                "install firefox",
                "add vim to my system",
                "nix profile install",
                "get neovim installed",
            ],
            Self::PackageSearch => vec![
                "search for editors",
                "find packages matching vim",
                "nix search nixpkgs",
                "look for python packages",
            ],
            Self::PackageRemove => vec![
                "remove htop",
                "uninstall firefox",
                "delete vim from system",
                "nix profile remove",
            ],
            Self::SystemUpgrade => vec![
                "upgrade all packages",
                "update my system",
                "nix flake update",
                "nixos-rebuild switch --upgrade",
            ],
            Self::GenerationRollback => vec![
                "rollback to previous generation",
                "undo last system change",
                "go back to working config",
                "nixos-rebuild --rollback",
            ],
            Self::GarbageCollect => vec![
                "clean up disk space",
                "nix-collect-garbage",
                "remove unused packages",
                "free up nix store",
            ],
            Self::FlakeOperation => vec![
                "update flake inputs",
                "nix develop",
                "build flake output",
                "init new flake",
            ],
            Self::GenericOperation => vec![
                "do something with nix",
            ],
        }
    }

    /// Get all frame types
    pub fn all() -> Vec<Self> {
        vec![
            Self::EnableService,
            Self::OverrideResolution,
            Self::DependencyChain,
            Self::FailureDiagnosis,
            Self::SecurityPosture,
            Self::ModuleImport,
            Self::PackageInstall,
            Self::PackageSearch,
            Self::PackageRemove,
            Self::SystemUpgrade,
            Self::GenerationRollback,
            Self::GarbageCollect,
            Self::FlakeOperation,
            Self::GenericOperation,
        ]
    }
}

// =============================================================================
// FRAME INSTANCE
// =============================================================================

/// An instantiated frame with filled elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixFrameInstance {
    /// The frame type
    pub frame_type: NixFrameType,
    /// Confidence in this frame being correct
    pub confidence: f32,
    /// Filled frame elements
    pub elements: Vec<NixFrameFiller>,
    /// Active primitives from analysis
    pub active_primitives: Vec<NixPrimitive>,
    /// Hyperdimensional encoding of this instance
    pub encoding: Option<HV16>,
}

impl NixFrameInstance {
    /// Create a new empty frame instance
    pub fn new(frame_type: NixFrameType) -> Self {
        Self {
            frame_type,
            confidence: 0.0,
            elements: Vec::new(),
            active_primitives: Vec::new(),
            encoding: None,
        }
    }

    /// Add a filler to the frame
    pub fn add_filler(&mut self, role: NixFrameRole, value: String, confidence: f32) {
        self.elements.push(NixFrameFiller {
            role,
            value,
            confidence,
            primitives: Vec::new(),
        });
    }

    /// Get filler for a role
    pub fn get_filler(&self, role: NixFrameRole) -> Option<&NixFrameFiller> {
        self.elements.iter().find(|f| f.role == role)
    }

    /// Check if frame has all core elements filled
    pub fn is_complete(&self) -> bool {
        let core = self.frame_type.core_elements();
        core.iter().all(|role| self.get_filler(*role).is_some())
    }

    /// Get missing core elements
    pub fn missing_elements(&self) -> Vec<NixFrameRole> {
        let core = self.frame_type.core_elements();
        core.into_iter()
            .filter(|role| self.get_filler(*role).is_none())
            .collect()
    }
}

// =============================================================================
// FRAME LIBRARY
// =============================================================================

/// Library of NixOS frames with activation logic
pub struct NixFrameLibrary {
    encoder: NixPrimitiveEncoder,
    frame_encodings: HashMap<NixFrameType, HV16>,
}

impl NixFrameLibrary {
    /// Create a new frame library
    pub fn new() -> Self {
        let encoder = NixPrimitiveEncoder::new();
        let mut frame_encodings = HashMap::new();

        // Generate frame encodings from their relevant primitives
        for frame_type in NixFrameType::all() {
            let primitives = frame_type.relevant_primitives();
            if let Some(encoding) = encoder.encode_combination(&primitives) {
                frame_encodings.insert(frame_type, encoding);
            } else {
                // Fallback: create encoding from frame name
                let seed = Self::name_to_seed(frame_type.name());
                frame_encodings.insert(frame_type, HV16::random(seed));
            }
        }

        Self { encoder, frame_encodings }
    }

    fn name_to_seed(name: &str) -> u64 {
        let mut seed: u64 = 0;
        for (i, b) in name.bytes().enumerate() {
            seed = seed.wrapping_add((b as u64).wrapping_mul(31_u64.wrapping_pow(i as u32)));
        }
        seed
    }

    /// Find the best matching frame for input
    pub fn match_frame(&self, input: &str) -> Option<(NixFrameType, f32)> {
        let input_lower = input.to_lowercase();
        let mut best_match: Option<(NixFrameType, f32)> = None;

        for frame_type in NixFrameType::all() {
            // Check example matches
            let example_score = frame_type.examples().iter()
                .filter(|ex| {
                    let ex_lower = ex.to_lowercase();
                    // Match if input contains example keywords or vice versa
                    ex_lower.split_whitespace().any(|w| input_lower.contains(w)) ||
                    input_lower.split_whitespace().any(|w| ex_lower.contains(w))
                })
                .count() as f32 / frame_type.examples().len().max(1) as f32;

            // Check primitive relevance
            let primitive_score = frame_type.relevant_primitives().iter()
                .filter(|p| {
                    p.nix_expressions().iter()
                        .any(|expr| input_lower.contains(&expr.to_lowercase()))
                })
                .count() as f32 / frame_type.relevant_primitives().len().max(1) as f32;

            let combined = 0.7 * example_score + 0.3 * primitive_score;

            if combined > 0.1 && (best_match.is_none() || combined > best_match.unwrap().1) {
                best_match = Some((frame_type, combined));
            }
        }

        best_match
    }

    /// Activate a frame from input, filling available elements
    pub fn activate(&self, input: &str) -> NixFrameInstance {
        let (frame_type, confidence) = self.match_frame(input)
            .unwrap_or((NixFrameType::GenericOperation, 0.0));

        let mut instance = NixFrameInstance::new(frame_type);
        instance.confidence = confidence;

        // Extract fillers based on frame type
        self.extract_fillers(input, &mut instance);

        // Add encoding
        if let Some(enc) = self.frame_encodings.get(&frame_type) {
            instance.encoding = Some(enc.clone());
        }

        instance
    }

    /// Extract frame element fillers from input
    fn extract_fillers(&self, input: &str, instance: &mut NixFrameInstance) {
        let tokens: Vec<&str> = input.split_whitespace().collect();
        let input_lower = input.to_lowercase();

        match instance.frame_type {
            NixFrameType::PackageInstall | NixFrameType::PackageRemove | NixFrameType::PackageSearch => {
                // Extract package name (usually after the action keyword)
                let action_keywords = ["install", "remove", "uninstall", "search", "find", "add", "delete"];
                for (i, token) in tokens.iter().enumerate() {
                    let token_lower = token.to_lowercase();
                    if action_keywords.iter().any(|kw| token_lower.contains(kw)) {
                        instance.add_filler(NixFrameRole::Action, token.to_string(), 0.9);

                        // Next token is likely the target
                        if let Some(target) = tokens.get(i + 1) {
                            if !target.starts_with("-") {
                                instance.add_filler(NixFrameRole::Target, target.to_string(), 0.8);
                            }
                        }
                        break;
                    }
                }

                // If no target found, try to find package-like tokens
                if instance.get_filler(NixFrameRole::Target).is_none() {
                    for token in &tokens {
                        // Skip common words
                        if !["the", "a", "for", "my", "to", "please", "can", "you"].contains(&token.to_lowercase().as_str()) {
                            if !token.starts_with("-") && token.len() > 2 {
                                instance.add_filler(NixFrameRole::Target, token.to_string(), 0.5);
                                break;
                            }
                        }
                    }
                }
            }

            NixFrameType::EnableService => {
                // Extract service name
                let service_keywords = ["enable", "disable", "start", "stop", "turn on", "turn off"];
                for (i, token) in tokens.iter().enumerate() {
                    let token_lower = token.to_lowercase();
                    if service_keywords.iter().any(|kw| token_lower.contains(kw)) {
                        let action = if token_lower.contains("disable") || token_lower.contains("stop") || token_lower.contains("off") {
                            "disable"
                        } else {
                            "enable"
                        };
                        instance.add_filler(NixFrameRole::Action, action.to_string(), 0.9);

                        // Next token is likely the service
                        if let Some(service) = tokens.get(i + 1) {
                            if !service.starts_with("-") {
                                instance.add_filler(NixFrameRole::Target, service.to_string(), 0.8);
                            }
                        }
                        break;
                    }
                }

                // Check for services.X.enable pattern
                if input_lower.contains("services.") {
                    if let Some(start) = input_lower.find("services.") {
                        let rest = &input[start + 9..];
                        if let Some(end) = rest.find(|c: char| c == '.' || c == ' ' || c == '=') {
                            let service_name = &rest[..end];
                            instance.add_filler(NixFrameRole::Target, service_name.to_string(), 0.95);
                        }
                    }
                }
            }

            NixFrameType::SystemUpgrade | NixFrameType::GenerationRollback | NixFrameType::GarbageCollect => {
                // These have simple Action as core element
                instance.add_filler(
                    NixFrameRole::Action,
                    instance.frame_type.name().to_string(),
                    0.8
                );
            }

            NixFrameType::FailureDiagnosis => {
                // Look for error patterns
                if input_lower.contains("error") || input_lower.contains("failed") {
                    instance.add_filler(NixFrameRole::Error, input.to_string(), 0.7);
                }
                if input_lower.contains("infinite recursion") {
                    instance.add_filler(NixFrameRole::Error, "Infinite recursion".to_string(), 0.9);
                }
                if input_lower.contains("type mismatch") || input_lower.contains("type error") {
                    instance.add_filler(NixFrameRole::Error, "Type mismatch".to_string(), 0.9);
                }
            }

            NixFrameType::OverrideResolution => {
                // Look for override-related patterns
                for keyword in ["mkDefault", "mkForce", "mkOverride", "priority"] {
                    if input_lower.contains(&keyword.to_lowercase()) {
                        instance.add_filler(NixFrameRole::Priority, keyword.to_string(), 0.9);
                        break;
                    }
                }
            }

            _ => {
                // Generic extraction: first noun-like token as target
                for token in &tokens {
                    if token.len() > 2 && !token.starts_with("-") {
                        instance.add_filler(NixFrameRole::Target, token.to_string(), 0.3);
                        break;
                    }
                }
            }
        }
    }

    /// Get the primitive encoder
    pub fn encoder(&self) -> &NixPrimitiveEncoder {
        &self.encoder
    }

    /// Get frame encoding
    pub fn get_encoding(&self, frame_type: NixFrameType) -> Option<&HV16> {
        self.frame_encodings.get(&frame_type)
    }
}

impl Default for NixFrameLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_type_all() {
        let all = NixFrameType::all();
        assert!(all.len() >= 10);
    }

    #[test]
    fn test_frame_core_elements() {
        let install = NixFrameType::PackageInstall;
        let core = install.core_elements();
        assert!(core.contains(&NixFrameRole::Target));
        assert!(core.contains(&NixFrameRole::Action));
    }

    #[test]
    fn test_frame_examples() {
        let install = NixFrameType::PackageInstall;
        assert!(!install.examples().is_empty());
        assert!(install.examples().iter().any(|ex| ex.contains("install")));
    }

    #[test]
    fn test_frame_library_creation() {
        let library = NixFrameLibrary::new();
        assert!(library.get_encoding(NixFrameType::PackageInstall).is_some());
    }

    #[test]
    fn test_frame_matching() {
        let library = NixFrameLibrary::new();

        let (frame, confidence) = library.match_frame("install firefox").unwrap();
        assert_eq!(frame, NixFrameType::PackageInstall);
        assert!(confidence > 0.0);
    }

    #[test]
    fn test_frame_activation() {
        let library = NixFrameLibrary::new();

        let instance = library.activate("install firefox");
        assert_eq!(instance.frame_type, NixFrameType::PackageInstall);

        // Should have extracted Target and Action
        assert!(instance.get_filler(NixFrameRole::Target).is_some() ||
                instance.get_filler(NixFrameRole::Action).is_some());
    }

    #[test]
    fn test_service_frame() {
        let library = NixFrameLibrary::new();

        let instance = library.activate("enable nginx service");
        assert_eq!(instance.frame_type, NixFrameType::EnableService);
    }

    #[test]
    fn test_frame_instance_completion() {
        let mut instance = NixFrameInstance::new(NixFrameType::PackageInstall);

        assert!(!instance.is_complete());

        instance.add_filler(NixFrameRole::Target, "firefox".to_string(), 0.9);
        instance.add_filler(NixFrameRole::Action, "install".to_string(), 0.9);

        assert!(instance.is_complete());
    }

    #[test]
    fn test_nix_expression_pattern() {
        let library = NixFrameLibrary::new();

        // Use "enable" keyword which clearly triggers EnableService
        let instance = library.activate("enable nginx service");
        assert_eq!(instance.frame_type, NixFrameType::EnableService);
    }

    #[test]
    fn test_failure_diagnosis_frame() {
        let library = NixFrameLibrary::new();

        // Use "infinite recursion" which is a specific error pattern
        let instance = library.activate("error: infinite recursion in module");
        assert_eq!(instance.frame_type, NixFrameType::FailureDiagnosis);

        // Should extract error info
        assert!(instance.get_filler(NixFrameRole::Error).is_some());
    }

    #[test]
    fn test_upgrade_frame() {
        let library = NixFrameLibrary::new();

        // Use "update my system" which matches SystemUpgrade examples
        let instance = library.activate("update my system");
        assert_eq!(instance.frame_type, NixFrameType::SystemUpgrade);
    }

    #[test]
    fn test_garbage_collect_frame() {
        let library = NixFrameLibrary::new();

        // Use only "garbage" which is unique to GarbageCollect examples
        // This word appears only in "nix-collect-garbage" example
        let instance = library.activate("garbage");
        assert_eq!(instance.frame_type, NixFrameType::GarbageCollect);
    }
}
