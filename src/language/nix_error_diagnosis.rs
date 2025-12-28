//! NixOS Error Diagnosis System
//!
//! Revolutionary error understanding that transforms cryptic NixOS errors into
//! actionable explanations with probable causes, minimal fixes, and verification steps.
//!
//! Covers the "big 6" error categories:
//! 1. Evaluation errors (infinite recursion, missing attribute, type mismatch)
//! 2. Derivation/build failures (fetch, hash mismatch, patch failure)
//! 3. Conflicting options / module merge conflicts
//! 4. Missing dependency / PATH confusion
//! 5. Permission/ownership issues
//! 6. Flake-specific errors

use std::collections::HashMap;
use crate::hdc::HV16;
use crate::observability::{SharedObserver, types::*};

// =============================================================================
// ERROR CATEGORIES
// =============================================================================

/// Top-level NixOS error categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NixErrorCategory {
    /// Nix expression evaluation errors
    Evaluation,
    /// Derivation build failures
    Build,
    /// Option/module conflicts
    Conflict,
    /// Missing dependencies or resources
    MissingResource,
    /// Permission and ownership errors
    Permission,
    /// Flake-specific errors
    Flake,
    /// Unknown error type
    Unknown,
}

impl NixErrorCategory {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Evaluation,
            Self::Build,
            Self::Conflict,
            Self::MissingResource,
            Self::Permission,
            Self::Flake,
            Self::Unknown,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Evaluation => "Evaluation Error",
            Self::Build => "Build Failure",
            Self::Conflict => "Configuration Conflict",
            Self::MissingResource => "Missing Resource",
            Self::Permission => "Permission Error",
            Self::Flake => "Flake Error",
            Self::Unknown => "Unknown Error",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Evaluation => "Error during Nix expression evaluation",
            Self::Build => "Error during derivation build process",
            Self::Conflict => "Conflicting module options or definitions",
            Self::MissingResource => "Required resource not found",
            Self::Permission => "Insufficient permissions for operation",
            Self::Flake => "Flake-specific configuration or lock error",
            Self::Unknown => "Error type could not be determined",
        }
    }
}

// =============================================================================
// ERROR SUBTYPES
// =============================================================================

/// Specific error types within each category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NixErrorType {
    // Evaluation errors
    InfiniteRecursion,
    MissingAttribute,
    TypeMismatch,
    AssertionFailed,
    UndefinedVariable,
    SyntaxError,

    // Build errors
    FetchFailure,
    HashMismatch,
    PatchFailed,
    CompilationError,
    TestFailure,
    DependencyBuildFailed,

    // Conflict errors
    OptionConflict,
    ModuleMergeConflict,
    PriorityConflict,
    DuplicateDefinition,
    MutuallyExclusive,

    // Missing resource errors
    PackageNotFound,
    FileNotFound,
    AttributePathNotFound,
    ChannelNotFound,
    InputNotFound,

    // Permission errors
    StorePermissionDenied,
    FilePermissionDenied,
    DaemonConnectionFailed,
    SudoRequired,
    TrustNotConfigured,

    // Flake errors
    LockfileOutdated,
    InputMismatch,
    DirtyWorkingTree,
    FlakeNotFound,
    RegistryError,
    PureEvalViolation,

    // Fallback
    UnknownError,
}

impl NixErrorType {
    pub fn category(&self) -> NixErrorCategory {
        match self {
            Self::InfiniteRecursion | Self::MissingAttribute | Self::TypeMismatch |
            Self::AssertionFailed | Self::UndefinedVariable | Self::SyntaxError
                => NixErrorCategory::Evaluation,

            Self::FetchFailure | Self::HashMismatch | Self::PatchFailed |
            Self::CompilationError | Self::TestFailure | Self::DependencyBuildFailed
                => NixErrorCategory::Build,

            Self::OptionConflict | Self::ModuleMergeConflict | Self::PriorityConflict |
            Self::DuplicateDefinition | Self::MutuallyExclusive
                => NixErrorCategory::Conflict,

            Self::PackageNotFound | Self::FileNotFound | Self::AttributePathNotFound |
            Self::ChannelNotFound | Self::InputNotFound
                => NixErrorCategory::MissingResource,

            Self::StorePermissionDenied | Self::FilePermissionDenied |
            Self::DaemonConnectionFailed | Self::SudoRequired | Self::TrustNotConfigured
                => NixErrorCategory::Permission,

            Self::LockfileOutdated | Self::InputMismatch | Self::DirtyWorkingTree |
            Self::FlakeNotFound | Self::RegistryError | Self::PureEvalViolation
                => NixErrorCategory::Flake,

            Self::UnknownError => NixErrorCategory::Unknown,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::InfiniteRecursion => "infinite recursion",
            Self::MissingAttribute => "missing attribute",
            Self::TypeMismatch => "type mismatch",
            Self::AssertionFailed => "assertion failed",
            Self::UndefinedVariable => "undefined variable",
            Self::SyntaxError => "syntax error",

            Self::FetchFailure => "fetch failure",
            Self::HashMismatch => "hash mismatch",
            Self::PatchFailed => "patch failed",
            Self::CompilationError => "compilation error",
            Self::TestFailure => "test failure",
            Self::DependencyBuildFailed => "dependency build failed",

            Self::OptionConflict => "option conflict",
            Self::ModuleMergeConflict => "module merge conflict",
            Self::PriorityConflict => "priority conflict",
            Self::DuplicateDefinition => "duplicate definition",
            Self::MutuallyExclusive => "mutually exclusive options",

            Self::PackageNotFound => "package not found",
            Self::FileNotFound => "file not found",
            Self::AttributePathNotFound => "attribute path not found",
            Self::ChannelNotFound => "channel not found",
            Self::InputNotFound => "input not found",

            Self::StorePermissionDenied => "store permission denied",
            Self::FilePermissionDenied => "file permission denied",
            Self::DaemonConnectionFailed => "daemon connection failed",
            Self::SudoRequired => "sudo required",
            Self::TrustNotConfigured => "trust not configured",

            Self::LockfileOutdated => "lockfile outdated",
            Self::InputMismatch => "input mismatch",
            Self::DirtyWorkingTree => "dirty working tree",
            Self::FlakeNotFound => "flake not found",
            Self::RegistryError => "registry error",
            Self::PureEvalViolation => "pure eval violation",

            Self::UnknownError => "unknown error",
        }
    }

    /// Patterns that indicate this error type in stderr output
    pub fn detection_patterns(&self) -> Vec<&'static str> {
        match self {
            Self::InfiniteRecursion => vec![
                "infinite recursion",
                "stack overflow",
                "maximum call depth",
                "evaluation depth exceeded",
            ],
            Self::MissingAttribute => vec![
                "attribute '",
                "' missing",
                "error: undefined variable",
                "does not have attribute",
                "has no attribute",
            ],
            Self::TypeMismatch => vec![
                "type mismatch",
                "expected a",
                "but got",
                "cannot coerce",
                "is not a",
                "value is a",
            ],
            Self::AssertionFailed => vec![
                "assertion failed",
                "assertion '",
                "' failed",
                "evaluation aborted",
            ],
            Self::UndefinedVariable => vec![
                "undefined variable",
                "error: undefined variable",
            ],
            Self::SyntaxError => vec![
                "syntax error",
                "unexpected",
                "expected",
                "at line",
                "parse error",
            ],

            Self::FetchFailure => vec![
                "unable to download",
                "fetch failed",
                "connection refused",
                "Could not resolve host",
                "SSL certificate problem",
                "curl: (7)",
                "curl: (6)",
            ],
            Self::HashMismatch => vec![
                "hash mismatch",
                "got:",
                "expected:",
                "sha256 hash",
                "specified:",
                "fixed-output derivation",
            ],
            Self::PatchFailed => vec![
                "patch failed",
                "patch did not apply",
                "hunks FAILED",
                "FAILED --",
                "can't find file to patch",
            ],
            Self::CompilationError => vec![
                "error:",
                "undefined reference",
                "cannot find",
                "compilation failed",
                "make: ***",
                "cc1:",
                "ld:",
            ],
            Self::TestFailure => vec![
                "test failed",
                "check failed",
                "tests failed",
                "FAIL:",
                "FAILED",
            ],
            Self::DependencyBuildFailed => vec![
                "dependency",
                "failed to build",
                "builder for",
                "failed with exit code",
            ],

            Self::OptionConflict => vec![
                "option",
                "conflict",
                "The option",
                "has conflicting definition values",
                "conflicting values",
            ],
            Self::ModuleMergeConflict => vec![
                "module",
                "conflict",
                "cannot merge",
                "merge conflict",
                "defined in both",
            ],
            Self::PriorityConflict => vec![
                "mkForce",
                "mkDefault",
                "mkOverride",
                "priority",
                "same priority",
            ],
            Self::DuplicateDefinition => vec![
                "duplicate",
                "already defined",
                "redefined",
                "multiple definitions",
            ],
            Self::MutuallyExclusive => vec![
                "mutually exclusive",
                "cannot be used together",
                "conflicts with",
                "incompatible with",
            ],

            Self::PackageNotFound => vec![
                "error: flake",
                "does not provide attribute",
                "Package '",
                "' not found",
                "attribute '",
                "' not found",
            ],
            Self::FileNotFound => vec![
                "No such file",
                "does not exist",
                "path '",
                "' does not exist",
                "cannot find",
            ],
            Self::AttributePathNotFound => vec![
                "attribute '",
                "' in selection path",
                "not found",
                "does not provide",
            ],
            Self::ChannelNotFound => vec![
                "channel",
                "not found",
                "no channel named",
                "NIX_PATH",
            ],
            Self::InputNotFound => vec![
                "input '",
                "not found",
                "does not exist in lock file",
            ],

            Self::StorePermissionDenied => vec![
                "/nix/store",
                "Permission denied",
                "cannot create",
                "cannot write",
            ],
            Self::FilePermissionDenied => vec![
                "Permission denied",
                "cannot open",
                "cannot read",
                "access denied",
            ],
            Self::DaemonConnectionFailed => vec![
                "cannot connect to daemon",
                "nix-daemon",
                "connection refused",
                "socket",
            ],
            Self::SudoRequired => vec![
                "sudo",
                "root",
                "permission denied",
                "must be run as root",
            ],
            Self::TrustNotConfigured => vec![
                "trusted",
                "not trusted",
                "trusted-users",
                "allowed-users",
            ],

            Self::LockfileOutdated => vec![
                "lock file",
                "out of date",
                "does not match",
                "needs to be updated",
            ],
            Self::InputMismatch => vec![
                "input",
                "mismatch",
                "does not match",
                "input '",
            ],
            Self::DirtyWorkingTree => vec![
                "dirty",
                "uncommitted",
                "not committed",
                "working tree",
                "--impure",
            ],
            Self::FlakeNotFound => vec![
                "flake '",
                "not found",
                "does not exist",
                "no flake.nix",
            ],
            Self::RegistryError => vec![
                "registry",
                "flake registry",
                "cannot fetch",
            ],
            Self::PureEvalViolation => vec![
                "--pure-eval",
                "pure evaluation",
                "impure",
                "access to path",
                "access to absolute path",
            ],

            Self::UnknownError => vec![],
        }
    }
}

// =============================================================================
// ERROR DIAGNOSIS FRAME
// =============================================================================

/// Role in the error diagnosis frame
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorFrameRole {
    /// The error symptom/message
    Symptom,
    /// File/line where error occurred
    Location,
    /// Most likely cause(s)
    LikelyCause,
    /// Evidence supporting the diagnosis
    Evidence,
    /// Suggested fix(es)
    Fix,
    /// Verification step after fix
    Verify,
    /// Risk level of the fix
    RiskLevel,
    /// Related configuration affected
    AffectedConfig,
    /// Module that caused the error
    SourceModule,
    /// The exact Nix expression that failed
    FailingExpression,
}

impl ErrorFrameRole {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Symptom,
            Self::Location,
            Self::LikelyCause,
            Self::Evidence,
            Self::Fix,
            Self::Verify,
            Self::RiskLevel,
            Self::AffectedConfig,
            Self::SourceModule,
            Self::FailingExpression,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Symptom => "symptom",
            Self::Location => "location",
            Self::LikelyCause => "likely_cause",
            Self::Evidence => "evidence",
            Self::Fix => "fix",
            Self::Verify => "verify",
            Self::RiskLevel => "risk_level",
            Self::AffectedConfig => "affected_config",
            Self::SourceModule => "source_module",
            Self::FailingExpression => "failing_expression",
        }
    }
}

/// Risk level for applying a fix
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FixRiskLevel {
    /// Safe to apply, no system changes
    Safe,
    /// Low risk, reversible changes
    Low,
    /// Medium risk, may require rollback
    Medium,
    /// High risk, significant changes
    High,
    /// Critical, potential system breakage
    Critical,
}

impl FixRiskLevel {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Safe => "No system changes, information only",
            Self::Low => "Minimal changes, easily reversible",
            Self::Medium => "Moderate changes, rollback recommended before applying",
            Self::High => "Significant changes, backup recommended",
            Self::Critical => "Major changes, could affect system boot",
        }
    }
}

// =============================================================================
// ERROR DIAGNOSIS RESULT
// =============================================================================

/// A suggested fix for an error
#[derive(Debug, Clone)]
pub struct SuggestedFix {
    /// Human-readable description
    pub description: String,
    /// Command(s) to apply the fix
    pub commands: Vec<String>,
    /// Risk level
    pub risk: FixRiskLevel,
    /// Verification command
    pub verify_command: Option<String>,
    /// Whether this is the primary recommendation
    pub primary: bool,
}

/// Complete error diagnosis result
#[derive(Debug, Clone)]
pub struct ErrorDiagnosis {
    /// The error category
    pub category: NixErrorCategory,
    /// Specific error type
    pub error_type: NixErrorType,
    /// Confidence in the diagnosis (0.0-1.0)
    pub confidence: f32,
    /// Human-readable symptom description
    pub symptom: String,
    /// File/location where error occurred
    pub location: Option<String>,
    /// Likely causes (ranked by probability)
    pub likely_causes: Vec<String>,
    /// Evidence supporting the diagnosis
    pub evidence: Vec<String>,
    /// Suggested fixes
    pub fixes: Vec<SuggestedFix>,
    /// Affected configuration paths
    pub affected_configs: Vec<String>,
    /// Source module (if identifiable)
    pub source_module: Option<String>,
    /// The failing expression
    pub failing_expression: Option<String>,
    /// Human-readable explanation
    pub explanation: String,
}

impl ErrorDiagnosis {
    /// Get the primary (recommended) fix
    pub fn primary_fix(&self) -> Option<&SuggestedFix> {
        self.fixes.iter().find(|f| f.primary).or_else(|| self.fixes.first())
    }

    /// Get the overall risk level
    pub fn overall_risk(&self) -> FixRiskLevel {
        self.fixes.iter()
            .filter(|f| f.primary)
            .map(|f| f.risk)
            .max()
            .unwrap_or(FixRiskLevel::Safe)
    }
}

// =============================================================================
// ERROR DIAGNOSER
// =============================================================================

/// The main error diagnosis engine
pub struct NixErrorDiagnoser {
    /// Known error patterns
    patterns: HashMap<NixErrorType, Vec<String>>,
    /// HDC encodings for semantic matching
    encodings: HashMap<NixErrorType, HV16>,
    /// Observer for tracing error diagnosis
    observer: Option<SharedObserver>,
}

impl NixErrorDiagnoser {
    pub fn new() -> Self {
        Self::with_observer(None)
    }

    pub fn with_observer(observer: Option<SharedObserver>) -> Self {
        let mut patterns = HashMap::new();
        let mut encodings = HashMap::new();

        // Initialize patterns and encodings for each error type
        for error_type in Self::all_error_types() {
            let detection_patterns: Vec<String> = error_type
                .detection_patterns()
                .iter()
                .map(|s| s.to_lowercase())
                .collect();
            patterns.insert(error_type, detection_patterns);

            // Generate HDC encoding from error name
            let seed = Self::name_to_seed(error_type.name());
            encodings.insert(error_type, HV16::random(seed));
        }

        Self {
            patterns,
            encodings,
            observer,
        }
    }

    fn all_error_types() -> Vec<NixErrorType> {
        vec![
            NixErrorType::InfiniteRecursion,
            NixErrorType::MissingAttribute,
            NixErrorType::TypeMismatch,
            NixErrorType::AssertionFailed,
            NixErrorType::UndefinedVariable,
            NixErrorType::SyntaxError,
            NixErrorType::FetchFailure,
            NixErrorType::HashMismatch,
            NixErrorType::PatchFailed,
            NixErrorType::CompilationError,
            NixErrorType::TestFailure,
            NixErrorType::DependencyBuildFailed,
            NixErrorType::OptionConflict,
            NixErrorType::ModuleMergeConflict,
            NixErrorType::PriorityConflict,
            NixErrorType::DuplicateDefinition,
            NixErrorType::MutuallyExclusive,
            NixErrorType::PackageNotFound,
            NixErrorType::FileNotFound,
            NixErrorType::AttributePathNotFound,
            NixErrorType::ChannelNotFound,
            NixErrorType::InputNotFound,
            NixErrorType::StorePermissionDenied,
            NixErrorType::FilePermissionDenied,
            NixErrorType::DaemonConnectionFailed,
            NixErrorType::SudoRequired,
            NixErrorType::TrustNotConfigured,
            NixErrorType::LockfileOutdated,
            NixErrorType::InputMismatch,
            NixErrorType::DirtyWorkingTree,
            NixErrorType::FlakeNotFound,
            NixErrorType::RegistryError,
            NixErrorType::PureEvalViolation,
        ]
    }

    fn name_to_seed(name: &str) -> u64 {
        let mut seed: u64 = 0;
        for (i, b) in name.bytes().enumerate() {
            seed = seed.wrapping_add((b as u64).wrapping_mul(31_u64.wrapping_pow(i as u32)));
        }
        seed
    }

    /// Diagnose an error from stderr output
    pub fn diagnose(&self, error_output: &str) -> ErrorDiagnosis {
        let lower_output = error_output.to_lowercase();

        // Find the best matching error type
        let (error_type, confidence, evidence) = self.classify_error(&lower_output, error_output);

        // Extract location from error output
        let location = self.extract_location(error_output);

        // Extract failing expression
        let failing_expression = self.extract_failing_expression(error_output);

        // Generate causes based on error type
        let likely_causes = self.generate_causes(error_type);

        // Generate fixes based on error type
        let fixes = self.generate_fixes(error_type, error_output);

        // Extract affected configs
        let affected_configs = self.extract_affected_configs(error_output);

        // Extract source module
        let source_module = self.extract_source_module(error_output);

        // Generate human-readable explanation
        let explanation = self.generate_explanation(error_type, &evidence, &location);

        // Generate symptom description
        let symptom = self.summarize_symptom(error_output);

        let diagnosis = ErrorDiagnosis {
            category: error_type.category(),
            error_type,
            confidence,
            symptom,
            location,
            likely_causes,
            evidence,
            fixes,
            affected_configs,
            source_module,
            failing_expression,
            explanation,
        };

        // Record error diagnosis event
        if let Some(ref observer) = self.observer {
            let mut context = HashMap::new();
            context.insert("category".to_string(), diagnosis.category.name().to_string());
            context.insert("confidence".to_string(), format!("{:.2}", diagnosis.confidence));

            if let Some(ref loc) = diagnosis.location {
                context.insert("location".to_string(), loc.clone());
            }

            if let Some(ref module) = diagnosis.source_module {
                context.insert("source_module".to_string(), module.clone());
            }

            if !diagnosis.affected_configs.is_empty() {
                context.insert("affected_configs".to_string(), diagnosis.affected_configs.join(", "));
            }

            if let Some(primary_fix) = diagnosis.primary_fix() {
                context.insert("suggested_fix".to_string(), primary_fix.description.clone());
                context.insert("fix_risk".to_string(), primary_fix.risk.name().to_string());
            }

            // Determine if error is recoverable based on fixes and risk
            let recoverable = !diagnosis.fixes.is_empty() &&
                diagnosis.fixes.iter().any(|f| f.risk < FixRiskLevel::Critical);

            let event = ErrorEvent {
                timestamp: chrono::Utc::now(),
                error_type: diagnosis.error_type.name().to_string(),
                message: diagnosis.explanation.clone(),
                context,
                recoverable,
            };

            if let Ok(mut obs) = observer.try_write() {
                if let Err(e) = obs.record_error(event) {
                    eprintln!("[OBSERVER ERROR] Failed to record error diagnosis: {}", e);
                }
            }
        }

        diagnosis
    }

    fn classify_error(&self, lower_output: &str, _original: &str) -> (NixErrorType, f32, Vec<String>) {
        let mut best_type = NixErrorType::UnknownError;
        let mut best_score = 0.0f32;
        let mut best_evidence = Vec::new();

        for (error_type, patterns) in &self.patterns {
            let mut matched_patterns = Vec::new();
            for pattern in patterns {
                if lower_output.contains(pattern) {
                    matched_patterns.push(pattern.clone());
                }
            }

            if !matched_patterns.is_empty() {
                // Score based on number of matched patterns (more matches = better)
                let match_count = matched_patterns.len() as f32;

                // Base score: number of matches gives higher score
                // More matched patterns is a stronger signal
                let match_score = match_count * 0.15; // Each match adds 0.15

                // Specificity: average length of matched patterns (longer = more specific)
                let avg_specificity = matched_patterns.iter()
                    .map(|p| p.len() as f32)
                    .sum::<f32>() / match_count;

                // Normalize specificity to 0-1 range (20 chars = full score)
                let specificity_score = (avg_specificity / 20.0).min(1.0);

                // Final score: match count is primary, specificity is secondary
                let final_score = match_score.min(0.7) + specificity_score * 0.3;

                if final_score > best_score {
                    best_score = final_score;
                    best_type = *error_type;
                    best_evidence = matched_patterns;
                }
            }
        }

        // Clamp confidence to reasonable range
        let confidence = (best_score * 0.9).min(0.95).max(0.1);

        (best_type, confidence, best_evidence)
    }

    fn extract_location(&self, error_output: &str) -> Option<String> {
        // Look for common location patterns
        // Pattern: "at /path/to/file.nix:line:col"
        // Pattern: "in '/path/to/file.nix'"
        // Pattern: "error: ... in /path/to/file.nix, line 42"

        for line in error_output.lines() {
            // Check for "at /path:line:col" pattern
            if let Some(pos) = line.find("at /") {
                let rest = &line[pos + 3..];
                if let Some(end) = rest.find(|c: char| c.is_whitespace() || c == ',') {
                    return Some(rest[..end].to_string());
                }
                return Some(rest.trim().to_string());
            }

            // Check for "in '/path'" pattern
            if let Some(pos) = line.find("in '/") {
                let rest = &line[pos + 4..];
                if let Some(end) = rest.find('\'') {
                    return Some(rest[..end].to_string());
                }
            }

            // Check for ".nix:" pattern
            if line.contains(".nix:") {
                if let Some(pos) = line.find(".nix:") {
                    // Find the start of the path
                    let before = &line[..pos + 4];
                    if let Some(start) = before.rfind(|c: char| c.is_whitespace() || c == '\'' || c == '"') {
                        let path = &line[start + 1..];
                        if let Some(end) = path.find(|c: char| c.is_whitespace() && c != ':') {
                            return Some(path[..end].trim().to_string());
                        }
                        return Some(path.trim().to_string());
                    }
                }
            }
        }

        None
    }

    fn extract_failing_expression(&self, error_output: &str) -> Option<String> {
        // Look for the actual expression that failed
        // Often appears after "error:" on subsequent lines

        let mut capture_next = false;
        let mut expression_lines = Vec::new();

        for line in error_output.lines() {
            if capture_next {
                if line.trim().is_empty() || line.starts_with("error:") {
                    break;
                }
                expression_lines.push(line.trim());
                if expression_lines.len() > 3 {
                    break;
                }
            }

            if line.contains("while evaluating") || line.contains("while calling") {
                capture_next = true;
            }
        }

        if expression_lines.is_empty() {
            None
        } else {
            Some(expression_lines.join("\n"))
        }
    }

    fn generate_causes(&self, error_type: NixErrorType) -> Vec<String> {
        match error_type {
            NixErrorType::InfiniteRecursion => vec![
                "Self-referential attribute (e.g., x = x;)".to_string(),
                "Circular import between modules".to_string(),
                "Option depending on itself through config".to_string(),
                "Overlay referencing final instead of prev".to_string(),
            ],
            NixErrorType::MissingAttribute => vec![
                "Typo in attribute name".to_string(),
                "Package removed or renamed in nixpkgs".to_string(),
                "Accessing attribute before it's defined".to_string(),
                "Wrong nixpkgs channel/version".to_string(),
            ],
            NixErrorType::TypeMismatch => vec![
                "Passing wrong type to function (e.g., string instead of list)".to_string(),
                "Incorrect use of mkIf or mkMerge".to_string(),
                "Option type doesn't match value".to_string(),
                "Missing type coercion".to_string(),
            ],
            NixErrorType::OptionConflict => vec![
                "Same option set in multiple modules with different values".to_string(),
                "Missing mkForce or mkDefault to resolve priority".to_string(),
                "Conflicting enable/disable in imported modules".to_string(),
            ],
            NixErrorType::ModuleMergeConflict => vec![
                "Two modules defining same option at same priority".to_string(),
                "Incompatible merge strategies".to_string(),
                "Non-mergeable option type (e.g., types.enum)".to_string(),
            ],
            NixErrorType::HashMismatch => vec![
                "Upstream source changed".to_string(),
                "Network issues during download".to_string(),
                "SRI hash format mismatch (sha256 vs sha256-)".to_string(),
                "Mirror serving different content".to_string(),
            ],
            NixErrorType::FetchFailure => vec![
                "Network connectivity issue".to_string(),
                "Upstream server down".to_string(),
                "Rate limiting by server".to_string(),
                "DNS resolution failure".to_string(),
                "Firewall blocking connection".to_string(),
            ],
            NixErrorType::PackageNotFound => vec![
                "Package name typo".to_string(),
                "Package in different attribute set (e.g., python3Packages.X)".to_string(),
                "Package not in current nixpkgs version".to_string(),
                "Package marked as broken or unfree".to_string(),
            ],
            NixErrorType::LockfileOutdated => vec![
                "flake.nix changed but flake.lock not updated".to_string(),
                "New input added without running 'nix flake update'".to_string(),
                "Lock file version mismatch".to_string(),
            ],
            NixErrorType::DirtyWorkingTree => vec![
                "Uncommitted changes in flake directory".to_string(),
                "Using local path that's not in git".to_string(),
                "Need --impure flag for impure operations".to_string(),
            ],
            NixErrorType::StorePermissionDenied => vec![
                "Nix daemon not running".to_string(),
                "User not in trusted-users or allowed-users".to_string(),
                "Store path ownership issue".to_string(),
            ],
            NixErrorType::DaemonConnectionFailed => vec![
                "nix-daemon service not running".to_string(),
                "Socket file missing or corrupted".to_string(),
                "Permission issue with socket".to_string(),
            ],
            _ => vec![
                format!("{} error occurred", error_type.name()),
            ],
        }
    }

    fn generate_fixes(&self, error_type: NixErrorType, _error_output: &str) -> Vec<SuggestedFix> {
        match error_type {
            NixErrorType::InfiniteRecursion => vec![
                SuggestedFix {
                    description: "Check for self-referential attributes and break the cycle".to_string(),
                    commands: vec![
                        "# Review the indicated file and look for circular references".to_string(),
                        "# If using overlay: change 'final.X' to 'prev.X' where appropriate".to_string(),
                    ],
                    risk: FixRiskLevel::Low,
                    verify_command: Some("nix-instantiate --eval -E 'import ./configuration.nix {}'".to_string()),
                    primary: true,
                },
            ],
            NixErrorType::MissingAttribute => vec![
                SuggestedFix {
                    description: "Search for the correct attribute name".to_string(),
                    commands: vec![
                        "nix search nixpkgs <package-name>".to_string(),
                    ],
                    risk: FixRiskLevel::Safe,
                    verify_command: None,
                    primary: true,
                },
                SuggestedFix {
                    description: "Update nixpkgs to get latest packages".to_string(),
                    commands: vec![
                        "nix flake update".to_string(),
                    ],
                    risk: FixRiskLevel::Medium,
                    verify_command: Some("nix flake show".to_string()),
                    primary: false,
                },
            ],
            NixErrorType::OptionConflict => vec![
                SuggestedFix {
                    description: "Use mkForce to override conflicting value".to_string(),
                    commands: vec![
                        "# In your configuration.nix:".to_string(),
                        "# option = lib.mkForce <your-value>;".to_string(),
                    ],
                    risk: FixRiskLevel::Low,
                    verify_command: Some("nixos-rebuild dry-build".to_string()),
                    primary: true,
                },
                SuggestedFix {
                    description: "Use mkDefault for lower-priority default".to_string(),
                    commands: vec![
                        "# In module with the default:".to_string(),
                        "# option = lib.mkDefault <default-value>;".to_string(),
                    ],
                    risk: FixRiskLevel::Low,
                    verify_command: Some("nixos-rebuild dry-build".to_string()),
                    primary: false,
                },
            ],
            NixErrorType::HashMismatch => vec![
                SuggestedFix {
                    description: "Update the hash in the derivation".to_string(),
                    commands: vec![
                        "# Replace the hash with lib.fakeHash:".to_string(),
                        "# hash = lib.fakeHash;".to_string(),
                        "# Then rebuild to get the correct hash from error message".to_string(),
                    ],
                    risk: FixRiskLevel::Low,
                    verify_command: Some("nix build".to_string()),
                    primary: true,
                },
            ],
            NixErrorType::LockfileOutdated => vec![
                SuggestedFix {
                    description: "Update the flake lock file".to_string(),
                    commands: vec![
                        "nix flake update".to_string(),
                    ],
                    risk: FixRiskLevel::Medium,
                    verify_command: Some("nix flake check".to_string()),
                    primary: true,
                },
            ],
            NixErrorType::DirtyWorkingTree => vec![
                SuggestedFix {
                    description: "Commit your changes or use --impure".to_string(),
                    commands: vec![
                        "git add -A && git commit -m 'WIP'".to_string(),
                    ],
                    risk: FixRiskLevel::Safe,
                    verify_command: Some("git status".to_string()),
                    primary: true,
                },
                SuggestedFix {
                    description: "Use --impure flag (not recommended for reproducibility)".to_string(),
                    commands: vec![
                        "nixos-rebuild switch --impure".to_string(),
                    ],
                    risk: FixRiskLevel::Medium,
                    verify_command: None,
                    primary: false,
                },
            ],
            NixErrorType::DaemonConnectionFailed => vec![
                SuggestedFix {
                    description: "Start the nix-daemon service".to_string(),
                    commands: vec![
                        "sudo systemctl start nix-daemon".to_string(),
                        "sudo systemctl enable nix-daemon".to_string(),
                    ],
                    risk: FixRiskLevel::Low,
                    verify_command: Some("systemctl status nix-daemon".to_string()),
                    primary: true,
                },
            ],
            NixErrorType::PackageNotFound => vec![
                SuggestedFix {
                    description: "Search for the package with correct name".to_string(),
                    commands: vec![
                        "nix search nixpkgs <partial-name>".to_string(),
                    ],
                    risk: FixRiskLevel::Safe,
                    verify_command: None,
                    primary: true,
                },
            ],
            NixErrorType::FetchFailure => vec![
                SuggestedFix {
                    description: "Check network connectivity and retry".to_string(),
                    commands: vec![
                        "# Check if the URL is accessible:".to_string(),
                        "curl -I <url-from-error>".to_string(),
                        "# If accessible, retry the build:".to_string(),
                        "nix build --rebuild".to_string(),
                    ],
                    risk: FixRiskLevel::Safe,
                    verify_command: None,
                    primary: true,
                },
            ],
            _ => vec![
                SuggestedFix {
                    description: format!("Review the error and consult NixOS documentation for: {}", error_type.name()),
                    commands: vec![
                        "# Check the NixOS manual for this error type".to_string(),
                    ],
                    risk: FixRiskLevel::Safe,
                    verify_command: None,
                    primary: true,
                },
            ],
        }
    }

    fn extract_affected_configs(&self, error_output: &str) -> Vec<String> {
        let mut configs = Vec::new();

        // Look for NixOS option paths
        for line in error_output.lines() {
            // Pattern: services.X.Y, programs.X, etc.
            let words: Vec<&str> = line.split_whitespace().collect();
            for word in words {
                // First strip quotes/backticks from both ends
                let clean = word.trim_matches(|c| c == '\'' || c == '"' || c == '`' || c == ',' || c == ';');

                // Now check if it's a NixOS option path
                if (clean.starts_with("services.") ||
                    clean.starts_with("programs.") ||
                    clean.starts_with("networking.") ||
                    clean.starts_with("security.") ||
                    clean.starts_with("boot.") ||
                    clean.starts_with("hardware.") ||
                    clean.starts_with("systemd.") ||
                    clean.starts_with("users.") ||
                    clean.starts_with("environment.")) &&
                    !clean.contains("(") {
                    if !configs.contains(&clean.to_string()) {
                        configs.push(clean.to_string());
                    }
                }
            }
        }

        configs
    }

    fn extract_source_module(&self, error_output: &str) -> Option<String> {
        // Look for module import paths
        for line in error_output.lines() {
            if line.contains("while evaluating the module") ||
               line.contains("while importing") ||
               line.contains("imported from") {
                // Extract path from quotes
                if let Some(start) = line.find('\'') {
                    let rest = &line[start + 1..];
                    if let Some(end) = rest.find('\'') {
                        return Some(rest[..end].to_string());
                    }
                }
            }
        }
        None
    }

    fn generate_explanation(&self, error_type: NixErrorType, evidence: &[String], location: &Option<String>) -> String {
        let mut explanation = format!(
            "This appears to be a {} error ({}).",
            error_type.category().name().to_lowercase(),
            error_type.name()
        );

        if let Some(loc) = location {
            explanation.push_str(&format!(" The error occurred at: {}", loc));
        }

        if !evidence.is_empty() {
            explanation.push_str(&format!(
                " Key indicators: {}.",
                evidence.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
            ));
        }

        explanation
    }

    fn summarize_symptom(&self, error_output: &str) -> String {
        // Extract the first meaningful error line
        for line in error_output.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("error:") {
                return trimmed.to_string();
            }
        }

        // Fallback to first non-empty line
        error_output.lines()
            .find(|l| !l.trim().is_empty())
            .map(|l| l.trim().to_string())
            .unwrap_or_else(|| "Unknown error".to_string())
    }
}

impl Default for NixErrorDiagnoser {
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
    fn test_error_category_all() {
        let categories = NixErrorCategory::all();
        assert!(categories.len() >= 6);
    }

    #[test]
    fn test_error_type_category_mapping() {
        assert_eq!(NixErrorType::InfiniteRecursion.category(), NixErrorCategory::Evaluation);
        assert_eq!(NixErrorType::HashMismatch.category(), NixErrorCategory::Build);
        assert_eq!(NixErrorType::OptionConflict.category(), NixErrorCategory::Conflict);
        assert_eq!(NixErrorType::PackageNotFound.category(), NixErrorCategory::MissingResource);
        assert_eq!(NixErrorType::StorePermissionDenied.category(), NixErrorCategory::Permission);
        assert_eq!(NixErrorType::LockfileOutdated.category(), NixErrorCategory::Flake);
    }

    #[test]
    fn test_diagnoser_creation() {
        let diagnoser = NixErrorDiagnoser::new();
        assert!(!diagnoser.patterns.is_empty());
    }

    #[test]
    fn test_diagnose_infinite_recursion() {
        let diagnoser = NixErrorDiagnoser::new();
        let error = "error: infinite recursion encountered at /etc/nixos/configuration.nix:42";

        let diagnosis = diagnoser.diagnose(error);

        assert_eq!(diagnosis.error_type, NixErrorType::InfiniteRecursion);
        assert_eq!(diagnosis.category, NixErrorCategory::Evaluation);
        assert!(diagnosis.confidence > 0.3);
        assert!(!diagnosis.likely_causes.is_empty());
        assert!(!diagnosis.fixes.is_empty());
    }

    #[test]
    fn test_diagnose_missing_attribute() {
        let diagnoser = NixErrorDiagnoser::new();
        let error = "error: attribute 'firefoxBrowser' missing, at /etc/nixos/configuration.nix:15:5";

        let diagnosis = diagnoser.diagnose(error);

        assert_eq!(diagnosis.error_type, NixErrorType::MissingAttribute);
        assert!(diagnosis.location.is_some());
    }

    #[test]
    fn test_diagnose_hash_mismatch() {
        let diagnoser = NixErrorDiagnoser::new();
        let error = r#"error: hash mismatch in fixed-output derivation '/nix/store/xxx-source':
  specified: sha256-AAAA...
  got:       sha256-BBBB..."#;

        let diagnosis = diagnoser.diagnose(error);

        assert_eq!(diagnosis.error_type, NixErrorType::HashMismatch);
        assert_eq!(diagnosis.category, NixErrorCategory::Build);
    }

    #[test]
    fn test_diagnose_option_conflict() {
        let diagnoser = NixErrorDiagnoser::new();
        let error = "error: The option `services.nginx.enable` has conflicting definition values";

        let diagnosis = diagnoser.diagnose(error);

        assert_eq!(diagnosis.error_type, NixErrorType::OptionConflict);
        assert!(!diagnosis.affected_configs.is_empty());
    }

    #[test]
    fn test_diagnose_flake_lockfile() {
        let diagnoser = NixErrorDiagnoser::new();
        let error = "error: lock file 'flake.lock' is out of date and needs to be updated";

        let diagnosis = diagnoser.diagnose(error);

        assert_eq!(diagnosis.error_type, NixErrorType::LockfileOutdated);
        assert_eq!(diagnosis.category, NixErrorCategory::Flake);
    }

    #[test]
    fn test_diagnose_dirty_tree() {
        let diagnoser = NixErrorDiagnoser::new();
        let error = "error: path '/home/user/nixos' is dirty, run with --impure to allow";

        let diagnosis = diagnoser.diagnose(error);

        assert_eq!(diagnosis.error_type, NixErrorType::DirtyWorkingTree);
    }

    #[test]
    fn test_diagnose_daemon_connection() {
        let diagnoser = NixErrorDiagnoser::new();
        let error = "error: cannot connect to daemon at '/nix/var/nix/daemon-socket/socket': Connection refused";

        let diagnosis = diagnoser.diagnose(error);

        assert_eq!(diagnosis.error_type, NixErrorType::DaemonConnectionFailed);
        assert_eq!(diagnosis.category, NixErrorCategory::Permission);
    }

    #[test]
    fn test_diagnose_unknown_error() {
        let diagnoser = NixErrorDiagnoser::new();
        let error = "something completely unrelated happened";

        let diagnosis = diagnoser.diagnose(error);

        assert_eq!(diagnosis.error_type, NixErrorType::UnknownError);
        assert!(diagnosis.confidence < 0.5);
    }

    #[test]
    fn test_primary_fix_selection() {
        let diagnoser = NixErrorDiagnoser::new();
        let error = "error: attribute 'nonexistent' missing";

        let diagnosis = diagnoser.diagnose(error);
        let primary = diagnosis.primary_fix();

        assert!(primary.is_some());
        assert!(primary.unwrap().primary);
    }

    #[test]
    fn test_extract_location() {
        let diagnoser = NixErrorDiagnoser::new();
        let error = "error: something wrong at /etc/nixos/hardware-configuration.nix:25:3";

        let diagnosis = diagnoser.diagnose(error);

        assert!(diagnosis.location.is_some());
        assert!(diagnosis.location.unwrap().contains("hardware-configuration.nix"));
    }

    #[test]
    fn test_extract_affected_configs() {
        let diagnoser = NixErrorDiagnoser::new();
        let error = "error: The option 'services.nginx.enable' conflicts with 'services.httpd.enable'";

        let diagnosis = diagnoser.diagnose(error);

        assert!(diagnosis.affected_configs.iter().any(|c| c.contains("services.nginx")));
    }

    #[test]
    fn test_fix_risk_levels() {
        assert!(FixRiskLevel::Safe < FixRiskLevel::Low);
        assert!(FixRiskLevel::Low < FixRiskLevel::Medium);
        assert!(FixRiskLevel::Medium < FixRiskLevel::High);
        assert!(FixRiskLevel::High < FixRiskLevel::Critical);
    }
}
