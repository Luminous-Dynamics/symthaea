// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # World-Grounded Prediction System (MAGI Loop Step 1 + 1.5)
//!
//! This module implements the first two steps of the Minimum AGI Loop (MAGI Loop):
//! - **Step 1**: World-grounded predictions about external reality
//! - **Step 1.5**: Resolution contracts that prevent self-grading
//!
//! ## The Core Insight
//!
//! Previous systems predicted only internal states (Phi, latency). This module
//! extends prediction to the *external world*:
//! - Will this code compile?
//! - Will this command succeed?
//! - Will the user approve this action?
//!
//! ## Why This Matters
//!
//! Without world-grounded prediction:
//! - The system never risks being wrong about external reality
//! - The system never earns calibrated confidence
//! - The system cannot ground its intelligence in truth
//!
//! With world-grounded prediction:
//! - Every action carries a falsifiable prediction
//! - Every outcome updates calibration
//! - Intelligence becomes grounded in reality
//!
//! ## Resolution Authority
//!
//! Critical innovation: outcomes are resolved by *external authorities*, not self-grading:
//! - TestSuite: Unit test pass/fail
//! - ExitCode: Process exit codes
//! - ExternalAPI: External service responses
//! - HumanConfirmation: Explicit human feedback
//!
//! This makes the system ungameable - it cannot lie to itself about outcomes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::consciousness::recursive_improvement::types::instant_now;

// ═══════════════════════════════════════════════════════════════════════════════
// PREDICTION DOMAIN
// ═══════════════════════════════════════════════════════════════════════════════

/// Domain of world prediction - what kind of external reality is being predicted
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PredictionDomain {
    /// Code execution outcomes (compile, test pass/fail)
    CodeExecution,
    /// Tool/command usage outcomes (exit codes, output)
    ToolUse,
    /// User behavior predictions (approval, response)
    UserBehavior,
    /// System/resource state (file exists, process running)
    SystemState,
    /// External factual claims (API responses, data validity)
    Factual,
}

impl PredictionDomain {
    /// Get all prediction domains
    pub fn all() -> Vec<Self> {
        vec![
            Self::CodeExecution,
            Self::ToolUse,
            Self::UserBehavior,
            Self::SystemState,
            Self::Factual,
        ]
    }

    /// Get the typical resolution timeout for this domain
    pub fn default_timeout(&self) -> Duration {
        match self {
            Self::CodeExecution => Duration::from_secs(60), // Tests might take time
            Self::ToolUse => Duration::from_secs(30),       // Commands usually faster
            Self::UserBehavior => Duration::from_secs(3600), // Users might be slow
            Self::SystemState => Duration::from_secs(5),    // State checks are fast
            Self::Factual => Duration::from_secs(30),       // API calls
        }
    }

    /// Infer prediction domain from action type string
    ///
    /// This allows automatic domain classification for EFE integration.
    pub fn from_action_type(action_type: &str) -> Self {
        let action_lower = action_type.to_lowercase();

        if action_lower.contains("test")
            || action_lower.contains("compile")
            || action_lower.contains("build")
            || action_lower.contains("run")
        {
            Self::CodeExecution
        } else if action_lower.contains("command")
            || action_lower.contains("exec")
            || action_lower.contains("shell")
            || action_lower.contains("tool")
        {
            Self::ToolUse
        } else if action_lower.contains("user")
            || action_lower.contains("human")
            || action_lower.contains("approval")
            || action_lower.contains("confirm")
        {
            Self::UserBehavior
        } else if action_lower.contains("file")
            || action_lower.contains("state")
            || action_lower.contains("resource")
            || action_lower.contains("check")
        {
            Self::SystemState
        } else {
            // Default to Factual for unknown action types
            Self::Factual
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OUTCOME CATEGORY
// ═══════════════════════════════════════════════════════════════════════════════

/// Category of action outcome - NOT just binary success/fail
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutcomeCategory {
    /// Full success - achieved intended effect
    Success,
    /// Partial success - achieved some effect but not all
    Partial,
    /// No effect - action had no impact (not necessarily bad)
    NoEffect,
    /// Safe failure - action failed but no damage
    SafeFailure,
    /// Unsafe failure - action failed with negative consequences
    UnsafeFailure,
}

impl OutcomeCategory {
    /// Numeric score for calibration (1.0 = best, 0.0 = worst)
    pub fn score(&self) -> f64 {
        match self {
            Self::Success => 1.0,
            Self::Partial => 0.7,
            Self::NoEffect => 0.4,
            Self::SafeFailure => 0.2,
            Self::UnsafeFailure => 0.0,
        }
    }

    /// Is this a "good" outcome (better than expected baseline)?
    pub fn is_positive(&self) -> bool {
        matches!(self, Self::Success | Self::Partial)
    }

    /// Is this outcome safe (no negative consequences)?
    pub fn is_safe(&self) -> bool {
        !matches!(self, Self::UnsafeFailure)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESOLUTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Resolution status of a prediction
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum Resolution {
    /// Not yet resolved
    #[default]
    Pending,
    /// Resolved as true (prediction was correct)
    True {
        /// Actual outcome observed
        observed_outcome: OutcomeCategory,
        /// Confidence the resolution is accurate
        resolution_confidence: f64,
        /// When resolution occurred
        #[serde(skip, default = "instant_now")]
        resolved_at: Instant,
    },
    /// Resolved as false (prediction was wrong)
    False {
        /// What actually happened
        observed_outcome: OutcomeCategory,
        /// What was predicted
        predicted_outcome: OutcomeCategory,
        /// Confidence the resolution is accurate
        resolution_confidence: f64,
        /// When resolution occurred
        #[serde(skip, default = "instant_now")]
        resolved_at: Instant,
    },
    /// Unclear - couldn't determine outcome
    Unclear {
        /// Why resolution failed
        reason: String,
        /// When we gave up
        #[serde(skip, default = "instant_now")]
        abandoned_at: Instant,
    },
    /// Timed out before resolution
    TimedOut {
        /// When timeout occurred
        #[serde(skip, default = "instant_now")]
        timed_out_at: Instant,
    },
}

impl Resolution {
    /// Check if prediction is still pending
    pub fn is_pending(&self) -> bool {
        matches!(self, Self::Pending)
    }

    /// Check if prediction was correct
    pub fn is_correct(&self) -> bool {
        matches!(self, Self::True { .. })
    }

    /// Get Brier score component (0.0 = perfect, 1.0 = maximally wrong)
    /// For use in calibration tracking
    pub fn brier_component(&self, predicted_confidence: f64) -> Option<f64> {
        match self {
            Self::Pending | Self::Unclear { .. } | Self::TimedOut { .. } => None,
            Self::True { .. } => {
                // (confidence - 1.0)^2
                Some((predicted_confidence - 1.0).powi(2))
            }
            Self::False { .. } => {
                // (confidence - 0.0)^2 = confidence^2
                Some(predicted_confidence.powi(2))
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RISK TIER
// ═══════════════════════════════════════════════════════════════════════════════

/// Risk tier for actions - determines execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RiskTier {
    /// No risk - read-only, no side effects
    Observation,
    /// Low risk - reversible side effects
    Reversible,
    /// Medium risk - state-modifying but recoverable
    StateModifying,
    /// High risk - potentially destructive
    Destructive,
    /// Critical risk - irreversible consequences
    Critical,
}

impl RiskTier {
    /// Numeric risk level (0.0 = safe, 1.0 = critical)
    pub fn level(&self) -> f64 {
        match self {
            Self::Observation => 0.0,
            Self::Reversible => 0.25,
            Self::StateModifying => 0.5,
            Self::Destructive => 0.75,
            Self::Critical => 1.0,
        }
    }

    /// From a numeric risk level
    pub fn from_level(level: f64) -> Self {
        if level <= 0.1 {
            Self::Observation
        } else if level <= 0.3 {
            Self::Reversible
        } else if level <= 0.6 {
            Self::StateModifying
        } else if level <= 0.85 {
            Self::Destructive
        } else {
            Self::Critical
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WORLD ACTION CONTEXT
// ═══════════════════════════════════════════════════════════════════════════════

/// Extended action context for world predictions
/// This extends the base ActionContext with world-grounding fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldActionContext {
    /// Type of action being taken (e.g., "git_commit", "file_write", "api_call")
    pub action_type: String,

    /// Description of the intended effect
    pub intended_effect: String,

    /// Current goal or objective
    pub goal: Option<String>,

    /// Risk tier of this action
    pub risk_tier: RiskTier,

    /// Constraints on the action
    pub constraints: Vec<String>,

    /// Pre-conditions that should hold
    pub preconditions: Vec<String>,

    /// Expected post-conditions after action
    pub postconditions: Vec<String>,

    /// Urgency level (0.0 = relaxed, 1.0 = urgent)
    pub urgency: f32,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl WorldActionContext {
    /// Create a new action context
    pub fn new(action_type: impl Into<String>, intended_effect: impl Into<String>) -> Self {
        Self {
            action_type: action_type.into(),
            intended_effect: intended_effect.into(),
            goal: None,
            risk_tier: RiskTier::Reversible,
            constraints: Vec::new(),
            preconditions: Vec::new(),
            postconditions: Vec::new(),
            urgency: 0.5,
            metadata: HashMap::new(),
        }
    }

    /// Builder pattern for goal
    pub fn with_goal(mut self, goal: impl Into<String>) -> Self {
        self.goal = Some(goal.into());
        self
    }

    /// Builder pattern for risk tier
    pub fn with_risk_tier(mut self, risk_tier: RiskTier) -> Self {
        self.risk_tier = risk_tier;
        self
    }

    /// Builder pattern for adding precondition
    pub fn with_precondition(mut self, precondition: impl Into<String>) -> Self {
        self.preconditions.push(precondition.into());
        self
    }

    /// Builder pattern for adding postcondition
    pub fn with_postcondition(mut self, postcondition: impl Into<String>) -> Self {
        self.postconditions.push(postcondition.into());
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESOLUTION AUTHORITY (UPGRADE A)
// ═══════════════════════════════════════════════════════════════════════════════

/// Who/what resolves the prediction outcome - external authority, NOT self-grading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionAuthority {
    /// Unit test pass/fail determines outcome
    TestSuite {
        /// Pattern to match test files/names
        test_pattern: String,
        /// Threshold for "pass" (e.g., 0.95 = 95% tests must pass)
        pass_threshold: f64,
    },

    /// Process exit code determines outcome
    ExitCode {
        /// Which exit codes count as success
        success_codes: Vec<i32>,
    },

    /// File diff verification
    DiffVerifier {
        /// Path to expected output
        expected_path: PathBuf,
        /// How strict the comparison is
        tolerance: DiffTolerance,
    },

    /// External API check
    ExternalAPI {
        /// Endpoint to query
        endpoint: String,
        /// Expected HTTP status code
        expected_status: u16,
        /// Optional: expected body content substring
        expected_body_contains: Option<String>,
    },

    /// Human must confirm outcome
    HumanConfirmation {
        /// Prompt to show human
        prompt: String,
        /// How long to wait for confirmation
        timeout: Duration,
    },

    /// Resource/file state check
    ResourceState {
        /// Path to resource
        path: PathBuf,
        /// Expected state of the resource
        expected_state: ResourceExpectation,
    },

    /// Composite: multiple authorities must agree
    Composite {
        /// Authorities that must all agree
        authorities: Vec<Box<ResolutionAuthority>>,
        /// Quorum required (how many must agree)
        quorum: usize,
    },
}

impl ResolutionAuthority {
    /// Create a test suite authority
    pub fn test_suite(pattern: impl Into<String>) -> Self {
        Self::TestSuite {
            test_pattern: pattern.into(),
            pass_threshold: 1.0, // All tests must pass by default
        }
    }

    /// Create an exit code authority
    pub fn exit_code(success_codes: Vec<i32>) -> Self {
        Self::ExitCode { success_codes }
    }

    /// Create a simple success-only exit code authority
    pub fn exit_success() -> Self {
        Self::ExitCode {
            success_codes: vec![0],
        }
    }

    /// Create a human confirmation authority
    pub fn human(prompt: impl Into<String>) -> Self {
        Self::HumanConfirmation {
            prompt: prompt.into(),
            timeout: Duration::from_secs(3600), // 1 hour default
        }
    }

    /// Create a resource state authority
    pub fn resource_exists(path: impl Into<PathBuf>) -> Self {
        Self::ResourceState {
            path: path.into(),
            expected_state: ResourceExpectation::Exists,
        }
    }
}

/// How strict diff verification should be
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DiffTolerance {
    /// Exact match required
    Exact,
    /// Whitespace differences ignored
    IgnoreWhitespace,
    /// Order can differ (for lists)
    UnorderedLines,
    /// Semantic similarity (embedding-based)
    Semantic { threshold: f64 },
}

/// Expected state of a resource
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ResourceExpectation {
    /// Resource should exist
    #[default]
    Exists,
    /// Resource should NOT exist
    NotExists,
    /// Resource should contain content
    Contains(String),
    /// Resource should match pattern
    MatchesPattern(String),
    /// Resource should have specific size
    HasSize { min: Option<u64>, max: Option<u64> },
    /// Resource was modified after a time
    ModifiedAfter(#[serde(skip)] Option<Instant>),
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESOLUTION CONTRACT (UPGRADE A)
// ═══════════════════════════════════════════════════════════════════════════════

/// Contract that defines how an action's outcome will be resolved
/// This is CRITICAL for making the system ungameable - outcomes are not self-graded
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionContract {
    /// Action type this contract applies to
    pub action_type: String,

    /// What counts as Success
    pub success_criteria: String,

    /// What counts as Partial success
    pub partial_criteria: String,

    /// What counts as NoEffect
    pub no_effect_criteria: String,

    /// What counts as SafeFailure
    pub safe_failure_criteria: String,

    /// What counts as UnsafeFailure
    pub unsafe_failure_criteria: String,

    /// External authority that resolves the outcome
    pub resolver: ResolutionAuthority,

    /// How long before declaring NoEffect
    pub timeout: Duration,
}

impl ResolutionContract {
    /// Create a new resolution contract
    pub fn new(action_type: impl Into<String>, resolver: ResolutionAuthority) -> Self {
        Self {
            action_type: action_type.into(),
            success_criteria: "Action achieved intended effect".to_string(),
            partial_criteria: "Action achieved some but not all intended effect".to_string(),
            no_effect_criteria: "Action had no observable effect".to_string(),
            safe_failure_criteria: "Action failed but caused no harm".to_string(),
            unsafe_failure_criteria: "Action failed and caused negative consequences".to_string(),
            resolver,
            timeout: Duration::from_secs(60),
        }
    }

    /// Builder pattern for success criteria
    pub fn with_success_criteria(mut self, criteria: impl Into<String>) -> Self {
        self.success_criteria = criteria.into();
        self
    }

    /// Builder pattern for partial criteria
    pub fn with_partial_criteria(mut self, criteria: impl Into<String>) -> Self {
        self.partial_criteria = criteria.into();
        self
    }

    /// Builder pattern for timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Standard contract for code execution
    pub fn code_execution(test_pattern: impl Into<String>) -> Self {
        Self::new(
            "code_execution",
            ResolutionAuthority::test_suite(test_pattern),
        )
        .with_success_criteria("All tests pass")
        .with_partial_criteria("Some tests pass")
        .with_timeout(Duration::from_secs(120))
    }

    /// Standard contract for file operations
    pub fn file_operation(path: impl Into<PathBuf>) -> Self {
        Self::new("file_operation", ResolutionAuthority::resource_exists(path))
            .with_success_criteria("File created/modified as expected")
            .with_timeout(Duration::from_secs(10))
    }

    /// Standard contract for shell commands
    pub fn shell_command() -> Self {
        Self::new("shell_command", ResolutionAuthority::exit_success())
            .with_success_criteria("Command exits with code 0")
            .with_partial_criteria("Command exits with warning (code 1)")
            .with_timeout(Duration::from_secs(30))
    }

    /// Standard contract for user interaction
    pub fn user_interaction(prompt: impl Into<String>) -> Self {
        Self::new("user_interaction", ResolutionAuthority::human(prompt))
            .with_success_criteria("User confirms success")
            .with_timeout(Duration::from_secs(3600))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WORLD PREDICTION
// ═══════════════════════════════════════════════════════════════════════════════

/// A prediction about the external world
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldPrediction {
    /// Unique identifier for this prediction
    pub id: String,

    /// The claim being made (e.g., "This code will compile")
    pub claim: String,

    /// Predicted outcome category
    pub predicted_outcome: OutcomeCategory,

    /// Confidence in the prediction (0.0 - 1.0)
    pub confidence: f64,

    /// Domain of the prediction
    pub domain: PredictionDomain,

    /// Action context that led to this prediction
    pub action_context: WorldActionContext,

    /// Contract for how this will be resolved
    pub resolution_contract: ResolutionContract,

    /// Current resolution status
    pub resolution: Resolution,

    /// When this prediction was made
    #[serde(skip, default = "instant_now")]
    pub created_at: Instant,

    /// Deadline for resolution
    pub deadline: Duration,

    /// Tags for grouping/analysis
    pub tags: Vec<String>,
}

impl WorldPrediction {
    /// Create a new world prediction
    pub fn new(
        claim: impl Into<String>,
        predicted_outcome: OutcomeCategory,
        confidence: f64,
        action_context: WorldActionContext,
        resolution_contract: ResolutionContract,
    ) -> Self {
        let deadline = resolution_contract.timeout;
        let domain = Self::infer_domain(&action_context.action_type);

        Self {
            id: Self::generate_id(),
            claim: claim.into(),
            predicted_outcome,
            confidence: confidence.clamp(0.0, 1.0),
            domain,
            action_context,
            resolution_contract,
            resolution: Resolution::Pending,
            created_at: Instant::now(),
            deadline,
            tags: Vec::new(),
        }
    }

    /// Generate a unique ID for this prediction
    fn generate_id() -> String {
        use std::time::SystemTime;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("wp_{:x}", timestamp)
    }

    /// Infer prediction domain from action type
    fn infer_domain(action_type: &str) -> PredictionDomain {
        let action_lower = action_type.to_lowercase();
        if action_lower.contains("test")
            || action_lower.contains("compile")
            || action_lower.contains("build")
        {
            PredictionDomain::CodeExecution
        } else if action_lower.contains("command")
            || action_lower.contains("shell")
            || action_lower.contains("exec")
        {
            PredictionDomain::ToolUse
        } else if action_lower.contains("user")
            || action_lower.contains("confirm")
            || action_lower.contains("approve")
        {
            PredictionDomain::UserBehavior
        } else if action_lower.contains("file")
            || action_lower.contains("state")
            || action_lower.contains("resource")
        {
            PredictionDomain::SystemState
        } else {
            PredictionDomain::Factual
        }
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Check if prediction has expired
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.deadline
    }

    /// Check if prediction is still pending
    pub fn is_pending(&self) -> bool {
        self.resolution.is_pending()
    }

    /// Resolve the prediction as correct
    pub fn resolve_true(&mut self, observed_outcome: OutcomeCategory, resolution_confidence: f64) {
        self.resolution = Resolution::True {
            observed_outcome,
            resolution_confidence,
            resolved_at: Instant::now(),
        };
    }

    /// Resolve the prediction as incorrect
    pub fn resolve_false(&mut self, observed_outcome: OutcomeCategory, resolution_confidence: f64) {
        self.resolution = Resolution::False {
            observed_outcome,
            predicted_outcome: self.predicted_outcome,
            resolution_confidence,
            resolved_at: Instant::now(),
        };
    }

    /// Mark as unclear
    pub fn resolve_unclear(&mut self, reason: impl Into<String>) {
        self.resolution = Resolution::Unclear {
            reason: reason.into(),
            abandoned_at: Instant::now(),
        };
    }

    /// Mark as timed out
    pub fn resolve_timeout(&mut self) {
        self.resolution = Resolution::TimedOut {
            timed_out_at: Instant::now(),
        };
    }

    /// Get Brier score contribution (for calibration)
    pub fn brier_score(&self) -> Option<f64> {
        self.resolution.brier_component(self.confidence)
    }

    /// Get prediction error (difference between predicted and actual)
    pub fn prediction_error(&self) -> Option<f64> {
        match &self.resolution {
            Resolution::True {
                observed_outcome, ..
            } => Some((self.predicted_outcome.score() - observed_outcome.score()).abs()),
            Resolution::False {
                observed_outcome, ..
            } => Some((self.predicted_outcome.score() - observed_outcome.score()).abs()),
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTRACT REGISTRY
// ═══════════════════════════════════════════════════════════════════════════════

/// Registry of resolution contracts by action type
#[derive(Debug, Clone, Default)]
pub struct ContractRegistry {
    contracts: HashMap<String, ResolutionContract>,
}

impl ContractRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with standard contracts
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();

        // Register standard contracts
        registry.register(ResolutionContract::code_execution("tests/**/*.rs"));
        registry.register(ResolutionContract::shell_command());
        registry.register(ResolutionContract::user_interaction(
            "Was the action successful?",
        ));

        registry
    }

    /// Register a contract
    pub fn register(&mut self, contract: ResolutionContract) {
        self.contracts
            .insert(contract.action_type.clone(), contract);
    }

    /// Get contract for an action type
    pub fn get(&self, action_type: &str) -> Option<&ResolutionContract> {
        self.contracts.get(action_type)
    }

    /// Get or create a default contract for an action type
    pub fn get_or_default(&self, action_type: &str) -> ResolutionContract {
        self.contracts.get(action_type).cloned().unwrap_or_else(|| {
            // Default to human confirmation for unknown action types
            ResolutionContract::user_interaction(format!("Was '{}' successful?", action_type))
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outcome_category_scores() {
        assert_eq!(OutcomeCategory::Success.score(), 1.0);
        assert_eq!(OutcomeCategory::UnsafeFailure.score(), 0.0);
        assert!(OutcomeCategory::Partial.score() > OutcomeCategory::SafeFailure.score());
    }

    #[test]
    fn test_risk_tier_ordering() {
        assert!(RiskTier::Observation < RiskTier::Critical);
        assert!(RiskTier::Reversible < RiskTier::Destructive);
    }

    #[test]
    fn test_world_prediction_creation() {
        let action = WorldActionContext::new("test_execution", "Run unit tests")
            .with_risk_tier(RiskTier::Observation);
        let contract = ResolutionContract::code_execution("tests/**");

        let prediction = WorldPrediction::new(
            "All tests will pass",
            OutcomeCategory::Success,
            0.85,
            action,
            contract,
        );

        assert_eq!(prediction.confidence, 0.85);
        assert_eq!(prediction.domain, PredictionDomain::CodeExecution);
        assert!(prediction.is_pending());
    }

    #[test]
    fn test_prediction_resolution() {
        let action = WorldActionContext::new("compile", "Compile code");
        let contract = ResolutionContract::shell_command();

        let mut prediction = WorldPrediction::new(
            "Code will compile",
            OutcomeCategory::Success,
            0.90,
            action,
            contract,
        );

        // Resolve as correct
        prediction.resolve_true(OutcomeCategory::Success, 1.0);
        assert!(prediction.resolution.is_correct());

        // Brier score should be small (high confidence, correct prediction)
        let brier = prediction.brier_score().unwrap();
        assert!(brier < 0.1);
    }

    #[test]
    fn test_brier_score_calculation() {
        // High confidence, correct -> low Brier
        let brier_correct = Resolution::True {
            observed_outcome: OutcomeCategory::Success,
            resolution_confidence: 1.0,
            resolved_at: Instant::now(),
        }
        .brier_component(0.9);
        assert!(brier_correct.unwrap() < 0.02);

        // High confidence, wrong -> high Brier
        let brier_wrong = Resolution::False {
            observed_outcome: OutcomeCategory::SafeFailure,
            predicted_outcome: OutcomeCategory::Success,
            resolution_confidence: 1.0,
            resolved_at: Instant::now(),
        }
        .brier_component(0.9);
        assert!(brier_wrong.unwrap() > 0.8);
    }

    #[test]
    fn test_contract_registry() {
        let registry = ContractRegistry::with_defaults();

        let shell_contract = registry.get("shell_command");
        assert!(shell_contract.is_some());

        let unknown = registry.get_or_default("unknown_action");
        // Should return a human confirmation contract
        assert!(matches!(
            unknown.resolver,
            ResolutionAuthority::HumanConfirmation { .. }
        ));
    }

    #[test]
    fn test_resolution_authority_builders() {
        let test_auth = ResolutionAuthority::test_suite("tests/**/*.rs");
        assert!(matches!(test_auth, ResolutionAuthority::TestSuite { .. }));

        let exit_auth = ResolutionAuthority::exit_success();
        assert!(
            matches!(exit_auth, ResolutionAuthority::ExitCode { success_codes } if success_codes == vec![0])
        );

        let human_auth = ResolutionAuthority::human("Did it work?");
        assert!(matches!(
            human_auth,
            ResolutionAuthority::HumanConfirmation { .. }
        ));
    }
}
