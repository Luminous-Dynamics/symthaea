// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit execution authority and cognitive telemetry.
//!
//! Nixward historically used a caller-supplied `phi: f32` as both a
//! consciousness-like signal and an execution gate. That conflates two
//! independent concerns:
//!
//! - **authority** answers whether a caller is permitted to perform an action;
//! - **cognition** describes how well-supported/integrated Nixward's current
//!   reasoning is.
//!
//! This module makes that separation explicit. Cognitive state can cause a
//! caller to deliberate more, gather evidence, or request stronger authority,
//! but it must never manufacture authority by itself.

use super::executor::SafetyLevel;
use serde::{Deserialize, Serialize};

/// Schema version for serialized execution contexts.
pub const EXECUTION_CONTEXT_SCHEMA_VERSION: u16 = 1;

/// Provenance of the authority presented for an execution request.
///
/// These values describe *where permission came from*. They are not confidence
/// scores and they make no claim about consciousness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthoritySource {
    /// No modifying authority is present. Read-only operations remain allowed.
    None,
    /// A human directly requested the operation, without a separate confirmation.
    DirectOperatorRequest,
    /// A human explicitly confirmed the exact operation after review.
    ExplicitOperatorConfirmation,
    /// A separate human-in-the-loop gate already approved the exact operation.
    UpstreamHumanGate,
    /// A deterministic policy engine authorized the exact operation.
    PolicyDecision,
}

/// Explicit permission envelope for a Nixward action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityContext {
    schema_version: u16,
    source: AuthoritySource,
    safety_ceiling: SafetyLevel,
    /// Optional digest binding this authority to an exact action/plan.
    ///
    /// The digest is advisory until a caller validates it against its own
    /// authorization protocol; this type deliberately does not pretend an
    /// unsigned string is cryptographic authority.
    action_digest: Option<String>,
}

impl AuthorityContext {
    /// Read-only observation context. This cannot authorize mutation.
    pub fn observe_only() -> Self {
        Self {
            schema_version: EXECUTION_CONTEXT_SCHEMA_VERSION,
            source: AuthoritySource::None,
            safety_ceiling: SafetyLevel::ReadOnly,
            action_digest: None,
        }
    }

    /// Direct human request without a separate confirmation step.
    ///
    /// This deliberately tops out at `UserModify`: system-wide or destructive
    /// changes require a stronger authority source.
    pub fn direct_operator_request() -> Self {
        Self {
            schema_version: EXECUTION_CONTEXT_SCHEMA_VERSION,
            source: AuthoritySource::DirectOperatorRequest,
            safety_ceiling: SafetyLevel::UserModify,
            action_digest: None,
        }
    }

    /// Explicit operator confirmation for actions up to `safety_ceiling`.
    pub fn explicit_operator_confirmation(safety_ceiling: SafetyLevel) -> Self {
        Self {
            schema_version: EXECUTION_CONTEXT_SCHEMA_VERSION,
            source: AuthoritySource::ExplicitOperatorConfirmation,
            safety_ceiling,
            action_digest: None,
        }
    }

    /// Human approval already completed in an upstream gate.
    pub fn upstream_human_gate(
        safety_ceiling: SafetyLevel,
        action_digest: Option<String>,
    ) -> Self {
        Self {
            schema_version: EXECUTION_CONTEXT_SCHEMA_VERSION,
            source: AuthoritySource::UpstreamHumanGate,
            safety_ceiling,
            action_digest,
        }
    }

    /// Deterministic policy authorization.
    ///
    /// A caller should normally bind this to an exact action digest. The model
    /// does not itself validate policy signatures or identities.
    pub fn policy_decision(safety_ceiling: SafetyLevel, action_digest: Option<String>) -> Self {
        Self {
            schema_version: EXECUTION_CONTEXT_SCHEMA_VERSION,
            source: AuthoritySource::PolicyDecision,
            safety_ceiling,
            action_digest,
        }
    }

    pub fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn source(&self) -> AuthoritySource {
        self.source
    }

    pub fn safety_ceiling(&self) -> SafetyLevel {
        self.safety_ceiling
    }

    pub fn action_digest(&self) -> Option<&str> {
        self.action_digest.as_deref()
    }

    /// Whether this authority envelope permits an action at `safety`.
    ///
    /// Read-only actions never require modifying authority. Any mutation needs
    /// a non-`None` authority source and must remain at or below the explicit
    /// safety ceiling.
    pub fn allows(&self, safety: SafetyLevel) -> bool {
        if safety == SafetyLevel::ReadOnly {
            return true;
        }
        if self.source == AuthoritySource::None {
            return false;
        }
        safety_rank(safety) <= safety_rank(self.safety_ceiling)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != EXECUTION_CONTEXT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported authority schema {}, expected {}",
                self.schema_version, EXECUTION_CONTEXT_SCHEMA_VERSION
            ));
        }
        if self.source == AuthoritySource::None && self.safety_ceiling != SafetyLevel::ReadOnly {
            return Err("authority source None cannot carry modifying authority".into());
        }
        if let Some(digest) = &self.action_digest
            && digest.trim().is_empty()
        {
            return Err("action digest must not be empty when present".into());
        }
        Ok(())
    }
}

/// A Φ observation with explicit provenance, or an honest absence of one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhiMeasurement {
    /// Nixward has no valid Φ measurement for this execution context.
    Unavailable,
    /// A real measurement or approximation supplied with provenance.
    Measured {
        value: f32,
        provenance: String,
    },
}

impl PhiMeasurement {
    /// Build a measured Φ value without silently accepting NaN/out-of-range data.
    pub fn measured(value: f32, provenance: impl Into<String>) -> Result<Self, String> {
        let provenance = provenance.into();
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(format!("phi must be finite and in [0,1], got {value}"));
        }
        if provenance.trim().is_empty() {
            return Err("phi provenance must not be empty".into());
        }
        Ok(Self::Measured { value, provenance })
    }

    pub fn value(&self) -> Option<f32> {
        match self {
            Self::Unavailable => None,
            Self::Measured { value, .. } => Some(*value),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Unavailable => Ok(()),
            Self::Measured { value, provenance } => {
                if !value.is_finite() || !(0.0..=1.0).contains(value) {
                    return Err(format!("phi must be finite and in [0,1], got {value}"));
                }
                if provenance.trim().is_empty() {
                    return Err("phi provenance must not be empty".into());
                }
                Ok(())
            }
        }
    }
}

/// Cognitive evidence associated with a proposed action.
///
/// Every field is optional because absence is preferable to inventing a score.
/// These values are telemetry/advisory inputs only; `AuthorityContext::allows`
/// never consults them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveContext {
    pub phi: PhiMeasurement,
    pub confidence: Option<f32>,
    pub free_energy: Option<f32>,
    pub prediction_error: Option<f32>,
    pub causal_support: Option<f32>,
}

impl Default for CognitiveContext {
    fn default() -> Self {
        Self {
            phi: PhiMeasurement::Unavailable,
            confidence: None,
            free_energy: None,
            prediction_error: None,
            causal_support: None,
        }
    }
}

impl CognitiveContext {
    pub fn validate(&self) -> Result<(), String> {
        self.phi.validate()?;
        validate_unit_interval("confidence", self.confidence)?;
        validate_non_negative("free_energy", self.free_energy)?;
        validate_non_negative("prediction_error", self.prediction_error)?;
        validate_unit_interval("causal_support", self.causal_support)?;
        Ok(())
    }
}

/// Complete context supplied to a future context-aware executor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionContext {
    pub schema_version: u16,
    pub authority: AuthorityContext,
    pub cognition: CognitiveContext,
}

impl ExecutionContext {
    pub fn new(authority: AuthorityContext, cognition: CognitiveContext) -> Self {
        Self {
            schema_version: EXECUTION_CONTEXT_SCHEMA_VERSION,
            authority,
            cognition,
        }
    }

    pub fn observe_only() -> Self {
        Self::new(
            AuthorityContext::observe_only(),
            CognitiveContext::default(),
        )
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != EXECUTION_CONTEXT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported execution context schema {}, expected {}",
                self.schema_version, EXECUTION_CONTEXT_SCHEMA_VERSION
            ));
        }
        self.authority.validate()?;
        self.cognition.validate()?;
        Ok(())
    }

    /// Authority decision for an action. Cognitive telemetry is intentionally
    /// absent from this decision path.
    pub fn allows(&self, safety: SafetyLevel) -> bool {
        self.authority.allows(safety)
    }
}

fn safety_rank(level: SafetyLevel) -> u8 {
    match level {
        SafetyLevel::ReadOnly => 0,
        SafetyLevel::UserModify => 1,
        SafetyLevel::SystemModify => 2,
        SafetyLevel::SystemCritical => 3,
        SafetyLevel::Destructive => 4,
    }
}

fn validate_unit_interval(name: &str, value: Option<f32>) -> Result<(), String> {
    if let Some(value) = value
        && (!value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(format!("{name} must be finite and in [0,1], got {value}"));
    }
    Ok(())
}

fn validate_non_negative(name: &str, value: Option<f32>) -> Result<(), String> {
    if let Some(value) = value
        && (!value.is_finite() || value < 0.0)
    {
        return Err(format!("{name} must be finite and non-negative, got {value}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_authority_still_allows_observation_but_never_mutation() {
        let ctx = ExecutionContext::observe_only();
        assert!(ctx.allows(SafetyLevel::ReadOnly));
        assert!(!ctx.allows(SafetyLevel::UserModify));
        assert!(!ctx.allows(SafetyLevel::SystemCritical));
        assert!(!ctx.allows(SafetyLevel::Destructive));
    }

    #[test]
    fn direct_operator_request_stops_before_system_wide_changes() {
        let ctx = ExecutionContext::new(
            AuthorityContext::direct_operator_request(),
            CognitiveContext::default(),
        );
        assert!(ctx.allows(SafetyLevel::ReadOnly));
        assert!(ctx.allows(SafetyLevel::UserModify));
        assert!(!ctx.allows(SafetyLevel::SystemModify));
        assert!(!ctx.allows(SafetyLevel::SystemCritical));
        assert!(!ctx.allows(SafetyLevel::Destructive));
    }

    #[test]
    fn explicit_confirmation_is_bounded_by_its_ceiling() {
        let ctx = ExecutionContext::new(
            AuthorityContext::explicit_operator_confirmation(SafetyLevel::SystemCritical),
            CognitiveContext::default(),
        );
        assert!(ctx.allows(SafetyLevel::SystemModify));
        assert!(ctx.allows(SafetyLevel::SystemCritical));
        assert!(!ctx.allows(SafetyLevel::Destructive));
    }

    #[test]
    fn destructive_authority_must_be_explicitly_present() {
        let critical = ExecutionContext::new(
            AuthorityContext::explicit_operator_confirmation(SafetyLevel::SystemCritical),
            CognitiveContext::default(),
        );
        let destructive = ExecutionContext::new(
            AuthorityContext::explicit_operator_confirmation(SafetyLevel::Destructive),
            CognitiveContext::default(),
        );
        assert!(!critical.allows(SafetyLevel::Destructive));
        assert!(destructive.allows(SafetyLevel::Destructive));
    }

    #[test]
    fn phi_never_expands_authority() {
        let authority = AuthorityContext::observe_only();
        let low = ExecutionContext::new(
            authority.clone(),
            CognitiveContext {
                phi: PhiMeasurement::measured(0.01, "test-low").unwrap(),
                confidence: Some(0.1),
                free_energy: Some(0.9),
                prediction_error: Some(0.8),
                causal_support: Some(0.1),
            },
        );
        let high = ExecutionContext::new(
            authority,
            CognitiveContext {
                phi: PhiMeasurement::measured(1.0, "test-high").unwrap(),
                confidence: Some(1.0),
                free_energy: Some(0.0),
                prediction_error: Some(0.0),
                causal_support: Some(1.0),
            },
        );

        for safety in [
            SafetyLevel::UserModify,
            SafetyLevel::SystemModify,
            SafetyLevel::SystemCritical,
            SafetyLevel::Destructive,
        ] {
            assert_eq!(low.allows(safety), high.allows(safety));
            assert!(!high.allows(safety));
        }
    }

    #[test]
    fn unavailable_phi_is_first_class_and_valid() {
        let ctx = CognitiveContext::default();
        assert_eq!(ctx.phi, PhiMeasurement::Unavailable);
        assert_eq!(ctx.phi.value(), None);
        ctx.validate().unwrap();
    }

    #[test]
    fn measured_phi_requires_range_and_provenance() {
        assert!(PhiMeasurement::measured(f32::NAN, "test").is_err());
        assert!(PhiMeasurement::measured(-0.1, "test").is_err());
        assert!(PhiMeasurement::measured(1.1, "test").is_err());
        assert!(PhiMeasurement::measured(0.5, "").is_err());
        assert!(PhiMeasurement::measured(0.5, "iit-small-network").is_ok());
    }

    #[test]
    fn malformed_serialized_authority_fails_validation() {
        let raw = r#"{
            "schema_version": 1,
            "source": "None",
            "safety_ceiling": "Destructive",
            "action_digest": null
        }"#;
        let authority: AuthorityContext = serde_json::from_str(raw).unwrap();
        assert!(authority.validate().is_err());
    }

    #[test]
    fn cognitive_metrics_are_validated_without_becoming_authority() {
        let invalid = CognitiveContext {
            phi: PhiMeasurement::Unavailable,
            confidence: Some(1.2),
            free_energy: Some(0.1),
            prediction_error: Some(0.2),
            causal_support: Some(0.8),
        };
        assert!(invalid.validate().is_err());

        let valid = CognitiveContext {
            phi: PhiMeasurement::Unavailable,
            confidence: Some(0.8),
            free_energy: Some(1.4),
            prediction_error: Some(0.2),
            causal_support: Some(0.8),
        };
        assert!(valid.validate().is_ok());
    }
}
