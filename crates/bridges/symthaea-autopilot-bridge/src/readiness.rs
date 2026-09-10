// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-backed PX4 operational-readiness profiles and assessments.
//!
//! Readiness here means only that the propositions required by one explicit
//! profile are currently evidenced as satisfied. It is not operator authority,
//! arming permission, safety approval, command acceptance, or physical execution.

use std::collections::HashSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_evidence::{EvidenceValidationError, TimestampV1};
use thiserror::Error;

/// Schema version for PX4 readiness profiles and assessments.
pub const PX4_READINESS_SCHEMA_V1: u16 = 1;
const PROFILE_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.readiness-profile.v1\0";
const ASSESSMENT_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.readiness-assessment.v1\0";

/// Stable prerequisite propositions that a PX4 operation profile may require.
///
/// This enum is intentionally a vocabulary, not a universal mandatory checklist.
/// The selected [`Px4ReadinessProfileV1`] determines which facts are required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Px4ReadinessRequirementV1 {
    /// Provider heartbeat / bound-instance liveness is fresh under its profile.
    HeartbeatFresh,
    /// Time synchronization quality satisfies the selected transport/profile policy.
    TimesyncQualified,
    /// PX4 angular-velocity estimate is valid for the selected operation.
    AngularVelocityValid,
    /// PX4 attitude estimate is valid for the selected operation.
    AttitudeValid,
    /// PX4 local-altitude estimate is valid for the selected operation.
    LocalAltitudeValid,
    /// PX4 local-position estimate is valid for the selected operation.
    LocalPositionValid,
    /// PX4 local-velocity estimate is valid for the selected operation.
    LocalVelocityValid,
    /// PX4 global-position estimate is valid for the selected operation.
    GlobalPositionValid,
    /// Required Offboard proof-of-life/control signal is currently present.
    OffboardSignalPresent,
    /// Current PX4 mode reports that it accepts offboard setpoints.
    AcceptsOffboardSetpoints,
    /// PX4 is not presently reporting a failsafe state under the selected profile.
    FailsafeInactive,
    /// PX4 reports the vehicle ready to arm; this does not authorize arming.
    ReadyToArm,
    /// Actuator lockdown is not active when the selected operation requires it absent.
    LockdownInactive,
    /// Manual kill state is not active when the selected operation requires it absent.
    KillInactive,
    /// Actuator termination is not active when the selected operation requires it absent.
    TerminationInactive,
    /// Selected transport authentication/signing requirement is currently satisfied.
    TransportAuthenticationSatisfied,
    /// Bound provider supports the selected semantic control profile.
    ControlProfileSupported,
}

impl Px4ReadinessRequirementV1 {
    /// Stable token used for canonical ordering and content commitments.
    pub const fn wire_token(self) -> &'static str {
        match self {
            Self::HeartbeatFresh => "heartbeat_fresh",
            Self::TimesyncQualified => "timesync_qualified",
            Self::AngularVelocityValid => "angular_velocity_valid",
            Self::AttitudeValid => "attitude_valid",
            Self::LocalAltitudeValid => "local_altitude_valid",
            Self::LocalPositionValid => "local_position_valid",
            Self::LocalVelocityValid => "local_velocity_valid",
            Self::GlobalPositionValid => "global_position_valid",
            Self::OffboardSignalPresent => "offboard_signal_present",
            Self::AcceptsOffboardSetpoints => "accepts_offboard_setpoints",
            Self::FailsafeInactive => "failsafe_inactive",
            Self::ReadyToArm => "ready_to_arm",
            Self::LockdownInactive => "lockdown_inactive",
            Self::KillInactive => "kill_inactive",
            Self::TerminationInactive => "termination_inactive",
            Self::TransportAuthenticationSatisfied => "transport_authentication_satisfied",
            Self::ControlProfileSupported => "control_profile_supported",
        }
    }
}

/// Observation state of one required PX4 readiness proposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Px4ReadinessFactStateV1 {
    /// Evidence establishes the required proposition as currently satisfied.
    Satisfied,
    /// Evidence establishes the required proposition as currently unsatisfied.
    Unsatisfied,
    /// The proposition has not been established either way.
    Unobserved,
}

impl Px4ReadinessFactStateV1 {
    fn wire_token(self) -> &'static str {
        match self {
            Self::Satisfied => "satisfied",
            Self::Unsatisfied => "unsatisfied",
            Self::Unobserved => "unobserved",
        }
    }
}

/// One readiness proposition and its evidence state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ReadinessFactV1 {
    /// Proposition being assessed.
    pub requirement: Px4ReadinessRequirementV1,
    /// Current evidence state of the proposition.
    pub state: Px4ReadinessFactStateV1,
    /// Evidence reference for observed positive/negative claims.
    ///
    /// Must be present for `Satisfied`/`Unsatisfied` and absent for `Unobserved`.
    pub evidence_id: Option<String>,
}

impl Px4ReadinessFactV1 {
    /// Construct an evidenced satisfied proposition.
    pub fn satisfied(
        requirement: Px4ReadinessRequirementV1,
        evidence_id: impl Into<String>,
    ) -> Result<Self, Px4ReadinessValidationError> {
        Self::observed(requirement, Px4ReadinessFactStateV1::Satisfied, evidence_id)
    }

    /// Construct an evidenced unsatisfied proposition.
    pub fn unsatisfied(
        requirement: Px4ReadinessRequirementV1,
        evidence_id: impl Into<String>,
    ) -> Result<Self, Px4ReadinessValidationError> {
        Self::observed(requirement, Px4ReadinessFactStateV1::Unsatisfied, evidence_id)
    }

    /// Construct an explicitly unobserved proposition without fabricated evidence.
    pub fn unobserved(requirement: Px4ReadinessRequirementV1) -> Self {
        Self {
            requirement,
            state: Px4ReadinessFactStateV1::Unobserved,
            evidence_id: None,
        }
    }

    /// Validate evidence-presence semantics for this fact.
    pub fn validate(&self) -> Result<(), Px4ReadinessValidationError> {
        match self.state {
            Px4ReadinessFactStateV1::Satisfied | Px4ReadinessFactStateV1::Unsatisfied => {
                let evidence_id = self
                    .evidence_id
                    .as_deref()
                    .ok_or(Px4ReadinessValidationError::ObservedFactMissingEvidence(
                        self.requirement,
                    ))?;
                validate_identifier(evidence_id, "evidence_id")?;
            }
            Px4ReadinessFactStateV1::Unobserved => {
                if self.evidence_id.is_some() {
                    return Err(Px4ReadinessValidationError::UnobservedFactHasEvidence(
                        self.requirement,
                    ));
                }
            }
        }
        Ok(())
    }

    fn observed(
        requirement: Px4ReadinessRequirementV1,
        state: Px4ReadinessFactStateV1,
        evidence_id: impl Into<String>,
    ) -> Result<Self, Px4ReadinessValidationError> {
        let value = Self {
            requirement,
            state,
            evidence_id: Some(evidence_id.into()),
        };
        value.validate()?;
        Ok(value)
    }
}

/// Canonical, content-addressed set of readiness propositions required by one
/// PX4 operation/control profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ReadinessProfileV1 {
    /// Schema version. Must equal [`PX4_READINESS_SCHEMA_V1`].
    pub schema_version: u16,
    /// Stable identity of the selected readiness/control policy profile.
    pub profile_id: String,
    /// Canonically ordered set of required readiness propositions.
    pub required_facts: Vec<Px4ReadinessRequirementV1>,
    /// Domain-separated content commitment over profile identity and requirements.
    pub profile_digest_hex: String,
}

impl Px4ReadinessProfileV1 {
    /// Construct, canonicalize, content-bind, and validate one readiness profile.
    pub fn new(
        profile_id: impl Into<String>,
        mut required_facts: Vec<Px4ReadinessRequirementV1>,
    ) -> Result<Self, Px4ReadinessValidationError> {
        canonicalize_requirements(&mut required_facts)?;
        let mut value = Self {
            schema_version: PX4_READINESS_SCHEMA_V1,
            profile_id: profile_id.into(),
            required_facts,
            profile_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.profile_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate schema, identity, canonical requirement set, and commitment.
    pub fn validate(&self) -> Result<(), Px4ReadinessValidationError> {
        self.validate_without_digest()?;
        if self.profile_digest_hex != self.compute_digest_hex() {
            return Err(Px4ReadinessValidationError::DigestMismatch("readiness_profile"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact readiness profile.
    pub fn profile_identity(&self) -> Result<String, Px4ReadinessValidationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.readiness-profile.v1:{}",
            self.profile_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), Px4ReadinessValidationError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.profile_id, "profile_id")?;
        if self.required_facts.is_empty() {
            return Err(Px4ReadinessValidationError::EmptyRequirements);
        }
        let mut seen = HashSet::with_capacity(self.required_facts.len());
        let mut previous = None;
        for requirement in &self.required_facts {
            if !seen.insert(*requirement) {
                return Err(Px4ReadinessValidationError::DuplicateRequirement(*requirement));
            }
            let token = requirement.wire_token();
            if previous.is_some_and(|prev: &str| prev >= token) {
                return Err(Px4ReadinessValidationError::NonCanonicalRequirementOrder);
            }
            previous = Some(token);
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROFILE_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.profile_id);
        hasher.update(&(self.required_facts.len() as u64).to_le_bytes());
        for requirement in &self.required_facts {
            feed_str(&mut hasher, requirement.wire_token());
        }
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Derived operational-readiness outcome under one exact profile.
///
/// This is a runtime view, not separately serialized evidence. `Qualified` does
/// not confer authority to arm, enter Offboard, publish a setpoint, or actuate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Px4ReadinessOutcomeV1 {
    /// Every required proposition is evidenced as satisfied.
    Qualified,
    /// One or more required propositions are evidenced as unsatisfied.
    Blocked {
        /// Canonically ordered blocking propositions.
        facts: Vec<Px4ReadinessRequirementV1>,
    },
    /// No required proposition is unsatisfied, but one or more remain unobserved.
    Incomplete {
        /// Canonically ordered propositions that still lack evidence.
        facts: Vec<Px4ReadinessRequirementV1>,
    },
}

/// Timestamped, content-addressed assessment of one exact PX4 readiness profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ReadinessAssessmentV1 {
    /// Schema version. Must equal [`PX4_READINESS_SCHEMA_V1`].
    pub schema_version: u16,
    /// Adapter/producer profile that assembled this assessment.
    pub producer_profile_id: String,
    /// Exact readiness profile being assessed.
    pub profile: Px4ReadinessProfileV1,
    /// Time at which the assessment proposition was assembled/observed.
    pub assessed_at: TimestampV1,
    /// Exactly one canonically ordered fact for every profile requirement.
    pub facts: Vec<Px4ReadinessFactV1>,
    /// Domain-separated content commitment over the complete assessment.
    pub assessment_digest_hex: String,
}

impl Px4ReadinessAssessmentV1 {
    /// Construct, canonicalize, content-bind, and validate one readiness assessment.
    pub fn new(
        producer_profile_id: impl Into<String>,
        profile: Px4ReadinessProfileV1,
        assessed_at: TimestampV1,
        mut facts: Vec<Px4ReadinessFactV1>,
    ) -> Result<Self, Px4ReadinessValidationError> {
        facts.sort_by_key(|fact| fact.requirement.wire_token());
        let mut value = Self {
            schema_version: PX4_READINESS_SCHEMA_V1,
            producer_profile_id: producer_profile_id.into(),
            profile,
            assessed_at,
            facts,
            assessment_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.assessment_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate profile, timestamp, complete fact set, and content commitment.
    pub fn validate(&self) -> Result<(), Px4ReadinessValidationError> {
        self.validate_without_digest()?;
        if self.assessment_digest_hex != self.compute_digest_hex() {
            return Err(Px4ReadinessValidationError::DigestMismatch(
                "readiness_assessment",
            ));
        }
        Ok(())
    }

    /// Content-addressed identity for this exact assessment.
    pub fn assessment_id(&self) -> Result<String, Px4ReadinessValidationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.readiness-assessment.v1:{}",
            self.assessment_digest_hex
        ))
    }

    /// Derive readiness under the exact content-bound profile.
    ///
    /// Negative evidence takes precedence over missing evidence: if any required
    /// fact is unsatisfied the result is `Blocked`; otherwise missing facts yield
    /// `Incomplete`; only a fully satisfied set yields `Qualified`.
    pub fn outcome(&self) -> Result<Px4ReadinessOutcomeV1, Px4ReadinessValidationError> {
        self.validate()?;
        let blocked: Vec<_> = self
            .facts
            .iter()
            .filter(|fact| fact.state == Px4ReadinessFactStateV1::Unsatisfied)
            .map(|fact| fact.requirement)
            .collect();
        if !blocked.is_empty() {
            return Ok(Px4ReadinessOutcomeV1::Blocked { facts: blocked });
        }
        let incomplete: Vec<_> = self
            .facts
            .iter()
            .filter(|fact| fact.state == Px4ReadinessFactStateV1::Unobserved)
            .map(|fact| fact.requirement)
            .collect();
        if !incomplete.is_empty() {
            return Ok(Px4ReadinessOutcomeV1::Incomplete { facts: incomplete });
        }
        Ok(Px4ReadinessOutcomeV1::Qualified)
    }

    fn validate_without_digest(&self) -> Result<(), Px4ReadinessValidationError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        self.profile.validate()?;
        self.assessed_at
            .validate()
            .map_err(Px4ReadinessValidationError::Timestamp)?;

        if self.facts.len() != self.profile.required_facts.len() {
            return Err(Px4ReadinessValidationError::FactSetMismatch);
        }

        let mut seen = HashSet::with_capacity(self.facts.len());
        for (expected, fact) in self.profile.required_facts.iter().zip(&self.facts) {
            fact.validate()?;
            if !seen.insert(fact.requirement) {
                return Err(Px4ReadinessValidationError::DuplicateFact(fact.requirement));
            }
            if expected != &fact.requirement {
                return Err(Px4ReadinessValidationError::FactSetMismatch);
            }
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(ASSESSMENT_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.producer_profile_id);
        feed_str(&mut hasher, &self.profile.profile_digest_hex);
        feed_str(&mut hasher, self.assessed_at.clock_domain.as_str());
        hasher.update(&self.assessed_at.nanoseconds.to_le_bytes());
        hasher.update(&(self.facts.len() as u64).to_le_bytes());
        for fact in &self.facts {
            feed_str(&mut hasher, fact.requirement.wire_token());
            feed_str(&mut hasher, fact.state.wire_token());
            match &fact.evidence_id {
                Some(evidence_id) => {
                    hasher.update(&[1]);
                    feed_str(&mut hasher, evidence_id);
                }
                None => hasher.update(&[0]),
            }
        }
        digest_hex(hasher.finalize().as_bytes())
    }
}

fn canonicalize_requirements(
    requirements: &mut Vec<Px4ReadinessRequirementV1>,
) -> Result<(), Px4ReadinessValidationError> {
    if requirements.is_empty() {
        return Err(Px4ReadinessValidationError::EmptyRequirements);
    }
    let mut seen = HashSet::with_capacity(requirements.len());
    for requirement in requirements.iter().copied() {
        if !seen.insert(requirement) {
            return Err(Px4ReadinessValidationError::DuplicateRequirement(requirement));
        }
    }
    requirements.sort_by_key(|requirement| requirement.wire_token());
    Ok(())
}

fn validate_schema(schema_version: u16) -> Result<(), Px4ReadinessValidationError> {
    if schema_version != PX4_READINESS_SCHEMA_V1 {
        return Err(Px4ReadinessValidationError::UnsupportedSchemaVersion {
            found: schema_version,
        });
    }
    Ok(())
}

fn validate_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), Px4ReadinessValidationError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(Px4ReadinessValidationError::InvalidIdentifier(field));
    }
    Ok(())
}

fn feed_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

/// Validation failure for PX4 readiness profiles, facts, and assessments.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Px4ReadinessValidationError {
    /// Record uses an unsupported schema version.
    #[error("unsupported PX4 readiness schema version {found}")]
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// Identifier was empty, padded, too long, or contained control characters.
    #[error("invalid PX4 readiness identifier field {0}")]
    InvalidIdentifier(&'static str),
    /// A profile contained no required readiness propositions.
    #[error("PX4 readiness profile must contain at least one requirement")]
    EmptyRequirements,
    /// A profile repeated one prerequisite proposition.
    #[error("duplicate PX4 readiness requirement: {0:?}")]
    DuplicateRequirement(Px4ReadinessRequirementV1),
    /// A deserialized profile requirement vector is not in canonical token order.
    #[error("PX4 readiness requirements are not in canonical order")]
    NonCanonicalRequirementOrder,
    /// An observed satisfied/unsatisfied fact omitted its evidence reference.
    #[error("observed PX4 readiness fact {0:?} is missing evidence")]
    ObservedFactMissingEvidence(Px4ReadinessRequirementV1),
    /// An unobserved fact carried a fabricated evidence reference.
    #[error("unobserved PX4 readiness fact {0:?} must not carry evidence")]
    UnobservedFactHasEvidence(Px4ReadinessRequirementV1),
    /// One readiness proposition appeared more than once in an assessment.
    #[error("duplicate PX4 readiness fact: {0:?}")]
    DuplicateFact(Px4ReadinessRequirementV1),
    /// Assessment facts do not exactly match the selected profile requirement set.
    #[error("PX4 readiness assessment fact set does not match its profile")]
    FactSetMismatch,
    /// Assessment timestamp/clock identity is invalid.
    #[error("invalid PX4 readiness assessment timestamp: {0}")]
    Timestamp(EvidenceValidationError),
    /// Stored profile or assessment commitment no longer matches its semantic fields.
    #[error("PX4 {0} content commitment mismatch")]
    DigestMismatch(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::embodiment_evidence::ClockDomainId;

    fn ts(nanoseconds: u64) -> TimestampV1 {
        TimestampV1::new(ClockDomainId::new("px4.hrt").unwrap(), nanoseconds)
    }

    fn profile() -> Px4ReadinessProfileV1 {
        Px4ReadinessProfileV1::new(
            "px4.multicopter.offboard.fixture.v1",
            vec![
                Px4ReadinessRequirementV1::OffboardSignalPresent,
                Px4ReadinessRequirementV1::AngularVelocityValid,
                Px4ReadinessRequirementV1::AttitudeValid,
            ],
        )
        .unwrap()
    }

    fn satisfied(requirement: Px4ReadinessRequirementV1) -> Px4ReadinessFactV1 {
        Px4ReadinessFactV1::satisfied(
            requirement,
            format!("evidence:{}", requirement.wire_token()),
        )
        .unwrap()
    }

    #[test]
    fn all_required_facts_satisfied_is_qualified() {
        let profile = profile();
        let facts = profile
            .required_facts
            .iter()
            .copied()
            .map(satisfied)
            .collect();
        let assessment = Px4ReadinessAssessmentV1::new(
            "px4.readiness.fixture.v1",
            profile,
            ts(1_000),
            facts,
        )
        .unwrap();

        assert_eq!(assessment.outcome().unwrap(), Px4ReadinessOutcomeV1::Qualified);
    }

    #[test]
    fn negative_fact_blocks_even_when_another_fact_is_unobserved() {
        let profile = profile();
        let mut facts: Vec<_> = profile
            .required_facts
            .iter()
            .copied()
            .map(satisfied)
            .collect();
        let block = Px4ReadinessRequirementV1::AttitudeValid;
        let missing = Px4ReadinessRequirementV1::OffboardSignalPresent;
        for fact in &mut facts {
            if fact.requirement == block {
                *fact = Px4ReadinessFactV1::unsatisfied(block, "evidence:attitude-invalid").unwrap();
            } else if fact.requirement == missing {
                *fact = Px4ReadinessFactV1::unobserved(missing);
            }
        }
        let assessment = Px4ReadinessAssessmentV1::new(
            "px4.readiness.fixture.v1",
            profile,
            ts(1_000),
            facts,
        )
        .unwrap();

        assert_eq!(
            assessment.outcome().unwrap(),
            Px4ReadinessOutcomeV1::Blocked { facts: vec![block] }
        );
    }

    #[test]
    fn unobserved_required_fact_yields_incomplete() {
        let profile = profile();
        let missing = Px4ReadinessRequirementV1::OffboardSignalPresent;
        let facts = profile
            .required_facts
            .iter()
            .copied()
            .map(|requirement| {
                if requirement == missing {
                    Px4ReadinessFactV1::unobserved(requirement)
                } else {
                    satisfied(requirement)
                }
            })
            .collect();
        let assessment = Px4ReadinessAssessmentV1::new(
            "px4.readiness.fixture.v1",
            profile,
            ts(1_000),
            facts,
        )
        .unwrap();

        assert_eq!(
            assessment.outcome().unwrap(),
            Px4ReadinessOutcomeV1::Incomplete { facts: vec![missing] }
        );
    }

    #[test]
    fn profile_order_is_canonicalized_for_stable_identity() {
        let first = Px4ReadinessProfileV1::new(
            "profile:v1",
            vec![
                Px4ReadinessRequirementV1::FailsafeInactive,
                Px4ReadinessRequirementV1::HeartbeatFresh,
            ],
        )
        .unwrap();
        let second = Px4ReadinessProfileV1::new(
            "profile:v1",
            vec![
                Px4ReadinessRequirementV1::HeartbeatFresh,
                Px4ReadinessRequirementV1::FailsafeInactive,
            ],
        )
        .unwrap();

        assert_eq!(first, second);
        assert_eq!(first.profile_identity().unwrap(), second.profile_identity().unwrap());
    }

    #[test]
    fn duplicate_or_missing_required_fact_is_rejected() {
        assert!(matches!(
            Px4ReadinessProfileV1::new(
                "profile:v1",
                vec![
                    Px4ReadinessRequirementV1::HeartbeatFresh,
                    Px4ReadinessRequirementV1::HeartbeatFresh,
                ],
            ),
            Err(Px4ReadinessValidationError::DuplicateRequirement(
                Px4ReadinessRequirementV1::HeartbeatFresh
            ))
        ));

        let profile = profile();
        let facts = vec![satisfied(profile.required_facts[0])];
        assert!(matches!(
            Px4ReadinessAssessmentV1::new(
                "px4.readiness.fixture.v1",
                profile,
                ts(1_000),
                facts,
            ),
            Err(Px4ReadinessValidationError::FactSetMismatch)
        ));
    }

    #[test]
    fn observed_fact_requires_evidence_and_unobserved_forbids_it() {
        let observed = Px4ReadinessFactV1 {
            requirement: Px4ReadinessRequirementV1::HeartbeatFresh,
            state: Px4ReadinessFactStateV1::Satisfied,
            evidence_id: None,
        };
        assert!(matches!(
            observed.validate(),
            Err(Px4ReadinessValidationError::ObservedFactMissingEvidence(
                Px4ReadinessRequirementV1::HeartbeatFresh
            ))
        ));

        let unobserved = Px4ReadinessFactV1 {
            requirement: Px4ReadinessRequirementV1::HeartbeatFresh,
            state: Px4ReadinessFactStateV1::Unobserved,
            evidence_id: Some("fabricated:evidence".into()),
        };
        assert!(matches!(
            unobserved.validate(),
            Err(Px4ReadinessValidationError::UnobservedFactHasEvidence(
                Px4ReadinessRequirementV1::HeartbeatFresh
            ))
        ));
    }

    #[test]
    fn profile_difference_changes_assessment_identity() {
        let first_profile = Px4ReadinessProfileV1::new(
            "profile:v1",
            vec![Px4ReadinessRequirementV1::HeartbeatFresh],
        )
        .unwrap();
        let second_profile = Px4ReadinessProfileV1::new(
            "profile:v1",
            vec![
                Px4ReadinessRequirementV1::HeartbeatFresh,
                Px4ReadinessRequirementV1::FailsafeInactive,
            ],
        )
        .unwrap();
        let first = Px4ReadinessAssessmentV1::new(
            "producer:v1",
            first_profile,
            ts(1_000),
            vec![satisfied(Px4ReadinessRequirementV1::HeartbeatFresh)],
        )
        .unwrap();
        let second = Px4ReadinessAssessmentV1::new(
            "producer:v1",
            second_profile,
            ts(1_000),
            vec![
                satisfied(Px4ReadinessRequirementV1::HeartbeatFresh),
                satisfied(Px4ReadinessRequirementV1::FailsafeInactive),
            ],
        )
        .unwrap();

        assert_ne!(first.assessment_id().unwrap(), second.assessment_id().unwrap());
    }

    #[test]
    fn timestamp_and_mutation_participate_in_assessment_identity() {
        let profile = Px4ReadinessProfileV1::new(
            "profile:v1",
            vec![Px4ReadinessRequirementV1::HeartbeatFresh],
        )
        .unwrap();
        let facts = vec![satisfied(Px4ReadinessRequirementV1::HeartbeatFresh)];
        let first = Px4ReadinessAssessmentV1::new(
            "producer:v1",
            profile.clone(),
            ts(1_000),
            facts.clone(),
        )
        .unwrap();
        let second = Px4ReadinessAssessmentV1::new(
            "producer:v1",
            profile,
            ts(1_001),
            facts,
        )
        .unwrap();
        assert_ne!(first.assessment_id().unwrap(), second.assessment_id().unwrap());

        let mut tampered = first;
        tampered.facts[0] = Px4ReadinessFactV1::unsatisfied(
            Px4ReadinessRequirementV1::HeartbeatFresh,
            "evidence:heartbeat-stale",
        )
        .unwrap();
        assert_eq!(
            tampered.validate(),
            Err(Px4ReadinessValidationError::DigestMismatch(
                "readiness_assessment"
            ))
        );
    }

    #[test]
    fn serde_round_trip_requires_validation_but_preserves_identity() {
        let profile = Px4ReadinessProfileV1::new(
            "profile:v1",
            vec![Px4ReadinessRequirementV1::HeartbeatFresh],
        )
        .unwrap();
        let assessment = Px4ReadinessAssessmentV1::new(
            "producer:v1",
            profile,
            ts(1_000),
            vec![satisfied(Px4ReadinessRequirementV1::HeartbeatFresh)],
        )
        .unwrap();
        let bytes = serde_json::to_vec(&assessment).unwrap();
        let restored: Px4ReadinessAssessmentV1 = serde_json::from_slice(&bytes).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored.assessment_id().unwrap(), assessment.assessment_id().unwrap());
    }
}
