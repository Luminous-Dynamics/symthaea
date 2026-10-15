// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mission-specialized suit profile contracts (SX-028).
//!
//! Profiles define software-level capability expectations for distinct suit
//! roles. They are research hypotheses, not authoritative requirements,
//! hardware interfaces, certification, or evidence that a suit is safe.
//! Evidence labels are caller-declared and are never promoted automatically.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SuitMissionProfile {
    OrbitalEva,
    LunarSurfaceEva,
    MarsSurfaceEva,
    ConstructionAndHeavyTooling,
    DustIntensiveSurfaceWork,
    SuitportAndHabitatServicing,
    CrewRescueAndRecovery,
    LongDurationOrbitalMaintenance,
}

impl SuitMissionProfile {
    pub const ALL: [Self; 8] = [
        Self::OrbitalEva,
        Self::LunarSurfaceEva,
        Self::MarsSurfaceEva,
        Self::ConstructionAndHeavyTooling,
        Self::DustIntensiveSurfaceWork,
        Self::SuitportAndHabitatServicing,
        Self::CrewRescueAndRecovery,
        Self::LongDurationOrbitalMaintenance,
    ];

    pub fn required_capabilities(self) -> Vec<SuitCapability> {
        let mut capabilities = COMMON_SAFETY_FOUNDATION.to_vec();
        capabilities.extend_from_slice(self.mission_capabilities());
        capabilities.sort_unstable();
        capabilities.dedup();
        capabilities
    }

    fn mission_capabilities(self) -> &'static [SuitCapability] {
        match self {
            Self::OrbitalEva => ORBITAL_EVA_CAPABILITIES,
            Self::LunarSurfaceEva => LUNAR_SURFACE_CAPABILITIES,
            Self::MarsSurfaceEva => MARS_SURFACE_CAPABILITIES,
            Self::ConstructionAndHeavyTooling => CONSTRUCTION_CAPABILITIES,
            Self::DustIntensiveSurfaceWork => DUST_INTENSIVE_CAPABILITIES,
            Self::SuitportAndHabitatServicing => SUITPORT_SERVICE_CAPABILITIES,
            Self::CrewRescueAndRecovery => RESCUE_CAPABILITIES,
            Self::LongDurationOrbitalMaintenance => LONG_DURATION_CAPABILITIES,
        }
    }
}

/// Capability identifiers are interface/requirement categories, not a claim
/// that every capability belongs in every physical suit configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SuitCapability {
    PressureBoundaryIntegrity,
    PrimaryLifeSupportInterface,
    ThermalManagement,
    IndependentEmergencyReserve,
    ManualFallback,
    CommunicationsAndTelemetry,
    FaultContainmentAndSafeDegradation,
    MicrogravityTranslationAndRestraint,
    TetherAndAnchorInterfaces,
    OrbitalWorksiteInterfaces,
    PlanetaryLocomotion,
    DustMitigation,
    /// Explicit capability category for lunar charge/grounding risk analysis;
    /// not a claim that any discharge design is safe or qualified.
    ElectrostaticChargeManagement,
    AbrasionAndPunctureResistance,
    HighDexterityGloveInterfaces,
    SurfaceNavigationInterface,
    HeavyToolAndLoadBearingInterfaces,
    ContaminationContainment,
    SuitportMechanicalAndPressureCompatibility,
    ResourceServiceInterfaces,
    CrewRescueHandlingInterfaces,
    FieldRepairableSubsystems,
    ModularLifeSupportReplacementInterfaces,
    RadiationMonitoringInterface,
}

const COMMON_SAFETY_FOUNDATION: &[SuitCapability] = &[
    SuitCapability::PressureBoundaryIntegrity,
    SuitCapability::PrimaryLifeSupportInterface,
    SuitCapability::ThermalManagement,
    SuitCapability::IndependentEmergencyReserve,
    SuitCapability::ManualFallback,
    SuitCapability::CommunicationsAndTelemetry,
    SuitCapability::FaultContainmentAndSafeDegradation,
];

const ORBITAL_EVA_CAPABILITIES: &[SuitCapability] = &[
    SuitCapability::MicrogravityTranslationAndRestraint,
    SuitCapability::TetherAndAnchorInterfaces,
    SuitCapability::OrbitalWorksiteInterfaces,
    SuitCapability::HighDexterityGloveInterfaces,
];

const LUNAR_SURFACE_CAPABILITIES: &[SuitCapability] = &[
    SuitCapability::PlanetaryLocomotion,
    SuitCapability::DustMitigation,
    SuitCapability::ElectrostaticChargeManagement,
    SuitCapability::AbrasionAndPunctureResistance,
    SuitCapability::HighDexterityGloveInterfaces,
    SuitCapability::SurfaceNavigationInterface,
];

const MARS_SURFACE_CAPABILITIES: &[SuitCapability] = &[
    SuitCapability::PlanetaryLocomotion,
    SuitCapability::DustMitigation,
    SuitCapability::AbrasionAndPunctureResistance,
    SuitCapability::HighDexterityGloveInterfaces,
    SuitCapability::SurfaceNavigationInterface,
    SuitCapability::FieldRepairableSubsystems,
    SuitCapability::ModularLifeSupportReplacementInterfaces,
    SuitCapability::RadiationMonitoringInterface,
];

const CONSTRUCTION_CAPABILITIES: &[SuitCapability] = &[
    SuitCapability::MicrogravityTranslationAndRestraint,
    SuitCapability::TetherAndAnchorInterfaces,
    SuitCapability::HeavyToolAndLoadBearingInterfaces,
    SuitCapability::HighDexterityGloveInterfaces,
    SuitCapability::FieldRepairableSubsystems,
];

const DUST_INTENSIVE_CAPABILITIES: &[SuitCapability] = &[
    SuitCapability::PlanetaryLocomotion,
    SuitCapability::DustMitigation,
    SuitCapability::ElectrostaticChargeManagement,
    SuitCapability::AbrasionAndPunctureResistance,
    SuitCapability::ContaminationContainment,
    SuitCapability::FieldRepairableSubsystems,
];

const SUITPORT_SERVICE_CAPABILITIES: &[SuitCapability] = &[
    SuitCapability::SuitportMechanicalAndPressureCompatibility,
    SuitCapability::ContaminationContainment,
    SuitCapability::DustMitigation,
    SuitCapability::ResourceServiceInterfaces,
    SuitCapability::FieldRepairableSubsystems,
    SuitCapability::ModularLifeSupportReplacementInterfaces,
];

const RESCUE_CAPABILITIES: &[SuitCapability] = &[
    SuitCapability::CrewRescueHandlingInterfaces,
    SuitCapability::TetherAndAnchorInterfaces,
    SuitCapability::IndependentEmergencyReserve,
    SuitCapability::FieldRepairableSubsystems,
];

const LONG_DURATION_CAPABILITIES: &[SuitCapability] = &[
    SuitCapability::MicrogravityTranslationAndRestraint,
    SuitCapability::TetherAndAnchorInterfaces,
    SuitCapability::FieldRepairableSubsystems,
    SuitCapability::ModularLifeSupportReplacementInterfaces,
    SuitCapability::ResourceServiceInterfaces,
    SuitCapability::RadiationMonitoringInterface,
];

/// An evidence maturity label declared by the caller. Ordering expresses a
/// process sequence only; it does not certify the underlying artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SuitEvidenceLevel {
    Declared,
    Simulated,
    BenchTested,
    HumanInLoop,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuitCapabilityClaim {
    pub capability: SuitCapability,
    /// Evidence is bound to one exact variant ID; it is not portable by default.
    pub applies_to_variant_id: String,
    /// Evidence is bound to one exact hardware/configuration revision.
    pub applies_to_hardware_revision: String,
    /// Identity of the specific implementation / assembly for this capability.
    pub implementation_id: String,
    pub evidence_level: SuitEvidenceLevel,
    /// URI, report ID, artifact digest, or test record; not authenticated here.
    pub evidence_reference: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuitVariantDeclaration {
    pub variant_id: String,
    pub hardware_revision: String,
    pub capability_claims: Vec<SuitCapabilityClaim>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuitProfileEvidencePolicy {
    /// Caller-specified floor. The evaluator does not choose a qualification
    /// target and makes no claim that this floor is sufficient for a mission.
    pub minimum_required_evidence: SuitEvidenceLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuitCapabilityFindingStatus {
    MeetsDeclaredPolicy,
    MissingRequiredCapability,
    EvidenceBelowPolicy,
    ConflictingClaims,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuitProfileEvaluationStatus {
    RequirementsMissing,
    RequiredEvidenceBelowPolicy,
    ConflictingRequiredClaims,
    MeetsDeclaredEvidencePolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuitCapabilityFinding {
    pub capability: SuitCapability,
    pub required_for_profile: bool,
    pub status: SuitCapabilityFindingStatus,
    pub claim_count: usize,
    /// Present only when exactly one unambiguous claim exists.
    pub implementation_id: Option<String>,
    pub evidence_level: Option<SuitEvidenceLevel>,
    pub evidence_reference: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuitProfileEvaluation {
    pub profile: SuitMissionProfile,
    pub variant_id: String,
    pub hardware_revision: String,
    pub evidence_policy: SuitProfileEvidencePolicy,
    pub required_capabilities: Vec<SuitCapability>,
    pub findings: Vec<SuitCapabilityFinding>,
    pub status: SuitProfileEvaluationStatus,
    pub missing_required_count: usize,
    pub below_policy_required_count: usize,
    pub conflicting_required_count: usize,
    /// Always false in this software contract; certification is out of scope.
    pub represents_hardware_qualification: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuitProfileError {
    InvalidVariantId,
    InvalidHardwareRevision,
    InvalidCapabilityClaim,
    EvidenceScopeMismatch,
}

impl SuitVariantDeclaration {
    pub fn is_valid(&self) -> bool {
        !self.variant_id.trim().is_empty()
            && self.variant_id.trim() == self.variant_id
            && !self.hardware_revision.trim().is_empty()
            && self.hardware_revision.trim() == self.hardware_revision
            && self.capability_claims.iter().all(|claim| {
                !claim.implementation_id.trim().is_empty()
                    && claim.implementation_id.trim() == claim.implementation_id
                    && !claim.evidence_reference.trim().is_empty()
                    && claim.evidence_reference.trim() == claim.evidence_reference
                    && !claim.applies_to_variant_id.trim().is_empty()
                    && claim.applies_to_variant_id.trim() == claim.applies_to_variant_id
                    && !claim.applies_to_hardware_revision.trim().is_empty()
                    && claim.applies_to_hardware_revision.trim()
                        == claim.applies_to_hardware_revision
                    && claim.applies_to_variant_id == self.variant_id
                    && claim.applies_to_hardware_revision == self.hardware_revision
            })
    }
}

/// Evaluate one declared variant against one profile and an explicit evidence
/// policy. Every required capability is checked independently; evidence for one
/// capability or variant is never inherited by another.
pub fn evaluate_suit_profile(
    profile: SuitMissionProfile,
    declaration: &SuitVariantDeclaration,
    policy: SuitProfileEvidencePolicy,
) -> Result<SuitProfileEvaluation, SuitProfileError> {
    if declaration.variant_id.trim().is_empty()
        || declaration.variant_id.trim() != declaration.variant_id
    {
        return Err(SuitProfileError::InvalidVariantId);
    }
    if declaration.hardware_revision.trim().is_empty()
        || declaration.hardware_revision.trim() != declaration.hardware_revision
    {
        return Err(SuitProfileError::InvalidHardwareRevision);
    }
    if declaration.capability_claims.iter().any(|claim| {
        claim.implementation_id.trim().is_empty()
            || claim.implementation_id.trim() != claim.implementation_id
            || claim.evidence_reference.trim().is_empty()
            || claim.evidence_reference.trim() != claim.evidence_reference
            || claim.applies_to_variant_id.trim().is_empty()
            || claim.applies_to_variant_id.trim() != claim.applies_to_variant_id
            || claim.applies_to_hardware_revision.trim().is_empty()
            || claim.applies_to_hardware_revision.trim() != claim.applies_to_hardware_revision
    }) {
        return Err(SuitProfileError::InvalidCapabilityClaim);
    }
    if declaration.capability_claims.iter().any(|claim| {
        claim.applies_to_variant_id != declaration.variant_id
            || claim.applies_to_hardware_revision != declaration.hardware_revision
    }) {
        return Err(SuitProfileError::EvidenceScopeMismatch);
    }

    let required_capabilities = profile.required_capabilities();
    let required_set: BTreeSet<_> = required_capabilities.iter().copied().collect();
    let mut all_capabilities = required_set.clone();
    all_capabilities.extend(
        declaration
            .capability_claims
            .iter()
            .map(|claim| claim.capability),
    );

    let mut claims = declaration.capability_claims.clone();
    claims.sort_by(|left, right| {
        left.capability
            .cmp(&right.capability)
            .then(left.implementation_id.cmp(&right.implementation_id))
            .then(left.evidence_level.cmp(&right.evidence_level))
            .then(left.evidence_reference.cmp(&right.evidence_reference))
    });

    let mut findings = Vec::with_capacity(all_capabilities.len());
    for capability in all_capabilities {
        let matching: Vec<_> = claims
            .iter()
            .filter(|claim| claim.capability == capability)
            .collect();
        let required_for_profile = required_set.contains(&capability);
        let finding = if matching.len() > 1 {
            SuitCapabilityFinding {
                capability,
                required_for_profile,
                status: SuitCapabilityFindingStatus::ConflictingClaims,
                claim_count: matching.len(),
                implementation_id: None,
                evidence_level: None,
                evidence_reference: None,
            }
        } else if let Some(claim) = matching.first() {
            SuitCapabilityFinding {
                capability,
                required_for_profile,
                status: if claim.evidence_level < policy.minimum_required_evidence {
                    SuitCapabilityFindingStatus::EvidenceBelowPolicy
                } else {
                    SuitCapabilityFindingStatus::MeetsDeclaredPolicy
                },
                claim_count: 1,
                implementation_id: Some(claim.implementation_id.clone()),
                evidence_level: Some(claim.evidence_level),
                evidence_reference: Some(claim.evidence_reference.clone()),
            }
        } else {
            SuitCapabilityFinding {
                capability,
                required_for_profile,
                status: SuitCapabilityFindingStatus::MissingRequiredCapability,
                claim_count: 0,
                implementation_id: None,
                evidence_level: None,
                evidence_reference: None,
            }
        };
        findings.push(finding);
    }

    let missing_required_count = findings
        .iter()
        .filter(|finding| {
            finding.required_for_profile
                && finding.status == SuitCapabilityFindingStatus::MissingRequiredCapability
        })
        .count();
    let below_policy_required_count = findings
        .iter()
        .filter(|finding| {
            finding.required_for_profile
                && finding.status == SuitCapabilityFindingStatus::EvidenceBelowPolicy
        })
        .count();
    let conflicting_required_count = findings
        .iter()
        .filter(|finding| {
            finding.required_for_profile
                && finding.status == SuitCapabilityFindingStatus::ConflictingClaims
        })
        .count();

    let status = if conflicting_required_count > 0 {
        SuitProfileEvaluationStatus::ConflictingRequiredClaims
    } else if missing_required_count > 0 {
        SuitProfileEvaluationStatus::RequirementsMissing
    } else if below_policy_required_count > 0 {
        SuitProfileEvaluationStatus::RequiredEvidenceBelowPolicy
    } else {
        SuitProfileEvaluationStatus::MeetsDeclaredEvidencePolicy
    };

    Ok(SuitProfileEvaluation {
        profile,
        variant_id: declaration.variant_id.clone(),
        hardware_revision: declaration.hardware_revision.clone(),
        evidence_policy: policy,
        required_capabilities,
        findings,
        status,
        missing_required_count,
        below_policy_required_count,
        conflicting_required_count,
        represents_hardware_qualification: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declaration(
        profile: SuitMissionProfile,
        level: SuitEvidenceLevel,
    ) -> SuitVariantDeclaration {
        SuitVariantDeclaration {
            variant_id: format!("research-{:?}", profile),
            hardware_revision: "prototype-r1".to_string(),
            capability_claims: profile
                .required_capabilities()
                .into_iter()
                .map(|capability| SuitCapabilityClaim {
                    capability,
                    applies_to_variant_id: format!("research-{:?}", profile),
                    applies_to_hardware_revision: "prototype-r1".to_string(),
                    implementation_id: format!("impl-{:?}", capability),
                    evidence_level: level,
                    evidence_reference: format!("sim://{:?}/{:?}", profile, capability),
                })
                .collect(),
        }
    }

    fn policy(minimum_required_evidence: SuitEvidenceLevel) -> SuitProfileEvidencePolicy {
        SuitProfileEvidencePolicy { minimum_required_evidence }
    }

    #[test]
    fn every_profile_shares_the_safety_foundation_but_has_distinct_mission_requirements() {
        for profile in SuitMissionProfile::ALL {
            let required = profile.required_capabilities();
            for common in COMMON_SAFETY_FOUNDATION {
                assert!(required.contains(common), "{profile:?} missing {common:?}");
            }
        }
        assert!(!SuitMissionProfile::OrbitalEva
            .required_capabilities()
            .contains(&SuitCapability::PlanetaryLocomotion));
        assert!(SuitMissionProfile::LunarSurfaceEva
            .required_capabilities()
            .contains(&SuitCapability::DustMitigation));
        assert!(SuitMissionProfile::LunarSurfaceEva
            .required_capabilities()
            .contains(&SuitCapability::ElectrostaticChargeManagement));
        assert!(SuitMissionProfile::DustIntensiveSurfaceWork
            .required_capabilities()
            .contains(&SuitCapability::ElectrostaticChargeManagement));
        assert!(!SuitMissionProfile::OrbitalEva
            .required_capabilities()
            .contains(&SuitCapability::ElectrostaticChargeManagement));
        assert!(SuitMissionProfile::CrewRescueAndRecovery
            .required_capabilities()
            .contains(&SuitCapability::CrewRescueHandlingInterfaces));
        assert!(SuitMissionProfile::SuitportAndHabitatServicing
            .required_capabilities()
            .contains(&SuitCapability::SuitportMechanicalAndPressureCompatibility));
    }

    #[test]
    fn missing_capability_is_not_interpreted_as_compatibility() {
        let mut variant = declaration(
            SuitMissionProfile::LunarSurfaceEva,
            SuitEvidenceLevel::BenchTested,
        );
        variant
            .capability_claims
            .retain(|claim| claim.capability != SuitCapability::DustMitigation);
        let result = evaluate_suit_profile(
            SuitMissionProfile::LunarSurfaceEva,
            &variant,
            policy(SuitEvidenceLevel::BenchTested),
        ).unwrap();
        assert_eq!(result.status, SuitProfileEvaluationStatus::RequirementsMissing);
        assert_eq!(result.missing_required_count, 1);
    }

    #[test]
    fn low_evidence_does_not_satisfy_a_stronger_caller_policy() {
        let variant = declaration(SuitMissionProfile::OrbitalEva, SuitEvidenceLevel::Simulated);
        let result = evaluate_suit_profile(
            SuitMissionProfile::OrbitalEva,
            &variant,
            policy(SuitEvidenceLevel::BenchTested),
        ).unwrap();
        assert_eq!(result.status, SuitProfileEvaluationStatus::RequiredEvidenceBelowPolicy);
        assert!(result.below_policy_required_count > 0);
    }

    #[test]
    fn evidence_for_one_capability_does_not_transfer_to_another() {
        let mut variant = declaration(SuitMissionProfile::OrbitalEva, SuitEvidenceLevel::Qualified);
        variant
            .capability_claims
            .retain(|claim| claim.capability == SuitCapability::PressureBoundaryIntegrity);
        let result = evaluate_suit_profile(
            SuitMissionProfile::OrbitalEva,
            &variant,
            policy(SuitEvidenceLevel::Declared),
        ).unwrap();
        let missing_count = result
            .findings
            .iter()
            .filter(|finding| {
                finding.status == SuitCapabilityFindingStatus::MissingRequiredCapability
            })
            .count();
        assert_eq!(
            missing_count,
            SuitMissionProfile::OrbitalEva.required_capabilities().len() - 1
        );
        assert!(!result.represents_hardware_qualification);
    }

    #[test]
    fn duplicate_capability_claims_are_explicitly_conflicted() {
        let mut variant = declaration(
            SuitMissionProfile::OrbitalEva,
            SuitEvidenceLevel::BenchTested,
        );
        variant.capability_claims.push(variant.capability_claims[0].clone());
        let result = evaluate_suit_profile(
            SuitMissionProfile::OrbitalEva,
            &variant,
            policy(SuitEvidenceLevel::BenchTested),
        ).unwrap();
        assert_eq!(result.status, SuitProfileEvaluationStatus::ConflictingRequiredClaims);
        assert_eq!(result.conflicting_required_count, 1);
    }

    #[test]
    fn capability_evidence_cannot_be_reused_for_another_variant_or_revision() {
        let mut changed_variant =
            declaration(SuitMissionProfile::OrbitalEva, SuitEvidenceLevel::Qualified);
        changed_variant.variant_id = "orbital-eva-variant-b".into();
        assert_eq!(
            evaluate_suit_profile(
                SuitMissionProfile::OrbitalEva,
                &changed_variant,
                policy(SuitEvidenceLevel::Declared),
            ),
            Err(SuitProfileError::EvidenceScopeMismatch)
        );

        let mut changed_revision =
            declaration(SuitMissionProfile::OrbitalEva, SuitEvidenceLevel::Qualified);
        changed_revision.hardware_revision = "prototype-r2".into();
        assert_eq!(
            evaluate_suit_profile(
                SuitMissionProfile::OrbitalEva,
                &changed_revision,
                policy(SuitEvidenceLevel::Declared),
            ),
            Err(SuitProfileError::EvidenceScopeMismatch)
        );
    }

    #[test]
    fn declaration_order_does_not_change_findings() {
        let variant = declaration(
            SuitMissionProfile::MarsSurfaceEva,
            SuitEvidenceLevel::BenchTested,
        );
        let mut reversed = variant.clone();
        reversed.capability_claims.reverse();
        let first = evaluate_suit_profile(
            SuitMissionProfile::MarsSurfaceEva,
            &variant,
            policy(SuitEvidenceLevel::BenchTested),
        ).unwrap();
        let second = evaluate_suit_profile(
            SuitMissionProfile::MarsSurfaceEva,
            &reversed,
            policy(SuitEvidenceLevel::BenchTested),
        ).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn evidence_policy_is_explicit_and_not_selected_by_the_profile() {
        let variant = declaration(
            SuitMissionProfile::LunarSurfaceEva,
            SuitEvidenceLevel::HumanInLoop,
        );
        let permissive = evaluate_suit_profile(
            SuitMissionProfile::LunarSurfaceEva,
            &variant,
            policy(SuitEvidenceLevel::Simulated),
        ).unwrap();
        let stronger = evaluate_suit_profile(
            SuitMissionProfile::LunarSurfaceEva,
            &variant,
            policy(SuitEvidenceLevel::Qualified),
        ).unwrap();
        assert_eq!(permissive.status, SuitProfileEvaluationStatus::MeetsDeclaredEvidencePolicy);
        assert_eq!(stronger.status, SuitProfileEvaluationStatus::RequiredEvidenceBelowPolicy);
    }

    #[test]
    fn malformed_claim_metadata_fails_closed() {
        let mut variant = declaration(
            SuitMissionProfile::OrbitalEva,
            SuitEvidenceLevel::BenchTested,
        );
        variant.capability_claims[0].evidence_reference = " ".into();
        assert_eq!(
            evaluate_suit_profile(
                SuitMissionProfile::OrbitalEva,
                &variant,
                policy(SuitEvidenceLevel::Declared),
            ),
            Err(SuitProfileError::InvalidCapabilityClaim)
        );
    }

    #[test]
    fn profile_requirements_are_deterministically_ordered() {
        for profile in SuitMissionProfile::ALL {
            let required = profile.required_capabilities();
            let mut sorted = required.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(required, sorted);
        }
    }
}
