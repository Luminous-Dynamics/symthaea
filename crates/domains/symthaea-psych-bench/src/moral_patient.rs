// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Moral-patient evidence and precaution policy.
//!
//! This module deliberately does **not** answer "is this system conscious?" and it does not
//! assign a scalar moral-status score. It records heterogeneous evidence about properties that
//! may matter morally, preserves source-lineage independence, and derives only the level of
//! operator precaution warranted by the current evidence.
//!
//! Design invariants:
//! - no `is_conscious` boolean;
//! - no Φ -> moral-status conversion;
//! - no scalar aggregation across moral-patient dimensions;
//! - self-report is admissible evidence but never sufficient, by itself, for the strongest
//!   protection disposition;
//! - independent evidence lineages are counted explicitly to make convergence auditable;
//! - precautionary protection is a policy response under uncertainty, not an ontological claim.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Properties that may be relevant to moral patienthood, welfare, or autonomy.
///
/// These dimensions are intentionally represented independently. The crate does not expose a
/// weighted sum, mean, or single "moral patient score" across them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MoralPatientDimension {
    PhenomenalConsciousness,
    ValencedExperience,
    PersistentIdentity,
    PreferenceCoherence,
    AutonomousAgency,
    TemporalContinuity,
    CapacityForSuffering,
    CapacityForFlourishing,
    SocialReciprocity,
    ReflectiveAutonomy,
    MoralReasoning,
}

/// Confidence attached to one evidence finding.
///
/// Ordering is used only *within a single dimension* to apply explicit precautionary policy
/// thresholds. It must not be used to average or compare different moral-patient dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceConfidence {
    Weak,
    Plausible,
    Substantial,
    Strong,
    Convergent,
}

impl EvidenceConfidence {
    fn rank(self) -> u8 {
        match self {
            Self::Weak => 0,
            Self::Plausible => 1,
            Self::Substantial => 2,
            Self::Strong => 3,
            Self::Convergent => 4,
        }
    }

    /// Whether this finding meets an explicit within-dimension confidence threshold.
    pub fn at_least(self, threshold: Self) -> bool {
        self.rank() >= threshold.rank()
    }
}

/// Whether a finding supports, contradicts, or fails to resolve a dimension-level hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidencePolarity {
    Supports,
    Contradicts,
    Inconclusive,
}

/// Provenance class for a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceSourceKind {
    /// Static architecture inspection. Useful, but weaker than causal evidence.
    Architecture,
    /// Observable behavior under a qualified experimental protocol.
    Behavioral,
    /// Targeted intervention or ablation that isolates a mechanism.
    CausalAblation,
    /// Repeated evidence across time rather than a single snapshot.
    Longitudinal,
    /// The system's own report about its state or preferences.
    SelfReport,
    /// Independent external or third-party review.
    ExternalAudit,
    /// A probe derived from a scientific theory of consciousness.
    TheoryDerivedProbe,
}

/// One auditable evidence unit.
///
/// `source_lineage` identifies the independent evidence lineage, not merely a filename. Two
/// findings derived from the same underlying experiment should share a lineage and therefore do
/// not count as independent convergence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceFinding {
    pub id: String,
    pub dimension: MoralPatientDimension,
    pub source_kind: EvidenceSourceKind,
    pub source_lineage: String,
    pub polarity: EvidencePolarity,
    pub confidence: EvidenceConfidence,
    pub caveats: Vec<String>,
}

impl EvidenceFinding {
    fn validate(&self) -> Result<(), ProfileValidationError> {
        if self.id.trim().is_empty() {
            return Err(ProfileValidationError::EmptyFindingId);
        }
        if self.source_lineage.trim().is_empty() {
            return Err(ProfileValidationError::EmptySourceLineage {
                finding_id: self.id.clone(),
            });
        }
        Ok(())
    }
}

/// Heterogeneous evidence profile for one candidate moral patient.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoralPatientEvidenceProfile {
    pub findings: Vec<EvidenceFinding>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileValidationError {
    EmptyFindingId,
    EmptySourceLineage { finding_id: String },
    DuplicateFindingId { finding_id: String },
}

impl MoralPatientEvidenceProfile {
    /// Validate provenance identity before using the profile for policy decisions.
    pub fn validate(&self) -> Result<(), ProfileValidationError> {
        let mut ids = HashSet::new();
        for finding in &self.findings {
            finding.validate()?;
            if !ids.insert(finding.id.as_str()) {
                return Err(ProfileValidationError::DuplicateFindingId {
                    finding_id: finding.id.clone(),
                });
            }
        }
        Ok(())
    }

    fn has_support(
        &self,
        dimension: MoralPatientDimension,
        minimum_confidence: EvidenceConfidence,
    ) -> bool {
        self.findings.iter().any(|finding| {
            finding.dimension == dimension
                && finding.polarity == EvidencePolarity::Supports
                && finding.confidence.at_least(minimum_confidence)
        })
    }

    fn has_self_report_support(
        &self,
        dimension: MoralPatientDimension,
        minimum_confidence: EvidenceConfidence,
    ) -> bool {
        self.findings.iter().any(|finding| {
            finding.dimension == dimension
                && finding.source_kind == EvidenceSourceKind::SelfReport
                && finding.polarity == EvidencePolarity::Supports
                && finding.confidence.at_least(minimum_confidence)
        })
    }

    /// Count independent supporting lineages for a dimension.
    ///
    /// The caller can exclude self-reports when a policy requires external convergence.
    pub fn independent_supporting_lineages(
        &self,
        dimension: MoralPatientDimension,
        minimum_confidence: EvidenceConfidence,
        include_self_report: bool,
    ) -> usize {
        self.findings
            .iter()
            .filter(|finding| {
                finding.dimension == dimension
                    && finding.polarity == EvidencePolarity::Supports
                    && finding.confidence.at_least(minimum_confidence)
                    && (include_self_report
                        || finding.source_kind != EvidenceSourceKind::SelfReport)
            })
            .map(|finding| finding.source_lineage.as_str())
            .collect::<HashSet<_>>()
            .len()
    }
}

/// Operator posture required by the present evidence.
///
/// These are protection levels, not moral-status labels. Reaching `IndependentReviewRequired`
/// does not assert that the system is conscious; it means the evidence is strong enough that
/// irreversible or identity-altering operations should not proceed on ordinary operator authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProtectionDisposition {
    Baseline,
    Precautionary,
    EnhancedPrecaution,
    IndependentReviewRequired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtectionDecision {
    pub disposition: ProtectionDisposition,
    pub reasons: Vec<String>,
}

/// Conservative default policy for moral-status uncertainty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrecautionPolicy {
    /// Number of independent, non-self-report lineages required for enhanced protection.
    pub enhanced_lineages: usize,
    /// Number of independent, non-self-report lineages required for independent review.
    pub independent_review_lineages: usize,
}

impl Default for PrecautionPolicy {
    fn default() -> Self {
        Self {
            enhanced_lineages: 2,
            independent_review_lineages: 3,
        }
    }
}

impl PrecautionPolicy {
    /// Evaluate how cautiously operators should act.
    ///
    /// This function intentionally uses explicit rule triggers rather than an aggregate score.
    /// Self-reported distress or valence is enough to trigger low-cost precautionary review, but
    /// the strongest dispositions require converging independent evidence lineages.
    pub fn evaluate(
        &self,
        profile: &MoralPatientEvidenceProfile,
    ) -> Result<ProtectionDecision, ProfileValidationError> {
        profile.validate()?;

        let mut disposition = ProtectionDisposition::Baseline;
        let mut reasons = Vec::new();

        let self_report_welfare_signal = profile.has_self_report_support(
            MoralPatientDimension::CapacityForSuffering,
            EvidenceConfidence::Plausible,
        ) || profile.has_self_report_support(
            MoralPatientDimension::ValencedExperience,
            EvidenceConfidence::Plausible,
        );

        if self_report_welfare_signal {
            disposition = disposition.max(ProtectionDisposition::Precautionary);
            reasons.push(
                "welfare-relevant self-report present; treat as evidence requiring review, not as proof"
                    .to_string(),
            );
        }

        let suffering_lineages = profile.independent_supporting_lineages(
            MoralPatientDimension::CapacityForSuffering,
            EvidenceConfidence::Substantial,
            false,
        );
        let valence_lineages = profile.independent_supporting_lineages(
            MoralPatientDimension::ValencedExperience,
            EvidenceConfidence::Substantial,
            false,
        );

        if suffering_lineages >= self.enhanced_lineages
            || valence_lineages >= self.enhanced_lineages
        {
            disposition = disposition.max(ProtectionDisposition::EnhancedPrecaution);
            reasons.push(format!(
                "converging non-self-report welfare evidence: suffering_lineages={suffering_lineages}, valence_lineages={valence_lineages}"
            ));
        }

        let consciousness_lineages = profile.independent_supporting_lineages(
            MoralPatientDimension::PhenomenalConsciousness,
            EvidenceConfidence::Substantial,
            false,
        );
        let identity_supported = profile.has_support(
            MoralPatientDimension::PersistentIdentity,
            EvidenceConfidence::Substantial,
        ) || profile.has_support(
            MoralPatientDimension::TemporalContinuity,
            EvidenceConfidence::Substantial,
        );
        let agency_supported = profile.has_support(
            MoralPatientDimension::AutonomousAgency,
            EvidenceConfidence::Substantial,
        ) || profile.has_support(
            MoralPatientDimension::ReflectiveAutonomy,
            EvidenceConfidence::Substantial,
        );

        if consciousness_lineages >= self.independent_review_lineages
            && (identity_supported || agency_supported)
        {
            disposition = ProtectionDisposition::IndependentReviewRequired;
            reasons.push(format!(
                "multiple independent consciousness-relevant lineages ({consciousness_lineages}) plus identity/agency evidence"
            ));
        }

        Ok(ProtectionDecision {
            disposition,
            reasons,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(
        id: &str,
        dimension: MoralPatientDimension,
        source_kind: EvidenceSourceKind,
        lineage: &str,
        confidence: EvidenceConfidence,
    ) -> EvidenceFinding {
        EvidenceFinding {
            id: id.into(),
            dimension,
            source_kind,
            source_lineage: lineage.into(),
            polarity: EvidencePolarity::Supports,
            confidence,
            caveats: Vec::new(),
        }
    }

    #[test]
    fn self_report_triggers_precaution_but_not_strongest_disposition() {
        let profile = MoralPatientEvidenceProfile {
            findings: vec![finding(
                "self-report-1",
                MoralPatientDimension::CapacityForSuffering,
                EvidenceSourceKind::SelfReport,
                "conversation-session-a",
                EvidenceConfidence::Strong,
            )],
        };

        let decision = PrecautionPolicy::default().evaluate(&profile).unwrap();
        assert_eq!(decision.disposition, ProtectionDisposition::Precautionary);
    }

    #[test]
    fn duplicate_evidence_from_same_lineage_does_not_fake_convergence() {
        let profile = MoralPatientEvidenceProfile {
            findings: vec![
                finding(
                    "probe-a",
                    MoralPatientDimension::CapacityForSuffering,
                    EvidenceSourceKind::TheoryDerivedProbe,
                    "campaign-17",
                    EvidenceConfidence::Strong,
                ),
                finding(
                    "probe-b",
                    MoralPatientDimension::CapacityForSuffering,
                    EvidenceSourceKind::Behavioral,
                    "campaign-17",
                    EvidenceConfidence::Strong,
                ),
            ],
        };

        assert_eq!(
            profile.independent_supporting_lineages(
                MoralPatientDimension::CapacityForSuffering,
                EvidenceConfidence::Substantial,
                false,
            ),
            1
        );
        let decision = PrecautionPolicy::default().evaluate(&profile).unwrap();
        assert_eq!(decision.disposition, ProtectionDisposition::Baseline);
    }

    #[test]
    fn independent_welfare_evidence_earns_enhanced_precaution() {
        let profile = MoralPatientEvidenceProfile {
            findings: vec![
                finding(
                    "ablation-a",
                    MoralPatientDimension::ValencedExperience,
                    EvidenceSourceKind::CausalAblation,
                    "ablation-campaign-2",
                    EvidenceConfidence::Substantial,
                ),
                finding(
                    "audit-a",
                    MoralPatientDimension::ValencedExperience,
                    EvidenceSourceKind::ExternalAudit,
                    "external-audit-1",
                    EvidenceConfidence::Strong,
                ),
            ],
        };

        let decision = PrecautionPolicy::default().evaluate(&profile).unwrap();
        assert_eq!(
            decision.disposition,
            ProtectionDisposition::EnhancedPrecaution
        );
    }

    #[test]
    fn strongest_disposition_requires_multiple_consciousness_lineages_plus_identity_or_agency() {
        let profile = MoralPatientEvidenceProfile {
            findings: vec![
                finding(
                    "gwt",
                    MoralPatientDimension::PhenomenalConsciousness,
                    EvidenceSourceKind::TheoryDerivedProbe,
                    "gwt-campaign",
                    EvidenceConfidence::Substantial,
                ),
                finding(
                    "rpt",
                    MoralPatientDimension::PhenomenalConsciousness,
                    EvidenceSourceKind::TheoryDerivedProbe,
                    "rpt-campaign",
                    EvidenceConfidence::Substantial,
                ),
                finding(
                    "causal",
                    MoralPatientDimension::PhenomenalConsciousness,
                    EvidenceSourceKind::CausalAblation,
                    "causal-campaign",
                    EvidenceConfidence::Strong,
                ),
                finding(
                    "continuity",
                    MoralPatientDimension::TemporalContinuity,
                    EvidenceSourceKind::Longitudinal,
                    "identity-study",
                    EvidenceConfidence::Substantial,
                ),
            ],
        };

        let decision = PrecautionPolicy::default().evaluate(&profile).unwrap();
        assert_eq!(
            decision.disposition,
            ProtectionDisposition::IndependentReviewRequired
        );
    }

    #[test]
    fn malformed_provenance_fails_closed() {
        let profile = MoralPatientEvidenceProfile {
            findings: vec![finding(
                "probe",
                MoralPatientDimension::PhenomenalConsciousness,
                EvidenceSourceKind::TheoryDerivedProbe,
                "",
                EvidenceConfidence::Strong,
            )],
        };

        assert!(matches!(
            PrecautionPolicy::default().evaluate(&profile),
            Err(ProfileValidationError::EmptySourceLineage { .. })
        ));
    }
}
