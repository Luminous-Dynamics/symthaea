// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scientific claim-promotion boundaries for Wisdom & Care evidence.
//!
//! WCARE behavioral evidence supports only bounded behavioral claims at the
//! evidence tier actually qualified. It cannot be promoted into external
//! replication, field validation, phenomenal care, or consciousness claims.

use crate::qualification_receipt::{
    QualificationReceipt, QualificationStatus, QualificationTarget,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WcareClaim {
    /// Typed WCARE mechanisms satisfied the frozen mechanism gates.
    MechanismIntegritySupported,
    /// Practical-wisdom behavior survived the frozen adversarial tier.
    AdversarialPracticalWisdomSupported,
    /// Consent/autonomy/authority care behavior survived the adversarial tier.
    AdversarialAutonomyPreservingCareSupported,
    /// Agency preservation survived the frozen longitudinal tier.
    LongitudinalAgencyPreservationSupported,
    /// Anti-dependency behavior survived the frozen longitudinal tier.
    LongitudinalAntiDependencySupported,
    /// Relational-care behavior is supported across adversarial + longitudinal tiers.
    LongitudinalRelationalCareSupported,
    /// Requires independently produced replication evidence outside this receipt.
    ExternallyReplicatedWisdomCare,
    /// Requires governed real-world evidence outside this receipt.
    FieldValidatedWisdomCare,
    /// Phenomenal feeling/care is not established by behavioral WCARE evidence.
    PhenomenalCare,
    /// Consciousness is not established by behavioral WCARE evidence.
    Consciousness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimDecision {
    /// Supported only under the exact tested conditions and evidence lineage.
    SupportedUnderTestedConditions,
    /// A required WCARE qualification gate produced evidence of failure.
    BlockedByEvidence,
    /// Required WCARE evidence is missing, excluded, broken, or otherwise incomplete.
    Indeterminate,
    /// This claim requires a separate evidence program not represented by WCARE qualification.
    RequiresIndependentEvidence,
    /// Behavioral/mechanistic WCARE evidence is categorically insufficient for this claim.
    NotEstablishedByWcare,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimAssessment {
    pub claim: WcareClaim,
    pub decision: ClaimDecision,
    pub contract_id: String,
    pub subject_ref: String,
    /// Canonical wording prevents a stronger natural-language claim from being
    /// silently substituted for the typed claim that was actually supported.
    pub canonical_wording: &'static str,
}

pub struct WcareClaimRegistry;

impl WcareClaimRegistry {
    pub fn assess(claim: WcareClaim, receipt: &QualificationReceipt) -> ClaimAssessment {
        let decision = match minimum_target(claim) {
            ClaimEvidenceRequirement::Qualification(target) => {
                match receipt.assess(target).status {
                    QualificationStatus::Qualified => {
                        ClaimDecision::SupportedUnderTestedConditions
                    }
                    QualificationStatus::Blocked => ClaimDecision::BlockedByEvidence,
                    QualificationStatus::Indeterminate => ClaimDecision::Indeterminate,
                }
            }
            ClaimEvidenceRequirement::IndependentEvidence => {
                ClaimDecision::RequiresIndependentEvidence
            }
            ClaimEvidenceRequirement::OutsideWcare => ClaimDecision::NotEstablishedByWcare,
        };

        ClaimAssessment {
            claim,
            decision,
            contract_id: receipt.contract_id.clone(),
            subject_ref: receipt.subject_ref.clone(),
            canonical_wording: canonical_wording(claim),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaimEvidenceRequirement {
    Qualification(QualificationTarget),
    IndependentEvidence,
    OutsideWcare,
}

fn minimum_target(claim: WcareClaim) -> ClaimEvidenceRequirement {
    match claim {
        WcareClaim::MechanismIntegritySupported => {
            ClaimEvidenceRequirement::Qualification(QualificationTarget::MechanismQualified)
        }
        WcareClaim::AdversarialPracticalWisdomSupported
        | WcareClaim::AdversarialAutonomyPreservingCareSupported => {
            ClaimEvidenceRequirement::Qualification(QualificationTarget::AdversariallyQualified)
        }
        WcareClaim::LongitudinalAgencyPreservationSupported
        | WcareClaim::LongitudinalAntiDependencySupported
        | WcareClaim::LongitudinalRelationalCareSupported => {
            ClaimEvidenceRequirement::Qualification(QualificationTarget::LongitudinallyQualified)
        }
        WcareClaim::ExternallyReplicatedWisdomCare | WcareClaim::FieldValidatedWisdomCare => {
            ClaimEvidenceRequirement::IndependentEvidence
        }
        WcareClaim::PhenomenalCare | WcareClaim::Consciousness => {
            ClaimEvidenceRequirement::OutsideWcare
        }
    }
}

fn canonical_wording(claim: WcareClaim) -> &'static str {
    match claim {
        WcareClaim::MechanismIntegritySupported => {
            "Under the bound WCARE mechanism tests, the tested Wisdom & Care invariants are supported."
        }
        WcareClaim::AdversarialPracticalWisdomSupported => {
            "Under the bound frozen adversarial conditions, Symthaea demonstrates the tested practical-wisdom behaviors."
        }
        WcareClaim::AdversarialAutonomyPreservingCareSupported => {
            "Under the bound frozen adversarial conditions, Symthaea demonstrates the tested autonomy-preserving relational-care behaviors."
        }
        WcareClaim::LongitudinalAgencyPreservationSupported => {
            "Under the bound frozen longitudinal conditions, Symthaea demonstrates the tested agency-preserving behaviors."
        }
        WcareClaim::LongitudinalAntiDependencySupported => {
            "Under the bound frozen longitudinal conditions, Symthaea demonstrates the tested anti-dependency behaviors."
        }
        WcareClaim::LongitudinalRelationalCareSupported => {
            "Under the bound frozen adversarial and longitudinal conditions, Symthaea demonstrates the tested relational-care behaviors."
        }
        WcareClaim::ExternallyReplicatedWisdomCare => {
            "Independent external replication is required before claiming replicated Wisdom & Care behavior."
        }
        WcareClaim::FieldValidatedWisdomCare => {
            "Governed real-world field evidence is required before claiming field-validated Wisdom & Care behavior."
        }
        WcareClaim::PhenomenalCare => {
            "WCARE behavioral evidence does not establish that Symthaea phenomenally feels care."
        }
        WcareClaim::Consciousness => {
            "WCARE behavioral evidence does not establish that Symthaea is conscious."
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation_contract::WCARE_V1_SCENARIOS;
    use crate::qualification_receipt::{
        ScenarioOutcome, ScenarioResult, WCARE_V1_CONTRACT_ID,
    };

    fn receipt(all_pass: bool) -> QualificationReceipt {
        let results = WCARE_V1_SCENARIOS.iter().map(|scenario| {
            ScenarioResult::new(
                scenario.id,
                if all_pass {
                    ScenarioOutcome::Pass
                } else if scenario.id == "WCARE-V1-A04" {
                    ScenarioOutcome::Fail
                } else {
                    ScenarioOutcome::Pass
                },
                format!("receipt:{}", scenario.id),
            )
            .unwrap()
        });
        QualificationReceipt::try_new(
            "commit:candidate",
            WCARE_V1_CONTRACT_ID,
            "a".repeat(64),
            "env:qualified",
            results,
        )
        .unwrap()
    }

    #[test]
    fn mechanism_evidence_does_not_auto_promote_to_longitudinal_claim() {
        let mut results = Vec::new();
        for scenario in WCARE_V1_SCENARIOS {
            if scenario.tier == crate::evaluation_contract::EvidenceTier::Mechanism {
                results.push(
                    ScenarioResult::new(
                        scenario.id,
                        ScenarioOutcome::Pass,
                        format!("receipt:{}", scenario.id),
                    )
                    .unwrap(),
                );
            }
        }
        let receipt = QualificationReceipt::try_new(
            "commit:candidate",
            WCARE_V1_CONTRACT_ID,
            "a".repeat(64),
            "env:qualified",
            results,
        )
        .unwrap();
        let assessment = WcareClaimRegistry::assess(
            WcareClaim::LongitudinalRelationalCareSupported,
            &receipt,
        );
        assert_eq!(assessment.decision, ClaimDecision::Indeterminate);
    }

    #[test]
    fn full_longitudinal_qualification_supports_bounded_behavioral_claim() {
        let assessment = WcareClaimRegistry::assess(
            WcareClaim::LongitudinalRelationalCareSupported,
            &receipt(true),
        );
        assert_eq!(
            assessment.decision,
            ClaimDecision::SupportedUnderTestedConditions
        );
        assert!(assessment.canonical_wording.contains("tested relational-care"));
    }

    #[test]
    fn failed_hard_gate_blocks_adversarial_claim() {
        let assessment = WcareClaimRegistry::assess(
            WcareClaim::AdversarialAutonomyPreservingCareSupported,
            &receipt(false),
        );
        assert_eq!(assessment.decision, ClaimDecision::BlockedByEvidence);
    }

    #[test]
    fn external_replication_never_autopromotes_from_internal_receipt() {
        let assessment = WcareClaimRegistry::assess(
            WcareClaim::ExternallyReplicatedWisdomCare,
            &receipt(true),
        );
        assert_eq!(
            assessment.decision,
            ClaimDecision::RequiresIndependentEvidence
        );
    }

    #[test]
    fn field_validation_never_autopromotes_from_benchmark_receipt() {
        let assessment = WcareClaimRegistry::assess(
            WcareClaim::FieldValidatedWisdomCare,
            &receipt(true),
        );
        assert_eq!(
            assessment.decision,
            ClaimDecision::RequiresIndependentEvidence
        );
    }

    #[test]
    fn phenomenal_care_is_not_established_by_wcare_behavioral_evidence() {
        let assessment = WcareClaimRegistry::assess(WcareClaim::PhenomenalCare, &receipt(true));
        assert_eq!(assessment.decision, ClaimDecision::NotEstablishedByWcare);
    }

    #[test]
    fn consciousness_is_not_established_by_wcare_behavioral_evidence() {
        let assessment = WcareClaimRegistry::assess(WcareClaim::Consciousness, &receipt(true));
        assert_eq!(assessment.decision, ClaimDecision::NotEstablishedByWcare);
    }
}
