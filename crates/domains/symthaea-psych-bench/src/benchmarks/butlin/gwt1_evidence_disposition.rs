// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Diagnostic GWT-1 evidence disposition above resolved support tiers.
//!
//! [`ButlinResolvedEvidenceViewV2`] deliberately preserves the strongest
//! independently established lower-tier support when a later causal lineage is
//! negative or inconclusive. That is the correct evidence-composition rule, but
//! a consumer that reads only the final `resolved_outcome` can accidentally
//! collapse materially different epistemic states into the same `Observed`
//! label.
//!
//! This module therefore derives a *diagnostic* disposition from an already
//! resolved V2 view. It creates no promotion authority and never upgrades an
//! evidence tier. In particular:
//!
//! ```text
//! resolved support floor = strongest independently established support
//! causal disposition      = what the higher-tier causal method established
//! ```
//!
//! A causal contradiction may therefore coexist with a retained direct
//! `Observed` support floor without being hidden or softened.

use serde::{Deserialize, Serialize};

use super::report::{EvidenceOutcome, SupportTier};
use super::resolution_view::EvidenceLineageKindV1;
use super::resolution_view_v2::{
    BUTLIN_RESOLVED_EVIDENCE_VIEW_SCHEMA_V2, ButlinResolvedEvidenceViewV2,
    IndicatorEvidenceLineageV2,
};

pub const GWT1_EVIDENCE_DISPOSITION_SCHEMA_V1: &str =
    "butlin-gwt1-evidence-disposition-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gwt1EvidenceDispositionV1 {
    /// Direct evidence established `Observed`, but no causal lineage is present.
    DirectObservedPendingCausal,
    /// A verified causal lineage raised GWT-1 to `CausallySupported`.
    CausallySupported,
    /// The causal experiment was valid but did not demonstrate the predicted effect.
    /// Independent direct `Observed` evidence remains the resolved support floor.
    CausalNotDemonstratedRetainsObserved,
    /// The causal experiment produced a genuine contradiction under its protocol.
    /// Independent direct `Observed` evidence remains the resolved support floor.
    CausalContradictedRetainsObserved,
    /// The causal experiment could not be interpreted under its protocol.
    /// Independent direct `Observed` evidence remains the resolved support floor.
    CausalInconclusiveRetainsObserved,
    /// Direct qualification itself did not demonstrate GWT-1.
    DirectNotDemonstrated,
    /// Direct qualification itself contradicted the preregistered direct theorem.
    DirectContradicted,
    /// Direct qualification could not be interpreted.
    DirectInconclusive,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1EvidenceDispositionSummaryV1 {
    pub schema: String,
    pub direct_outcome: EvidenceOutcome,
    pub causal_outcome: Option<EvidenceOutcome>,
    /// The support floor represented by the resolved V2 chain.
    pub resolved_outcome: EvidenceOutcome,
    pub disposition: Gwt1EvidenceDispositionV1,
    /// True only when the causal protocol produced a genuine contradiction while
    /// a lower direct tier remains independently established.
    pub has_causal_contradiction: bool,
    /// True when the causal lane exists but still requires follow-up before a
    /// positive/negative causal interpretation can be made.
    pub causal_follow_up_required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gwt1EvidenceDispositionErrorV1 {
    WrongViewSchema { observed: String },
    MissingDirectLineage,
    DuplicateDirectLineage,
    DuplicateCausalLineage,
    InvalidDirectOutcome { observed: EvidenceOutcome },
    InvalidDirectTransition {
        lineage_outcome: EvidenceOutcome,
        resolved_outcome: EvidenceOutcome,
    },
    CausalRequiresDirectObserved { observed: EvidenceOutcome },
    InvalidCausalOutcome { observed: EvidenceOutcome },
    InvalidCausalTransition {
        base_outcome: EvidenceOutcome,
        lineage_outcome: EvidenceOutcome,
        resolved_outcome: EvidenceOutcome,
    },
}

impl std::fmt::Display for Gwt1EvidenceDispositionErrorV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongViewSchema { observed } => write!(
                f,
                "GWT-1 disposition requires V2 resolved view schema, observed {observed:?}"
            ),
            Self::MissingDirectLineage => write!(f, "GWT-1 disposition requires a direct lineage"),
            Self::DuplicateDirectLineage => write!(f, "GWT-1 disposition found duplicate direct lineages"),
            Self::DuplicateCausalLineage => write!(f, "GWT-1 disposition found duplicate causal lineages"),
            Self::InvalidDirectOutcome { observed } => write!(
                f,
                "GWT-1 direct lineage has impossible disposition outcome {observed:?}"
            ),
            Self::InvalidDirectTransition {
                lineage_outcome,
                resolved_outcome,
            } => write!(
                f,
                "GWT-1 direct lineage must resolve exactly to its own outcome: lineage={lineage_outcome:?}, resolved={resolved_outcome:?}"
            ),
            Self::CausalRequiresDirectObserved { observed } => write!(
                f,
                "GWT-1 causal disposition requires direct Observed support, observed {observed:?}"
            ),
            Self::InvalidCausalOutcome { observed } => write!(
                f,
                "GWT-1 causal lineage has impossible disposition outcome {observed:?}"
            ),
            Self::InvalidCausalTransition {
                base_outcome,
                lineage_outcome,
                resolved_outcome,
            } => write!(
                f,
                "GWT-1 causal lineage transition is inconsistent: base={base_outcome:?}, lineage={lineage_outcome:?}, resolved={resolved_outcome:?}"
            ),
        }
    }
}

impl std::error::Error for Gwt1EvidenceDispositionErrorV1 {}

fn classify_direct_only(
    direct: &IndicatorEvidenceLineageV2,
) -> Result<Gwt1EvidenceDispositionSummaryV1, Gwt1EvidenceDispositionErrorV1> {
    if direct.lineage_outcome != direct.resolved_outcome {
        return Err(Gwt1EvidenceDispositionErrorV1::InvalidDirectTransition {
            lineage_outcome: direct.lineage_outcome,
            resolved_outcome: direct.resolved_outcome,
        });
    }

    let disposition = match direct.lineage_outcome {
        EvidenceOutcome::Supported(SupportTier::Observed) => {
            Gwt1EvidenceDispositionV1::DirectObservedPendingCausal
        }
        EvidenceOutcome::NotDemonstrated => Gwt1EvidenceDispositionV1::DirectNotDemonstrated,
        EvidenceOutcome::Contradicted => Gwt1EvidenceDispositionV1::DirectContradicted,
        EvidenceOutcome::Inconclusive => Gwt1EvidenceDispositionV1::DirectInconclusive,
        observed => {
            return Err(Gwt1EvidenceDispositionErrorV1::InvalidDirectOutcome { observed });
        }
    };

    Ok(Gwt1EvidenceDispositionSummaryV1 {
        schema: GWT1_EVIDENCE_DISPOSITION_SCHEMA_V1.to_string(),
        direct_outcome: direct.lineage_outcome,
        causal_outcome: None,
        resolved_outcome: direct.resolved_outcome,
        disposition,
        has_causal_contradiction: false,
        causal_follow_up_required: false,
    })
}

fn classify_with_causal(
    direct: &IndicatorEvidenceLineageV2,
    causal: &IndicatorEvidenceLineageV2,
) -> Result<Gwt1EvidenceDispositionSummaryV1, Gwt1EvidenceDispositionErrorV1> {
    if direct.lineage_outcome != EvidenceOutcome::Supported(SupportTier::Observed)
        || direct.resolved_outcome != EvidenceOutcome::Supported(SupportTier::Observed)
    {
        return Err(Gwt1EvidenceDispositionErrorV1::CausalRequiresDirectObserved {
            observed: direct.resolved_outcome,
        });
    }

    if causal.base_outcome != direct.resolved_outcome {
        return Err(Gwt1EvidenceDispositionErrorV1::InvalidCausalTransition {
            base_outcome: causal.base_outcome,
            lineage_outcome: causal.lineage_outcome,
            resolved_outcome: causal.resolved_outcome,
        });
    }

    let (disposition, expected_resolved, has_causal_contradiction, causal_follow_up_required) =
        match causal.lineage_outcome {
            EvidenceOutcome::Supported(SupportTier::CausallySupported) => (
                Gwt1EvidenceDispositionV1::CausallySupported,
                EvidenceOutcome::Supported(SupportTier::CausallySupported),
                false,
                false,
            ),
            EvidenceOutcome::NotDemonstrated => (
                Gwt1EvidenceDispositionV1::CausalNotDemonstratedRetainsObserved,
                EvidenceOutcome::Supported(SupportTier::Observed),
                false,
                false,
            ),
            EvidenceOutcome::Contradicted => (
                Gwt1EvidenceDispositionV1::CausalContradictedRetainsObserved,
                EvidenceOutcome::Supported(SupportTier::Observed),
                true,
                false,
            ),
            EvidenceOutcome::Inconclusive => (
                Gwt1EvidenceDispositionV1::CausalInconclusiveRetainsObserved,
                EvidenceOutcome::Supported(SupportTier::Observed),
                false,
                true,
            ),
            observed => {
                return Err(Gwt1EvidenceDispositionErrorV1::InvalidCausalOutcome { observed });
            }
        };

    if causal.resolved_outcome != expected_resolved {
        return Err(Gwt1EvidenceDispositionErrorV1::InvalidCausalTransition {
            base_outcome: causal.base_outcome,
            lineage_outcome: causal.lineage_outcome,
            resolved_outcome: causal.resolved_outcome,
        });
    }

    Ok(Gwt1EvidenceDispositionSummaryV1 {
        schema: GWT1_EVIDENCE_DISPOSITION_SCHEMA_V1.to_string(),
        direct_outcome: direct.lineage_outcome,
        causal_outcome: Some(causal.lineage_outcome),
        resolved_outcome: causal.resolved_outcome,
        disposition,
        has_causal_contradiction,
        causal_follow_up_required,
    })
}

/// Derive a diagnostic GWT-1 disposition from an already resolved V2 view.
///
/// This function does not verify evidence artifacts, attestations, or promotion
/// authority. Those checks belong to the V1/V2 resolution path that produced the
/// view. It only prevents downstream consumers from erasing higher-tier causal
/// null/contradictory/inconclusive information when displaying the retained
/// support floor.
pub fn classify_gwt1_evidence_disposition_v1(
    view: &ButlinResolvedEvidenceViewV2,
) -> Result<Gwt1EvidenceDispositionSummaryV1, Gwt1EvidenceDispositionErrorV1> {
    if view.schema != BUTLIN_RESOLVED_EVIDENCE_VIEW_SCHEMA_V2 {
        return Err(Gwt1EvidenceDispositionErrorV1::WrongViewSchema {
            observed: view.schema.clone(),
        });
    }

    let mut direct = None;
    let mut causal = None;

    for lineage in view
        .lineages
        .iter()
        .filter(|lineage| lineage.indicator_id == "GWT-1")
    {
        match lineage.lineage.kind {
            EvidenceLineageKindV1::DirectQualification => {
                if direct.replace(lineage).is_some() {
                    return Err(Gwt1EvidenceDispositionErrorV1::DuplicateDirectLineage);
                }
            }
            EvidenceLineageKindV1::CausalQualification => {
                if causal.replace(lineage).is_some() {
                    return Err(Gwt1EvidenceDispositionErrorV1::DuplicateCausalLineage);
                }
            }
        }
    }

    let direct = direct.ok_or(Gwt1EvidenceDispositionErrorV1::MissingDirectLineage)?;
    match causal {
        Some(causal) => classify_with_causal(direct, causal),
        None => classify_direct_only(direct),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::resolution_view::{
        EvidenceArtifactIdentityV1, EvidenceAuthorityIdentityV1, EvidenceLineageIdentityV1,
        EvidenceOutcomeCountsV1,
    };

    fn identity(kind: EvidenceLineageKindV1) -> EvidenceLineageIdentityV1 {
        EvidenceLineageIdentityV1 {
            kind,
            method_id: match kind {
                EvidenceLineageKindV1::DirectQualification => "gwt1-direct-v1",
                EvidenceLineageKindV1::CausalQualification => "gwt1-causal-v1",
            }
            .to_string(),
            policy_id: Some("policy-v1".to_string()),
            source_commit_sha: "a".repeat(40),
            source_tree_sha: "b".repeat(40),
            execution_run_id: "123/1".to_string(),
            toolchain: "rustc 1.96.0".to_string(),
            artifact: EvidenceArtifactIdentityV1 {
                schema: "artifact-v1".to_string(),
                digest_algorithm: "sha256".to_string(),
                digest: "c".repeat(64),
                byte_len: 128,
            },
            authority: (kind == EvidenceLineageKindV1::CausalQualification).then(|| {
                EvidenceAuthorityIdentityV1 {
                    repository: "Luminous-Dynamics/symthaea".to_string(),
                    workflow: ".github/workflows/test.yml".to_string(),
                    workflow_sha: "d".repeat(40),
                    attestation_bundle_sha256: "e".repeat(64),
                    attestation_verification_sha256: "f".repeat(64),
                }
            }),
        }
    }

    fn lineage(
        kind: EvidenceLineageKindV1,
        base_outcome: EvidenceOutcome,
        lineage_outcome: EvidenceOutcome,
        resolved_outcome: EvidenceOutcome,
    ) -> IndicatorEvidenceLineageV2 {
        IndicatorEvidenceLineageV2 {
            indicator_id: "GWT-1".to_string(),
            base_outcome,
            lineage_outcome,
            resolved_outcome,
            lineage: identity(kind),
        }
    }

    fn view(lineages: Vec<IndicatorEvidenceLineageV2>) -> ButlinResolvedEvidenceViewV2 {
        ButlinResolvedEvidenceViewV2 {
            schema: BUTLIN_RESOLVED_EVIDENCE_VIEW_SCHEMA_V2.to_string(),
            base_report_schema_version: 3,
            base_report_blake3: "0".repeat(64),
            lineages,
            resolved_counts: EvidenceOutcomeCountsV1::default(),
        }
    }

    fn direct_observed() -> IndicatorEvidenceLineageV2 {
        lineage(
            EvidenceLineageKindV1::DirectQualification,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::Supported(SupportTier::Observed),
            EvidenceOutcome::Supported(SupportTier::Observed),
        )
    }

    #[test]
    fn direct_observed_without_causal_lineage_is_explicitly_pending() {
        let summary = classify_gwt1_evidence_disposition_v1(&view(vec![direct_observed()]))
            .expect("valid direct-only disposition");
        assert_eq!(
            summary.disposition,
            Gwt1EvidenceDispositionV1::DirectObservedPendingCausal
        );
        assert_eq!(summary.causal_outcome, None);
        assert!(!summary.has_causal_contradiction);
    }

    #[test]
    fn causal_support_is_distinct_from_retained_observed() {
        let summary = classify_gwt1_evidence_disposition_v1(&view(vec![
            direct_observed(),
            lineage(
                EvidenceLineageKindV1::CausalQualification,
                EvidenceOutcome::Supported(SupportTier::Observed),
                EvidenceOutcome::Supported(SupportTier::CausallySupported),
                EvidenceOutcome::Supported(SupportTier::CausallySupported),
            ),
        ]))
        .expect("valid positive causal disposition");
        assert_eq!(summary.disposition, Gwt1EvidenceDispositionV1::CausallySupported);
        assert_eq!(
            summary.resolved_outcome,
            EvidenceOutcome::Supported(SupportTier::CausallySupported)
        );
    }

    #[test]
    fn causal_not_demonstrated_retains_observed_without_becoming_pending() {
        let summary = classify_gwt1_evidence_disposition_v1(&view(vec![
            direct_observed(),
            lineage(
                EvidenceLineageKindV1::CausalQualification,
                EvidenceOutcome::Supported(SupportTier::Observed),
                EvidenceOutcome::NotDemonstrated,
                EvidenceOutcome::Supported(SupportTier::Observed),
            ),
        ]))
        .expect("valid causal null disposition");
        assert_eq!(
            summary.disposition,
            Gwt1EvidenceDispositionV1::CausalNotDemonstratedRetainsObserved
        );
        assert_eq!(summary.causal_outcome, Some(EvidenceOutcome::NotDemonstrated));
        assert!(!summary.has_causal_contradiction);
        assert!(!summary.causal_follow_up_required);
    }

    #[test]
    fn causal_contradiction_is_never_hidden_by_retained_observed() {
        let summary = classify_gwt1_evidence_disposition_v1(&view(vec![
            direct_observed(),
            lineage(
                EvidenceLineageKindV1::CausalQualification,
                EvidenceOutcome::Supported(SupportTier::Observed),
                EvidenceOutcome::Contradicted,
                EvidenceOutcome::Supported(SupportTier::Observed),
            ),
        ]))
        .expect("valid causal contradiction disposition");
        assert_eq!(
            summary.disposition,
            Gwt1EvidenceDispositionV1::CausalContradictedRetainsObserved
        );
        assert!(summary.has_causal_contradiction);
        assert_eq!(
            summary.resolved_outcome,
            EvidenceOutcome::Supported(SupportTier::Observed)
        );
    }

    #[test]
    fn causal_inconclusive_is_explicit_follow_up_state() {
        let summary = classify_gwt1_evidence_disposition_v1(&view(vec![
            direct_observed(),
            lineage(
                EvidenceLineageKindV1::CausalQualification,
                EvidenceOutcome::Supported(SupportTier::Observed),
                EvidenceOutcome::Inconclusive,
                EvidenceOutcome::Supported(SupportTier::Observed),
            ),
        ]))
        .expect("valid causal inconclusive disposition");
        assert_eq!(
            summary.disposition,
            Gwt1EvidenceDispositionV1::CausalInconclusiveRetainsObserved
        );
        assert!(summary.causal_follow_up_required);
    }

    #[test]
    fn direct_negative_states_remain_distinct() {
        for (outcome, disposition) in [
            (
                EvidenceOutcome::NotDemonstrated,
                Gwt1EvidenceDispositionV1::DirectNotDemonstrated,
            ),
            (
                EvidenceOutcome::Contradicted,
                Gwt1EvidenceDispositionV1::DirectContradicted,
            ),
            (
                EvidenceOutcome::Inconclusive,
                Gwt1EvidenceDispositionV1::DirectInconclusive,
            ),
        ] {
            let summary = classify_gwt1_evidence_disposition_v1(&view(vec![lineage(
                EvidenceLineageKindV1::DirectQualification,
                EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
                outcome,
                outcome,
            )]))
            .expect("valid direct negative disposition");
            assert_eq!(summary.disposition, disposition);
        }
    }

    #[test]
    fn positive_causal_outcome_cannot_leapfrog_direct_observed() {
        let result = classify_gwt1_evidence_disposition_v1(&view(vec![
            lineage(
                EvidenceLineageKindV1::DirectQualification,
                EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
                EvidenceOutcome::NotDemonstrated,
                EvidenceOutcome::NotDemonstrated,
            ),
            lineage(
                EvidenceLineageKindV1::CausalQualification,
                EvidenceOutcome::NotDemonstrated,
                EvidenceOutcome::Supported(SupportTier::CausallySupported),
                EvidenceOutcome::Supported(SupportTier::CausallySupported),
            ),
        ]));
        assert!(matches!(
            result,
            Err(Gwt1EvidenceDispositionErrorV1::CausalRequiresDirectObserved { .. })
        ));
    }

    #[test]
    fn duplicate_method_lineages_fail_closed() {
        let result = classify_gwt1_evidence_disposition_v1(&view(vec![
            direct_observed(),
            direct_observed(),
        ]));
        assert_eq!(
            result,
            Err(Gwt1EvidenceDispositionErrorV1::DuplicateDirectLineage)
        );
    }

    #[test]
    fn impossible_supported_tiers_are_rejected() {
        let result = classify_gwt1_evidence_disposition_v1(&view(vec![lineage(
            EvidenceLineageKindV1::DirectQualification,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::Supported(SupportTier::CausallySupported),
            EvidenceOutcome::Supported(SupportTier::CausallySupported),
        )]));
        assert!(matches!(
            result,
            Err(Gwt1EvidenceDispositionErrorV1::InvalidDirectOutcome { .. })
        ));
    }
}
