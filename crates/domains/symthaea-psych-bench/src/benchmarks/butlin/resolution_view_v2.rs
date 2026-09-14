// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! V2 non-destructive multi-lineage evidence resolution.
//!
//! V1 represented each lineage as a transition whose output became the next
//! indicator outcome. That is correct for positive tier promotion, but too
//! destructive for a negative higher-tier experiment: failure to demonstrate
//! causality does not erase independently established direct observation.
//!
//! V2 therefore records two distinct quantities for every lineage:
//!
//! - `lineage_outcome`: what that evidence method itself established;
//! - `resolved_outcome`: the strongest indicator state after incorporating it.
//!
//! A negative causal lineage remains visible as negative science while a prior
//! direct `Observed` result is retained. Only a positively verified causal
//! lineage may raise the resolved indicator to `CausallySupported`.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::report::{ButlinIndicatorReport, EvidenceOutcome, SupportTier};
use super::resolution_view::{
    EvidenceLineageIdentityV1, EvidenceLineageKindV1, EvidenceOutcomeCountsV1,
    EvidenceResolutionViewErrorV1, IndicatorOutcomeOverlayV1, base_report_blake3_v1,
};

pub const BUTLIN_RESOLVED_EVIDENCE_VIEW_SCHEMA_V2: &str =
    "butlin-resolved-evidence-view-v2";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndicatorEvidenceLineageV2 {
    pub indicator_id: String,
    /// Resolved indicator outcome immediately before this lineage is applied.
    pub base_outcome: EvidenceOutcome,
    /// Scientific result of this lineage itself.
    pub lineage_outcome: EvidenceOutcome,
    /// Strongest supported indicator outcome after incorporating this lineage.
    pub resolved_outcome: EvidenceOutcome,
    pub lineage: EvidenceLineageIdentityV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ButlinResolvedEvidenceViewV2 {
    pub schema: String,
    pub base_report_schema_version: u32,
    pub base_report_blake3: String,
    /// Ordered by indicator then evidence-lineage rank.
    pub lineages: Vec<IndicatorEvidenceLineageV2>,
    pub resolved_counts: EvidenceOutcomeCountsV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceResolutionViewErrorV2 {
    V1(EvidenceResolutionViewErrorV1),
    DuplicateBaseIndicator {
        indicator_id: String,
    },
    UnknownIndicator {
        indicator_id: String,
    },
    DuplicateLineage {
        indicator_id: String,
        kind: EvidenceLineageKindV1,
    },
    LineageInputMismatch {
        indicator_id: String,
        kind: EvidenceLineageKindV1,
        expected: EvidenceOutcome,
        observed: EvidenceOutcome,
    },
    InvalidDirectResolution {
        lineage_outcome: EvidenceOutcome,
        resolved_outcome: EvidenceOutcome,
    },
    InvalidCausalResolution {
        base_outcome: EvidenceOutcome,
        lineage_outcome: EvidenceOutcome,
        resolved_outcome: EvidenceOutcome,
    },
    MissingExpectedLineage {
        kind: EvidenceLineageKindV1,
    },
}

impl From<EvidenceResolutionViewErrorV1> for EvidenceResolutionViewErrorV2 {
    fn from(value: EvidenceResolutionViewErrorV1) -> Self {
        Self::V1(value)
    }
}

impl std::fmt::Display for EvidenceResolutionViewErrorV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V1(error) => write!(f, "V1 evidence verification failed: {error}"),
            Self::DuplicateBaseIndicator { indicator_id } => {
                write!(f, "base report contains duplicate indicator {indicator_id:?}")
            }
            Self::UnknownIndicator { indicator_id } => {
                write!(f, "evidence lineage targets unknown indicator {indicator_id:?}")
            }
            Self::DuplicateLineage { indicator_id, kind } => write!(
                f,
                "multiple {kind:?} lineages target indicator {indicator_id:?}"
            ),
            Self::LineageInputMismatch {
                indicator_id,
                kind,
                expected,
                observed,
            } => write!(
                f,
                "{kind:?} lineage for {indicator_id:?} expects prior outcome {expected:?}, but the resolved chain currently has {observed:?}"
            ),
            Self::InvalidDirectResolution {
                lineage_outcome,
                resolved_outcome,
            } => write!(
                f,
                "direct lineage resolution is invalid: lineage={lineage_outcome:?}, resolved={resolved_outcome:?}"
            ),
            Self::InvalidCausalResolution {
                base_outcome,
                lineage_outcome,
                resolved_outcome,
            } => write!(
                f,
                "causal lineage resolution is invalid: base={base_outcome:?}, lineage={lineage_outcome:?}, resolved={resolved_outcome:?}"
            ),
            Self::MissingExpectedLineage { kind } => {
                write!(f, "verified V1 view did not contain expected {kind:?} lineage")
            }
        }
    }
}

impl std::error::Error for EvidenceResolutionViewErrorV2 {}

const fn lineage_rank(kind: EvidenceLineageKindV1) -> u8 {
    match kind {
        EvidenceLineageKindV1::DirectQualification => 10,
        EvidenceLineageKindV1::CausalQualification => 20,
    }
}

fn validate_transition(
    item: &IndicatorEvidenceLineageV2,
) -> Result<(), EvidenceResolutionViewErrorV2> {
    match item.lineage.kind {
        EvidenceLineageKindV1::DirectQualification => match item.lineage_outcome {
            EvidenceOutcome::Supported(SupportTier::Observed) => {
                if item.resolved_outcome != EvidenceOutcome::Supported(SupportTier::Observed) {
                    return Err(EvidenceResolutionViewErrorV2::InvalidDirectResolution {
                        lineage_outcome: item.lineage_outcome,
                        resolved_outcome: item.resolved_outcome,
                    });
                }
            }
            EvidenceOutcome::NotDemonstrated
            | EvidenceOutcome::Contradicted
            | EvidenceOutcome::Inconclusive => {
                if item.resolved_outcome != item.base_outcome {
                    return Err(EvidenceResolutionViewErrorV2::InvalidDirectResolution {
                        lineage_outcome: item.lineage_outcome,
                        resolved_outcome: item.resolved_outcome,
                    });
                }
            }
            EvidenceOutcome::Supported(_) => {
                return Err(EvidenceResolutionViewErrorV2::InvalidDirectResolution {
                    lineage_outcome: item.lineage_outcome,
                    resolved_outcome: item.resolved_outcome,
                });
            }
        },
        EvidenceLineageKindV1::CausalQualification => match item.lineage_outcome {
            EvidenceOutcome::Supported(SupportTier::CausallySupported) => {
                if item.base_outcome != EvidenceOutcome::Supported(SupportTier::Observed)
                    || item.resolved_outcome
                        != EvidenceOutcome::Supported(SupportTier::CausallySupported)
                {
                    return Err(EvidenceResolutionViewErrorV2::InvalidCausalResolution {
                        base_outcome: item.base_outcome,
                        lineage_outcome: item.lineage_outcome,
                        resolved_outcome: item.resolved_outcome,
                    });
                }
            }
            EvidenceOutcome::NotDemonstrated
            | EvidenceOutcome::Contradicted
            | EvidenceOutcome::Inconclusive => {
                if item.resolved_outcome != item.base_outcome {
                    return Err(EvidenceResolutionViewErrorV2::InvalidCausalResolution {
                        base_outcome: item.base_outcome,
                        lineage_outcome: item.lineage_outcome,
                        resolved_outcome: item.resolved_outcome,
                    });
                }
            }
            EvidenceOutcome::Supported(_) => {
                return Err(EvidenceResolutionViewErrorV2::InvalidCausalResolution {
                    base_outcome: item.base_outcome,
                    lineage_outcome: item.lineage_outcome,
                    resolved_outcome: item.resolved_outcome,
                });
            }
        },
    }
    Ok(())
}

fn resolve_validated_lineages_v2(
    report: &ButlinIndicatorReport,
    mut lineages: Vec<IndicatorEvidenceLineageV2>,
) -> Result<ButlinResolvedEvidenceViewV2, EvidenceResolutionViewErrorV2> {
    let mut current_outcomes = BTreeMap::new();
    for indicator in &report.indicators {
        if current_outcomes
            .insert(indicator.id.clone(), indicator.outcome)
            .is_some()
        {
            return Err(EvidenceResolutionViewErrorV2::DuplicateBaseIndicator {
                indicator_id: indicator.id.clone(),
            });
        }
    }

    lineages.sort_by(|a, b| {
        a.indicator_id
            .cmp(&b.indicator_id)
            .then_with(|| lineage_rank(a.lineage.kind).cmp(&lineage_rank(b.lineage.kind)))
    });

    let mut seen = BTreeSet::new();
    for item in &lineages {
        let key = (item.indicator_id.clone(), item.lineage.kind);
        if !seen.insert(key) {
            return Err(EvidenceResolutionViewErrorV2::DuplicateLineage {
                indicator_id: item.indicator_id.clone(),
                kind: item.lineage.kind,
            });
        }

        let observed = current_outcomes
            .get(&item.indicator_id)
            .copied()
            .ok_or_else(|| EvidenceResolutionViewErrorV2::UnknownIndicator {
                indicator_id: item.indicator_id.clone(),
            })?;
        if observed != item.base_outcome {
            return Err(EvidenceResolutionViewErrorV2::LineageInputMismatch {
                indicator_id: item.indicator_id.clone(),
                kind: item.lineage.kind,
                expected: item.base_outcome,
                observed,
            });
        }

        validate_transition(item)?;
        current_outcomes.insert(item.indicator_id.clone(), item.resolved_outcome);
    }

    let mut resolved_counts = EvidenceOutcomeCountsV1::default();
    for indicator in &report.indicators {
        let outcome = current_outcomes
            .get(&indicator.id)
            .copied()
            .unwrap_or(indicator.outcome);
        match outcome {
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly) => {
                resolved_counts.architectural_only += 1;
            }
            EvidenceOutcome::Supported(SupportTier::Observed) => {
                resolved_counts.observed += 1;
            }
            EvidenceOutcome::Supported(SupportTier::CausallySupported) => {
                resolved_counts.causally_supported += 1;
            }
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported) => {
                resolved_counts.functionally_supported += 1;
            }
            EvidenceOutcome::NotDemonstrated => resolved_counts.not_demonstrated += 1,
            EvidenceOutcome::Contradicted => resolved_counts.contradicted += 1,
            EvidenceOutcome::Inconclusive => resolved_counts.inconclusive += 1,
        }
    }

    Ok(ButlinResolvedEvidenceViewV2 {
        schema: BUTLIN_RESOLVED_EVIDENCE_VIEW_SCHEMA_V2.to_string(),
        base_report_schema_version: report.schema_version,
        base_report_blake3: base_report_blake3_v1(report)?,
        lineages,
        resolved_counts,
    })
}

fn direct_lineage_from_v1(direct: IndicatorOutcomeOverlayV1) -> IndicatorEvidenceLineageV2 {
    IndicatorEvidenceLineageV2 {
        indicator_id: direct.indicator_id,
        base_outcome: direct.base_outcome,
        lineage_outcome: direct.lineage_outcome,
        resolved_outcome: direct.resolved_outcome,
        lineage: direct.lineage,
    }
}

fn causal_lineage_from_v1(
    direct: &IndicatorEvidenceLineageV2,
    causal: IndicatorOutcomeOverlayV1,
) -> Result<IndicatorEvidenceLineageV2, EvidenceResolutionViewErrorV2> {
    let lineage_outcome = causal.lineage_outcome;
    let resolved_outcome = match lineage_outcome {
        EvidenceOutcome::Supported(SupportTier::CausallySupported) => {
            if direct.resolved_outcome != EvidenceOutcome::Supported(SupportTier::Observed) {
                return Err(EvidenceResolutionViewErrorV2::InvalidCausalResolution {
                    base_outcome: direct.resolved_outcome,
                    lineage_outcome,
                    resolved_outcome: causal.resolved_outcome,
                });
            }
            causal.resolved_outcome
        }
        EvidenceOutcome::NotDemonstrated
        | EvidenceOutcome::Contradicted
        | EvidenceOutcome::Inconclusive => direct.resolved_outcome,
        EvidenceOutcome::Supported(_) => {
            return Err(EvidenceResolutionViewErrorV2::InvalidCausalResolution {
                base_outcome: direct.resolved_outcome,
                lineage_outcome,
                resolved_outcome: causal.resolved_outcome,
            });
        }
    };

    Ok(IndicatorEvidenceLineageV2 {
        indicator_id: causal.indicator_id,
        base_outcome: direct.resolved_outcome,
        lineage_outcome,
        resolved_outcome,
        lineage: causal.lineage,
    })
}

#[cfg(feature = "symthaea-backend")]
pub fn resolve_gwt1_evidence_view_v2(
    report: &ButlinIndicatorReport,
    evidence: &super::gwt1_end_to_end::Gwt1EndToEndEvidenceV1,
) -> Result<ButlinResolvedEvidenceViewV2, EvidenceResolutionViewErrorV2> {
    let verified_v1 = super::resolution_view::resolve_gwt1_evidence_view_v1(report, evidence)?;
    let direct = verified_v1
        .overlays
        .into_iter()
        .find(|item| item.lineage.kind == EvidenceLineageKindV1::DirectQualification)
        .ok_or(EvidenceResolutionViewErrorV2::MissingExpectedLineage {
            kind: EvidenceLineageKindV1::DirectQualification,
        })?;

    resolve_validated_lineages_v2(report, vec![direct_lineage_from_v1(direct)])
}

#[cfg(feature = "symthaea-backend")]
pub fn resolve_gwt1_causal_evidence_view_v2(
    report: &ButlinIndicatorReport,
    direct_evidence: &super::gwt1_end_to_end::Gwt1EndToEndEvidenceV1,
    verified_causal: &super::gwt1_causal_verified_promotion::VerifiedGwt1CausalPromotionV1,
) -> Result<ButlinResolvedEvidenceViewV2, EvidenceResolutionViewErrorV2> {
    // V1 remains the authority/verifier path. V2 changes only the resolution
    // semantics after both evidence methods have passed V1 validation.
    let verified_v1 = super::resolution_view::resolve_gwt1_causal_evidence_view_v1(
        report,
        direct_evidence,
        verified_causal,
    )?;

    let mut direct = None;
    let mut causal = None;
    for item in verified_v1.overlays {
        match item.lineage.kind {
            EvidenceLineageKindV1::DirectQualification => direct = Some(item),
            EvidenceLineageKindV1::CausalQualification => causal = Some(item),
        }
    }

    let direct = direct.ok_or(EvidenceResolutionViewErrorV2::MissingExpectedLineage {
        kind: EvidenceLineageKindV1::DirectQualification,
    })?;
    let causal = causal.ok_or(EvidenceResolutionViewErrorV2::MissingExpectedLineage {
        kind: EvidenceLineageKindV1::CausalQualification,
    })?;

    let direct_v2 = direct_lineage_from_v1(direct);
    let causal_v2 = causal_lineage_from_v1(&direct_v2, causal)?;

    resolve_validated_lineages_v2(report, vec![direct_v2, causal_v2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::resolution_view::{
        EvidenceArtifactIdentityV1, EvidenceAuthorityIdentityV1,
    };
    use crate::benchmarks::butlin::report::{EvidenceAnnotation, IndicatorEvidence};

    fn indicator(id: &str, outcome: EvidenceOutcome) -> IndicatorEvidence {
        IndicatorEvidence {
            id: id.to_string(),
            theory: "test theory".to_string(),
            description: "test indicator".to_string(),
            outcome,
            evidence: "test evidence".to_string(),
            architectural_score: 0.5,
            live_score: Some(0.5),
            probe_quality: None,
            causal_effect: None,
            functional_effect: None,
            annotations: Vec::<EvidenceAnnotation>::new(),
        }
    }

    fn report() -> ButlinIndicatorReport {
        ButlinIndicatorReport::from_indicators(vec![
            indicator(
                "GWT-1",
                EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            ),
            indicator("AE-1", EvidenceOutcome::NotDemonstrated),
        ])
    }

    fn identity(kind: EvidenceLineageKindV1) -> EvidenceLineageIdentityV1 {
        EvidenceLineageIdentityV1 {
            kind,
            method_id: "method-v1".to_string(),
            policy_id: Some("policy-v1".to_string()),
            source_commit_sha: "a".repeat(40),
            source_tree_sha: "b".repeat(40),
            execution_run_id: "run-1".to_string(),
            toolchain: "rustc test".to_string(),
            artifact: EvidenceArtifactIdentityV1 {
                schema: "artifact-v1".to_string(),
                digest_algorithm: if kind == EvidenceLineageKindV1::DirectQualification {
                    "blake3".to_string()
                } else {
                    "sha256".to_string()
                },
                digest: "c".repeat(64),
                byte_len: 64,
            },
            authority: if kind == EvidenceLineageKindV1::CausalQualification {
                Some(EvidenceAuthorityIdentityV1 {
                    repository: "Luminous-Dynamics/symthaea".to_string(),
                    workflow: ".github/workflows/trusted.yml".to_string(),
                    workflow_sha: "d".repeat(40),
                    attestation_bundle_sha256: "e".repeat(64),
                    attestation_verification_sha256: "f".repeat(64),
                })
            } else {
                None
            },
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

    fn overlay_v1(
        kind: EvidenceLineageKindV1,
        base_outcome: EvidenceOutcome,
        lineage_outcome: EvidenceOutcome,
        resolved_outcome: EvidenceOutcome,
    ) -> IndicatorOutcomeOverlayV1 {
        IndicatorOutcomeOverlayV1 {
            indicator_id: "GWT-1".to_string(),
            base_outcome,
            lineage_outcome,
            resolved_outcome,
            lineage: identity(kind),
        }
    }

    fn observed_direct() -> IndicatorEvidenceLineageV2 {
        lineage(
            EvidenceLineageKindV1::DirectQualification,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::Supported(SupportTier::Observed),
            EvidenceOutcome::Supported(SupportTier::Observed),
        )
    }

    #[test]
    fn v1_direct_negative_conversion_preserves_method_outcome_and_support_floor() {
        for outcome in [
            EvidenceOutcome::NotDemonstrated,
            EvidenceOutcome::Contradicted,
            EvidenceOutcome::Inconclusive,
        ] {
            let converted = direct_lineage_from_v1(overlay_v1(
                EvidenceLineageKindV1::DirectQualification,
                EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
                outcome,
                EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            ));
            assert_eq!(converted.lineage_outcome, outcome);
            assert_eq!(
                converted.resolved_outcome,
                EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly)
            );
            validate_transition(&converted).expect("valid converted direct negative");
        }
    }

    #[test]
    fn v1_causal_negative_conversion_preserves_method_outcome_and_observed_floor() {
        let direct = observed_direct();
        for outcome in [
            EvidenceOutcome::NotDemonstrated,
            EvidenceOutcome::Contradicted,
            EvidenceOutcome::Inconclusive,
        ] {
            let converted = causal_lineage_from_v1(
                &direct,
                overlay_v1(
                    EvidenceLineageKindV1::CausalQualification,
                    EvidenceOutcome::Supported(SupportTier::Observed),
                    outcome,
                    EvidenceOutcome::Supported(SupportTier::Observed),
                ),
            )
            .expect("valid converted causal negative");
            assert_eq!(converted.lineage_outcome, outcome);
            assert_eq!(
                converted.resolved_outcome,
                EvidenceOutcome::Supported(SupportTier::Observed)
            );
            validate_transition(&converted).expect("valid causal transition");
        }
    }

    #[test]
    fn paired_negative_methods_preserve_both_results_and_architectural_floor() {
        let direct = direct_lineage_from_v1(overlay_v1(
            EvidenceLineageKindV1::DirectQualification,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::NotDemonstrated,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        ));
        let causal = causal_lineage_from_v1(
            &direct,
            overlay_v1(
                EvidenceLineageKindV1::CausalQualification,
                EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
                EvidenceOutcome::Contradicted,
                EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            ),
        )
        .expect("negative causal result does not require direct Observed");
        let view = resolve_validated_lineages_v2(&report(), vec![direct, causal])
            .expect("paired negative methods remain representable");
        assert_eq!(view.lineages[0].lineage_outcome, EvidenceOutcome::NotDemonstrated);
        assert_eq!(view.lineages[1].lineage_outcome, EvidenceOutcome::Contradicted);
        assert_eq!(view.resolved_counts.architectural_only, 1);
    }

    #[test]
    fn direct_architectural_only_cannot_masquerade_as_method_result() {
        let invalid = lineage(
            EvidenceLineageKindV1::DirectQualification,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        );
        assert!(matches!(
            resolve_validated_lineages_v2(&report(), vec![invalid]),
            Err(EvidenceResolutionViewErrorV2::InvalidDirectResolution { .. })
        ));
    }

    #[test]
    fn positive_causal_lineage_raises_observed_to_causal() {
        let causal = lineage(
            EvidenceLineageKindV1::CausalQualification,
            EvidenceOutcome::Supported(SupportTier::Observed),
            EvidenceOutcome::Supported(SupportTier::CausallySupported),
            EvidenceOutcome::Supported(SupportTier::CausallySupported),
        );
        let view = resolve_validated_lineages_v2(&report(), vec![causal, observed_direct()])
            .expect("resolved causal chain");
        assert_eq!(view.resolved_counts.causally_supported, 1);
        assert_eq!(
            view.lineages[1].lineage_outcome,
            EvidenceOutcome::Supported(SupportTier::CausallySupported)
        );
    }

    #[test]
    fn causal_not_demonstrated_is_retained_without_erasing_observed() {
        let causal = lineage(
            EvidenceLineageKindV1::CausalQualification,
            EvidenceOutcome::Supported(SupportTier::Observed),
            EvidenceOutcome::NotDemonstrated,
            EvidenceOutcome::Supported(SupportTier::Observed),
        );
        let view = resolve_validated_lineages_v2(&report(), vec![observed_direct(), causal])
            .expect("resolved causal null");
        assert_eq!(view.resolved_counts.observed, 1);
        assert_eq!(view.lineages[1].lineage_outcome, EvidenceOutcome::NotDemonstrated);
        assert_eq!(
            view.lineages[1].resolved_outcome,
            EvidenceOutcome::Supported(SupportTier::Observed)
        );
    }

    #[test]
    fn causal_contradiction_is_retained_without_erasing_observed() {
        let causal = lineage(
            EvidenceLineageKindV1::CausalQualification,
            EvidenceOutcome::Supported(SupportTier::Observed),
            EvidenceOutcome::Contradicted,
            EvidenceOutcome::Supported(SupportTier::Observed),
        );
        let view = resolve_validated_lineages_v2(&report(), vec![observed_direct(), causal])
            .expect("resolved contradiction");
        assert_eq!(view.resolved_counts.observed, 1);
        assert_eq!(view.lineages[1].lineage_outcome, EvidenceOutcome::Contradicted);
    }

    #[test]
    fn causal_inconclusive_is_retained_without_erasing_observed() {
        let causal = lineage(
            EvidenceLineageKindV1::CausalQualification,
            EvidenceOutcome::Supported(SupportTier::Observed),
            EvidenceOutcome::Inconclusive,
            EvidenceOutcome::Supported(SupportTier::Observed),
        );
        let view = resolve_validated_lineages_v2(&report(), vec![observed_direct(), causal])
            .expect("resolved inconclusive");
        assert_eq!(view.resolved_counts.observed, 1);
        assert_eq!(view.lineages[1].lineage_outcome, EvidenceOutcome::Inconclusive);
    }

    #[test]
    fn negative_causal_lineage_cannot_demote_lower_tier() {
        let invalid = lineage(
            EvidenceLineageKindV1::CausalQualification,
            EvidenceOutcome::Supported(SupportTier::Observed),
            EvidenceOutcome::NotDemonstrated,
            EvidenceOutcome::NotDemonstrated,
        );
        assert!(matches!(
            resolve_validated_lineages_v2(&report(), vec![observed_direct(), invalid]),
            Err(EvidenceResolutionViewErrorV2::InvalidCausalResolution { .. })
        ));
    }

    #[test]
    fn positive_causal_lineage_requires_observed_base() {
        let direct = lineage(
            EvidenceLineageKindV1::DirectQualification,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::NotDemonstrated,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        );
        let causal = lineage(
            EvidenceLineageKindV1::CausalQualification,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::Supported(SupportTier::CausallySupported),
            EvidenceOutcome::Supported(SupportTier::CausallySupported),
        );
        assert!(matches!(
            resolve_validated_lineages_v2(&report(), vec![direct, causal]),
            Err(EvidenceResolutionViewErrorV2::InvalidCausalResolution { .. })
        ));
    }

    #[test]
    fn duplicate_lineage_is_rejected() {
        let direct = observed_direct();
        assert!(matches!(
            resolve_validated_lineages_v2(&report(), vec![direct.clone(), direct]),
            Err(EvidenceResolutionViewErrorV2::DuplicateLineage { .. })
        ));
    }
}
