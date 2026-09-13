// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Non-destructive evidence resolution over a base Butlin report.
//!
//! `ButlinIndicatorReport` remains the historical architectural/live-probe
//! snapshot. Independent evidence lineages (for example the direct GWT-1
//! qualification) are represented as typed overlays rather than being forced
//! into scalar `live_score`/`ProbeQuality` fields that do not fit them.
//!
//! The resolved view is cryptographically bound to the exact serialized base
//! report and recomputes tier counts without mutating the report itself.
//!
//! Crucially, the generic overlay resolver is private. Public callers cannot
//! submit an arbitrary `IndicatorOutcomeOverlayV1` and ask first-party code to
//! bless its tier counts. Each public resolver must start from the actual typed
//! evidence object for its method and construct the overlay internally.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::report::{ButlinIndicatorReport, EvidenceOutcome, SupportTier};

pub const BUTLIN_RESOLVED_EVIDENCE_VIEW_SCHEMA_V1: &str =
    "butlin-resolved-evidence-view-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceLineageKindV1 {
    DirectQualification,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLineageIdentityV1 {
    pub kind: EvidenceLineageKindV1,
    pub method_id: String,
    pub policy_id: Option<String>,
    pub source_commit_sha: String,
    pub source_tree_sha: String,
    pub execution_run_id: String,
    pub toolchain: String,
    pub raw_artifact_schema: String,
    pub raw_artifact_blake3: String,
    pub raw_artifact_byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndicatorOutcomeOverlayV1 {
    pub indicator_id: String,
    /// Outcome in the exact base report this overlay was built against.
    pub base_outcome: EvidenceOutcome,
    /// Outcome after resolving the independent lineage.
    pub resolved_outcome: EvidenceOutcome,
    pub lineage: EvidenceLineageIdentityV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct EvidenceOutcomeCountsV1 {
    pub architectural_only: usize,
    pub observed: usize,
    pub causally_supported: usize,
    pub functionally_supported: usize,
    pub not_demonstrated: usize,
    pub contradicted: usize,
    pub inconclusive: usize,
}

impl EvidenceOutcomeCountsV1 {
    fn record(&mut self, outcome: EvidenceOutcome) {
        match outcome {
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly) => {
                self.architectural_only += 1;
            }
            EvidenceOutcome::Supported(SupportTier::Observed) => self.observed += 1,
            EvidenceOutcome::Supported(SupportTier::CausallySupported) => {
                self.causally_supported += 1;
            }
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported) => {
                self.functionally_supported += 1;
            }
            EvidenceOutcome::NotDemonstrated => self.not_demonstrated += 1,
            EvidenceOutcome::Contradicted => self.contradicted += 1,
            EvidenceOutcome::Inconclusive => self.inconclusive += 1,
        }
    }

    pub fn total(&self) -> usize {
        self.architectural_only
            + self.observed
            + self.causally_supported
            + self.functionally_supported
            + self.not_demonstrated
            + self.contradicted
            + self.inconclusive
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ButlinResolvedEvidenceViewV1 {
    pub schema: String,
    pub base_report_schema_version: u32,
    pub base_report_blake3: String,
    /// Deterministic indicator-id ordering. These are derived records, not
    /// authority-bearing inputs to a public generic resolver.
    pub overlays: Vec<IndicatorOutcomeOverlayV1>,
    pub resolved_counts: EvidenceOutcomeCountsV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceResolutionViewErrorV1 {
    BaseSerialization(String),
    DuplicateBaseIndicator { indicator_id: String },
    UnknownIndicator { indicator_id: String },
    DuplicateOverlay { indicator_id: String },
    BaseOutcomeMismatch {
        indicator_id: String,
        expected: EvidenceOutcome,
        observed: EvidenceOutcome,
    },
    InvalidLineageIdentity {
        indicator_id: String,
        field: String,
    },
    DirectEvidenceResolutionMismatch { indicator_id: String },
}

impl std::fmt::Display for EvidenceResolutionViewErrorV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BaseSerialization(error) => write!(f, "failed to serialize base report: {error}"),
            Self::DuplicateBaseIndicator { indicator_id } => {
                write!(f, "base report contains duplicate indicator {indicator_id:?}")
            }
            Self::UnknownIndicator { indicator_id } => {
                write!(f, "overlay targets unknown indicator {indicator_id:?}")
            }
            Self::DuplicateOverlay { indicator_id } => {
                write!(f, "multiple overlays target indicator {indicator_id:?}")
            }
            Self::BaseOutcomeMismatch {
                indicator_id,
                expected,
                observed,
            } => write!(
                f,
                "overlay for {indicator_id:?} was built against base outcome {expected:?}, but the supplied report contains {observed:?}"
            ),
            Self::InvalidLineageIdentity {
                indicator_id,
                field,
            } => write!(
                f,
                "overlay for {indicator_id:?} has invalid lineage field {field:?}"
            ),
            Self::DirectEvidenceResolutionMismatch { indicator_id } => write!(
                f,
                "direct evidence for {indicator_id:?} carries a cached resolution that does not match recomputation from its envelope and raw bytes"
            ),
        }
    }
}

impl std::error::Error for EvidenceResolutionViewErrorV1 {}

fn is_hex_len(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_lower_hex_len(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_lineage(
    overlay: &IndicatorOutcomeOverlayV1,
) -> Result<(), EvidenceResolutionViewErrorV1> {
    let invalid = |field: &str| EvidenceResolutionViewErrorV1::InvalidLineageIdentity {
        indicator_id: overlay.indicator_id.clone(),
        field: field.to_string(),
    };

    if overlay.lineage.method_id.trim().is_empty() {
        return Err(invalid("method_id"));
    }
    if overlay.lineage.kind == EvidenceLineageKindV1::DirectQualification
        && overlay
            .lineage
            .policy_id
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
    {
        return Err(invalid("policy_id"));
    }
    if overlay.lineage.kind == EvidenceLineageKindV1::DirectQualification
        && matches!(
            overlay.resolved_outcome,
            EvidenceOutcome::Supported(SupportTier::CausallySupported)
                | EvidenceOutcome::Supported(SupportTier::FunctionallySupported)
        )
    {
        return Err(invalid("resolved_outcome"));
    }
    if !is_hex_len(&overlay.lineage.source_commit_sha, 40) {
        return Err(invalid("source_commit_sha"));
    }
    if !is_hex_len(&overlay.lineage.source_tree_sha, 40) {
        return Err(invalid("source_tree_sha"));
    }
    if overlay.lineage.execution_run_id.trim().is_empty() {
        return Err(invalid("execution_run_id"));
    }
    if overlay.lineage.toolchain.trim().is_empty() {
        return Err(invalid("toolchain"));
    }
    if overlay.lineage.raw_artifact_schema.trim().is_empty() {
        return Err(invalid("raw_artifact_schema"));
    }
    if !is_lower_hex_len(&overlay.lineage.raw_artifact_blake3, 64) {
        return Err(invalid("raw_artifact_blake3"));
    }
    if overlay.lineage.raw_artifact_byte_len == 0 {
        return Err(invalid("raw_artifact_byte_len"));
    }

    Ok(())
}

pub fn base_report_blake3_v1(
    report: &ButlinIndicatorReport,
) -> Result<String, EvidenceResolutionViewErrorV1> {
    let bytes = serde_json::to_vec(report)
        .map_err(|error| EvidenceResolutionViewErrorV1::BaseSerialization(error.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

/// Internal-only generic resolver. Public callers must use a method-specific
/// resolver that starts from its typed evidence object and constructs overlays
/// after independently verifying that evidence.
fn resolve_validated_overlays_v1(
    report: &ButlinIndicatorReport,
    mut overlays: Vec<IndicatorOutcomeOverlayV1>,
) -> Result<ButlinResolvedEvidenceViewV1, EvidenceResolutionViewErrorV1> {
    let mut base_outcomes = BTreeMap::new();
    for indicator in &report.indicators {
        if base_outcomes
            .insert(indicator.id.clone(), indicator.outcome)
            .is_some()
        {
            return Err(EvidenceResolutionViewErrorV1::DuplicateBaseIndicator {
                indicator_id: indicator.id.clone(),
            });
        }
    }

    overlays.sort_by(|a, b| a.indicator_id.cmp(&b.indicator_id));
    let mut seen_overlays = BTreeSet::new();
    let mut resolved_overrides = BTreeMap::new();

    for overlay in &overlays {
        if !seen_overlays.insert(overlay.indicator_id.clone()) {
            return Err(EvidenceResolutionViewErrorV1::DuplicateOverlay {
                indicator_id: overlay.indicator_id.clone(),
            });
        }
        let observed = base_outcomes
            .get(&overlay.indicator_id)
            .copied()
            .ok_or_else(|| EvidenceResolutionViewErrorV1::UnknownIndicator {
                indicator_id: overlay.indicator_id.clone(),
            })?;
        if observed != overlay.base_outcome {
            return Err(EvidenceResolutionViewErrorV1::BaseOutcomeMismatch {
                indicator_id: overlay.indicator_id.clone(),
                expected: overlay.base_outcome,
                observed,
            });
        }
        validate_lineage(overlay)?;
        resolved_overrides.insert(overlay.indicator_id.clone(), overlay.resolved_outcome);
    }

    let mut resolved_counts = EvidenceOutcomeCountsV1::default();
    for indicator in &report.indicators {
        let outcome = resolved_overrides
            .get(&indicator.id)
            .copied()
            .unwrap_or(indicator.outcome);
        resolved_counts.record(outcome);
    }

    Ok(ButlinResolvedEvidenceViewV1 {
        schema: BUTLIN_RESOLVED_EVIDENCE_VIEW_SCHEMA_V1.to_string(),
        base_report_schema_version: report.schema_version,
        base_report_blake3: base_report_blake3_v1(report)?,
        overlays,
        resolved_counts,
    })
}

#[cfg(feature = "symthaea-backend")]
fn gwt1_direct_overlay_v1(
    report: &ButlinIndicatorReport,
    evidence: &super::gwt1_end_to_end::Gwt1EndToEndEvidenceV1,
) -> Result<IndicatorOutcomeOverlayV1, EvidenceResolutionViewErrorV1> {
    use super::gwt1_evidence_envelope::resolve_gwt1_evidence_envelope_v1;
    use super::gwt1_promotion::promote_direct_gwt1_v1;

    let recomputed = resolve_gwt1_evidence_envelope_v1(
        &evidence.envelope,
        &evidence.raw_observation_bytes,
    );
    if recomputed != evidence.resolution {
        return Err(EvidenceResolutionViewErrorV1::DirectEvidenceResolutionMismatch {
            indicator_id: "GWT-1".to_string(),
        });
    }

    let mut matches = report
        .indicators
        .iter()
        .filter(|indicator| indicator.id == "GWT-1");
    let base = matches
        .next()
        .ok_or_else(|| EvidenceResolutionViewErrorV1::UnknownIndicator {
            indicator_id: "GWT-1".to_string(),
        })?;
    if matches.next().is_some() {
        return Err(EvidenceResolutionViewErrorV1::DuplicateBaseIndicator {
            indicator_id: "GWT-1".to_string(),
        });
    }

    let promotion = promote_direct_gwt1_v1(&recomputed);
    let receipt = &evidence.envelope.receipt;
    let raw = &evidence.envelope.raw_observations;

    let overlay = IndicatorOutcomeOverlayV1 {
        indicator_id: "GWT-1".to_string(),
        base_outcome: base.outcome,
        resolved_outcome: promotion.evidence_outcome,
        lineage: EvidenceLineageIdentityV1 {
            kind: EvidenceLineageKindV1::DirectQualification,
            method_id: receipt.schema.clone(),
            policy_id: Some(promotion.policy),
            source_commit_sha: receipt.source_commit_sha.clone(),
            source_tree_sha: receipt.source_tree_sha.clone(),
            execution_run_id: receipt.execution_run_id.clone(),
            toolchain: receipt.toolchain.clone(),
            raw_artifact_schema: raw.schema.clone(),
            raw_artifact_blake3: raw.blake3.clone(),
            raw_artifact_byte_len: raw.byte_len,
        },
    };
    validate_lineage(&overlay)?;
    Ok(overlay)
}

/// Resolve the exact direct GWT-1 evidence object into a non-destructive view
/// over the supplied base report.
///
/// This is the only public V1 route that can add a GWT-1 direct-qualification
/// overlay. It recomputes the envelope resolution from the raw bytes, applies
/// the frozen conservative promotion policy, binds source/artifact provenance,
/// and leaves the base report's scalar diagnostic lineage unchanged.
#[cfg(feature = "symthaea-backend")]
pub fn resolve_gwt1_evidence_view_v1(
    report: &ButlinIndicatorReport,
    evidence: &super::gwt1_end_to_end::Gwt1EndToEndEvidenceV1,
) -> Result<ButlinResolvedEvidenceViewV1, EvidenceResolutionViewErrorV1> {
    let overlay = gwt1_direct_overlay_v1(report, evidence)?;
    resolve_validated_overlays_v1(report, vec![overlay])
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn overlay(outcome: EvidenceOutcome) -> IndicatorOutcomeOverlayV1 {
        IndicatorOutcomeOverlayV1 {
            indicator_id: "GWT-1".to_string(),
            base_outcome: EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            resolved_outcome: outcome,
            lineage: EvidenceLineageIdentityV1 {
                kind: EvidenceLineageKindV1::DirectQualification,
                method_id: "direct-gwt1-v1".to_string(),
                policy_id: Some("policy-v1".to_string()),
                source_commit_sha: "a".repeat(40),
                source_tree_sha: "b".repeat(40),
                execution_run_id: "run-1".to_string(),
                toolchain: "rustc test".to_string(),
                raw_artifact_schema: "raw-v1".to_string(),
                raw_artifact_blake3: "c".repeat(64),
                raw_artifact_byte_len: 128,
            },
        }
    }

    #[test]
    fn overlay_recomputes_counts_without_mutating_base_report() {
        let base = report();
        let original = base.clone();
        let view = resolve_validated_overlays_v1(
            &base,
            vec![overlay(EvidenceOutcome::Supported(SupportTier::Observed))],
        )
        .expect("resolved view");

        assert_eq!(base, original);
        assert_eq!(base.architectural_only_count, 1);
        assert_eq!(view.resolved_counts.architectural_only, 0);
        assert_eq!(view.resolved_counts.observed, 1);
        assert_eq!(view.resolved_counts.not_demonstrated, 1);
        assert_eq!(view.resolved_counts.total(), base.indicators.len());
    }

    #[test]
    fn direct_qualification_cannot_claim_causal_or_functional_support() {
        for forbidden in [
            EvidenceOutcome::Supported(SupportTier::CausallySupported),
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported),
        ] {
            assert!(matches!(
                resolve_validated_overlays_v1(&report(), vec![overlay(forbidden)]),
                Err(EvidenceResolutionViewErrorV1::InvalidLineageIdentity { .. })
            ));
        }
    }

    #[test]
    fn stale_overlay_cannot_attach_to_changed_base_outcome() {
        let mut stale = overlay(EvidenceOutcome::Supported(SupportTier::Observed));
        stale.base_outcome = EvidenceOutcome::Contradicted;
        assert!(matches!(
            resolve_validated_overlays_v1(&report(), vec![stale]),
            Err(EvidenceResolutionViewErrorV1::BaseOutcomeMismatch { .. })
        ));
    }

    #[test]
    fn duplicate_overlay_is_rejected() {
        let item = overlay(EvidenceOutcome::Supported(SupportTier::Observed));
        assert!(matches!(
            resolve_validated_overlays_v1(&report(), vec![item.clone(), item]),
            Err(EvidenceResolutionViewErrorV1::DuplicateOverlay { .. })
        ));
    }

    #[test]
    fn unknown_indicator_is_rejected() {
        let mut item = overlay(EvidenceOutcome::Supported(SupportTier::Observed));
        item.indicator_id = "UNKNOWN".to_string();
        assert!(matches!(
            resolve_validated_overlays_v1(&report(), vec![item]),
            Err(EvidenceResolutionViewErrorV1::UnknownIndicator { .. })
        ));
    }

    #[test]
    fn malformed_lineage_digest_is_rejected() {
        let mut item = overlay(EvidenceOutcome::Supported(SupportTier::Observed));
        item.lineage.raw_artifact_blake3 = "NOT-A-DIGEST".to_string();
        assert!(matches!(
            resolve_validated_overlays_v1(&report(), vec![item]),
            Err(EvidenceResolutionViewErrorV1::InvalidLineageIdentity { .. })
        ));
    }

    #[test]
    fn base_report_digest_is_deterministic() {
        let base = report();
        assert_eq!(
            base_report_blake3_v1(&base).unwrap(),
            base_report_blake3_v1(&base).unwrap()
        );
    }
}
