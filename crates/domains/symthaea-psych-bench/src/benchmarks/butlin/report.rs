// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Butlin indicator report structures.
//!
//! See `BUTLIN_EVIDENCE_TIER_DESIGN.md` (crate root) for the full rationale.
//! Core principle: architectural rationale, live observation, causal
//! support, and functional support are separate evidence dimensions. None
//! may be averaged into the appearance of stronger evidence than exists.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Bump on any breaking change to `IndicatorEvidence`/`ButlinIndicatorReport`
/// shape, so consumers of serialized reports can detect incompatible data.
pub const REPORT_SCHEMA_VERSION: u32 = 2;

/// Runtime consciousness data from the structural Phi engine.
///
/// When available, feeds live-probe evidence for indicators that have one.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeConsciousnessData {
    /// Micro-level Phi (within-cluster integration).
    pub micro_phi: f64,
    /// Meso-level Phi (inter-cluster integration).
    pub meso_phi: f64,
    /// Macro-level Phi (global integration).
    pub macro_phi: f64,
    /// Bottleneck score: gap between global and inter-cluster integration [0, 1].
    pub bottleneck_score: f64,
    /// Emergence ratio: macro / (micro + meso). > 1.0 means whole > sum of parts.
    pub emergence_ratio: f64,
    /// Number of detected clusters.
    pub num_clusters: usize,
    /// Real, mechanism-specific behavioral measurements from
    /// `ablation::measure_indicator` — the same probes the ablation matrix
    /// uses to prove a mechanism load-bearing. When present, these replace
    /// the structural-Phi-sigmoid proxy for the indicators they cover.
    #[serde(default)]
    pub behavioral: Option<BehavioralIndicatorSignals>,
}

impl RuntimeConsciousnessData {
    /// Construct from structural Phi fields (typically extracted from CycleMetadata).
    pub fn from_structural(
        micro_phi: f64,
        meso_phi: f64,
        macro_phi: f64,
        bottleneck_score: f64,
        emergence_ratio: f64,
        num_clusters: usize,
    ) -> Self {
        Self {
            micro_phi,
            meso_phi,
            macro_phi,
            bottleneck_score,
            emergence_ratio,
            num_clusters,
            behavioral: None,
        }
    }

    /// Attach real behavioral measurements (see `ablation::measure_indicator`).
    pub fn with_behavioral(mut self, behavioral: BehavioralIndicatorSignals) -> Self {
        self.behavioral = Some(behavioral);
        self
    }
}

/// Real, mechanism-specific measurements for 12 of the 14 indicators (see
/// `ablation::run_ablation_matrix`'s per-row causal effects). All fields are
/// the same probes `ablation::measure_indicator` computes, run here against
/// a live (non-ablated) service rather than a baseline-vs-ablated pair.
///
/// GWT-1 is deliberately not a field here — it's derived from the other
/// fields' aggregate in `indicators.rs::specialization_fraction`.
///
/// As of 2026-07-26 this suite matches Butlin et al. (2023) Table 1 exactly
/// (arXiv:2308.08708) — 14 indicators: RPT-1/2, GWT-1/2/3/4, HOT-1/2/3/4,
/// AST-1, PP-1 (one, not two), AE-1/2 (Agency and Embodiment).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BehavioralIndicatorSignals {
    /// RPT-1: input-discrimination / temporal-coherence proxy (0-1).
    pub rpt1_temporal_coherence: f64,
    /// RPT-2: fraction of cycles with active cross-modal binding (0-1).
    pub rpt2_binding_activity: f64,
    /// GWT-2: fraction of cycles with a non-empty, bounded GWT coalition (0-1).
    pub gwt2_bounded_coalition: f64,
    /// GWT-3: fraction of cycles with an active GWT broadcast (0-1).
    pub gwt3_broadcast_activity: f64,
    /// GWT-4: mean deviation of phi_attention_weight from neutral (0-1).
    pub gwt4_state_dependent_attention: f64,
    /// HOT-1: variance-based signal for whether prediction_error actually
    /// differentiates across inputs (0-1) — honestly near-zero while PE is
    /// frozen (see memory/symthaea_prediction_error_frozen_investigation.md).
    pub hot1_prediction_differentiation: f64,
    /// HOT-2: metacognitive monitoring accuracy (0-1).
    pub hot2_meta_cognitive_accuracy: f64,
    /// HOT-3: effective learning rate actually applied this cycle (raw units;
    /// treated as a presence signal — see `indicators.rs`'s use site). Same
    /// underlying signal as PP-1, different Butlin theoretical claim.
    pub hot3_effective_lr: f64,
    /// PP-1: effective learning rate actually applied this cycle (raw units;
    /// treated as a presence signal — see `indicators.rs`'s use site).
    pub pp1_effective_lr: f64,
    /// AE-1: fraction of distinct FEP actions (exploit/consolidate/explore/
    /// tighten) selected across distinct inputs (0-1, distinct_count/4.0) —
    /// "flexible responsiveness to competing goals" per Butlin et al.'s AE-1.
    pub ae1_action_diversity: f64,
    /// AE-2: embodied_agency from CycleMetadata.embodied (0-1, already
    /// 0.0 when embodied cognition is disabled) — "modeling output-input
    /// contingencies... in perception or control" per Butlin et al.'s AE-2.
    pub ae2_embodied_agency: f64,
    /// AST-1: attention-schema focus signal (0-1, non-zero fallback per
    /// `ablation::extract_indicator_score`).
    pub ast1_attention_focus: f64,
    /// HOT-4: fraction of near-zero output dimensions, averaged over several
    /// distinct inputs (0-1). Needs no cognitive-loop ablation at all — see
    /// `live_runner::CognitiveLoopBenchmarkRunner::measure_hot4_sparse_smooth_coding`.
    pub hot4_sparsity: f64,
    /// HOT-4: fraction of perturbation steps for which output dissimilarity
    /// grows non-decreasingly with perturbation size (0-1) — a genuinely
    /// smooth code shouldn't respond discontinuously to small changes.
    pub hot4_smoothness: f64,
}

/// How strongly an indicator's live behavior is supported by evidence, when
/// evidence supports it at all. See `EvidenceOutcome` for the two distinct
/// negative findings this ladder does NOT include.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupportTier {
    /// The mechanism exists and is wired into the loop; no live signal has
    /// been measured. Positive but limited evidence — an architectural
    /// claim, not an empirical one.
    ArchitecturalOnly,
    /// A live signal was measured and passed `ProbeQuality` checks (finite,
    /// non-fallback, demonstrably responsive) but no ablation has been run
    /// against it.
    Observed,
    /// A targeted ablation dropped the indicator's own signal
    /// (`indicator_dropped` in ablation terms).
    CausallySupported,
    /// As `CausallySupported`, and the paired downstream benchmark also
    /// degraded — the mechanism's removal harmed a real behavioral
    /// competency, not just an internal proxy metric.
    FunctionallySupported,
}

/// The overall evidentiary outcome for an indicator: either some degree of
/// positive support, or one of two distinct negative findings.
///
/// Deliberately NOT a single ordered enum with `SupportTier`'s variants —
/// `NotDemonstrated` (a relevant test was attempted and failed to show the
/// predicted effect) and `Contradicted` (evidence moved significantly the
/// wrong direction) are different failure modes with different
/// implications, not "below ArchitecturalOnly." Treating them as ordinal
/// would invite comparisons like `tier >= Observed` that don't make sense
/// for a negative outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceOutcome {
    Supported(SupportTier),
    /// A relevant test (typically an ablation) was attempted and did not
    /// demonstrate the predicted effect. Not the same as "never tried."
    NotDemonstrated,
    /// Evidence moved significantly in the direction opposite the
    /// prediction (e.g. structural Phi *rising* under severe network
    /// collapse). General-purpose — not tied to any one indicator.
    Contradicted,
}

impl EvidenceOutcome {
    /// Whether this is one of the two negative findings.
    pub fn is_negative(&self) -> bool {
        matches!(
            self,
            EvidenceOutcome::NotDemonstrated | EvidenceOutcome::Contradicted
        )
    }

    /// The `SupportTier` if this outcome is positive, else `None`.
    pub fn support_tier(&self) -> Option<SupportTier> {
        match self {
            EvidenceOutcome::Supported(t) => Some(*t),
            _ => None,
        }
    }
}

impl std::fmt::Display for EvidenceOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly) => {
                write!(f, "ARCHITECTURAL-ONLY")
            }
            EvidenceOutcome::Supported(SupportTier::Observed) => write!(f, "OBSERVED"),
            EvidenceOutcome::Supported(SupportTier::CausallySupported) => {
                write!(f, "CAUSALLY-SUPPORTED")
            }
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported) => {
                write!(f, "FUNCTIONALLY-SUPPORTED")
            }
            EvidenceOutcome::NotDemonstrated => write!(f, "NOT-DEMONSTRATED"),
            EvidenceOutcome::Contradicted => write!(f, "CONTRADICTED"),
        }
    }
}

/// Why a probe failed to qualify for `Observed` (or, absent any of these,
/// why it qualifies).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DegeneracyReason {
    /// All samples were exactly equal (or within numerical noise) — a frozen constant.
    Frozen,
    /// The value returned is a documented fallback/default, not a computed measurement.
    FallbackValue,
    /// Fewer than the minimum required samples were collected.
    InsufficientSamples,
    /// One or more samples were non-finite (NaN/inf).
    NonFinite,
}

/// How a probe's responsiveness was assessed, and what it was found to do.
/// The exact test differs by probe shape — see `BUTLIN_EVIDENCE_TIER_DESIGN.md`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Responsiveness {
    /// Not yet assessed (e.g. static-only evaluation, no live data at all).
    NotAssessed,
    /// A graded probe varied meaningfully across distinct inputs/conditions/seeds.
    VariesAcrossConditions,
    /// A binary/discrete probe activated when the mechanism had genuine
    /// opportunity to fire, and stayed inactive when it didn't.
    OpportunityGated,
    /// A theoretically stable quantity was shown sensitive to at least one
    /// relevant manipulation.
    SensitiveToManipulation,
    /// The probe returned a value but showed none of the above — this is a
    /// disqualifying signal for `Observed`, not merely "unknown."
    Unresponsive,
}

/// Quality/provenance metadata for a live probe measurement — the gate
/// `Observed` must pass. A populated numeric field alone is not evidence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProbeQuality {
    pub sample_count: usize,
    pub finite_fraction: f64,
    pub variance: Option<f64>,
    pub responsiveness: Responsiveness,
    pub fallback_used: bool,
    pub degeneracy: Option<DegeneracyReason>,
}

impl ProbeQuality {
    /// A probe with no samples at all — always disqualified.
    pub fn none_collected() -> Self {
        Self {
            sample_count: 0,
            finite_fraction: 0.0,
            variance: None,
            responsiveness: Responsiveness::NotAssessed,
            fallback_used: false,
            degeneracy: Some(DegeneracyReason::InsufficientSamples),
        }
    }

    /// Whether this probe's quality is strong enough to earn `Observed`.
    /// A frozen constant, a fallback default, or a probe that returned a
    /// value without demonstrating responsiveness all fail this check even
    /// though a finite number exists.
    pub fn qualifies_as_observed(&self) -> bool {
        self.degeneracy.is_none()
            && !self.fallback_used
            && self.finite_fraction >= 1.0
            && matches!(
                self.responsiveness,
                Responsiveness::VariesAcrossConditions
                    | Responsiveness::OpportunityGated
                    | Responsiveness::SensitiveToManipulation
            )
    }
}

/// Magnitude of an observed effect between a baseline and an intervention
/// (ablation) condition, across some number of seeds. Carries the actual
/// numbers rather than only a boolean gate, so future multi-seed thresholds
/// don't require another schema redesign.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EffectEstimate {
    pub baseline_mean: f64,
    pub intervention_mean: f64,
    pub absolute_change: f64,
    pub relative_change: Option<f64>,
    pub seed_count: usize,
    pub standard_deviation: Option<f64>,
}

impl EffectEstimate {
    /// Construct from a single baseline/intervention pair (`seed_count: 1`);
    /// use `with_std_dev` to attach cross-seed spread once available.
    pub fn new(baseline_mean: f64, intervention_mean: f64, seed_count: usize) -> Self {
        let absolute_change = intervention_mean - baseline_mean;
        let relative_change = if baseline_mean.abs() > f64::EPSILON {
            Some(absolute_change / baseline_mean)
        } else {
            None
        };
        Self {
            baseline_mean,
            intervention_mean,
            absolute_change,
            relative_change,
            seed_count,
            standard_deviation: None,
        }
    }

    pub fn with_std_dev(mut self, std_dev: f64) -> Self {
        self.standard_deviation = Some(std_dev);
        self
    }
}

/// Caveats about what KIND of evidence an indicator's outcome represents,
/// orthogonal to the outcome itself — a signal can be `Observed` AND a
/// derived aggregate at the same time. The outcome answers "how strongly
/// supported"; annotations answer "what kind of evidence is this."
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvidenceAnnotation {
    /// Derived from other indicators' signals rather than an independent probe (e.g. GWT-1).
    DerivedAggregate { sources: Vec<String> },
    /// A proxy for the theoretical construct, not a direct measurement of it.
    ProxyMeasure,
    /// Reuses the same underlying signal as another indicator (different theoretical claim).
    SharedUnderlyingSignal { with: Vec<String> },
    /// Internal telemetry, not externally observable behavior.
    InternalTelemetry,
    /// Externally observable behavior (e.g. a downstream benchmark outcome).
    ExternalBehavior,
    /// A known confound affecting interpretation, described in free text.
    KnownConfound(String),
    /// An ablation shows the indicator's own signal moves, but it hasn't
    /// been shown that the effect is specific to the targeted mechanism
    /// rather than a broad, non-specific system-wide degradation.
    TargetSpecificityNotYetEstablished,
}

/// Evidence for a single consciousness indicator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorEvidence {
    /// Indicator ID (e.g., "RPT-1", "GWT-3").
    pub id: String,
    /// Theory of origin (e.g., "Recurrent Processing Theory").
    pub theory: String,
    /// Description of the indicator.
    pub description: String,
    /// The evidentiary outcome — see `EvidenceOutcome`.
    pub outcome: EvidenceOutcome,
    /// Detailed evidence string (hand-written rationale).
    pub evidence: String,
    /// Hand-assigned architectural-plausibility constant (0.0-1.0). An
    /// expert heuristic, not an empirical result — always present.
    pub architectural_score: f64,
    /// Raw, unblended live probe value when one was measured. `None` at
    /// `ArchitecturalOnly`. Populated even when `probe_quality` disqualifies
    /// it from `Observed` — the number is reported regardless of what it
    /// means.
    pub live_score: Option<f64>,
    /// Quality/provenance of the live probe, when one was measured.
    pub probe_quality: Option<ProbeQuality>,
    /// Effect of a targeted ablation on this indicator's own signal.
    pub causal_effect: Option<EffectEstimate>,
    /// Effect of the same ablation on a paired downstream benchmark.
    pub functional_effect: Option<EffectEstimate>,
    /// Caveats about the kind of evidence this outcome represents.
    #[serde(default)]
    pub annotations: Vec<EvidenceAnnotation>,
}

/// Complete report of all consciousness indicators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ButlinIndicatorReport {
    pub schema_version: u32,
    pub indicators: Vec<IndicatorEvidence>,
    pub architectural_only_count: usize,
    pub observed_count: usize,
    pub causally_supported_count: usize,
    pub functionally_supported_count: usize,
    pub not_demonstrated_count: usize,
    pub contradicted_count: usize,
}

impl ButlinIndicatorReport {
    /// Build from a list of indicator evaluations.
    pub fn from_indicators(indicators: Vec<IndicatorEvidence>) -> Self {
        let count = |want: EvidenceOutcome| indicators.iter().filter(|i| i.outcome == want).count();
        Self {
            schema_version: REPORT_SCHEMA_VERSION,
            architectural_only_count: count(EvidenceOutcome::Supported(
                SupportTier::ArchitecturalOnly,
            )),
            observed_count: count(EvidenceOutcome::Supported(SupportTier::Observed)),
            causally_supported_count: count(EvidenceOutcome::Supported(
                SupportTier::CausallySupported,
            )),
            functionally_supported_count: count(EvidenceOutcome::Supported(
                SupportTier::FunctionallySupported,
            )),
            not_demonstrated_count: count(EvidenceOutcome::NotDemonstrated),
            contradicted_count: count(EvidenceOutcome::Contradicted),
            indicators,
        }
    }

    /// Tier-count breakdown, replacing the old single blended
    /// `mean_quality_score` — a distribution across evidence tiers is
    /// honest in a way one scalar mean can't be, since tiers aren't
    /// commensurable quantities to average.
    pub fn tier_summary(&self) -> String {
        format!(
            "ArchitecturalOnly:      {}\nObserved:               {}\nCausallySupported:      {}\nFunctionallySupported:  {}\nNotDemonstrated:        {}\nContradicted:           {}",
            self.architectural_only_count,
            self.observed_count,
            self.causally_supported_count,
            self.functionally_supported_count,
            self.not_demonstrated_count,
            self.contradicted_count,
        )
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "=== Butlin et al. Consciousness Indicators ===".to_string(),
            self.tier_summary(),
        ];
        for ind in &self.indicators {
            let live_str = ind
                .live_score
                .map(|s| format!(" live={:.2}", s))
                .unwrap_or_default();
            lines.push(format!(
                "  [{}] {} - {}: {} (architectural={:.2}{})",
                ind.id,
                ind.outcome,
                ind.description,
                ind.evidence,
                ind.architectural_score,
                live_str
            ));
        }
        lines.join("\n")
    }
}

/// Result of a single ablation row. Lives here (always compiled) rather than
/// in `ablation.rs` (feature-gated behind `symthaea-backend`, since it needs
/// `symthaea::cognitive_loop` types throughout) because `ButlinEvidenceBundle`
/// below embeds it and must compile without that feature — that's the whole
/// point of the cheap `butlin_regression.rs` gate. Owned `String` fields (not
/// `&'static str`) so this round-trips through serde for a persisted
/// evidence-baseline artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationResult {
    /// Which ablation was performed.
    pub name: String,
    /// Target indicator ID.
    pub target_indicator: String,
    /// Indicator score with mechanism ON (baseline).
    pub baseline_indicator_score: f64,
    /// Indicator score with mechanism OFF (ablated).
    pub ablated_indicator_score: f64,
    /// Downstream benchmark accuracy with mechanism ON.
    pub baseline_benchmark_accuracy: f64,
    /// Downstream benchmark accuracy with mechanism OFF.
    pub ablated_benchmark_accuracy: f64,
    /// Whether the indicator dropped sufficiently (ablated < baseline * 0.5).
    pub indicator_dropped: bool,
    /// Whether the downstream benchmark degraded (ablated_acc < baseline_acc * 0.7).
    pub benchmark_degraded: bool,
    /// Whether the indicator moved significantly in the WRONG direction
    /// (ablated > baseline * 1.5) — e.g. the structural-Phi-inverse-response
    /// finding (17.33 → 27.63, +59%, under severe network collapse). Distinct
    /// from `indicator_dropped == false`: this requires evidence of active
    /// movement the wrong way, not merely no drop.
    pub contradicted: bool,
}

/// Everything needed to merge ablation/functional evidence onto a report
/// without detaching the resulting claims from the conditions under which
/// they were measured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ButlinEvidenceBundle {
    pub schema_version: u32,
    pub commit_sha: String,
    pub config_hash: String,
    pub seeds: Vec<u64>,
    pub generated_at: String,
    pub ablations: Vec<AblationResult>,
}

/// Failure modes for `annotate_with_ablation_results` — a strict, provenance-checking merge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceMergeError {
    /// An `AblationResult` names an indicator ID not present in the report.
    UnknownIndicatorId(String),
    /// Two or more `AblationResult`s target the same indicator ID.
    DuplicateIndicatorId(String),
    /// The bundle's schema version doesn't match what this code understands.
    SchemaVersionMismatch { expected: u32, found: u32 },
}

impl std::fmt::Display for EvidenceMergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvidenceMergeError::UnknownIndicatorId(id) => {
                write!(f, "ablation result targets unknown indicator id {id:?}")
            }
            EvidenceMergeError::DuplicateIndicatorId(id) => {
                write!(f, "duplicate ablation result for indicator id {id:?}")
            }
            EvidenceMergeError::SchemaVersionMismatch { expected, found } => write!(
                f,
                "evidence bundle schema version {found} does not match expected {expected}"
            ),
        }
    }
}
impl std::error::Error for EvidenceMergeError {}

/// Pure, fallible merge of ablation/functional evidence onto a
/// static-evaluation report. Never silently upgrades a tier past what the
/// evidence in `bundle` actually supports; deterministic and idempotent —
/// reapplying the same bundle to its own output is a no-op.
pub fn annotate_with_ablation_results(
    mut report: ButlinIndicatorReport,
    bundle: &ButlinEvidenceBundle,
) -> Result<ButlinIndicatorReport, EvidenceMergeError> {
    if bundle.schema_version != REPORT_SCHEMA_VERSION {
        return Err(EvidenceMergeError::SchemaVersionMismatch {
            expected: REPORT_SCHEMA_VERSION,
            found: bundle.schema_version,
        });
    }

    let known_ids: HashSet<&str> = report.indicators.iter().map(|i| i.id.as_str()).collect();
    let mut seen_ids: HashSet<&str> = HashSet::new();
    for result in &bundle.ablations {
        if !known_ids.contains(result.target_indicator.as_str()) {
            return Err(EvidenceMergeError::UnknownIndicatorId(
                result.target_indicator.clone(),
            ));
        }
        if !seen_ids.insert(result.target_indicator.as_str()) {
            return Err(EvidenceMergeError::DuplicateIndicatorId(
                result.target_indicator.clone(),
            ));
        }
    }

    for indicator in &mut report.indicators {
        let Some(result) = bundle
            .ablations
            .iter()
            .find(|r| r.target_indicator == indicator.id)
        else {
            continue;
        };

        let causal_effect = EffectEstimate::new(
            result.baseline_indicator_score,
            result.ablated_indicator_score,
            bundle.seeds.len().max(1),
        );
        let functional_effect = EffectEstimate::new(
            result.baseline_benchmark_accuracy,
            result.ablated_benchmark_accuracy,
            bundle.seeds.len().max(1),
        );

        indicator.outcome = if result.contradicted {
            EvidenceOutcome::Contradicted
        } else if !result.indicator_dropped {
            EvidenceOutcome::NotDemonstrated
        } else if result.benchmark_degraded {
            indicator
                .annotations
                .push(EvidenceAnnotation::ExternalBehavior);
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported)
        } else {
            indicator
                .annotations
                .push(EvidenceAnnotation::TargetSpecificityNotYetEstablished);
            EvidenceOutcome::Supported(SupportTier::CausallySupported)
        };
        indicator.causal_effect = Some(causal_effect);
        indicator.functional_effect = Some(functional_effect);
    }

    Ok(ButlinIndicatorReport::from_indicators(report.indicators))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_consciousness_from_structural() {
        let data = RuntimeConsciousnessData::from_structural(0.1, 0.2, 0.3, 0.05, 1.5, 4);
        assert!((data.micro_phi - 0.1).abs() < f64::EPSILON);
        assert!((data.meso_phi - 0.2).abs() < f64::EPSILON);
        assert!((data.macro_phi - 0.3).abs() < f64::EPSILON);
        assert!((data.bottleneck_score - 0.05).abs() < f64::EPSILON);
        assert!((data.emergence_ratio - 1.5).abs() < f64::EPSILON);
        assert_eq!(data.num_clusters, 4);
    }

    fn stub_indicator(id: &str, outcome: EvidenceOutcome) -> IndicatorEvidence {
        IndicatorEvidence {
            id: id.to_string(),
            theory: "Test Theory".to_string(),
            description: "test".to_string(),
            outcome,
            evidence: "test evidence".to_string(),
            architectural_score: 0.85,
            live_score: None,
            probe_quality: None,
            causal_effect: None,
            functional_effect: None,
            annotations: Vec::new(),
        }
    }

    fn stub_ablation_result(
        id: &str,
        indicator_dropped: bool,
        benchmark_degraded: bool,
        contradicted: bool,
    ) -> AblationResult {
        AblationResult {
            name: format!("disable_{id}"),
            target_indicator: id.to_string(),
            baseline_indicator_score: 0.9,
            ablated_indicator_score: if contradicted {
                1.2
            } else if indicator_dropped {
                0.1
            } else {
                0.89
            },
            baseline_benchmark_accuracy: 0.8,
            ablated_benchmark_accuracy: if benchmark_degraded { 0.2 } else { 0.79 },
            indicator_dropped,
            benchmark_degraded,
            contradicted,
        }
    }

    #[test]
    fn test_merge_causally_supported_without_downstream() {
        let report = ButlinIndicatorReport::from_indicators(vec![stub_indicator(
            "AE-1",
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        )]);
        let bundle = ButlinEvidenceBundle {
            schema_version: REPORT_SCHEMA_VERSION,
            commit_sha: "test".into(),
            config_hash: "test".into(),
            seeds: vec![1],
            generated_at: "test".into(),
            ablations: vec![stub_ablation_result("AE-1", true, false, false)],
        };
        let merged = annotate_with_ablation_results(report, &bundle).unwrap();
        assert_eq!(
            merged.indicators[0].outcome,
            EvidenceOutcome::Supported(SupportTier::CausallySupported)
        );
        assert!(
            merged.indicators[0]
                .annotations
                .contains(&EvidenceAnnotation::TargetSpecificityNotYetEstablished)
        );
    }

    #[test]
    fn test_merge_functionally_supported_with_downstream() {
        let report = ButlinIndicatorReport::from_indicators(vec![stub_indicator(
            "AE-2",
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        )]);
        let bundle = ButlinEvidenceBundle {
            schema_version: REPORT_SCHEMA_VERSION,
            commit_sha: "test".into(),
            config_hash: "test".into(),
            seeds: vec![1],
            generated_at: "test".into(),
            ablations: vec![stub_ablation_result("AE-2", true, true, false)],
        };
        let merged = annotate_with_ablation_results(report, &bundle).unwrap();
        assert_eq!(
            merged.indicators[0].outcome,
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported)
        );
    }

    #[test]
    fn test_merge_not_demonstrated_when_indicator_did_not_drop() {
        let report = ButlinIndicatorReport::from_indicators(vec![stub_indicator(
            "RPT-2",
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        )]);
        let bundle = ButlinEvidenceBundle {
            schema_version: REPORT_SCHEMA_VERSION,
            commit_sha: "test".into(),
            config_hash: "test".into(),
            seeds: vec![1],
            generated_at: "test".into(),
            ablations: vec![stub_ablation_result("RPT-2", false, false, false)],
        };
        let merged = annotate_with_ablation_results(report, &bundle).unwrap();
        assert_eq!(
            merged.indicators[0].outcome,
            EvidenceOutcome::NotDemonstrated
        );
    }

    #[test]
    fn test_merge_contradicted_on_inverse_effect() {
        let report = ButlinIndicatorReport::from_indicators(vec![stub_indicator(
            "GWT-2",
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        )]);
        let bundle = ButlinEvidenceBundle {
            schema_version: REPORT_SCHEMA_VERSION,
            commit_sha: "test".into(),
            config_hash: "test".into(),
            seeds: vec![1],
            generated_at: "test".into(),
            ablations: vec![stub_ablation_result("GWT-2", false, false, true)],
        };
        let merged = annotate_with_ablation_results(report, &bundle).unwrap();
        assert_eq!(merged.indicators[0].outcome, EvidenceOutcome::Contradicted);
    }

    #[test]
    fn test_merge_missing_evidence_stays_architectural_only() {
        // No ablation for this indicator at all -- distinct from a
        // negative finding, which requires an attempted test.
        let report = ButlinIndicatorReport::from_indicators(vec![stub_indicator(
            "HOT-4",
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        )]);
        let bundle = ButlinEvidenceBundle {
            schema_version: REPORT_SCHEMA_VERSION,
            commit_sha: "test".into(),
            config_hash: "test".into(),
            seeds: vec![1],
            generated_at: "test".into(),
            ablations: vec![],
        };
        let merged = annotate_with_ablation_results(report, &bundle).unwrap();
        assert_eq!(
            merged.indicators[0].outcome,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly)
        );
    }

    #[test]
    fn test_merge_rejects_unknown_indicator_id() {
        let report = ButlinIndicatorReport::from_indicators(vec![stub_indicator(
            "AE-1",
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        )]);
        let bundle = ButlinEvidenceBundle {
            schema_version: REPORT_SCHEMA_VERSION,
            commit_sha: "test".into(),
            config_hash: "test".into(),
            seeds: vec![1],
            generated_at: "test".into(),
            ablations: vec![stub_ablation_result("NOT-A-REAL-ID", true, true, false)],
        };
        assert_eq!(
            annotate_with_ablation_results(report, &bundle).unwrap_err(),
            EvidenceMergeError::UnknownIndicatorId("NOT-A-REAL-ID".into())
        );
    }

    #[test]
    fn test_merge_rejects_duplicate_indicator_id() {
        let report = ButlinIndicatorReport::from_indicators(vec![stub_indicator(
            "AE-1",
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        )]);
        let bundle = ButlinEvidenceBundle {
            schema_version: REPORT_SCHEMA_VERSION,
            commit_sha: "test".into(),
            config_hash: "test".into(),
            seeds: vec![1],
            generated_at: "test".into(),
            ablations: vec![
                stub_ablation_result("AE-1", true, true, false),
                stub_ablation_result("AE-1", true, false, false),
            ],
        };
        assert_eq!(
            annotate_with_ablation_results(report, &bundle).unwrap_err(),
            EvidenceMergeError::DuplicateIndicatorId("AE-1".into())
        );
    }

    #[test]
    fn test_merge_is_deterministic_and_idempotent() {
        let make_report = || {
            ButlinIndicatorReport::from_indicators(vec![stub_indicator(
                "AE-2",
                EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            )])
        };
        let bundle = ButlinEvidenceBundle {
            schema_version: REPORT_SCHEMA_VERSION,
            commit_sha: "test".into(),
            config_hash: "test".into(),
            seeds: vec![1],
            generated_at: "test".into(),
            ablations: vec![stub_ablation_result("AE-2", true, true, false)],
        };
        let once = annotate_with_ablation_results(make_report(), &bundle).unwrap();
        let twice = annotate_with_ablation_results(once.clone(), &bundle).unwrap();
        assert_eq!(once.indicators[0].outcome, twice.indicators[0].outcome);
        assert_eq!(
            once.functionally_supported_count,
            twice.functionally_supported_count
        );
    }

    #[test]
    fn test_probe_quality_rejects_frozen_signal() {
        let frozen = ProbeQuality {
            sample_count: 10,
            finite_fraction: 1.0,
            variance: Some(0.0),
            responsiveness: Responsiveness::Unresponsive,
            fallback_used: false,
            degeneracy: Some(DegeneracyReason::Frozen),
        };
        assert!(!frozen.qualifies_as_observed());
    }

    #[test]
    fn test_probe_quality_accepts_responsive_signal() {
        let responsive = ProbeQuality {
            sample_count: 10,
            finite_fraction: 1.0,
            variance: Some(0.05),
            responsiveness: Responsiveness::VariesAcrossConditions,
            fallback_used: false,
            degeneracy: None,
        };
        assert!(responsive.qualifies_as_observed());
    }
}
