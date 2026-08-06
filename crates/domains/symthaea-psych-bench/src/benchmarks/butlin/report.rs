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
///
/// v3 (2026-07-26): added `EvidenceOutcome::Inconclusive` and
/// `ButlinIndicatorReport::inconclusive_count` (PR #30 review fix — probe
/// quality gating). A `ButlinEvidenceBundle` built against v2 would still
/// deserialize structurally, but any consumer matching exhaustively on
/// `EvidenceOutcome` needs to handle the new variant, hence the bump rather
/// than treating this as additive-only.
pub const REPORT_SCHEMA_VERSION: u32 = 3;

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
    /// An ablation was attempted, but the probe itself was not interpretable
    /// (frozen/unresponsive to its own targeted intervention, non-finite, or
    /// otherwise degenerate — see `ProbeQuality`). Deliberately distinct from
    /// both `NotDemonstrated` (a qualified probe that simply didn't move) and
    /// `Contradicted` (a qualified probe that moved the wrong way): a broken
    /// probe crossing the `contradicted` threshold by accident is not a
    /// scientific refutation, and reporting it as one would misrepresent a
    /// measurement failure as a finding.
    Inconclusive,
}

impl EvidenceOutcome {
    /// Whether this is one of the genuine negative findings (a qualified
    /// probe that either didn't move or moved the wrong way). `Inconclusive`
    /// is deliberately NOT included: it isn't a finding about the
    /// architecture at all, it's a statement that the measurement itself
    /// couldn't be interpreted.
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
            EvidenceOutcome::Inconclusive => write!(f, "INCONCLUSIVE"),
        }
    }
}

/// Why a probe failed to qualify for `Observed` (or, absent any of these,
/// why it qualifies).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DegeneracyReason {
    /// All samples were exactly equal (or within numerical noise) — a frozen constant.
    /// NOT inferred from a single ablation delta reading zero (that's circular — see
    /// `ablation_probe_quality`'s doc comment); reserved for when independent evidence
    /// (e.g. a separate responsiveness control, or the static evaluate() layer's own
    /// repeated-measurement check) establishes the probe itself can't move, distinct
    /// from the mechanism it targets genuinely having no effect.
    Frozen,
    /// The value returned is a documented fallback/default, not a computed measurement.
    FallbackValue,
    /// Fewer than the minimum required samples were collected.
    InsufficientSamples,
    /// The measured quantity itself has too little dynamic range to distinguish a real
    /// drop or reversal from noise (e.g. an ablation baseline at or below the presence
    /// epsilon) — a floor-effect problem, not a sample-count problem; kept distinct from
    /// `InsufficientSamples`.
    InsufficientDynamicRange,
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

/// Whether a boolean fact about probe provenance is actually known, or
/// merely defaulted because the layer producing the data doesn't track it.
/// Introduced because `ablation_probe_quality()` used to write `false` for
/// `fallback_used` unconditionally — collapsing "confirmed not a fallback"
/// and "we have no idea" into the same value, which then silently fed a
/// positive quality assertion (`qualifies_for_ablation_interpretation()`
/// reading `!fallback_used`). An unknown fact must not be usable as
/// affirmative evidence of quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FallbackStatus {
    /// Confirmed: this measurement did not take a fallback/default path.
    NotUsed,
    /// Confirmed: this measurement took a fallback/default path.
    Used,
    /// The layer that produced this measurement doesn't track whether a
    /// fallback was used. Distinct from `NotUsed`: permitted for
    /// provisional ablation interpretation (`qualifies_for_ablation_
    /// interpretation()` lets it through, paired with a mandatory
    /// provenance-disclosure annotation — see `annotate_with_ablation_
    /// results`), but insufficient for the stricter `Observed` tier
    /// (`qualifies_as_observed()` requires confirmed `NotUsed`) or for any
    /// future provenance-complete/publication-claim gate. "Unknown" cannot
    /// license the same trust as a confirmed absence of fallback behavior —
    /// it's just not treated as equivalent to a *confirmed* fallback
    /// (`Used`) either.
    Unknown,
}

/// Quality/provenance metadata for a live probe measurement — the gate
/// `Observed` must pass. A populated numeric field alone is not evidence.
///
/// `sample_count` is `None` when the producing layer doesn't actually know
/// how many underlying measurements went into the score it's reporting (a
/// single baseline/ablated pair from `AblationResult` is one *comparison*,
/// not necessarily one *sample* — if either side aggregates multiple
/// cognitive-loop cycles, claiming `sample_count: 1` would understate that).
/// `Some(n)` is reserved for callers that actually know `n`.
///
/// `finite_fraction` has the same layering caveat: for an ablation-derived
/// probe it means "both aggregate values this layer can see (baseline,
/// ablated) are finite" — i.e. the fraction is over `sample_count`'s same
/// unknown-cardinality aggregate comparison, not over whatever number of
/// underlying cognitive-loop cycles fed into computing those two aggregate
/// values. `1.0` here is not itself evidence that every one of the ~200
/// underlying per-arm cycles was finite — only that the two numbers this
/// layer actually receives are.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProbeQuality {
    pub sample_count: Option<usize>,
    pub finite_fraction: f64,
    pub variance: Option<f64>,
    pub responsiveness: Responsiveness,
    pub fallback_status: FallbackStatus,
    pub degeneracy: Option<DegeneracyReason>,
}

impl ProbeQuality {
    /// A probe with no samples at all — always disqualified.
    pub fn none_collected() -> Self {
        Self {
            sample_count: Some(0),
            finite_fraction: 0.0,
            variance: None,
            responsiveness: Responsiveness::NotAssessed,
            // `Unknown`, not `NotUsed`: with nothing collected, absence of
            // fallback behavior was never established -- `degeneracy`
            // already disqualifies this probe regardless, but the metadata
            // itself should stay truthful rather than assert a fact this
            // constructor has no basis for.
            fallback_status: FallbackStatus::Unknown,
            degeneracy: Some(DegeneracyReason::InsufficientSamples),
        }
    }

    /// Whether this probe's quality is strong enough to earn `Observed`.
    /// A frozen constant, a fallback default, an unknown fallback status,
    /// or a probe that returned a value without demonstrating
    /// responsiveness all fail this check even though a finite number
    /// exists.
    pub fn qualifies_as_observed(&self) -> bool {
        self.degeneracy.is_none()
            && self.fallback_status == FallbackStatus::NotUsed
            && self.finite_fraction >= 1.0
            && matches!(
                self.responsiveness,
                Responsiveness::VariesAcrossConditions
                    | Responsiveness::OpportunityGated
                    | Responsiveness::SensitiveToManipulation
            )
    }

    /// Whether an ablation-derived probe's comparison result — whatever it
    /// is, moved, didn't move, or moved the wrong way — can be trusted at
    /// all. Deliberately **not** the same bar as `qualifies_as_observed()`:
    /// that method requires a *demonstrated response*, which is exactly
    /// what an ablation test is trying to measure, not a precondition for
    /// trusting the measurement. Gating on responsiveness here would be
    /// circular — inferring "the probe is broken" from the same null delta
    /// the test produced, with no independent evidence. A qualified probe
    /// that shows no effect is a real null result (`NotDemonstrated`), not
    /// a disqualified one; only `degeneracy`/`finite_fraction`/
    /// `fallback_status` — genuine data-quality problems independent of what
    /// the comparison came out as — gate this.
    ///
    /// `FallbackStatus::Unknown` is deliberately **not** disqualifying here
    /// (only `Used` is): `AblationResult` doesn't currently track fallback
    /// provenance at all, so every ablation-derived probe reports `Unknown`
    /// — treating that as disqualifying would make every ablation row
    /// `Inconclusive` forever, which is worse than the honesty gap it would
    /// close. Instead `annotate_with_ablation_results` attaches an explicit
    /// `KnownConfound` annotation disclosing the gap, so the report stays
    /// honest about what isn't known without discarding the whole matrix's
    /// evidentiary value.
    pub fn qualifies_for_ablation_interpretation(&self) -> bool {
        self.degeneracy.is_none()
            && self.fallback_status != FallbackStatus::Used
            && self.finite_fraction >= 1.0
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
///
/// **Design invariant (do not violate):** no method on this type returns, or
/// ever will return, a single scalar computed across indicators that carry
/// different `EvidenceOutcome`s. The old `mean_quality_score` did exactly
/// that — averaged a hand-assigned architectural constant together with
/// whatever live signal existed, regardless of how well-supported each
/// indicator's claim actually was — and it's how a suite with real gaps
/// still reported "14/14, mean 0.85". The tier-count fields below and
/// `tier_summary()` are a vector/distribution over outcomes, not a
/// reduction to one number, and that's deliberate: outcomes aren't
/// commensurable quantities to average. This can't stop a caller from
/// pulling `architectural_score`/`live_score` out of `indicators` and
/// averaging them anyway — Rust's type system doesn't prevent arbitrary
/// downstream arithmetic — but no *first-party* report, summary, or
/// serialized field will ever do that itself. See
/// `test_no_first_party_scalar_aggregates_across_outcomes`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ButlinIndicatorReport {
    pub schema_version: u32,
    pub indicators: Vec<IndicatorEvidence>,
    pub architectural_only_count: usize,
    pub observed_count: usize,
    pub causally_supported_count: usize,
    pub functionally_supported_count: usize,
    pub not_demonstrated_count: usize,
    pub contradicted_count: usize,
    pub inconclusive_count: usize,
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
            inconclusive_count: count(EvidenceOutcome::Inconclusive),
            indicators,
        }
    }

    /// Tier-count breakdown, replacing the old single blended
    /// `mean_quality_score` — a distribution across evidence tiers is
    /// honest in a way one scalar mean can't be, since tiers aren't
    /// commensurable quantities to average.
    pub fn tier_summary(&self) -> String {
        format!(
            "ArchitecturalOnly:      {}\nObserved:               {}\nCausallySupported:      {}\nFunctionallySupported:  {}\nNotDemonstrated:        {}\nContradicted:           {}\nInconclusive:           {}",
            self.architectural_only_count,
            self.observed_count,
            self.causally_supported_count,
            self.functionally_supported_count,
            self.not_demonstrated_count,
            self.contradicted_count,
            self.inconclusive_count,
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
    /// An `AblationResult`'s cached `indicator_dropped`/`benchmark_degraded`/
    /// `contradicted` booleans disagree with what `classify_ablation`
    /// recomputes from its own raw scores. Since the normal producer
    /// (`ablation.rs::run_ablation_matrix`) now derives those cached fields
    /// from this exact same canonical classifier, any disagreement in
    /// legitimate evidence should be impossible — a mismatch means the
    /// bundle is malformed, stale (built against an older classifier
    /// version), corrupted, or tampered with. Rejected outright rather than
    /// silently recomputed-and-accepted: a "strict provenance-checking
    /// merge" must not paper over a self-contradictory evidence row, even
    /// when the raw scores alone would still determine a defensible
    /// scientific outcome.
    ClassificationMismatch {
        indicator_id: String,
        stored: AblationClassification,
        recomputed: AblationClassification,
    },
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
            EvidenceMergeError::ClassificationMismatch {
                indicator_id,
                stored,
                recomputed,
            } => write!(
                f,
                "ablation result for indicator {indicator_id:?} has cached \
                 classification {stored:?} that disagrees with the \
                 recomputed classification {recomputed:?} from its own raw \
                 scores — bundle rejected as malformed"
            ),
        }
    }
}
impl std::error::Error for EvidenceMergeError {}

/// Assess an ablation row's probe quality from what `AblationResult` already
/// records — the guard between "the ablation matrix produced a genuine
/// scientific result" and "the probe itself wasn't interpretable, and any
/// resulting `Contradicted`/`CausallySupported`/`FunctionallySupported`
/// verdict would misrepresent a measurement failure as a finding".
///
/// **Deliberately does NOT treat `baseline == ablated` as evidence the probe
/// is frozen.** An earlier version of this function did, and that was
/// circular: it inferred "the probe is broken" from the very same null
/// delta the ablation test produced, with no independent evidence. A
/// qualified probe showing no effect from a real, targeted intervention is
/// exactly what `NotDemonstrated` is *for* — there are several honest
/// explanations for a null delta (the mechanism genuinely has no causal
/// effect, the effect is below measurement resolution, the intervention
/// didn't manipulate the intended mechanism, sampling noise) and this
/// function cannot and does not try to distinguish among them from one
/// baseline/ablated pair. Only two things here are treated as genuine
/// *measurement* problems, independent of what the comparison came out as:
///
/// - non-finite values — an outright data error;
/// - a baseline at or below the same 0.0005 presence epsilon
///   `classify_ablation`'s own `indicator_dropped`/`contradicted` logic uses —
///   there's no dynamic range to measure a drop or reversal *from* at all
///   (this is the exact condition the `KNOWN_LIMITATIONS`/
///   `near_zero_baseline` carve-outs in `butlin_ablation_integration.rs`
///   were disclosing by hand; this makes it structural instead).
///
/// `DegeneracyReason::Frozen` is reserved for when *independent* evidence
/// (a separate responsiveness control, not this same delta) establishes the
/// probe itself can't move — not currently produced by this function; see
/// `BUTLIN_EVIDENCE_TIER_DESIGN.md`'s "Corrections after review" for why
/// this is disclosed as an open gap rather than a fake fix.
fn ablation_probe_quality(result: &AblationResult) -> ProbeQuality {
    if !result.baseline_indicator_score.is_finite() || !result.ablated_indicator_score.is_finite() {
        return ProbeQuality {
            sample_count: None,
            finite_fraction: 0.0,
            variance: None,
            responsiveness: Responsiveness::NotAssessed,
            fallback_status: FallbackStatus::Unknown,
            degeneracy: Some(DegeneracyReason::NonFinite),
        };
    }
    if result.baseline_indicator_score <= 0.0005 {
        return ProbeQuality {
            sample_count: None,
            finite_fraction: 1.0,
            variance: None,
            responsiveness: Responsiveness::NotAssessed,
            fallback_status: FallbackStatus::Unknown,
            degeneracy: Some(DegeneracyReason::InsufficientDynamicRange),
        };
    }
    // Whether it moved is the ablation's actual result, not a precondition
    // for trusting the measurement -- deliberately not gated on here. See
    // `qualifies_for_ablation_interpretation()` for the epistemic reasoning.
    let moved = (result.ablated_indicator_score - result.baseline_indicator_score).abs() > 1e-9;
    ProbeQuality {
        // `None`, not `Some(1)`: `measure_indicator` aggregates many
        // cognitive-loop cycles (200 per arm) into the single scalar stored
        // in `baseline_indicator_score`/`ablated_indicator_score` -- this
        // layer only sees the aggregate, so claiming a sample count here
        // would either understate the real cycle count or overstate this
        // function's own knowledge of it. Neither is honest; `None` is.
        sample_count: None,
        finite_fraction: 1.0,
        // Not `variance`: this is a single baseline/ablated pair, not a
        // repeated-sample dispersion estimate -- there is no variance to
        // report from one paired observation. Left `None` rather than
        // mislabeling the delta magnitude as variance; multi-seed work
        // (issue #7 follow-up) is what would give this a real value.
        variance: None,
        responsiveness: if moved {
            Responsiveness::SensitiveToManipulation
        } else {
            Responsiveness::NotAssessed
        },
        // `AblationResult` doesn't currently record whether a
        // fallback/default path was taken anywhere in producing these
        // scores -- `Unknown`, not a silent `NotUsed` assumption. See
        // `qualifies_for_ablation_interpretation()`'s doc comment for why
        // `Unknown` doesn't disqualify the row outright, and
        // `annotate_with_ablation_results` for the disclosure annotation
        // this attaches instead.
        fallback_status: FallbackStatus::Unknown,
        degeneracy: None,
    }
}

/// Whether a downstream-benchmark accuracy pair is trustworthy enough to
/// license `FunctionallySupported` — a lighter-weight, benchmark-specific
/// counterpart to `ablation_probe_quality`'s indicator-score checks. An
/// accuracy is a proportion; a non-finite value or one outside `[0.0, 1.0]`
/// indicates a broken measurement pipeline, not a legitimate extreme
/// result, and `classify_ablation`'s `benchmark_degraded` comparison
/// doesn't itself guard against this — e.g. an infinite baseline accuracy
/// paired with any finite ablated value satisfies `ablated_acc <
/// baseline_acc * 0.7` trivially, which would otherwise let a broken
/// benchmark measurement manufacture false functional evidence.
///
/// Deliberately narrower than a full `ProbeQuality` for the indicator score:
/// this only gates the `CausallySupported` → `FunctionallySupported`
/// transition, not the indicator's own probe interpretation (already
/// covered by `ablation_probe_quality`/`qualifies_for_ablation_
/// interpretation`). A future redesign might promote this into its own
/// `downstream_benchmark_quality` field alongside `probe_quality`, mirroring
/// the indicator side more fully — not done here to avoid a second schema
/// break in the same round; see `BUTLIN_EVIDENCE_TIER_DESIGN.md`.
fn benchmark_measurement_is_valid(result: &AblationResult) -> bool {
    let in_unit_range = |x: f64| x.is_finite() && (0.0..=1.0).contains(&x);
    in_unit_range(result.baseline_benchmark_accuracy)
        && in_unit_range(result.ablated_benchmark_accuracy)
}

/// The three derived booleans an `AblationResult` carries
/// (`indicator_dropped`, `benchmark_degraded`, `contradicted`), recomputed
/// from raw scores rather than read off a struct field. `pub` (not
/// `pub(crate)`) because `EvidenceMergeError::ClassificationMismatch`
/// embeds it — a public error type can't hold a private field type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AblationClassification {
    pub indicator_dropped: bool,
    pub benchmark_degraded: bool,
    pub contradicted: bool,
}

/// Canonical classification of an ablation's raw baseline/ablated scores
/// into the three derived booleans `AblationResult` also carries as cached
/// fields. This is the **single source of truth** both `ablation.rs` (which
/// computes these at measurement time, via this same function) and
/// `annotate_with_ablation_results` (which recomputes them at merge time
/// rather than trusting the stored booleans, see below) use — there is no
/// second copy of these thresholds anywhere to drift out of sync.
///
/// Thresholds match the pre-existing behavior exactly: a drop requires the
/// baseline to be above the same `0.0005` presence epsilon used elsewhere in
/// this module, with the ablated score below half the baseline; a benchmark
/// degradation requires the baseline accuracy above `0.01` with the ablated
/// accuracy below 70% of it; a contradiction requires the baseline above the
/// presence epsilon with the ablated score above 1.5x it. All three default
/// to `false` when their respective baseline has no usable dynamic range —
/// consistent with `ablation_probe_quality`'s own `InsufficientDynamicRange`
/// gate treating that state as unable to support any positive claim.
pub fn classify_ablation(
    baseline_indicator: f64,
    ablated_indicator: f64,
    baseline_acc: f64,
    ablated_acc: f64,
) -> AblationClassification {
    let indicator_dropped =
        baseline_indicator > 0.0005 && ablated_indicator < baseline_indicator * 0.5;
    let benchmark_degraded = baseline_acc > 0.01 && ablated_acc < baseline_acc * 0.7;
    // Mutually exclusive with indicator_dropped by construction (can't be
    // both < baseline*0.5 and > baseline*1.5).
    let contradicted = baseline_indicator > 0.0005 && ablated_indicator > baseline_indicator * 1.5;
    AblationClassification {
        indicator_dropped,
        benchmark_degraded,
        contradicted,
    }
}

/// Push `annotation` onto `annotations` only if an equal one isn't already
/// present. Without this, reapplying the same bundle to an already-merged
/// report (`annotate_with_ablation_results` is documented as idempotent)
/// would duplicate every annotation the merge unconditionally attaches —
/// fallback-provenance disclosure, `ExternalBehavior`,
/// `TargetSpecificityNotYetEstablished`, the invalid-benchmark confound —
/// on every reapplication, silently violating that guarantee. Static
/// annotations from `evaluate()` (e.g. `ProxyMeasure`, `DerivedAggregate`)
/// are untouched; this only guards the merge's own additions, which are
/// deterministic given the same `result`, so an equality check is exactly
/// the right dedup key.
fn push_annotation_once(annotations: &mut Vec<EvidenceAnnotation>, annotation: EvidenceAnnotation) {
    if !annotations.contains(&annotation) {
        annotations.push(annotation);
    }
}

/// Pure, fallible merge of ablation/functional evidence onto a
/// static-evaluation report. Never silently upgrades a tier past what the
/// evidence in `bundle` actually supports; deterministic and idempotent —
/// reapplying the same bundle to its own output is a no-op (verified
/// structurally, not just by outcome/tier-count, by
/// `test_evidence_merge_is_structurally_idempotent`).
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

        // Reject rather than silently recompute-and-accept: the normal
        // producer now derives its cached booleans from this exact
        // classifier (see `ablation.rs::run_ablation_matrix`), so any
        // disagreement in legitimate evidence should be impossible. A
        // mismatch means the bundle is malformed, stale, or tampered with —
        // see `EvidenceMergeError::ClassificationMismatch`'s doc comment.
        let stored = AblationClassification {
            indicator_dropped: result.indicator_dropped,
            benchmark_degraded: result.benchmark_degraded,
            contradicted: result.contradicted,
        };
        let recomputed = classify_ablation(
            result.baseline_indicator_score,
            result.ablated_indicator_score,
            result.baseline_benchmark_accuracy,
            result.ablated_benchmark_accuracy,
        );
        if stored != recomputed {
            return Err(EvidenceMergeError::ClassificationMismatch {
                indicator_id: result.target_indicator.clone(),
                stored,
                recomputed,
            });
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

        let quality = ablation_probe_quality(result);
        indicator.probe_quality = Some(quality);

        // Fallback provenance is genuinely unknown at this layer (see
        // `ablation_probe_quality`'s doc comment) -- disclose that plainly
        // rather than let `Unknown`'s pass-through of the quality gate read
        // as tacit confirmation nothing untoward happened. Gated on
        // `Unknown` specifically (not attached unconditionally): if
        // `AblationResult` ever learns to report a confirmed
        // `FallbackStatus`, this disclosure would become inaccurate noise
        // rather than a genuine caveat.
        if quality.fallback_status == FallbackStatus::Unknown {
            push_annotation_once(
                &mut indicator.annotations,
                EvidenceAnnotation::KnownConfound(
                    "fallback status unavailable in AblationResult schema".to_string(),
                ),
            );
        }

        // Recomputed from the raw scores via the canonical classifier, NOT
        // read off `result.indicator_dropped`/`contradicted`/
        // `benchmark_degraded` -- the validation pass above has already
        // confirmed these agree (rejecting the whole bundle otherwise), so
        // this recomputation is guaranteed consistent with the cached
        // fields by this point. Still recomputed here rather than threading
        // the validation pass's value through, to keep this loop
        // self-contained; the cached booleans remain diagnostics only,
        // never a merge input in their own right.
        let classification = classify_ablation(
            result.baseline_indicator_score,
            result.ablated_indicator_score,
            result.baseline_benchmark_accuracy,
            result.ablated_benchmark_accuracy,
        );

        // Probe quality gates everything below on genuine measurement
        // problems only (non-finite, no dynamic range) -- NOT on whether
        // the probe moved, which is the ablation's actual result, not a
        // precondition for trusting it. See
        // `qualifies_for_ablation_interpretation()`'s doc comment for why
        // this differs from `qualifies_as_observed()`.
        indicator.outcome = if !quality.qualifies_for_ablation_interpretation() {
            push_annotation_once(
                &mut indicator.annotations,
                EvidenceAnnotation::KnownConfound(format!(
                    "ablation probe quality insufficient to interpret ({:?}); \
                     baseline={:.4}, ablated={:.4} — not treated as a scientific result",
                    quality.degeneracy,
                    result.baseline_indicator_score,
                    result.ablated_indicator_score
                )),
            );
            EvidenceOutcome::Inconclusive
        } else if classification.contradicted {
            EvidenceOutcome::Contradicted
        } else if !classification.indicator_dropped {
            EvidenceOutcome::NotDemonstrated
        } else if classification.benchmark_degraded && benchmark_measurement_is_valid(result) {
            push_annotation_once(
                &mut indicator.annotations,
                EvidenceAnnotation::ExternalBehavior,
            );
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported)
        } else {
            // Either the downstream benchmark genuinely didn't degrade, or
            // it looked like it did but the accuracy pair itself is
            // non-finite or outside [0,1] -- a broken measurement, not
            // legitimate functional evidence (e.g. an infinite baseline
            // accuracy trivially satisfies the degradation comparison
            // regardless of the ablated value). Either way this caps at
            // CausallySupported: the indicator's own probe already passed
            // its quality gate above, so that half of the evidence stands;
            // only the *functional* claim is capped. Exactly one of these
            // two annotations is pushed, never both -- each explains why
            // this specific row didn't reach FunctionallySupported.
            if classification.benchmark_degraded {
                push_annotation_once(
                    &mut indicator.annotations,
                    EvidenceAnnotation::KnownConfound(format!(
                        "downstream benchmark measurement invalid (non-finite or \
                         outside [0,1]): baseline_acc={:.4}, ablated_acc={:.4} — \
                         capped at CausallySupported rather than treated as \
                         functional evidence",
                        result.baseline_benchmark_accuracy, result.ablated_benchmark_accuracy
                    )),
                );
            } else {
                push_annotation_once(
                    &mut indicator.annotations,
                    EvidenceAnnotation::TargetSpecificityNotYetEstablished,
                );
            }
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
                // Must exceed baseline * 1.5 = 1.35 to actually satisfy
                // `classify_ablation`'s own contradiction threshold -- 1.2
                // (the original value here) does NOT, which the new
                // classification-consistency check in
                // `annotate_with_ablation_results` now catches as a
                // ClassificationMismatch. This stub must stay
                // self-consistent with the canonical classifier, same as
                // any real `AblationResult`.
                1.4
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

    /// Full-control variant for probe-quality and classification-consistency
    /// tests: sets baseline/ablated directly rather than deriving them from
    /// the flags. Some callers deliberately pass a stored-flag combination
    /// that disagrees with what `classify_ablation` would compute from the
    /// given raw scores -- those exercise `EvidenceMergeError::
    /// ClassificationMismatch` (the bundle gets rejected outright) rather
    /// than a quality-gate override, so any *quality*-focused test using
    /// this helper must keep its stored flags self-consistent with the raw
    /// scores or it will be rejected before the quality gate ever runs.
    #[allow(clippy::too_many_arguments)]
    fn stub_ablation_result_raw(
        id: &str,
        baseline_indicator_score: f64,
        ablated_indicator_score: f64,
        indicator_dropped: bool,
        benchmark_degraded: bool,
        contradicted: bool,
    ) -> AblationResult {
        AblationResult {
            name: format!("disable_{id}"),
            target_indicator: id.to_string(),
            baseline_indicator_score,
            ablated_indicator_score,
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

    // ── Probe-quality gating (arkh-node's PR #30 review) ────────────────

    #[test]
    fn test_merge_equal_baseline_and_ablated_is_not_demonstrated_not_inconclusive() {
        // A qualified probe (finite, real dynamic range) that shows no
        // effect from a real targeted intervention is a genuine null
        // result -- NotDemonstrated, not Inconclusive. An earlier version
        // of the quality gate treated baseline == ablated as automatic
        // proof the probe was "frozen", which was circular: it inferred a
        // measurement failure from the very same null delta the test
        // produced, with no independent evidence the probe itself can't
        // move. This is exactly the RPT-2/HOT-1 real finding (byte-
        // identical baseline/ablated arms) -- it must stay NotDemonstrated,
        // per the disclosed KNOWN_LIMITATIONS in
        // butlin_ablation_integration.rs.
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
            ablations: vec![stub_ablation_result_raw(
                "RPT-2", 0.5, 0.5, // baseline == ablated: a real null result
                false, false, false, // indicator_dropped/contradicted correctly false
            )],
        };
        let merged = annotate_with_ablation_results(report, &bundle).unwrap();
        assert_eq!(
            merged.indicators[0].outcome,
            EvidenceOutcome::NotDemonstrated
        );
        assert_eq!(
            merged.indicators[0].probe_quality.map(|q| q.degeneracy),
            Some(None),
            "a merely-null result is not a data-quality problem"
        );
    }

    #[test]
    fn test_merge_qualifies_for_ablation_interpretation_does_not_require_movement() {
        // Direct unit test of the epistemic distinction: a probe reading
        // exactly zero delta is still "qualified" (trustworthy) -- movement
        // is the ablation's result, not the quality gate's precondition.
        let unmoved = ProbeQuality {
            sample_count: Some(1),
            finite_fraction: 1.0,
            variance: None,
            responsiveness: Responsiveness::NotAssessed,
            fallback_status: FallbackStatus::NotUsed,
            degeneracy: None,
        };
        assert!(unmoved.qualifies_for_ablation_interpretation());
        assert!(
            !unmoved.qualifies_as_observed(),
            "NotAssessed responsiveness correctly still fails the stricter Observed bar"
        );

        // Unknown fallback status must not disqualify ablation
        // interpretation (see `qualifies_for_ablation_interpretation`'s doc
        // comment) -- only a *confirmed* fallback (`Used`) should.
        let unknown_fallback = ProbeQuality {
            fallback_status: FallbackStatus::Unknown,
            ..unmoved
        };
        assert!(unknown_fallback.qualifies_for_ablation_interpretation());
        let confirmed_fallback = ProbeQuality {
            fallback_status: FallbackStatus::Used,
            ..unmoved
        };
        assert!(!confirmed_fallback.qualifies_for_ablation_interpretation());
    }

    #[test]
    fn test_merge_non_finite_probe_produces_inconclusive() {
        let report = ButlinIndicatorReport::from_indicators(vec![stub_indicator(
            "HOT-1",
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        )]);
        let bundle = ButlinEvidenceBundle {
            schema_version: REPORT_SCHEMA_VERSION,
            commit_sha: "test".into(),
            config_hash: "test".into(),
            seeds: vec![1],
            generated_at: "test".into(),
            ablations: vec![stub_ablation_result_raw(
                "HOT-1",
                f64::NAN,
                0.1,
                // Stored flags must stay consistent with what
                // `classify_ablation` actually computes for a NaN baseline
                // (all comparisons against NaN are false) -- a real
                // producer would never emit `true`/`true` here, and the new
                // classification-consistency check would otherwise reject
                // this as a malformed bundle before ever reaching the
                // probe-quality gate this test means to exercise.
                false,
                false,
                false,
            )],
        };
        let merged = annotate_with_ablation_results(report, &bundle).unwrap();
        assert_eq!(merged.indicators[0].outcome, EvidenceOutcome::Inconclusive);
        assert_eq!(
            merged.indicators[0]
                .probe_quality
                .and_then(|q| q.degeneracy),
            Some(DegeneracyReason::NonFinite)
        );
    }

    #[test]
    fn test_merge_near_zero_baseline_probe_produces_inconclusive() {
        let report = ButlinIndicatorReport::from_indicators(vec![stub_indicator(
            "PP-1",
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        )]);
        let bundle = ButlinEvidenceBundle {
            schema_version: REPORT_SCHEMA_VERSION,
            commit_sha: "test".into(),
            config_hash: "test".into(),
            seeds: vec![1],
            generated_at: "test".into(),
            ablations: vec![stub_ablation_result_raw(
                "PP-1", 0.0002, 0.0001, false, false, false,
            )],
        };
        let merged = annotate_with_ablation_results(report, &bundle).unwrap();
        assert_eq!(merged.indicators[0].outcome, EvidenceOutcome::Inconclusive);
        assert_eq!(
            merged.indicators[0]
                .probe_quality
                .and_then(|q| q.degeneracy),
            Some(DegeneracyReason::InsufficientDynamicRange)
        );
    }

    #[test]
    fn test_merge_responsive_probe_is_not_inconclusive() {
        // Sanity check: a well-behaved probe (the common case exercised by
        // the other merge tests) must NOT be caught by the quality gate.
        //
        // Checks `qualifies_for_ablation_interpretation()`, NOT
        // `qualifies_as_observed()` -- the latter requires a *confirmed*
        // `FallbackStatus::NotUsed`, which `ablation_probe_quality()` can
        // never produce (it always reports `Unknown`, since
        // `AblationResult` doesn't track fallback provenance -- see that
        // function's doc comment). An earlier version of this test checked
        // `qualifies_as_observed()` and passed only by accident, back when
        // `fallback_used` was a bare bool defaulted to `false`: that
        // silently satisfied `!fallback_used` regardless of whether
        // "not used" was actually known. `annotate_with_ablation_results`
        // itself never produces `SupportTier::Observed` as an outcome
        // anyway (see `BUTLIN_EVIDENCE_TIER_DESIGN.md`) -- the relevant
        // gate for this code path has always been
        // `qualifies_for_ablation_interpretation()`.
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
        assert_ne!(merged.indicators[0].outcome, EvidenceOutcome::Inconclusive);
        assert_eq!(
            merged.indicators[0]
                .probe_quality
                .map(|q| q.qualifies_for_ablation_interpretation()),
            Some(true)
        );
        assert_eq!(
            merged.indicators[0]
                .probe_quality
                .map(|q| q.qualifies_as_observed()),
            Some(false),
            "ablation-derived probes never satisfy the stricter Observed \
             bar -- fallback provenance is never confirmed at this layer"
        );
    }

    // ── Aggregation guarantee (arkh-node's PR #30 review) ───────────────

    /// The design invariant on `ButlinIndicatorReport`: no first-party field
    /// or serialized key is a scalar aggregate across indicators with
    /// different outcomes. Enumerates the exact expected top-level JSON keys
    /// so an added scalar-mean-style field would have to change this test,
    /// not slip in silently.
    #[test]
    fn test_no_first_party_scalar_aggregates_across_outcomes() {
        let report = ButlinIndicatorReport::from_indicators(vec![
            stub_indicator("RPT-1", EvidenceOutcome::Supported(SupportTier::Observed)),
            stub_indicator("HOT-1", EvidenceOutcome::NotDemonstrated),
            stub_indicator("GWT-2", EvidenceOutcome::Contradicted),
        ]);
        let json = serde_json::to_value(&report).unwrap();
        let mut keys: Vec<&str> = json
            .as_object()
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "architectural_only_count",
                "causally_supported_count",
                "contradicted_count",
                "functionally_supported_count",
                "inconclusive_count",
                "indicators",
                "not_demonstrated_count",
                "observed_count",
                "schema_version",
            ],
            "ButlinIndicatorReport gained or lost a top-level field -- if this \
             added a scalar computed across indicators with differing \
             outcomes (a mean, a weighted score, etc.), that violates the \
             no-mixed-tier-scalar design invariant documented on the struct"
        );

        // Belt-and-suspenders: the exact-key-list check above is a brittle
        // schema snapshot that would need updating for any new field,
        // legitimate or not. This deny-list is narrower and more durable --
        // it specifically names the kind of field that must never reappear,
        // so it stays meaningful even if the exact-key-list assertion above
        // is ever loosened or replaced.
        for forbidden in [
            "mean_score",
            "mean_quality_score",
            "composite_score",
            "overall_quality",
            "overall_score",
            "average_score",
        ] {
            assert!(
                !keys.contains(&forbidden),
                "ButlinIndicatorReport must never expose a scalar aggregate \
                 field like {forbidden:?}"
            );
        }
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
    fn test_evidence_merge_is_structurally_idempotent() {
        // Stronger than the outcome/tier-count check above: the merge
        // unconditionally attaches several annotations (fallback-provenance
        // disclosure, ExternalBehavior/TargetSpecificityNotYetEstablished,
        // the invalid-benchmark confound) -- without `push_annotation_once`
        // dedup, reapplying the same bundle to an already-merged report
        // would duplicate every one of them, silently breaking the
        // documented idempotence guarantee. Full struct equality (not just
        // outcome/count) is what actually catches that class of bug.
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
        let once = annotate_with_ablation_results(report, &bundle).unwrap();
        let twice = annotate_with_ablation_results(once.clone(), &bundle).unwrap();
        assert_eq!(
            once, twice,
            "reapplying the same bundle to an already-merged report must be \
             a structural no-op, including annotation counts"
        );
        assert_eq!(
            once.indicators[0].annotations.len(),
            twice.indicators[0].annotations.len(),
            "annotations must not accumulate duplicates across reapplication"
        );
    }

    #[test]
    fn test_probe_quality_rejects_frozen_signal() {
        let frozen = ProbeQuality {
            sample_count: Some(10),
            finite_fraction: 1.0,
            variance: Some(0.0),
            responsiveness: Responsiveness::Unresponsive,
            fallback_status: FallbackStatus::NotUsed,
            degeneracy: Some(DegeneracyReason::Frozen),
        };
        assert!(!frozen.qualifies_as_observed());
    }

    #[test]
    fn test_probe_quality_accepts_responsive_signal() {
        let responsive = ProbeQuality {
            sample_count: Some(10),
            finite_fraction: 1.0,
            variance: Some(0.05),
            responsiveness: Responsiveness::VariesAcrossConditions,
            fallback_status: FallbackStatus::NotUsed,
            degeneracy: None,
        };
        assert!(responsive.qualifies_as_observed());
    }

    #[test]
    fn test_probe_quality_rejects_unknown_fallback_status_for_observed() {
        // The stricter `Observed` bar requires *confirmed* absence of
        // fallback behavior -- `Unknown` must not pass it, even though
        // `qualifies_for_ablation_interpretation()` deliberately does let
        // `Unknown` through (see that method's doc comment).
        let unknown_fallback = ProbeQuality {
            sample_count: Some(10),
            finite_fraction: 1.0,
            variance: Some(0.05),
            responsiveness: Responsiveness::VariesAcrossConditions,
            fallback_status: FallbackStatus::Unknown,
            degeneracy: None,
        };
        assert!(!unknown_fallback.qualifies_as_observed());
    }

    #[test]
    fn test_none_collected_fallback_status_is_unknown_not_confirmed() {
        // With nothing collected, absence of fallback behavior was never
        // established -- `degeneracy` already disqualifies this probe, but
        // the metadata itself must stay truthful (`Unknown`, not a silent
        // `NotUsed` claim).
        let none = ProbeQuality::none_collected();
        assert_eq!(none.fallback_status, FallbackStatus::Unknown);
        assert_eq!(none.degeneracy, Some(DegeneracyReason::InsufficientSamples));
    }

    #[test]
    fn test_merge_always_discloses_fallback_status_unavailable() {
        // Every ablation-derived indicator must carry the disclosure
        // annotation -- `AblationResult` never tracks fallback provenance,
        // so `Unknown`'s pass-through of the quality gate must never read
        // as tacit confirmation nothing untoward happened.
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
        assert!(merged.indicators[0].annotations.iter().any(|a| matches!(
            a,
            EvidenceAnnotation::KnownConfound(msg) if msg.contains("fallback status unavailable")
        )));
    }

    // ── Cached classification must agree with the recomputed one, or the
    // ── whole bundle is rejected outright (arkh-node round 3) ────────────

    #[test]
    fn test_merge_rejects_inconsistent_contradicted_flag() {
        // A malformed/stale/externally-constructed evidence row claims
        // `contradicted: true` alongside baseline == ablated -- the raw
        // scores say nothing moved at all. Since the real producer now
        // derives this flag from the same canonical classifier, this
        // disagreement can only mean the bundle is malformed -- reject the
        // whole bundle rather than silently recomputing and accepting it.
        let report = ButlinIndicatorReport::from_indicators(vec![stub_indicator(
            "GWT-2",
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        )]);
        let bad_result = stub_ablation_result_raw(
            "GWT-2", 0.5, 0.5, // raw scores: nothing moved
            false, false, true, // stored flag lies: contradicted=true
        );
        let bundle = ButlinEvidenceBundle {
            schema_version: REPORT_SCHEMA_VERSION,
            commit_sha: "test".into(),
            config_hash: "test".into(),
            seeds: vec![1],
            generated_at: "test".into(),
            ablations: vec![bad_result],
        };
        assert_eq!(
            annotate_with_ablation_results(report, &bundle).unwrap_err(),
            EvidenceMergeError::ClassificationMismatch {
                indicator_id: "GWT-2".into(),
                stored: AblationClassification {
                    indicator_dropped: false,
                    benchmark_degraded: false,
                    contradicted: true,
                },
                recomputed: AblationClassification {
                    indicator_dropped: false,
                    benchmark_degraded: false,
                    contradicted: false,
                },
            }
        );
    }

    #[test]
    fn test_merge_rejects_inconsistent_indicator_dropped_flag() {
        // Stored flag says the indicator did NOT drop, but the raw scores
        // show a genuine >50% drop from a usable baseline -- rejected as a
        // malformed bundle rather than silently corrected.
        let report = ButlinIndicatorReport::from_indicators(vec![stub_indicator(
            "RPT-2",
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        )]);
        let bad_result = stub_ablation_result_raw(
            "RPT-2", 0.9, 0.1, // raw scores: a real drop
            false, false, false, // stored flag lies: indicator_dropped=false
        );
        let bundle = ButlinEvidenceBundle {
            schema_version: REPORT_SCHEMA_VERSION,
            commit_sha: "test".into(),
            config_hash: "test".into(),
            seeds: vec![1],
            generated_at: "test".into(),
            ablations: vec![bad_result],
        };
        let err = annotate_with_ablation_results(report, &bundle).unwrap_err();
        assert_eq!(
            err,
            EvidenceMergeError::ClassificationMismatch {
                indicator_id: "RPT-2".into(),
                stored: AblationClassification {
                    indicator_dropped: false,
                    benchmark_degraded: false,
                    contradicted: false,
                },
                recomputed: AblationClassification {
                    indicator_dropped: true,
                    benchmark_degraded: false,
                    contradicted: false,
                },
            }
        );
    }

    #[test]
    fn test_merge_rejects_inconsistent_benchmark_degraded_flag() {
        // Stored flag says the benchmark degraded, but the raw accuracies
        // show no real degradation -- rejected rather than silently
        // downgraded to CausallySupported behind the caller's back.
        let mut bad_result = stub_ablation_result_raw("AE-2", 0.9, 0.1, true, true, false);
        bad_result.baseline_benchmark_accuracy = 0.8;
        bad_result.ablated_benchmark_accuracy = 0.79; // no real degradation
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
            ablations: vec![bad_result],
        };
        let err = annotate_with_ablation_results(report, &bundle).unwrap_err();
        assert_eq!(
            err,
            EvidenceMergeError::ClassificationMismatch {
                indicator_id: "AE-2".into(),
                stored: AblationClassification {
                    indicator_dropped: true,
                    benchmark_degraded: true,
                    contradicted: false,
                },
                recomputed: AblationClassification {
                    indicator_dropped: true,
                    benchmark_degraded: false,
                    contradicted: false,
                },
            }
        );
    }

    #[test]
    fn test_merge_accepts_consistent_classification() {
        // Sanity check: a row whose cached flags genuinely agree with
        // `classify_ablation`'s recomputation (the normal, non-malformed
        // case) must merge successfully, not be rejected.
        let good_result = stub_ablation_result("RPT-1", true, true, false);
        let report = ButlinIndicatorReport::from_indicators(vec![stub_indicator(
            "RPT-1",
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        )]);
        let bundle = ButlinEvidenceBundle {
            schema_version: REPORT_SCHEMA_VERSION,
            commit_sha: "test".into(),
            config_hash: "test".into(),
            seeds: vec![1],
            generated_at: "test".into(),
            ablations: vec![good_result],
        };
        let merged = annotate_with_ablation_results(report, &bundle).unwrap();
        assert_eq!(
            merged.indicators[0].outcome,
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported)
        );
    }

    // ── Downstream-benchmark measurement validity gates FunctionallySupported ──

    #[test]
    fn test_merge_infinite_benchmark_baseline_caps_at_causally_supported() {
        // An infinite baseline accuracy trivially satisfies
        // `classify_ablation`'s degradation comparison (anything finite is
        // "below" infinity*0.7), so `benchmark_degraded` recomputes to
        // `true` and agrees with the stored flag -- fix #2's
        // classification-consistency check alone would NOT catch this,
        // since stored and recomputed agree. `benchmark_measurement_is_valid`
        // is the guard that actually catches it.
        let mut result = stub_ablation_result_raw("AE-1", 0.9, 0.1, true, true, false);
        result.baseline_benchmark_accuracy = f64::INFINITY;
        result.ablated_benchmark_accuracy = 0.2;
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
            ablations: vec![result],
        };
        let merged = annotate_with_ablation_results(report, &bundle).unwrap();
        assert_eq!(
            merged.indicators[0].outcome,
            EvidenceOutcome::Supported(SupportTier::CausallySupported),
            "an infinite benchmark baseline must not manufacture functional \
             evidence, even though the indicator's own probe is genuinely \
             causally supported"
        );
        assert!(merged.indicators[0].annotations.iter().any(|a| matches!(
            a,
            EvidenceAnnotation::KnownConfound(msg) if msg.contains("benchmark measurement invalid")
        )));
    }

    #[test]
    fn test_merge_out_of_unit_range_benchmark_accuracy_caps_at_causally_supported() {
        // Finite but outside [0.0, 1.0] -- still not a legitimate accuracy
        // value, even though it isn't caught by a finiteness check alone.
        let mut result = stub_ablation_result_raw("AE-2", 0.9, 0.1, true, true, false);
        result.baseline_benchmark_accuracy = 2.0;
        result.ablated_benchmark_accuracy = 0.1;
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
            ablations: vec![result],
        };
        let merged = annotate_with_ablation_results(report, &bundle).unwrap();
        assert_eq!(
            merged.indicators[0].outcome,
            EvidenceOutcome::Supported(SupportTier::CausallySupported)
        );
    }

    #[test]
    fn test_merge_rejects_v2_evidence_bundle() {
        // A bundle built against the pre-Inconclusive schema (v2) must be
        // refused outright, not silently accepted with a missing variant.
        let report = ButlinIndicatorReport::from_indicators(vec![stub_indicator(
            "AE-1",
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        )]);
        let bundle = ButlinEvidenceBundle {
            schema_version: 2,
            commit_sha: "test".into(),
            config_hash: "test".into(),
            seeds: vec![1],
            generated_at: "test".into(),
            ablations: vec![],
        };
        assert_eq!(
            annotate_with_ablation_results(report, &bundle).unwrap_err(),
            EvidenceMergeError::SchemaVersionMismatch {
                expected: REPORT_SCHEMA_VERSION,
                found: 2,
            }
        );
    }
}
