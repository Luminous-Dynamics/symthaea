// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed, verifier-owned authority contract for Butlin functional evidence.
//!
//! This module is deliberately **conformance logic, not empirical evidence**.
//! It defines what a future real runner must prove before
//! `SupportTier::FunctionallySupported` can be authorized. Synthetic tests
//! here establish only that the software contract fails closed.
//!
//! The legacy `ButlinEvidenceBundle` does not carry a typed identity binding
//! its indicator-ablation and downstream observations to the same
//! intervention instance. That path is handled separately by issue #979.
//! This module is the positive re-qualification path from #1015: both
//! observations must carry exactly one shared `InterventionIdentity`, the
//! declared target/probe/benchmark must match `QualificationDesign`, and the
//! verifier derives outcomes from raw measurements rather than trusting
//! caller-supplied effect booleans.
//!
//! The module is crate-internal until a real runner owns construction of its
//! inputs. Exposing a public constructor-only conformance API before that
//! runner exists would let downstream callers manufacture attractive
//! synthetic inputs and mistake a software-contract pass for empirical
//! authority.

use super::qualification_design::QualificationDesign;
use super::qualification_runtime::RuntimeQualification;
use super::report::{classify_ablation, EffectEstimate, EvidenceOutcome, SupportTier};

/// Stable identity for one concrete intervention lineage.
///
/// Equality is intentionally strict. Matching only a human-readable lever
/// name is insufficient: source commit, intervention configuration, exact
/// seed identities, execution context, and run identity all participate in
/// the authority boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InterventionIdentity {
    target_lever: String,
    target_lever_group: String,
    source_commit_sha: String,
    intervention_config_hash: String,
    seeds: Vec<u64>,
    execution_context_hash: String,
    run_id: String,
}

impl InterventionIdentity {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        target_lever: impl Into<String>,
        target_lever_group: impl Into<String>,
        source_commit_sha: impl Into<String>,
        intervention_config_hash: impl Into<String>,
        mut seeds: Vec<u64>,
        execution_context_hash: impl Into<String>,
        run_id: impl Into<String>,
    ) -> Result<Self, FunctionalAuthorityError> {
        let source_commit_sha = source_commit_sha.into();
        if !is_full_git_oid(&source_commit_sha) {
            return Err(FunctionalAuthorityError::InvalidSourceCommitIdentity);
        }
        if seeds.is_empty() {
            return Err(FunctionalAuthorityError::MissingSeedIdentity);
        }
        // A seed set is identity, not evidence multiplicity supplied by a
        // caller. Canonicalize order and reject duplicates rather than
        // silently counting repeated seed IDs as independent runs.
        seeds.sort_unstable();
        if seeds.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(FunctionalAuthorityError::DuplicateSeedIdentity);
        }

        let identity = Self {
            target_lever: target_lever.into(),
            target_lever_group: target_lever_group.into(),
            source_commit_sha,
            intervention_config_hash: intervention_config_hash.into(),
            seeds,
            execution_context_hash: execution_context_hash.into(),
            run_id: run_id.into(),
        };

        for (field, value) in [
            ("target_lever", identity.target_lever.as_str()),
            ("target_lever_group", identity.target_lever_group.as_str()),
            (
                "intervention_config_hash",
                identity.intervention_config_hash.as_str(),
            ),
            (
                "execution_context_hash",
                identity.execution_context_hash.as_str(),
            ),
            ("run_id", identity.run_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(FunctionalAuthorityError::MissingIdentityField(field));
            }
        }

        Ok(identity)
    }

    pub(crate) fn target_lever(&self) -> &str {
        &self.target_lever
    }

    pub(crate) fn target_lever_group(&self) -> &str {
        &self.target_lever_group
    }

    pub(crate) fn source_commit_sha(&self) -> &str {
        &self.source_commit_sha
    }

    pub(crate) fn intervention_config_hash(&self) -> &str {
        &self.intervention_config_hash
    }

    pub(crate) fn seeds(&self) -> &[u64] {
        &self.seeds
    }

    pub(crate) fn execution_context_hash(&self) -> &str {
        &self.execution_context_hash
    }

    pub(crate) fn run_id(&self) -> &str {
        &self.run_id
    }
}

fn is_full_git_oid(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Causal indicator observation bound to one intervention identity.
///
/// The effect estimate is constructed internally from raw means so callers
/// cannot smuggle inconsistent cached `absolute_change`, `relative_change`,
/// or `seed_count` fields across the authority boundary.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CausalObservation {
    intervention: InterventionIdentity,
    metric_id: String,
    effect: EffectEstimate,
}

impl CausalObservation {
    pub(crate) fn new(
        intervention: InterventionIdentity,
        metric_id: impl Into<String>,
        baseline_mean: f64,
        intervention_mean: f64,
        standard_deviation: Option<f64>,
    ) -> Result<Self, FunctionalAuthorityError> {
        let metric_id = metric_id.into();
        if metric_id.trim().is_empty() {
            return Err(FunctionalAuthorityError::MissingMetricIdentity("causal"));
        }
        validate_finite_measurement(&metric_id, baseline_mean, intervention_mean)?;
        validate_std_dev(&metric_id, standard_deviation)?;

        let mut effect = EffectEstimate::new(
            baseline_mean,
            intervention_mean,
            intervention.seeds().len(),
        );
        if let Some(sd) = standard_deviation {
            effect = effect.with_std_dev(sd);
        }
        Ok(Self {
            intervention,
            metric_id,
            effect,
        })
    }

    pub(crate) fn intervention(&self) -> &InterventionIdentity {
        &self.intervention
    }

    pub(crate) fn metric_id(&self) -> &str {
        &self.metric_id
    }

    pub(crate) fn effect(&self) -> &EffectEstimate {
        &self.effect
    }
}

/// Downstream behavioral accuracy observation bound to one intervention.
///
/// The current Butlin functional endpoint is an accuracy/proportion, so this
/// constructor validates the `[0, 1]` measurement domain as well as
/// finiteness. A future non-accuracy endpoint should receive its own typed
/// observation rather than weakening this constructor.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FunctionalAccuracyObservation {
    intervention: InterventionIdentity,
    benchmark_id: String,
    effect: EffectEstimate,
}

impl FunctionalAccuracyObservation {
    pub(crate) fn new(
        intervention: InterventionIdentity,
        benchmark_id: impl Into<String>,
        baseline_mean: f64,
        intervention_mean: f64,
        standard_deviation: Option<f64>,
    ) -> Result<Self, FunctionalAuthorityError> {
        let benchmark_id = benchmark_id.into();
        if benchmark_id.trim().is_empty() {
            return Err(FunctionalAuthorityError::MissingMetricIdentity(
                "functional",
            ));
        }
        validate_finite_measurement(&benchmark_id, baseline_mean, intervention_mean)?;
        if !(0.0..=1.0).contains(&baseline_mean) || !(0.0..=1.0).contains(&intervention_mean) {
            return Err(FunctionalAuthorityError::InvalidMeasurement {
                metric_id: benchmark_id,
                reason: MeasurementInvalidReason::OutsideUnitInterval,
            });
        }
        validate_std_dev(&benchmark_id, standard_deviation)?;

        let mut effect = EffectEstimate::new(
            baseline_mean,
            intervention_mean,
            intervention.seeds().len(),
        );
        if let Some(sd) = standard_deviation {
            effect = effect.with_std_dev(sd);
        }
        Ok(Self {
            intervention,
            benchmark_id,
            effect,
        })
    }

    pub(crate) fn intervention(&self) -> &InterventionIdentity {
        &self.intervention
    }

    pub(crate) fn benchmark_id(&self) -> &str {
        &self.benchmark_id
    }

    pub(crate) fn effect(&self) -> &EffectEstimate {
        &self.effect
    }
}

fn validate_finite_measurement(
    metric_id: &str,
    baseline_mean: f64,
    intervention_mean: f64,
) -> Result<(), FunctionalAuthorityError> {
    if !baseline_mean.is_finite() || !intervention_mean.is_finite() {
        return Err(FunctionalAuthorityError::InvalidMeasurement {
            metric_id: metric_id.to_string(),
            reason: MeasurementInvalidReason::NonFinite,
        });
    }
    Ok(())
}

fn validate_std_dev(
    metric_id: &str,
    standard_deviation: Option<f64>,
) -> Result<(), FunctionalAuthorityError> {
    if let Some(sd) = standard_deviation {
        if !sd.is_finite() || sd < 0.0 {
            return Err(FunctionalAuthorityError::InvalidMeasurement {
                metric_id: metric_id.to_string(),
                reason: MeasurementInvalidReason::InvalidStandardDeviation,
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MeasurementInvalidReason {
    NonFinite,
    OutsideUnitInterval,
    InvalidStandardDeviation,
}

/// Hard failures in the authority contract. These are malformed or
/// self-inconsistent evidence lineages, not ordinary negative findings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FunctionalAuthorityError {
    MissingIdentityField(&'static str),
    InvalidSourceCommitIdentity,
    MissingSeedIdentity,
    DuplicateSeedIdentity,
    MissingMetricIdentity(&'static str),
    InvalidMeasurement {
        metric_id: String,
        reason: MeasurementInvalidReason,
    },
    InterventionIdentityMismatch,
    DesignBindingMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    QualificationDesignMismatch {
        declared: bool,
        actual: bool,
    },
}

impl std::fmt::Display for FunctionalAuthorityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingIdentityField(field) => {
                write!(f, "intervention identity field {field:?} is empty")
            }
            Self::InvalidSourceCommitIdentity => write!(
                f,
                "source commit identity must be a full 40- or 64-hex Git object id"
            ),
            Self::MissingSeedIdentity => {
                write!(f, "intervention identity contains no exact seed identities")
            }
            Self::DuplicateSeedIdentity => write!(
                f,
                "intervention identity repeats a seed id; duplicate seeds are not independent evidence"
            ),
            Self::MissingMetricIdentity(kind) => {
                write!(f, "{kind} observation has no metric identity")
            }
            Self::InvalidMeasurement { metric_id, reason } => {
                write!(f, "measurement {metric_id:?} is invalid: {reason:?}")
            }
            Self::InterventionIdentityMismatch => write!(
                f,
                "causal and functional observations are not bound to the exact same intervention identity"
            ),
            Self::DesignBindingMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "functional authority {field} binding mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::QualificationDesignMismatch { declared, actual } => write!(
                f,
                "runtime qualification's static-design state ({declared}) disagrees with current design ({actual})"
            ),
        }
    }
}
impl std::error::Error for FunctionalAuthorityError {}

fn require_design_binding(
    field: &'static str,
    expected: &str,
    actual: &str,
) -> Result<(), FunctionalAuthorityError> {
    if expected != actual {
        return Err(FunctionalAuthorityError::DesignBindingMismatch {
            field,
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

/// Verifier-owned transition from qualified, intervention-bound observations
/// to an evidence outcome.
///
/// This remains crate-internal until the real empirical runner owns these
/// inputs. The verifier checks the current static design directly, binds the
/// intervention/probe/benchmark identities to that design, requires the
/// causal and functional observations to share one exact intervention
/// lineage, and derives the current Butlin classification from raw means.
///
/// A run that fails runtime qualification is `Inconclusive`; a qualified
/// causal null is `NotDemonstrated`; an inverse causal response is
/// `Contradicted`; a causal drop without downstream degradation is
/// `CausallySupported`; only a qualified causal + functional effect under
/// the same design-bound intervention reaches `FunctionallySupported`.
pub(crate) fn authorize_functional_evidence(
    design: &QualificationDesign,
    qualification: &RuntimeQualification,
    causal: &CausalObservation,
    functional: &FunctionalAccuracyObservation,
) -> Result<EvidenceOutcome, FunctionalAuthorityError> {
    let static_design_qualifies = design.static_design_qualifies();
    if qualification.static_design_qualifies != static_design_qualifies {
        return Err(FunctionalAuthorityError::QualificationDesignMismatch {
            declared: qualification.static_design_qualifies,
            actual: static_design_qualifies,
        });
    }

    if causal.intervention() != functional.intervention() {
        return Err(FunctionalAuthorityError::InterventionIdentityMismatch);
    }

    require_design_binding(
        "target_lever",
        design.target_lever,
        causal.intervention().target_lever(),
    )?;
    require_design_binding(
        "target_lever_group",
        design.target_lever_group,
        causal.intervention().target_lever_group(),
    )?;
    require_design_binding("probe_metric", design.probe_metric, causal.metric_id())?;
    require_design_binding(
        "functional_benchmark",
        design.functional_benchmark,
        functional.benchmark_id(),
    )?;

    if !static_design_qualifies || !qualification.qualifies_run() {
        return Ok(EvidenceOutcome::Inconclusive);
    }

    // Seed count comes from the exact canonicalized seed identities inside
    // InterventionIdentity. Callers cannot independently supply a count.
    debug_assert_eq!(
        causal.effect().seed_count,
        causal.intervention().seeds().len()
    );
    debug_assert_eq!(
        functional.effect().seed_count,
        functional.intervention().seeds().len()
    );

    let classification = classify_ablation(
        causal.effect().baseline_mean,
        causal.effect().intervention_mean,
        functional.effect().baseline_mean,
        functional.effect().intervention_mean,
    );

    if classification.contradicted {
        Ok(EvidenceOutcome::Contradicted)
    } else if !classification.indicator_dropped {
        Ok(EvidenceOutcome::NotDemonstrated)
    } else if classification.benchmark_degraded {
        Ok(EvidenceOutcome::Supported(
            SupportTier::FunctionallySupported,
        ))
    } else {
        Ok(EvidenceOutcome::Supported(SupportTier::CausallySupported))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::butlin::qualification_design::planned_designs;

    const SOURCE_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn design() -> QualificationDesign {
        *planned_designs()
            .iter()
            .find(|d| d.indicator == "AE-2")
            .expect("AE-2 design must exist")
    }

    fn identity_for(design: &QualificationDesign, run_id: &str) -> InterventionIdentity {
        InterventionIdentity::new(
            design.target_lever,
            design.target_lever_group,
            SOURCE_SHA,
            "cfg:abc123",
            vec![1, 2, 3, 4],
            "env:nix-lock-xyz",
            run_id,
        )
        .unwrap()
    }

    fn qualified(design: &QualificationDesign) -> RuntimeQualification {
        RuntimeQualification {
            static_design_qualifies: design.static_design_qualifies(),
            intervention_applied: true,
            intervention_specificity_passed: true,
            positive_control_effect_observed: true,
            sham_behaved_as_expected: true,
            probe_signal_usable: true,
            identity_and_config_match: true,
        }
    }

    fn causal(
        design: &QualificationDesign,
        id: InterventionIdentity,
        baseline: f64,
        intervention: f64,
    ) -> CausalObservation {
        CausalObservation::new(id, design.probe_metric, baseline, intervention, None).unwrap()
    }

    fn functional(
        design: &QualificationDesign,
        id: InterventionIdentity,
        baseline: f64,
        intervention: f64,
    ) -> FunctionalAccuracyObservation {
        FunctionalAccuracyObservation::new(
            id,
            design.functional_benchmark,
            baseline,
            intervention,
            None,
        )
        .unwrap()
    }

    #[test]
    fn exact_same_design_bound_intervention_can_authorize_functional_support() {
        let design = design();
        assert!(design.static_design_qualifies());
        let id = identity_for(&design, "run-001");
        let outcome = authorize_functional_evidence(
            &design,
            &qualified(&design),
            &causal(&design, id.clone(), 0.9, 0.1),
            &functional(&design, id, 0.8, 0.4),
        )
        .unwrap();
        assert_eq!(
            outcome,
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported)
        );
    }

    #[test]
    fn different_run_identity_fails_closed() {
        let design = design();
        let err = authorize_functional_evidence(
            &design,
            &qualified(&design),
            &causal(&design, identity_for(&design, "run-001"), 0.9, 0.1),
            &functional(&design, identity_for(&design, "run-002"), 0.8, 0.4),
        )
        .unwrap_err();
        assert_eq!(err, FunctionalAuthorityError::InterventionIdentityMismatch);
    }

    #[test]
    fn different_config_identity_fails_closed() {
        let design = design();
        let causal_id = identity_for(&design, "run-001");
        let functional_id = InterventionIdentity::new(
            design.target_lever,
            design.target_lever_group,
            SOURCE_SHA,
            "cfg:different",
            vec![1, 2, 3, 4],
            "env:nix-lock-xyz",
            "run-001",
        )
        .unwrap();
        let err = authorize_functional_evidence(
            &design,
            &qualified(&design),
            &causal(&design, causal_id, 0.9, 0.1),
            &functional(&design, functional_id, 0.8, 0.4),
        )
        .unwrap_err();
        assert_eq!(err, FunctionalAuthorityError::InterventionIdentityMismatch);
    }

    #[test]
    fn wrong_target_lever_cannot_hide_behind_all_true_runtime_flags() {
        let design = design();
        let wrong_id = InterventionIdentity::new(
            "some_other_lever",
            design.target_lever_group,
            SOURCE_SHA,
            "cfg:abc123",
            vec![1, 2, 3, 4],
            "env:nix-lock-xyz",
            "run-001",
        )
        .unwrap();
        let err = authorize_functional_evidence(
            &design,
            &qualified(&design),
            &causal(&design, wrong_id.clone(), 0.9, 0.1),
            &functional(&design, wrong_id, 0.8, 0.4),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            FunctionalAuthorityError::DesignBindingMismatch {
                field: "target_lever",
                ..
            }
        ));
    }

    #[test]
    fn wrong_probe_metric_fails_closed() {
        let design = design();
        let id = identity_for(&design, "run-001");
        let bad_causal = CausalObservation::new(id.clone(), "wrong_probe", 0.9, 0.1, None).unwrap();
        let err = authorize_functional_evidence(
            &design,
            &qualified(&design),
            &bad_causal,
            &functional(&design, id, 0.8, 0.4),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            FunctionalAuthorityError::DesignBindingMismatch {
                field: "probe_metric",
                ..
            }
        ));
    }

    #[test]
    fn wrong_functional_benchmark_fails_closed() {
        let design = design();
        let id = identity_for(&design, "run-001");
        let bad_functional =
            FunctionalAccuracyObservation::new(id.clone(), "Wrong::Benchmark", 0.8, 0.4, None)
                .unwrap();
        let err = authorize_functional_evidence(
            &design,
            &qualified(&design),
            &causal(&design, id, 0.9, 0.1),
            &bad_functional,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            FunctionalAuthorityError::DesignBindingMismatch {
                field: "functional_benchmark",
                ..
            }
        ));
    }

    #[test]
    fn stale_static_qualification_state_is_rejected() {
        let design = design();
        let id = identity_for(&design, "run-001");
        let mut q = qualified(&design);
        q.static_design_qualifies = !design.static_design_qualifies();
        let err = authorize_functional_evidence(
            &design,
            &q,
            &causal(&design, id.clone(), 0.9, 0.1),
            &functional(&design, id, 0.8, 0.4),
        )
        .unwrap_err();
        assert_eq!(
            err,
            FunctionalAuthorityError::QualificationDesignMismatch {
                declared: false,
                actual: true,
            }
        );
    }

    #[test]
    fn failed_runtime_qualification_is_inconclusive_even_when_effects_look_good() {
        let design = design();
        let id = identity_for(&design, "run-001");
        let mut q = qualified(&design);
        q.positive_control_effect_observed = false;
        let outcome = authorize_functional_evidence(
            &design,
            &q,
            &causal(&design, id.clone(), 0.9, 0.1),
            &functional(&design, id, 0.8, 0.4),
        )
        .unwrap();
        assert_eq!(outcome, EvidenceOutcome::Inconclusive);
    }

    #[test]
    fn causal_drop_without_functional_drop_caps_at_causal_support() {
        let design = design();
        let id = identity_for(&design, "run-001");
        let outcome = authorize_functional_evidence(
            &design,
            &qualified(&design),
            &causal(&design, id.clone(), 0.9, 0.1),
            &functional(&design, id, 0.8, 0.79),
        )
        .unwrap();
        assert_eq!(
            outcome,
            EvidenceOutcome::Supported(SupportTier::CausallySupported)
        );
    }

    #[test]
    fn qualified_causal_null_is_not_demonstrated() {
        let design = design();
        let id = identity_for(&design, "run-001");
        let outcome = authorize_functional_evidence(
            &design,
            &qualified(&design),
            &causal(&design, id.clone(), 0.9, 0.89),
            &functional(&design, id, 0.8, 0.4),
        )
        .unwrap();
        assert_eq!(outcome, EvidenceOutcome::NotDemonstrated);
    }

    #[test]
    fn qualified_inverse_causal_effect_is_contradicted() {
        let design = design();
        let id = identity_for(&design, "run-001");
        let outcome = authorize_functional_evidence(
            &design,
            &qualified(&design),
            &causal(&design, id.clone(), 0.9, 1.4),
            &functional(&design, id, 0.8, 0.4),
        )
        .unwrap();
        assert_eq!(outcome, EvidenceOutcome::Contradicted);
    }

    #[test]
    fn seed_order_is_canonicalized_into_identity() {
        let design = design();
        let a = InterventionIdentity::new(
            design.target_lever,
            design.target_lever_group,
            SOURCE_SHA,
            "cfg:abc123",
            vec![4, 2, 1, 3],
            "env:nix-lock-xyz",
            "run-001",
        )
        .unwrap();
        let b = identity_for(&design, "run-001");
        assert_eq!(a, b);
        assert_eq!(a.seeds(), &[1, 2, 3, 4]);
    }

    #[test]
    fn duplicate_seed_identities_are_rejected() {
        let design = design();
        let err = InterventionIdentity::new(
            design.target_lever,
            design.target_lever_group,
            SOURCE_SHA,
            "cfg:abc123",
            vec![1, 2, 2, 3],
            "env:nix-lock-xyz",
            "run-001",
        )
        .unwrap_err();
        assert_eq!(err, FunctionalAuthorityError::DuplicateSeedIdentity);
    }

    #[test]
    fn empty_seed_identity_is_rejected_at_identity_construction() {
        let design = design();
        let err = InterventionIdentity::new(
            design.target_lever,
            design.target_lever_group,
            SOURCE_SHA,
            "cfg:abc123",
            vec![],
            "env:nix-lock-xyz",
            "run-001",
        )
        .unwrap_err();
        assert_eq!(err, FunctionalAuthorityError::MissingSeedIdentity);
    }

    #[test]
    fn invalid_source_commit_is_rejected() {
        let design = design();
        let err = InterventionIdentity::new(
            design.target_lever,
            design.target_lever_group,
            "short-sha",
            "cfg:abc123",
            vec![1, 2, 3, 4],
            "env:nix-lock-xyz",
            "run-001",
        )
        .unwrap_err();
        assert_eq!(err, FunctionalAuthorityError::InvalidSourceCommitIdentity);
    }

    #[test]
    fn functional_accuracy_constructor_rejects_out_of_range_values() {
        let design = design();
        let err = FunctionalAccuracyObservation::new(
            identity_for(&design, "run-001"),
            design.functional_benchmark,
            1.2,
            0.4,
            None,
        )
        .unwrap_err();
        assert_eq!(
            err,
            FunctionalAuthorityError::InvalidMeasurement {
                metric_id: design.functional_benchmark.into(),
                reason: MeasurementInvalidReason::OutsideUnitInterval,
            }
        );
    }

    #[test]
    fn negative_or_nonfinite_dispersion_is_rejected() {
        let design = design();
        let id = identity_for(&design, "run-001");
        let err = CausalObservation::new(id, design.probe_metric, 0.9, 0.1, Some(-0.1))
            .unwrap_err();
        assert_eq!(
            err,
            FunctionalAuthorityError::InvalidMeasurement {
                metric_id: design.probe_metric.into(),
                reason: MeasurementInvalidReason::InvalidStandardDeviation,
            }
        );
    }

    #[test]
    fn intervention_identity_refuses_missing_provenance_fields() {
        let design = design();
        let err = InterventionIdentity::new(
            design.target_lever,
            design.target_lever_group,
            SOURCE_SHA,
            "",
            vec![1, 2, 3, 4],
            "env:nix-lock-xyz",
            "run-001",
        )
        .unwrap_err();
        assert_eq!(
            err,
            FunctionalAuthorityError::MissingIdentityField("intervention_config_hash")
        );
    }

    #[test]
    fn identity_exposes_exact_provenance_read_only() {
        let design = design();
        let id = identity_for(&design, "run-001");
        assert_eq!(id.source_commit_sha(), SOURCE_SHA);
        assert_eq!(id.intervention_config_hash(), "cfg:abc123");
        assert_eq!(id.execution_context_hash(), "env:nix-lock-xyz");
        assert_eq!(id.run_id(), "run-001");
    }
}
