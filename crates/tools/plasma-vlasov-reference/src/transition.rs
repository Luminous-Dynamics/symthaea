//! Fail-closed qualification for proposed kinetic phase-space representation handoffs.
//!
//! This module evaluates *measured transition evidence*. It does not perform a
//! transfer, establish a rigorous continuum error bound, or authorize a live
//! solver to change representation. Its tolerances must be frozen by an
//! independently calibrated benchmark profile before evaluating held-out runs.
//!
//! All distribution discrepancies must be computed on the same declared
//! diagnostic domain and quadrature. Relative errors must use a documented,
//! frozen denominator/floor convention. A successful receipt is therefore an
//! empirical, profile-scoped qualification—not a mathematical certificate.

/// Physical semantics of the represented distribution.
///
/// Positivity is checked only for a genuinely nonnegative distribution. It is
/// deliberately not inferred from the numerical representation's name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributionSemantics {
    /// The represented state is a physical distribution known to satisfy f >= 0.
    Nonnegative,
    /// The represented state may be signed (for example, delta-f).
    Signed,
}

/// A measured transition from one identified representation state to another.
///
/// Representation identifiers should include the numerical family, revision,
/// discretization/basis and relevant configuration (for example,
/// `dense-fourier:nx=64:nv=128:v=[-6,6):solver=sha256:...`).
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionEvidence {
    /// Stable identifier for this attempted transition.
    pub transition_id: String,
    /// Exact source representation and configuration identity.
    pub source_representation_id: String,
    /// Exact destination representation and configuration identity.
    pub destination_representation_id: String,
    /// SHA-256 of canonical source-state bytes or a canonical state manifest.
    pub source_state_sha256: String,
    /// SHA-256 of canonical destination-state bytes or a canonical state manifest.
    pub destination_state_sha256: String,
    /// Whether positivity is a valid physical constraint for this state.
    pub distribution_semantics: DistributionSemantics,

    /// Weighted L1 discrepancy relative to the source-state norm.
    pub weighted_l1_relative_error: Option<f64>,
    /// Weighted L2 discrepancy relative to the source-state norm.
    pub weighted_l2_relative_error: Option<f64>,
    /// Relative change in mass using the benchmark's frozen denominator floor.
    pub mass_relative_error: Option<f64>,
    /// Absolute change in momentum in the benchmark's declared units.
    pub momentum_absolute_error: Option<f64>,
    /// Relative change in total energy using the frozen denominator floor.
    pub total_energy_relative_error: Option<f64>,
    /// Destination field/Poisson residual for its own declared discrete operator.
    pub destination_field_residual: Option<f64>,

    /// Minimum destination value; required only for Nonnegative semantics.
    pub destination_minimum_distribution: Option<f64>,
    /// Negative mass fraction: integral(max(-f, 0)) / integral(abs(f));
    /// required only for Nonnegative semantics.
    pub destination_negative_mass_fraction: Option<f64>,

    /// Peak measured memory attributable to the transition, including overlap.
    pub peak_memory_bytes: Option<u64>,
    /// Measured end-to-end transition wall time, including allocation and checks.
    pub wall_time_nanoseconds: Option<u64>,
}

/// Frozen acceptance budget for one declared transition profile.
///
/// These are empirical guardrails. They are not automatically suitable for
/// every equation cohort or physical regime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitionBudget {
    pub max_weighted_l1_relative_error: f64,
    pub max_weighted_l2_relative_error: f64,
    pub max_mass_relative_error: f64,
    pub max_momentum_absolute_error: f64,
    pub max_total_energy_relative_error: f64,
    pub max_destination_field_residual: f64,
    /// Minimum allowed value for a nonnegative distribution. A small negative
    /// floor may be used only when justified by the frozen discretization.
    pub minimum_destination_distribution: f64,
    pub max_destination_negative_mass_fraction: f64,
    pub max_peak_memory_bytes: u64,
    pub max_wall_time_nanoseconds: u64,
}

impl TransitionBudget {
    /// Check that all configured limits are finite and physically meaningful.
    pub fn validate(self) -> Result<(), InvalidTransitionBudget> {
        let nonnegative_finite = [
            (
                "max_weighted_l1_relative_error",
                self.max_weighted_l1_relative_error,
            ),
            (
                "max_weighted_l2_relative_error",
                self.max_weighted_l2_relative_error,
            ),
            ("max_mass_relative_error", self.max_mass_relative_error),
            (
                "max_momentum_absolute_error",
                self.max_momentum_absolute_error,
            ),
            (
                "max_total_energy_relative_error",
                self.max_total_energy_relative_error,
            ),
            (
                "max_destination_field_residual",
                self.max_destination_field_residual,
            ),
            (
                "max_destination_negative_mass_fraction",
                self.max_destination_negative_mass_fraction,
            ),
        ];
        for (field, value) in nonnegative_finite {
            if !value.is_finite() || value < 0.0 {
                return Err(InvalidTransitionBudget { field });
            }
        }
        if !self.minimum_destination_distribution.is_finite() {
            return Err(InvalidTransitionBudget {
                field: "minimum_destination_distribution",
            });
        }
        if self.max_destination_negative_mass_fraction > 1.0 {
            return Err(InvalidTransitionBudget {
                field: "max_destination_negative_mass_fraction",
            });
        }
        if self.max_peak_memory_bytes == 0 {
            return Err(InvalidTransitionBudget {
                field: "max_peak_memory_bytes",
            });
        }
        if self.max_wall_time_nanoseconds == 0 {
            return Err(InvalidTransitionBudget {
                field: "max_wall_time_nanoseconds",
            });
        }
        Ok(())
    }
}

/// Invalid frozen budget configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidTransitionBudget {
    pub field: &'static str,
}

/// Numeric quantities that can be checked against a transition budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionMetric {
    WeightedL1RelativeError,
    WeightedL2RelativeError,
    MassRelativeError,
    MomentumAbsoluteError,
    TotalEnergyRelativeError,
    DestinationFieldResidual,
    DestinationMinimumDistribution,
    DestinationNegativeMassFraction,
    PeakMemoryBytes,
    WallTimeNanoseconds,
}

/// Which digest field failed canonical SHA-256 validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionDigest {
    SourceState,
    DestinationState,
}

/// Why a transition could not be accepted.
#[derive(Debug, Clone, PartialEq)]
pub enum TransitionReason {
    EmptyIdentifier,
    InvalidDigest {
        digest: TransitionDigest,
    },
    InvalidBudget {
        field: &'static str,
    },
    MissingMetric {
        metric: TransitionMetric,
    },
    InvalidMetric {
        metric: TransitionMetric,
    },
    LimitExceeded {
        metric: TransitionMetric,
        observed: f64,
        limit: f64,
    },
    MemoryBudgetExceeded {
        observed_bytes: u64,
        limit_bytes: u64,
    },
    WallTimeBudgetExceeded {
        observed_nanoseconds: u64,
        limit_nanoseconds: u64,
    },
}

/// The outcome of evaluating the available evidence against a frozen budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionDecision {
    /// All required evidence was present, valid, and within the frozen budget.
    Accepted,
    /// At least one measured requirement exceeded its frozen limit.
    Rejected,
    /// Evidence was incomplete or malformed; no pass may be inferred.
    Inconclusive,
}

/// Auditable result of one representation-transition qualification attempt.
///
/// The receipt preserves the identifying evidence and every reason emitted by
/// the evaluator. Callers should persist the full input evidence and budget
/// alongside this decision, bound to the run manifest and source revision.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionReceipt {
    pub transition_id: String,
    pub source_representation_id: String,
    pub destination_representation_id: String,
    pub source_state_sha256: String,
    pub destination_state_sha256: String,
    pub decision: TransitionDecision,
    pub reasons: Vec<TransitionReason>,
}

/// Evaluate one proposed transition. Missing or malformed measurements are
/// inconclusive. A measured budget violation is a rejection and takes
/// precedence over missing evidence: a known failure cannot be rescued by an
/// unrelated missing measurement.
pub fn qualify_transition(
    evidence: &TransitionEvidence,
    budget: TransitionBudget,
) -> TransitionReceipt {
    let mut reasons = Vec::new();

    if evidence.transition_id.trim().is_empty()
        || evidence.source_representation_id.trim().is_empty()
        || evidence.destination_representation_id.trim().is_empty()
    {
        reasons.push(TransitionReason::EmptyIdentifier);
    }

    if !is_sha256_hex(&evidence.source_state_sha256) {
        reasons.push(TransitionReason::InvalidDigest {
            digest: TransitionDigest::SourceState,
        });
    }
    if !is_sha256_hex(&evidence.destination_state_sha256) {
        reasons.push(TransitionReason::InvalidDigest {
            digest: TransitionDigest::DestinationState,
        });
    }

    if let Err(error) = budget.validate() {
        reasons.push(TransitionReason::InvalidBudget { field: error.field });
    } else {
        check_bounded_metric(
            &mut reasons,
            TransitionMetric::WeightedL1RelativeError,
            evidence.weighted_l1_relative_error,
            budget.max_weighted_l1_relative_error,
        );
        check_bounded_metric(
            &mut reasons,
            TransitionMetric::WeightedL2RelativeError,
            evidence.weighted_l2_relative_error,
            budget.max_weighted_l2_relative_error,
        );
        check_bounded_metric(
            &mut reasons,
            TransitionMetric::MassRelativeError,
            evidence.mass_relative_error,
            budget.max_mass_relative_error,
        );
        check_bounded_metric(
            &mut reasons,
            TransitionMetric::MomentumAbsoluteError,
            evidence.momentum_absolute_error,
            budget.max_momentum_absolute_error,
        );
        check_bounded_metric(
            &mut reasons,
            TransitionMetric::TotalEnergyRelativeError,
            evidence.total_energy_relative_error,
            budget.max_total_energy_relative_error,
        );
        check_bounded_metric(
            &mut reasons,
            TransitionMetric::DestinationFieldResidual,
            evidence.destination_field_residual,
            budget.max_destination_field_residual,
        );

        if evidence.distribution_semantics == DistributionSemantics::Nonnegative {
            match evidence.destination_minimum_distribution {
                None => reasons.push(TransitionReason::MissingMetric {
                    metric: TransitionMetric::DestinationMinimumDistribution,
                }),
                Some(value) if !value.is_finite() => {
                    reasons.push(TransitionReason::InvalidMetric {
                        metric: TransitionMetric::DestinationMinimumDistribution,
                    });
                }
                Some(value) if value < budget.minimum_destination_distribution => {
                    reasons.push(TransitionReason::LimitExceeded {
                        metric: TransitionMetric::DestinationMinimumDistribution,
                        observed: value,
                        limit: budget.minimum_destination_distribution,
                    });
                }
                Some(_) => {}
            }
            check_fraction_metric(
                &mut reasons,
                evidence.destination_negative_mass_fraction,
                budget.max_destination_negative_mass_fraction,
            );
        }

        match evidence.peak_memory_bytes {
            None => reasons.push(TransitionReason::MissingMetric {
                metric: TransitionMetric::PeakMemoryBytes,
            }),
            Some(0) => reasons.push(TransitionReason::InvalidMetric {
                metric: TransitionMetric::PeakMemoryBytes,
            }),
            Some(value) if value > budget.max_peak_memory_bytes => {
                reasons.push(TransitionReason::MemoryBudgetExceeded {
                    observed_bytes: value,
                    limit_bytes: budget.max_peak_memory_bytes,
                });
            }
            Some(_) => {}
        }

        match evidence.wall_time_nanoseconds {
            None => reasons.push(TransitionReason::MissingMetric {
                metric: TransitionMetric::WallTimeNanoseconds,
            }),
            Some(0) => reasons.push(TransitionReason::InvalidMetric {
                metric: TransitionMetric::WallTimeNanoseconds,
            }),
            Some(value) if value > budget.max_wall_time_nanoseconds => {
                reasons.push(TransitionReason::WallTimeBudgetExceeded {
                    observed_nanoseconds: value,
                    limit_nanoseconds: budget.max_wall_time_nanoseconds,
                });
            }
            Some(_) => {}
        }
    }

    let has_known_violation = reasons.iter().any(|reason| {
        matches!(
            reason,
            TransitionReason::LimitExceeded { .. }
                | TransitionReason::MemoryBudgetExceeded { .. }
                | TransitionReason::WallTimeBudgetExceeded { .. }
        )
    });
    let decision = if has_known_violation {
        TransitionDecision::Rejected
    } else if reasons.is_empty() {
        TransitionDecision::Accepted
    } else {
        TransitionDecision::Inconclusive
    };

    TransitionReceipt {
        transition_id: evidence.transition_id.clone(),
        source_representation_id: evidence.source_representation_id.clone(),
        destination_representation_id: evidence.destination_representation_id.clone(),
        source_state_sha256: evidence.source_state_sha256.clone(),
        destination_state_sha256: evidence.destination_state_sha256.clone(),
        decision,
        reasons,
    }
}

fn check_bounded_metric(
    reasons: &mut Vec<TransitionReason>,
    metric: TransitionMetric,
    observed: Option<f64>,
    limit: f64,
) {
    match observed {
        None => reasons.push(TransitionReason::MissingMetric { metric }),
        Some(value) if !value.is_finite() || value < 0.0 => {
            reasons.push(TransitionReason::InvalidMetric { metric });
        }
        Some(value) if value > limit => {
            reasons.push(TransitionReason::LimitExceeded {
                metric,
                observed: value,
                limit,
            });
        }
        Some(_) => {}
    }
}

fn check_fraction_metric(reasons: &mut Vec<TransitionReason>, observed: Option<f64>, limit: f64) {
    match observed {
        None => reasons.push(TransitionReason::MissingMetric {
            metric: TransitionMetric::DestinationNegativeMassFraction,
        }),
        Some(value) if !value.is_finite() || !(0.0..=1.0).contains(&value) => {
            reasons.push(TransitionReason::InvalidMetric {
                metric: TransitionMetric::DestinationNegativeMassFraction,
            });
        }
        Some(value) if value > limit => {
            reasons.push(TransitionReason::LimitExceeded {
                metric: TransitionMetric::DestinationNegativeMassFraction,
                observed: value,
                limit,
            });
        }
        Some(_) => {}
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn budget() -> TransitionBudget {
        TransitionBudget {
            max_weighted_l1_relative_error: 1.0e-6,
            max_weighted_l2_relative_error: 1.0e-6,
            max_mass_relative_error: 1.0e-10,
            max_momentum_absolute_error: 1.0e-9,
            max_total_energy_relative_error: 1.0e-8,
            max_destination_field_residual: 1.0e-10,
            minimum_destination_distribution: -1.0e-13,
            max_destination_negative_mass_fraction: 1.0e-12,
            max_peak_memory_bytes: 1_000_000,
            max_wall_time_nanoseconds: 1_000_000,
        }
    }

    fn evidence() -> TransitionEvidence {
        TransitionEvidence {
            transition_id: "transition-0001".to_owned(),
            source_representation_id: "dense-fourier:nx=64:nv=128".to_owned(),
            destination_representation_id: "dense-fourier:nx=128:nv=128".to_owned(),
            source_state_sha256: HASH_A.to_owned(),
            destination_state_sha256: HASH_B.to_owned(),
            distribution_semantics: DistributionSemantics::Nonnegative,
            weighted_l1_relative_error: Some(2.0e-8),
            weighted_l2_relative_error: Some(3.0e-8),
            mass_relative_error: Some(2.0e-12),
            momentum_absolute_error: Some(2.0e-11),
            total_energy_relative_error: Some(4.0e-10),
            destination_field_residual: Some(2.0e-12),
            destination_minimum_distribution: Some(-2.0e-14),
            destination_negative_mass_fraction: Some(4.0e-14),
            peak_memory_bytes: Some(512_000),
            wall_time_nanoseconds: Some(100_000),
        }
    }

    #[test]
    fn accepts_complete_evidence_inside_every_budget() {
        let receipt = qualify_transition(&evidence(), budget());
        assert_eq!(receipt.decision, TransitionDecision::Accepted);
        assert!(receipt.reasons.is_empty());
        assert_eq!(receipt.transition_id, "transition-0001");
    }

    #[test]
    fn rejects_a_known_distribution_error_even_when_other_metrics_are_missing() {
        let mut observed = evidence();
        observed.weighted_l1_relative_error = Some(2.0e-6);
        observed.weighted_l2_relative_error = None;
        let receipt = qualify_transition(&observed, budget());

        assert_eq!(receipt.decision, TransitionDecision::Rejected);
        assert!(receipt.reasons.iter().any(|reason| matches!(
            reason,
            TransitionReason::LimitExceeded {
                metric: TransitionMetric::WeightedL1RelativeError,
                ..
            }
        )));
        assert!(receipt.reasons.iter().any(|reason| matches!(
            reason,
            TransitionReason::MissingMetric {
                metric: TransitionMetric::WeightedL2RelativeError
            }
        )));
    }

    #[test]
    fn missing_required_evidence_is_inconclusive_not_a_pass() {
        let mut observed = evidence();
        observed.weighted_l2_relative_error = None;
        observed.peak_memory_bytes = None;
        let receipt = qualify_transition(&observed, budget());

        assert_eq!(receipt.decision, TransitionDecision::Inconclusive);
        assert!(!receipt.reasons.is_empty());
    }

    #[test]
    fn signed_delta_f_does_not_receive_an_invalid_positivity_gate() {
        let mut observed = evidence();
        observed.distribution_semantics = DistributionSemantics::Signed;
        observed.destination_minimum_distribution = None;
        observed.destination_negative_mass_fraction = None;
        let receipt = qualify_transition(&observed, budget());

        assert_eq!(receipt.decision, TransitionDecision::Accepted);
    }

    #[test]
    fn nonnegative_distribution_rejects_negative_mass_outside_budget() {
        let mut observed = evidence();
        observed.destination_minimum_distribution = Some(-1.0e-4);
        observed.destination_negative_mass_fraction = Some(1.0e-5);
        let receipt = qualify_transition(&observed, budget());

        assert_eq!(receipt.decision, TransitionDecision::Rejected);
        assert!(receipt.reasons.iter().any(|reason| matches!(
            reason,
            TransitionReason::LimitExceeded {
                metric: TransitionMetric::DestinationMinimumDistribution,
                ..
            }
        )));
        assert!(receipt.reasons.iter().any(|reason| matches!(
            reason,
            TransitionReason::LimitExceeded {
                metric: TransitionMetric::DestinationNegativeMassFraction,
                ..
            }
        )));
    }

    #[test]
    fn resource_budget_exhaustion_rejects_the_transition() {
        let mut observed = evidence();
        observed.peak_memory_bytes = Some(1_000_001);
        observed.wall_time_nanoseconds = Some(1_000_001);
        let receipt = qualify_transition(&observed, budget());

        assert_eq!(receipt.decision, TransitionDecision::Rejected);
        assert!(
            receipt
                .reasons
                .iter()
                .any(|reason| matches!(reason, TransitionReason::MemoryBudgetExceeded { .. }))
        );
        assert!(
            receipt
                .reasons
                .iter()
                .any(|reason| matches!(reason, TransitionReason::WallTimeBudgetExceeded { .. }))
        );
    }

    #[test]
    fn malformed_hashes_and_non_finite_measurements_are_inconclusive() {
        let mut observed = evidence();
        observed.source_state_sha256 = "not-a-sha256".to_owned();
        observed.destination_field_residual = Some(f64::NAN);
        let receipt = qualify_transition(&observed, budget());

        assert_eq!(receipt.decision, TransitionDecision::Inconclusive);
        assert!(receipt.reasons.iter().any(|reason| matches!(
            reason,
            TransitionReason::InvalidDigest {
                digest: TransitionDigest::SourceState
            }
        )));
        assert!(receipt.reasons.iter().any(|reason| matches!(
            reason,
            TransitionReason::InvalidMetric {
                metric: TransitionMetric::DestinationFieldResidual
            }
        )));
    }

    #[test]
    fn invalid_budget_cannot_produce_a_pass() {
        let mut limits = budget();
        limits.max_weighted_l1_relative_error = f64::NAN;
        let receipt = qualify_transition(&evidence(), limits);

        assert_eq!(receipt.decision, TransitionDecision::Inconclusive);
        assert!(matches!(
            receipt.reasons.as_slice(),
            [TransitionReason::InvalidBudget {
                field: "max_weighted_l1_relative_error"
            }]
        ));
    }

    #[test]
    fn empty_identity_and_zero_resource_measurements_are_not_accepted() {
        let mut observed = evidence();
        observed.transition_id.clear();
        observed.peak_memory_bytes = Some(0);
        observed.wall_time_nanoseconds = Some(0);
        let receipt = qualify_transition(&observed, budget());

        assert_eq!(receipt.decision, TransitionDecision::Inconclusive);
        assert!(receipt.reasons.contains(&TransitionReason::EmptyIdentifier));
        assert!(receipt.reasons.iter().any(|reason| matches!(
            reason,
            TransitionReason::InvalidMetric {
                metric: TransitionMetric::PeakMemoryBytes
            }
        )));
        assert!(receipt.reasons.iter().any(|reason| matches!(
            reason,
            TransitionReason::InvalidMetric {
                metric: TransitionMetric::WallTimeNanoseconds
            }
        )));
    }

    #[test]
    fn negative_fraction_above_one_is_malformed_not_accepted() {
        let mut observed = evidence();
        observed.destination_negative_mass_fraction = Some(1.1);
        let receipt = qualify_transition(&observed, budget());

        assert_eq!(receipt.decision, TransitionDecision::Inconclusive);
        assert!(receipt.reasons.iter().any(|reason| matches!(
            reason,
            TransitionReason::InvalidMetric {
                metric: TransitionMetric::DestinationNegativeMassFraction
            }
        )));
    }
}
