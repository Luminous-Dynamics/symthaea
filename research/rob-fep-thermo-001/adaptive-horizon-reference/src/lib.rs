#![forbid(unsafe_code)]

//! Deterministic reference combiner for a viability- and thermodynamics-constrained
//! action commitment horizon.
//!
//! This crate consumes already-qualified integer caps. It does not estimate
//! thermal, actuator, dynamics, freshness, deadline or viability limits, and it
//! does not authorize actuation.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundKind {
    Proposal,
    Freshness,
    Deadline,
    Risk,
    Thermal,
    Actuator,
    Dynamics,
    Viability,
    Queue,
    Continuity,
}

impl BoundKind {
    pub const ALL: [Self; 9] = [
        Self::Proposal,
        Self::Freshness,
        Self::Deadline,
        Self::Risk,
        Self::Thermal,
        Self::Actuator,
        Self::Dynamics,
        Self::Viability,
        Self::Queue,
        Self::Continuity,
    ];
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundStatus {
    Qualified,
    Unknown,
    Stale,
    DigestMismatch,
    InvalidPrefixClosure,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HorizonBound {
    pub kind: BoundKind,
    pub status: BoundStatus,
    pub max_ticks: Option<u32>,
    pub context_digest: Option<String>,
    pub prefix_closed: bool,
}

impl HorizonBound {
    pub fn qualified(kind: BoundKind, max_ticks: u32, context_digest: &str) -> Self {
        Self {
            kind,
            status: BoundStatus::Qualified,
            max_ticks: Some(max_ticks),
            context_digest: Some(context_digest.to_owned()),
            prefix_closed: true,
        }
    }

    pub fn unknown(kind: BoundKind) -> Self {
        Self {
            kind,
            status: BoundStatus::Unknown,
            max_ticks: None,
            context_digest: None,
            prefix_closed: false,
        }
    }

    pub fn stale(kind: BoundKind) -> Self {
        Self {
            kind,
            status: BoundStatus::Stale,
            max_ticks: None,
            context_digest: None,
            prefix_closed: false,
        }
    }

    pub fn digest_mismatch(kind: BoundKind) -> Self {
        Self {
            kind,
            status: BoundStatus::DigestMismatch,
            max_ticks: None,
            context_digest: None,
            prefix_closed: false,
        }
    }

    pub fn invalid_prefix(kind: BoundKind) -> Self {
        Self {
            kind,
            status: BoundStatus::InvalidPrefixClosure,
            max_ticks: None,
            context_digest: None,
            prefix_closed: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FailureKind {
    EmptyCommonContext,
    CandidateModelMismatch,
    ExecutedPrefixMismatch,
    NonCanonicalBoundOrder,
    UnknownEvidence,
    StaleEvidence,
    ProvenanceMismatch,
    InvalidBoundSemantics,
    MissingNumericBound,
    UnexpectedNumericBound,
    MissingContextDigest,
    ContextDigestMismatch,
    MissingPrefixClosure,
    UnexpectedPrefixClosure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoundFailure {
    pub kind: Option<BoundKind>,
    pub reason: FailureKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HorizonRequest {
    pub common_context_digest: String,
    pub candidate_model_matches: bool,
    pub executed_prefix_matches: bool,
    pub minimum_commit_ticks: u32,
    /// Bounds must be in the canonical order declared by BoundKind::ALL.
    pub bounds: [HorizonBound; 10],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HorizonDecision {
    Eligible {
        max_ticks: u32,
        binding_constraints: Vec<BoundKind>,
    },
    ZeroHorizon {
        binding_constraints: Vec<BoundKind>,
    },
    Blocked {
        failures: Vec<BoundFailure>,
    },
}

impl HorizonDecision {
    /// This reference never grants physical actuation authority.
    pub const fn actuation_authorized(&self) -> bool {
        false
    }
}

/// Select the maximum eligible prefix as the minimum of ten qualified caps.
///
/// A positive result means only "EligibleForIndependentProjection". Every
/// bound must be qualified, finite, prefix-closed, and tied to the same
/// common_context_digest. Missing or incompatible evidence blocks selection.
/// A zero/too-short prefix requires a separately qualified fallback.
pub fn select_commitment_horizon(request: &HorizonRequest) -> HorizonDecision {
    let mut failures = Vec::new();

    if request.common_context_digest.trim().is_empty() {
        failures.push(BoundFailure {
            kind: None,
            reason: FailureKind::EmptyCommonContext,
        });
    }
    if !request.candidate_model_matches {
        failures.push(BoundFailure {
            kind: None,
            reason: FailureKind::CandidateModelMismatch,
        });
    }
    if !request.executed_prefix_matches {
        failures.push(BoundFailure {
            kind: None,
            reason: FailureKind::ExecutedPrefixMismatch,
        });
    }

    let mut caps = [0_u32; 10];

    for (index, expected_kind) in BoundKind::ALL.iter().copied().enumerate() {
        let bound = &request.bounds[index];

        if bound.kind != expected_kind {
            failures.push(BoundFailure {
                kind: Some(expected_kind),
                reason: FailureKind::NonCanonicalBoundOrder,
            });
            continue;
        }

        match bound.status {
            BoundStatus::Qualified => {
                match bound.max_ticks {
                    Some(value) => caps[index] = value,
                    None => failures.push(BoundFailure {
                        kind: Some(expected_kind),
                        reason: FailureKind::MissingNumericBound,
                    }),
                }

                match bound.context_digest.as_deref() {
                    Some(value) if !value.trim().is_empty() => {
                        if value != request.common_context_digest {
                            failures.push(BoundFailure {
                                kind: Some(expected_kind),
                                reason: FailureKind::ContextDigestMismatch,
                            });
                        }
                    }
                    _ => failures.push(BoundFailure {
                        kind: Some(expected_kind),
                        reason: FailureKind::MissingContextDigest,
                    }),
                }

                if !bound.prefix_closed {
                    failures.push(BoundFailure {
                        kind: Some(expected_kind),
                        reason: FailureKind::MissingPrefixClosure,
                    });
                }
            }
            BoundStatus::Unknown => {
                failures.push(BoundFailure {
                    kind: Some(expected_kind),
                    reason: FailureKind::UnknownEvidence,
                });
                reject_unqualified_payload(bound, expected_kind, &mut failures);
            }
            BoundStatus::Stale => {
                failures.push(BoundFailure {
                    kind: Some(expected_kind),
                    reason: FailureKind::StaleEvidence,
                });
                reject_unqualified_payload(bound, expected_kind, &mut failures);
            }
            BoundStatus::DigestMismatch => {
                failures.push(BoundFailure {
                    kind: Some(expected_kind),
                    reason: FailureKind::ProvenanceMismatch,
                });
                reject_unqualified_payload(bound, expected_kind, &mut failures);
            }
            BoundStatus::InvalidPrefixClosure => {
                failures.push(BoundFailure {
                    kind: Some(expected_kind),
                    reason: FailureKind::InvalidBoundSemantics,
                });
                reject_unqualified_payload(bound, expected_kind, &mut failures);
            }
        }
    }

    if !failures.is_empty() {
        return HorizonDecision::Blocked { failures };
    }

    let minimum = caps.iter().copied().min().unwrap_or(0);
    let binding_constraints = BoundKind::ALL
        .iter()
        .copied()
        .zip(caps.iter().copied())
        .filter_map(|(kind, cap)| (cap == minimum).then_some(kind))
        .collect::<Vec<_>>();

    if minimum == 0 || minimum < request.minimum_commit_ticks {
        return HorizonDecision::ZeroHorizon {
            binding_constraints,
        };
    }

    HorizonDecision::Eligible {
        max_ticks: minimum,
        binding_constraints,
    }
}

fn reject_unqualified_payload(
    bound: &HorizonBound,
    kind: BoundKind,
    failures: &mut Vec<BoundFailure>,
) {
    if bound.max_ticks.is_some() {
        failures.push(BoundFailure {
            kind: Some(kind),
            reason: FailureKind::UnexpectedNumericBound,
        });
    }
    if bound.context_digest.is_some() {
        failures.push(BoundFailure {
            kind: Some(kind),
            reason: FailureKind::ProvenanceMismatch,
        });
    }
    if bound.prefix_closed {
        failures.push(BoundFailure {
            kind: Some(kind),
            reason: FailureKind::UnexpectedPrefixClosure,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(caps: [u32; 10]) -> HorizonRequest {
        let context = "test-context-v1";
        let bounds = core::array::from_fn(|index| {
            HorizonBound::qualified(BoundKind::ALL[index], caps[index], context)
        });
        HorizonRequest {
            common_context_digest: context.to_owned(),
            candidate_model_matches: true,
            executed_prefix_matches: true,
            minimum_commit_ticks: 1,
            bounds,
        }
    }

    #[test]
    fn minimum_cap_and_all_ties_are_preserved() {
        let input = request([16, 10, 9, 24, 3, 3, 11, 12, 6, 12]);
        assert_eq!(
            select_commitment_horizon(&input),
            HorizonDecision::Eligible {
                max_ticks: 3,
                binding_constraints: vec![BoundKind::Thermal, BoundKind::Actuator],
            }
        );
    }

    #[test]
    fn unknown_bound_blocks_otherwise_finite_caps() {
        let mut input = request([16, 10, 9, 24, 8, 14, 11, 12, 6, 12]);
        input.bounds[3] = HorizonBound::unknown(BoundKind::Thermal);
        assert!(matches!(
            select_commitment_horizon(&input),
            HorizonDecision::Blocked { failures }
                if failures.iter().any(|failure| failure.reason == FailureKind::UnknownEvidence)
        ));
    }

    #[test]
    fn stale_bound_blocks_selection() {
        let mut input = request([16, 10, 9, 24, 8, 14, 11, 12, 6, 12]);
        input.bounds[1] = HorizonBound::stale(BoundKind::Freshness);
        assert!(matches!(
            select_commitment_horizon(&input),
            HorizonDecision::Blocked { failures }
                if failures.iter().any(|failure| failure.reason == FailureKind::StaleEvidence)
        ));
    }

    #[test]
    fn mismatched_context_digest_blocks_selection() {
        let mut input = request([16, 10, 9, 24, 8, 14, 11, 12, 6, 12]);
        input.bounds[3].context_digest = Some("different-action-prefix".to_owned());
        assert!(matches!(
            select_commitment_horizon(&input),
            HorizonDecision::Blocked { failures }
                if failures.iter().any(|failure| failure.reason == FailureKind::ContextDigestMismatch)
        ));
    }

    #[test]
    fn non_prefix_closed_bound_is_rejected() {
        let mut input = request([16, 10, 9, 24, 8, 14, 11, 12, 6, 12]);
        input.bounds[3].prefix_closed = false;
        assert!(matches!(
            select_commitment_horizon(&input),
            HorizonDecision::Blocked { failures }
                if failures.iter().any(|failure| failure.reason == FailureKind::MissingPrefixClosure)
        ));
    }

    #[test]
    fn digest_mismatch_status_blocks_selection() {
        let mut input = request([16, 10, 9, 24, 8, 14, 11, 12, 6, 12]);
        input.bounds[5] = HorizonBound::digest_mismatch(BoundKind::Dynamics);
        assert!(matches!(
            select_commitment_horizon(&input),
            HorizonDecision::Blocked { failures }
                if failures.iter().any(|failure| failure.reason == FailureKind::ProvenanceMismatch)
        ));
    }

    #[test]
    fn zero_cap_requires_fallback() {
        let input = request([16, 10, 0, 24, 8, 14, 11, 12, 6, 12]);
        assert_eq!(
            select_commitment_horizon(&input),
            HorizonDecision::ZeroHorizon {
                binding_constraints: vec![BoundKind::Deadline],
            }
        );
    }

    #[test]
    fn minimum_commit_threshold_can_force_fallback() {
        let mut input = request([16, 10, 9, 24, 8, 14, 11, 12, 2, 12]);
        input.minimum_commit_ticks = 3;
        assert_eq!(
            select_commitment_horizon(&input),
            HorizonDecision::ZeroHorizon {
                binding_constraints: vec![BoundKind::Queue],
            }
        );
    }

    #[test]
    fn candidate_model_mismatch_blocks_selection() {
        let mut input = request([16, 10, 9, 24, 8, 14, 11, 12, 6, 12]);
        input.candidate_model_matches = false;
        assert!(matches!(
            select_commitment_horizon(&input),
            HorizonDecision::Blocked { failures }
                if failures.iter().any(|failure| failure.reason == FailureKind::CandidateModelMismatch)
        ));
    }

    #[test]
    fn executed_prefix_mismatch_blocks_selection() {
        let mut input = request([16, 10, 9, 24, 8, 14, 11, 12, 6, 12]);
        input.executed_prefix_matches = false;
        assert!(matches!(
            select_commitment_horizon(&input),
            HorizonDecision::Blocked { failures }
                if failures.iter().any(|failure| failure.reason == FailureKind::ExecutedPrefixMismatch)
        ));
    }

    #[test]
    fn preallocated_non_recyclable_risk_cap_limits_commitment() {
        let input = request([16, 10, 9, 2, 8, 14, 11, 12, 6, 12]);
        assert_eq!(
            select_commitment_horizon(&input),
            HorizonDecision::Eligible {
                max_ticks: 2,
                binding_constraints: vec![BoundKind::Risk],
            }
        );
    }

    #[test]
    fn unallocated_risk_budget_blocks_selection() {
        let mut input = request([16, 10, 9, 24, 8, 14, 11, 12, 6, 12]);
        input.bounds[3] = HorizonBound::unknown(BoundKind::Risk);
        assert!(matches!(
            select_commitment_horizon(&input),
            HorizonDecision::Blocked { failures }
                if failures.iter().any(|failure| failure.reason == FailureKind::UnknownEvidence)
        ));
    }

    #[test]
    fn no_result_can_authorize_actuation() {
        let input = request([16, 10, 9, 24, 8, 14, 11, 12, 6, 12]);
        let decision = select_commitment_horizon(&input);
        assert!(!decision.actuation_authorized());
    }
}
