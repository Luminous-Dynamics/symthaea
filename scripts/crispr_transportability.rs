//! Robust transportability bounds for paired Recall@K deltas under uncertain target-group mix.
//!
//! This module addresses a specific failure mode: a model may improve macro/micro scores on
//! the source cohort yet lose the advantage when the target population contains different
//! proportions of guide/Cas9, assay, cell-context, or cohort groups.
//!
//! Inputs are group-specific paired Recall@K-delta intervals and preregistered lower/upper
//! bounds on target-group prevalence. The routine computes sharp bounds for the linear
//! mixture delta over the bounded probability simplex, treating each supplied group delta
//! conservatively at its lower/upper bound. It does not estimate the intervals or group
//! proportions, establish exchangeability, or infer performance for unsupported groups.
//!
//! Every target-relevant context must have a permitted, independently supported estimate.
//! Unmeasured or unsupported target contexts fail closed rather than getting zero effect.
//! This is a conditional transportability envelope, not a confidence interval or clinical
//! safety guarantee. Group-delta intervals and target-prevalence bounds need independent
//! provenance, and dependence/joint-coverage limitations must be recorded by the caller.
//!
//! Compile without Cargo/network:
//!   rustc --edition=2024 --test scripts/crispr_transportability.rs -o /tmp/crispr-transport-tests
//!   /tmp/crispr-transport-tests

use std::collections::HashSet;
use std::fmt;

const TOLERANCE: f64 = 1e-12;

#[derive(Clone, Debug, PartialEq)]
pub struct ContextRecallDelta {
    pub context_id: String,
    /// Lower bound for policy Recall@K minus baseline Recall@K in this context.
    pub delta_lower: f64,
    /// Upper bound for policy Recall@K minus baseline Recall@K in this context.
    pub delta_upper: f64,
    /// Permitted target-population prevalence lower bound for this context.
    pub target_prevalence_lower: f64,
    /// Permitted target-population prevalence upper bound for this context.
    pub target_prevalence_upper: f64,
    /// True only if a source/measurement pipeline supports this target context under the
    /// declared benchmark contract. This flag is a gate, not proof of transportability.
    pub context_supported: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MixtureBoundWitness {
    /// Feasible context weights summing to one, keyed by context ID in sorted order.
    pub prevalence_by_context: Vec<(String, f64)>,
    /// Weighted bound value attained by these weights for the selected group delta bounds.
    pub mixture_delta: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransportabilityConclusion {
    /// The entire conservative mixture interval is strictly above zero.
    RobustPositive,
    /// The entire conservative mixture interval is at or below zero.
    RobustNonPositive,
    /// Positive and non-positive target effects both remain compatible with the inputs.
    Inconclusive,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransportabilityEnvelope {
    pub lower_delta: f64,
    pub upper_delta: f64,
    pub conclusion: TransportabilityConclusion,
    pub lower_witness: MixtureBoundWitness,
    pub upper_witness: MixtureBoundWitness,
    pub target_prevalence_lower_sum: f64,
    pub target_prevalence_upper_sum: f64,
    pub context_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TransportabilityError {
    NoContexts,
    EmptyContextId,
    DuplicateContextId(String),
    UnsupportedTargetContext(String),
    InvalidDeltaInterval { context_id: String, lower: f64, upper: f64 },
    InvalidPrevalenceInterval { context_id: String, lower: f64, upper: f64 },
    LowerPrevalenceMassExceedsOne(f64),
    UpperPrevalenceMassBelowOne(f64),
    NumericalInfeasibility(String),
}

impl fmt::Display for TransportabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoContexts => write!(f, "at least one target context is required"),
            Self::EmptyContextId => write!(f, "context ID must not be blank"),
            Self::DuplicateContextId(id) => write!(f, "duplicate context ID: {id}"),
            Self::UnsupportedTargetContext(id) => {
                write!(f, "target context {id} lacks supported context-specific evidence")
            }
            Self::InvalidDeltaInterval { context_id, lower, upper } => write!(
                f,
                "context {context_id} has invalid Recall@K delta interval [{lower}, {upper}]"
            ),
            Self::InvalidPrevalenceInterval { context_id, lower, upper } => write!(
                f,
                "context {context_id} has invalid target prevalence interval [{lower}, {upper}]"
            ),
            Self::LowerPrevalenceMassExceedsOne(sum) => write!(
                f,
                "sum of target prevalence lower bounds is {sum}, greater than one"
            ),
            Self::UpperPrevalenceMassBelowOne(sum) => write!(
                f,
                "sum of target prevalence upper bounds is {sum}, below one"
            ),
            Self::NumericalInfeasibility(message) => {
                write!(f, "bounded-simplex calculation infeasible: {message}")
            }
        }
    }
}

impl std::error::Error for TransportabilityError {}

fn validate_contexts(
    contexts: &[ContextRecallDelta],
) -> Result<(f64, f64), TransportabilityError> {
    if contexts.is_empty() {
        return Err(TransportabilityError::NoContexts);
    }
    let mut ids = HashSet::with_capacity(contexts.len());
    let mut lower_sum = 0.0;
    let mut upper_sum = 0.0;

    for context in contexts {
        if context.context_id.trim().is_empty() {
            return Err(TransportabilityError::EmptyContextId);
        }
        if !ids.insert(context.context_id.as_str()) {
            return Err(TransportabilityError::DuplicateContextId(
                context.context_id.clone(),
            ));
        }
        if !context.context_supported {
            return Err(TransportabilityError::UnsupportedTargetContext(
                context.context_id.clone(),
            ));
        }
        if !context.delta_lower.is_finite()
            || !context.delta_upper.is_finite()
            || context.delta_lower < -1.0
            || context.delta_upper > 1.0
            || context.delta_lower > context.delta_upper
        {
            return Err(TransportabilityError::InvalidDeltaInterval {
                context_id: context.context_id.clone(),
                lower: context.delta_lower,
                upper: context.delta_upper,
            });
        }
        if !context.target_prevalence_lower.is_finite()
            || !context.target_prevalence_upper.is_finite()
            || context.target_prevalence_lower < 0.0
            || context.target_prevalence_upper > 1.0
            || context.target_prevalence_lower > context.target_prevalence_upper
        {
            return Err(TransportabilityError::InvalidPrevalenceInterval {
                context_id: context.context_id.clone(),
                lower: context.target_prevalence_lower,
                upper: context.target_prevalence_upper,
            });
        }
        lower_sum += context.target_prevalence_lower;
        upper_sum += context.target_prevalence_upper;
    }

    if lower_sum > 1.0 + TOLERANCE {
        return Err(TransportabilityError::LowerPrevalenceMassExceedsOne(
            lower_sum,
        ));
    }
    if upper_sum < 1.0 - TOLERANCE {
        return Err(TransportabilityError::UpperPrevalenceMassBelowOne(
            upper_sum,
        ));
    }
    Ok((lower_sum, upper_sum))
}

/// Optimize a linear mixture over a bounded simplex by starting at every lower bound,
/// then assigning remaining mass in objective order up to each upper bound.
/// This is exact for the declared coordinate-wise independent bounds and a single
/// sum-to-one constraint.
fn optimize_bounded_simplex(
    contexts: &[ContextRecallDelta],
    use_lower_delta: bool,
    minimize: bool,
) -> Result<MixtureBoundWitness, TransportabilityError> {
    let mut ordered: Vec<&ContextRecallDelta> = contexts.iter().collect();
    ordered.sort_by(|a, b| {
        let va = if use_lower_delta { a.delta_lower } else { a.delta_upper };
        let vb = if use_lower_delta { b.delta_lower } else { b.delta_upper };
        let cmp = va.total_cmp(&vb);
        let cmp = if minimize { cmp } else { cmp.reverse() };
        cmp.then_with(|| a.context_id.cmp(&b.context_id))
    });

    let base_mass: f64 = contexts.iter().map(|c| c.target_prevalence_lower).sum();
    let mut remaining = (1.0 - base_mass).max(0.0);
    let mut weights: Vec<(String, f64)> = contexts
        .iter()
        .map(|c| (c.context_id.clone(), c.target_prevalence_lower))
        .collect();

    for context in ordered {
        if remaining <= TOLERANCE {
            break;
        }
        let capacity = (context.target_prevalence_upper - context.target_prevalence_lower).max(0.0);
        let add = remaining.min(capacity);
        if add > 0.0 {
            let item = weights
                .iter_mut()
                .find(|(id, _)| id == &context.context_id)
                .ok_or_else(|| {
                    TransportabilityError::NumericalInfeasibility(
                        "validated context not found in weight vector".to_owned(),
                    )
                })?;
            item.1 += add;
            remaining -= add;
        }
    }

    if remaining > TOLERANCE {
        return Err(TransportabilityError::NumericalInfeasibility(format!(
            "unallocated prevalence mass remains: {remaining}"
        )));
    }

    weights.sort_by(|a, b| a.0.cmp(&b.0));
    let mixture_delta = contexts
        .iter()
        .map(|context| {
            let weight = weights
                .iter()
                .find(|(id, _)| id == &context.context_id)
                .map(|(_, weight)| *weight)
                .unwrap_or(0.0);
            let delta = if use_lower_delta {
                context.delta_lower
            } else {
                context.delta_upper
            };
            weight * delta
        })
        .sum();

    Ok(MixtureBoundWitness {
        prevalence_by_context: weights,
        mixture_delta,
    })
}

/// Compute conservative target-mixture bounds for policy-vs-baseline Recall@K improvement.
/// The lower bound uses per-context lower delta endpoints and the target prevalence vector
/// that minimizes that weighted sum; the upper bound analogously maximizes upper endpoints.
/// The context-effect intervals and prevalence ranges are varied independently, so bounds
/// may be conservative when their dependencies are known but not represented.
pub fn transportability_envelope(
    contexts: &[ContextRecallDelta],
) -> Result<TransportabilityEnvelope, TransportabilityError> {
    let (target_prevalence_lower_sum, target_prevalence_upper_sum) =
        validate_contexts(contexts)?;

    let lower_witness = optimize_bounded_simplex(contexts, true, true)?;
    let upper_witness = optimize_bounded_simplex(contexts, false, false)?;

    let lower_delta = lower_witness.mixture_delta;
    let upper_delta = upper_witness.mixture_delta;
    if lower_delta > upper_delta + TOLERANCE {
        return Err(TransportabilityError::NumericalInfeasibility(format!(
            "computed lower delta {lower_delta} exceeds upper delta {upper_delta}"
        )));
    }

    let conclusion = if lower_delta > 0.0 {
        TransportabilityConclusion::RobustPositive
    } else if upper_delta <= 0.0 {
        TransportabilityConclusion::RobustNonPositive
    } else {
        TransportabilityConclusion::Inconclusive
    };

    Ok(TransportabilityEnvelope {
        lower_delta,
        upper_delta,
        conclusion,
        lower_witness,
        upper_witness,
        target_prevalence_lower_sum,
        target_prevalence_upper_sum,
        context_count: contexts.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(
        id: &str,
        delta: (f64, f64),
        prevalence: (f64, f64),
        supported: bool,
    ) -> ContextRecallDelta {
        ContextRecallDelta {
            context_id: id.to_owned(),
            delta_lower: delta.0,
            delta_upper: delta.1,
            target_prevalence_lower: prevalence.0,
            target_prevalence_upper: prevalence.1,
            context_supported: supported,
        }
    }

    fn fixture_mixed() -> Vec<ContextRecallDelta> {
        vec![
            ctx("A", (0.08, 0.12), (0.2, 0.5), true),
            ctx("B", (0.02, 0.07), (0.3, 0.6), true),
            ctx("C", (-0.25, -0.15), (0.1, 0.4), true),
        ]
    }

    #[test]
    fn uncertain_target_mix_can_erase_source_weighted_gain() {
        let result = transportability_envelope(&fixture_mixed()).unwrap();
        assert!((result.lower_delta - (-0.076)).abs() < 1e-12);
        assert!((result.upper_delta - 0.073).abs() < 1e-12);
        assert_eq!(result.conclusion, TransportabilityConclusion::Inconclusive);
        assert_eq!(
            result.lower_witness.prevalence_by_context,
            vec![("A".to_owned(), 0.2), ("B".to_owned(), 0.4), ("C".to_owned(), 0.4)]
        );
        assert_eq!(
            result.upper_witness.prevalence_by_context,
            vec![("A".to_owned(), 0.5), ("B".to_owned(), 0.4), ("C".to_owned(), 0.1)]
        );
    }

    #[test]
    fn positive_effect_in_every_context_is_robustly_positive() {
        let contexts = vec![
            ctx("A", (0.1, 0.12), (0.2, 0.5), true),
            ctx("B", (0.05, 0.07), (0.3, 0.6), true),
            ctx("C", (0.02, 0.03), (0.1, 0.4), true),
        ];
        let result = transportability_envelope(&contexts).unwrap();
        assert!(result.lower_delta > 0.0);
        assert_eq!(result.conclusion, TransportabilityConclusion::RobustPositive);
    }

    #[test]
    fn nonpositive_effects_prove_no_positive_target_gain() {
        let contexts = vec![
            ctx("A", (-0.1, -0.05), (0.2, 0.5), true),
            ctx("B", (-0.08, -0.01), (0.3, 0.6), true),
            ctx("C", (-0.2, -0.1), (0.1, 0.4), true),
        ];
        let result = transportability_envelope(&contexts).unwrap();
        assert!(result.upper_delta <= 0.0);
        assert_eq!(result.conclusion, TransportabilityConclusion::RobustNonPositive);
    }

    #[test]
    fn fixed_target_mix_is_weighted_exactly() {
        let contexts = vec![
            ctx("A", (0.1, 0.2), (0.25, 0.25), true),
            ctx("B", (-0.2, -0.1), (0.75, 0.75), true),
        ];
        let result = transportability_envelope(&contexts).unwrap();
        assert!((result.lower_delta - (-0.125)).abs() < 1e-12);
        assert!((result.upper_delta - (-0.025)).abs() < 1e-12);
    }

    #[test]
    fn feasible_witness_weights_sum_to_one_and_obey_ranges() {
        let result = transportability_envelope(&fixture_mixed()).unwrap();
        for witness in [&result.lower_witness, &result.upper_witness] {
            let sum: f64 = witness.prevalence_by_context.iter().map(|x| x.1).sum();
            assert!((sum - 1.0).abs() < 1e-12);
            for (id, weight) in &witness.prevalence_by_context {
                let context = fixture_mixed().into_iter().find(|c| c.context_id == *id).unwrap();
                assert!(*weight >= context.target_prevalence_lower - TOLERANCE);
                assert!(*weight <= context.target_prevalence_upper + TOLERANCE);
            }
        }
    }

    #[test]
    fn missing_supported_context_fails_closed() {
        let contexts = vec![
            ctx("A", (0.1, 0.2), (0.5, 0.8), true),
            ctx("unseen", (0.0, 0.0), (0.2, 0.5), false),
        ];
        assert_eq!(
            transportability_envelope(&contexts),
            Err(TransportabilityError::UnsupportedTargetContext("unseen".to_owned()))
        );
    }

    #[test]
    fn lower_prevalence_mass_over_one_is_rejected() {
        let contexts = vec![
            ctx("A", (0.0, 0.1), (0.7, 0.9), true),
            ctx("B", (0.0, 0.1), (0.4, 0.6), true),
        ];
        assert!(matches!(
            transportability_envelope(&contexts),
            Err(TransportabilityError::LowerPrevalenceMassExceedsOne(_))
        ));
    }

    #[test]
    fn upper_prevalence_mass_below_one_is_rejected() {
        let contexts = vec![
            ctx("A", (0.0, 0.1), (0.1, 0.2), true),
            ctx("B", (0.0, 0.1), (0.1, 0.3), true),
        ];
        assert!(matches!(
            transportability_envelope(&contexts),
            Err(TransportabilityError::UpperPrevalenceMassBelowOne(_))
        ));
    }

    #[test]
    fn reversed_prevalence_interval_is_rejected() {
        let contexts = vec![
            ctx("A", (0.0, 0.1), (0.7, 0.4), true),
            ctx("B", (0.0, 0.1), (0.1, 0.8), true),
        ];
        assert!(matches!(
            transportability_envelope(&contexts),
            Err(TransportabilityError::InvalidPrevalenceInterval { .. })
        ));
    }

    #[test]
    fn invalid_delta_interval_is_rejected() {
        let contexts = vec![
            ctx("A", (0.2, 0.1), (0.2, 0.8), true),
            ctx("B", (0.0, 0.1), (0.1, 0.8), true),
        ];
        assert!(matches!(
            transportability_envelope(&contexts),
            Err(TransportabilityError::InvalidDeltaInterval { .. })
        ));
    }

    #[test]
    fn recall_delta_outside_possible_range_is_rejected() {
        let contexts = vec![
            ctx("A", (0.0, 1.1), (0.2, 0.8), true),
            ctx("B", (0.0, 0.1), (0.1, 0.8), true),
        ];
        assert!(matches!(
            transportability_envelope(&contexts),
            Err(TransportabilityError::InvalidDeltaInterval { .. })
        ));
    }

    #[test]
    fn non_finite_bound_is_rejected() {
        let contexts = vec![
            ctx("A", (0.0, f64::NAN), (0.2, 0.8), true),
            ctx("B", (0.0, 0.1), (0.1, 0.8), true),
        ];
        assert!(matches!(
            transportability_envelope(&contexts),
            Err(TransportabilityError::InvalidDeltaInterval { .. })
        ));
    }

    #[test]
    fn duplicate_context_id_is_rejected() {
        let contexts = vec![
            ctx("A", (0.0, 0.1), (0.2, 0.8), true),
            ctx("A", (0.0, 0.1), (0.2, 0.8), true),
        ];
        assert!(matches!(
            transportability_envelope(&contexts),
            Err(TransportabilityError::DuplicateContextId(_))
        ));
    }

    #[test]
    fn blank_context_id_is_rejected() {
        let contexts = vec![
            ctx(" ", (0.0, 0.1), (0.2, 0.8), true),
            ctx("B", (0.0, 0.1), (0.2, 0.8), true),
        ];
        assert_eq!(
            transportability_envelope(&contexts),
            Err(TransportabilityError::EmptyContextId)
        );
    }

    #[test]
    fn sum_constraints_allow_exact_feasible_boundary() {
        let contexts = vec![
            ctx("A", (0.1, 0.2), (0.4, 0.4), true),
            ctx("B", (0.0, 0.1), (0.6, 0.6), true),
        ];
        let result = transportability_envelope(&contexts).unwrap();
        assert!((result.lower_delta - 0.04).abs() < 1e-12);
        assert!((result.upper_delta - 0.14).abs() < 1e-12);
    }

    #[test]
    fn deterministic_ties_use_context_id() {
        let contexts = vec![
            ctx("B", (-0.1, 0.1), (0.0, 1.0), true),
            ctx("A", (-0.1, 0.1), (0.0, 1.0), true),
        ];
        let result = transportability_envelope(&contexts).unwrap();
        assert_eq!(
            result.lower_witness.prevalence_by_context,
            vec![("A".to_owned(), 1.0), ("B".to_owned(), 0.0)]
        );
        assert_eq!(
            result.upper_witness.prevalence_by_context,
            vec![("A".to_owned(), 1.0), ("B".to_owned(), 0.0)]
        );
    }
}
