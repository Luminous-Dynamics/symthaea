use crate::{EpistemicDisposition, EpistemicReadinessError, EpistemicReadinessInput, EpistemicReadinessPolicy};

/// Result of applying epistemic readiness immediately before proposal construction.
///
/// This type deliberately contains no action/edit permission, authority token, or
/// receipt. A formulated proposal must still traverse the existing authority layer.
#[derive(Debug, Clone, PartialEq)]
pub enum ProposalFormulation<T> {
    /// The readiness policy returned `ReadyToPropose`, so the proposal builder ran.
    Formulated {
        proposal: T,
        readiness_input: EpistemicReadinessInput,
    },
    /// Proposal construction was suppressed. The disposition states what epistemic
    /// work should happen next, without creating any permission to mutate a world.
    Blocked {
        disposition: EpistemicDisposition,
        readiness_input: EpistemicReadinessInput,
    },
}

impl<T> ProposalFormulation<T> {
    pub const fn is_formulated(&self) -> bool {
        matches!(self, Self::Formulated { .. })
    }

    pub const fn readiness_input(&self) -> &EpistemicReadinessInput {
        match self {
            Self::Formulated {
                readiness_input, ..
            }
            | Self::Blocked {
                readiness_input, ..
            } => readiness_input,
        }
    }

    pub const fn blocked_disposition(&self) -> Option<EpistemicDisposition> {
        match self {
            Self::Formulated { .. } => None,
            Self::Blocked { disposition, .. } => Some(*disposition),
        }
    }
}

/// Evaluate epistemic readiness and invoke `build_proposal` iff the policy returns
/// `ReadyToPropose`.
///
/// Readiness is evaluated inside this function so callers cannot bypass the policy
/// by supplying a preconstructed positive disposition. The builder is `FnOnce` and
/// is never evaluated on blocked paths.
///
/// A successful result means only that proposal *formulation* was permitted. It does
/// not authorize execution, mutation, commitment, or receipt creation.
pub fn formulate_if_epistemically_ready<T, F>(
    policy: &EpistemicReadinessPolicy,
    input: EpistemicReadinessInput,
    build_proposal: F,
) -> Result<ProposalFormulation<T>, EpistemicReadinessError>
where
    F: FnOnce() -> T,
{
    let disposition = policy.evaluate(input)?;
    match disposition {
        EpistemicDisposition::ReadyToPropose => Ok(ProposalFormulation::Formulated {
            proposal: build_proposal(),
            readiness_input: input,
        }),
        blocked => Ok(ProposalFormulation::Blocked {
            disposition: blocked,
            readiness_input: input,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn policy() -> EpistemicReadinessPolicy {
        EpistemicReadinessPolicy::new(0.7, 2, 2, true, true, true).unwrap()
    }

    fn ready_input() -> EpistemicReadinessInput {
        EpistemicReadinessInput {
            grounded_evidence_count: 2,
            independent_source_count: 2,
            confidence: 0.8,
            conflicting_evidence: false,
            counterfactual_only: false,
            unresolvable: false,
        }
    }

    fn assert_builder_not_run(
        input: EpistemicReadinessInput,
        expected: EpistemicDisposition,
    ) {
        let calls = Cell::new(0u32);
        let result = formulate_if_epistemically_ready(&policy(), input, || {
            calls.set(calls.get() + 1);
            "proposal"
        })
        .unwrap();

        assert_eq!(calls.get(), 0);
        assert!(!result.is_formulated());
        assert_eq!(result.blocked_disposition(), Some(expected));
        assert_eq!(*result.readiness_input(), input);
    }

    #[test]
    fn ready_path_constructs_exactly_one_proposal() {
        let input = ready_input();
        let calls = Cell::new(0u32);
        let result = formulate_if_epistemically_ready(&policy(), input, || {
            calls.set(calls.get() + 1);
            "proposal"
        })
        .unwrap();

        assert_eq!(calls.get(), 1);
        assert!(result.is_formulated());
        assert_eq!(*result.readiness_input(), input);
        match result {
            ProposalFormulation::Formulated { proposal, .. } => assert_eq!(proposal, "proposal"),
            ProposalFormulation::Blocked { .. } => panic!("ready input must formulate"),
        }
    }

    #[test]
    fn observe_more_never_constructs_proposal() {
        let mut input = ready_input();
        input.grounded_evidence_count = 0;
        assert_builder_not_run(input, EpistemicDisposition::ObserveMore);
    }

    #[test]
    fn counterfactual_only_never_constructs_proposal() {
        let mut input = ready_input();
        input.counterfactual_only = true;
        assert_builder_not_run(input, EpistemicDisposition::ObserveMore);
    }

    #[test]
    fn corroboration_request_never_constructs_proposal() {
        let mut input = ready_input();
        input.conflicting_evidence = true;
        assert_builder_not_run(input, EpistemicDisposition::RequestCorroboration);
    }

    #[test]
    fn abstention_never_constructs_proposal() {
        let mut input = ready_input();
        input.unresolvable = true;
        assert_builder_not_run(input, EpistemicDisposition::Abstain);
    }

    #[test]
    fn invalid_readiness_input_never_constructs_proposal() {
        let mut input = ready_input();
        input.confidence = f32::NAN;
        let calls = Cell::new(0u32);
        let result = formulate_if_epistemically_ready(&policy(), input, || {
            calls.set(calls.get() + 1);
            "proposal"
        });
        assert!(matches!(
            result,
            Err(EpistemicReadinessError::InvalidInputConfidence(value)) if value.is_nan()
        ));
        assert_eq!(calls.get(), 0);
    }
}
