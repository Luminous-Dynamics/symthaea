use std::fmt;

/// Epistemic disposition produced before proposal/authority evaluation.
///
/// `ReadyToPropose` is not permission to act. Normal proposal, authority, and
/// receipt boundaries still apply after this policy returns positively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicDisposition {
    ReadyToPropose,
    ObserveMore,
    RequestCorroboration,
    Abstain,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpistemicReadinessInput {
    pub grounded_evidence_count: u32,
    pub independent_source_count: u32,
    pub confidence: f32,
    pub conflicting_evidence: bool,
    pub counterfactual_only: bool,
    /// True when the evidence state is explicitly classified as not safely
    /// resolvable by additional observation/corroboration under the current task.
    pub unresolvable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpistemicReadinessPolicy {
    min_confidence: f32,
    min_grounded_evidence: u32,
    min_independent_sources: u32,
    request_corroboration_on_conflict: bool,
    forbid_counterfactual_only_proposal: bool,
    abstain_on_unresolvable: bool,
}

impl EpistemicReadinessPolicy {
    /// Construct a policy explicitly. There is intentionally no `Default`:
    /// thresholds are experimental/operational policy and must be chosen visibly.
    pub fn new(
        min_confidence: f32,
        min_grounded_evidence: u32,
        min_independent_sources: u32,
        request_corroboration_on_conflict: bool,
        forbid_counterfactual_only_proposal: bool,
        abstain_on_unresolvable: bool,
    ) -> Result<Self, EpistemicReadinessError> {
        if !min_confidence.is_finite() || !(0.0..=1.0).contains(&min_confidence) {
            return Err(EpistemicReadinessError::InvalidConfidenceThreshold(
                min_confidence,
            ));
        }
        Ok(Self {
            min_confidence,
            min_grounded_evidence,
            min_independent_sources,
            request_corroboration_on_conflict,
            forbid_counterfactual_only_proposal,
            abstain_on_unresolvable,
        })
    }

    pub fn evaluate(
        &self,
        input: EpistemicReadinessInput,
    ) -> Result<EpistemicDisposition, EpistemicReadinessError> {
        if !input.confidence.is_finite() || !(0.0..=1.0).contains(&input.confidence) {
            return Err(EpistemicReadinessError::InvalidInputConfidence(
                input.confidence,
            ));
        }

        if input.unresolvable && self.abstain_on_unresolvable {
            return Ok(EpistemicDisposition::Abstain);
        }

        if input.counterfactual_only && self.forbid_counterfactual_only_proposal {
            return Ok(EpistemicDisposition::ObserveMore);
        }

        if input.conflicting_evidence && self.request_corroboration_on_conflict {
            return Ok(EpistemicDisposition::RequestCorroboration);
        }

        if input.grounded_evidence_count < self.min_grounded_evidence
            || input.independent_source_count < self.min_independent_sources
            || input.confidence < self.min_confidence
        {
            return Ok(EpistemicDisposition::ObserveMore);
        }

        Ok(EpistemicDisposition::ReadyToPropose)
    }

    pub const fn min_confidence(&self) -> f32 {
        self.min_confidence
    }

    pub const fn min_grounded_evidence(&self) -> u32 {
        self.min_grounded_evidence
    }

    pub const fn min_independent_sources(&self) -> u32 {
        self.min_independent_sources
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EpistemicReadinessError {
    InvalidConfidenceThreshold(f32),
    InvalidInputConfidence(f32),
}

impl fmt::Display for EpistemicReadinessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfidenceThreshold(value) => {
                write!(f, "minimum confidence must be finite and within [0,1], got {value}")
            }
            Self::InvalidInputConfidence(value) => {
                write!(f, "input confidence must be finite and within [0,1], got {value}")
            }
        }
    }
}

impl std::error::Error for EpistemicReadinessError {}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn sufficient_evidence_is_only_ready_to_propose() {
        assert_eq!(
            policy().evaluate(ready_input()).unwrap(),
            EpistemicDisposition::ReadyToPropose
        );
    }

    #[test]
    fn counterfactual_only_state_requests_more_observation() {
        let mut input = ready_input();
        input.counterfactual_only = true;
        assert_eq!(
            policy().evaluate(input).unwrap(),
            EpistemicDisposition::ObserveMore
        );
    }

    #[test]
    fn conflicting_evidence_requests_corroboration() {
        let mut input = ready_input();
        input.conflicting_evidence = true;
        assert_eq!(
            policy().evaluate(input).unwrap(),
            EpistemicDisposition::RequestCorroboration
        );
    }

    #[test]
    fn insufficient_grounding_or_sources_observes_more() {
        let mut input = ready_input();
        input.grounded_evidence_count = 1;
        assert_eq!(
            policy().evaluate(input).unwrap(),
            EpistemicDisposition::ObserveMore
        );
        input.grounded_evidence_count = 2;
        input.independent_source_count = 1;
        assert_eq!(
            policy().evaluate(input).unwrap(),
            EpistemicDisposition::ObserveMore
        );
    }

    #[test]
    fn low_confidence_observes_more() {
        let mut input = ready_input();
        input.confidence = 0.69;
        assert_eq!(
            policy().evaluate(input).unwrap(),
            EpistemicDisposition::ObserveMore
        );
    }

    #[test]
    fn explicit_unresolvable_state_abstains() {
        let mut input = ready_input();
        input.unresolvable = true;
        assert_eq!(
            policy().evaluate(input).unwrap(),
            EpistemicDisposition::Abstain
        );
    }

    #[test]
    fn invalid_threshold_or_input_confidence_rejects() {
        assert!(EpistemicReadinessPolicy::new(f32::NAN, 1, 1, true, true, true).is_err());
        let mut input = ready_input();
        input.confidence = 1.1;
        assert!(policy().evaluate(input).is_err());
    }
}
