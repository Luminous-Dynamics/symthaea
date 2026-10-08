#![forbid(unsafe_code)]

//! Standalone reference model for separating provider-effect observation
//! from causal operation attribution.
//!
//! This file intentionally contains no provider credentials or network calls.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionCausalAttributionV1 {
    Established,
    Unestablished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionEffectSourceV1 {
    DirectProviderResult,
    DurableSubjectObservation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionEffectObservationV1 {
    pub source: PromotionEffectSourceV1,
    pub causal_attribution: PromotionCausalAttributionV1,
    pub local_promotion_operation_id: Option<String>,
    pub provider_operation_uuid: Option<String>,
    pub expected_pr_head_sha: String,
    pub observed_merge_commit: String,
}

impl PromotionEffectObservationV1 {
    fn provider_direct(
        local_promotion_operation_id: impl Into<String>,
        provider_operation_uuid: impl Into<String>,
        expected_pr_head_sha: impl Into<String>,
        observed_merge_commit: impl Into<String>,
    ) -> Self {
        Self {
            source: PromotionEffectSourceV1::DirectProviderResult,
            causal_attribution: PromotionCausalAttributionV1::Established,
            local_promotion_operation_id: Some(local_promotion_operation_id.into()),
            provider_operation_uuid: Some(provider_operation_uuid.into()),
            expected_pr_head_sha: expected_pr_head_sha.into(),
            observed_merge_commit: observed_merge_commit.into(),
        }
    }

    fn durable_only(
        expected_pr_head_sha: impl Into<String>,
        observed_merge_commit: impl Into<String>,
    ) -> Self {
        Self {
            source: PromotionEffectSourceV1::DurableSubjectObservation,
            causal_attribution: PromotionCausalAttributionV1::Unestablished,
            provider_operation_uuid: None,
            expected_pr_head_sha: expected_pr_head_sha.into(),
            observed_merge_commit: observed_merge_commit.into(),
        }
    }

    fn validate(&self) -> Result<(), &'static str> {
        if self.expected_pr_head_sha.is_empty() {
            return Err("missing expected PR head SHA");
        }
        if self.observed_merge_commit.is_empty() {
            return Err("missing observed merge commit");
        }

        match self.source {
            PromotionEffectSourceV1::DirectProviderResult => {
                if self.causal_attribution != PromotionCausalAttributionV1::Established
                    || self.local_promotion_operation_id
                        .as_deref()
                        .unwrap_or_default()
                        .is_empty()
                    || self.provider_operation_uuid.as_deref().unwrap_or_default().is_empty()
                {
                    return Err("direct provider result is missing local/provider operation binding");
                }
            }
            PromotionEffectSourceV1::DurableSubjectObservation => {
                if self.causal_attribution != PromotionCausalAttributionV1::Unestablished
                    || self.provider_operation_uuid.is_some()
                {
                    return Err("durable-only observation illegally claims provider causality");
                }
            }
        }

        Ok(())
    }
}

/// A provider-reported merged result contains the async operation UUID
/// and the resulting merge commit, so the narrow operation-to-effect edge
/// is directly represented.
pub fn from_async_merged(
    local_promotion_operation_id: &str,
    provider_operation_uuid: &str,
    expected_pr_head_sha: &str,
    merge_commit: &str,
) -> Result<PromotionEffectObservationV1, &'static str> {
    let observation = PromotionEffectObservationV1::provider_direct(
        local_promotion_operation_id,
        provider_operation_uuid,
        expected_pr_head_sha,
        merge_commit,
    );
    observation.validate()?;
    Ok(observation)
}

/// An enqueued async request is a final provider result, but eventual PR
/// merged state is a separate durable observation and does not retain the
/// original causal edge.
pub fn from_enqueued_then_durable_merge(
    expected_pr_head_sha: &str,
    merge_commit: &str,
) -> Result<PromotionEffectObservationV1, &'static str> {
    let observation =
        PromotionEffectObservationV1::durable_only(expected_pr_head_sha, merge_commit);
    observation.validate()?;
    Ok(observation)
}

/// An expired UUID plus a durable merged PR remains effect evidence without
/// operation-level causal attribution.
pub fn from_expired_uuid_then_durable_merge(
    expected_pr_head_sha: &str,
    merge_commit: &str,
) -> Result<PromotionEffectObservationV1, &'static str> {
    from_enqueued_then_durable_merge(expected_pr_head_sha, merge_commit)
}

/// A retry seeing an already-merged PR receives effect state but no historical
/// async operation UUID, so the retry itself is not causally attributed.
pub fn from_already_merged_retry(
    expected_pr_head_sha: &str,
    merge_commit: &str,
) -> Result<PromotionEffectObservationV1, &'static str> {
    from_enqueued_then_durable_merge(expected_pr_head_sha, merge_commit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_provider_merge_establishes_narrow_causal_attribution() {
        let observation = from_async_merged("OP-1", "uuid-1", "H1", "M1").unwrap();
        assert_eq!(observation.source, PromotionEffectSourceV1::DirectProviderResult);
        assert_eq!(
            observation.causal_attribution,
            PromotionCausalAttributionV1::Established
        );
        assert_eq!(
            observation.local_promotion_operation_id.as_deref(),
            Some("OP-1")
        );
        assert_eq!(observation.provider_operation_uuid.as_deref(), Some("uuid-1"));
    }

    #[test]
    fn enqueued_then_durable_merge_is_effect_only() {
        let observation = from_enqueued_then_durable_merge("H1", "M1").unwrap();
        assert_eq!(observation.source, PromotionEffectSourceV1::DurableSubjectObservation);
        assert_eq!(
            observation.causal_attribution,
            PromotionCausalAttributionV1::Unestablished
        );
        assert!(observation.provider_operation_uuid.is_none());
    }

    #[test]
    fn expired_uuid_then_durable_merge_is_effect_only() {
        let observation = from_expired_uuid_then_durable_merge("H1", "M1").unwrap();
        assert_eq!(
            observation.causal_attribution,
            PromotionCausalAttributionV1::Unestablished
        );
    }

    #[test]
    fn already_merged_retry_is_not_backdated_as_causal() {
        let observation = from_already_merged_retry("H1", "M1").unwrap();
        assert_eq!(
            observation.causal_attribution,
            PromotionCausalAttributionV1::Unestablished
        );
        assert!(observation.provider_operation_uuid.is_none());
    }

    #[test]
    fn another_actor_merge_is_not_provider_causal() {
        let observation = PromotionEffectObservationV1::durable_only("H1", "M2");
        observation.validate().unwrap();
        assert_eq!(
            observation.causal_attribution,
            PromotionCausalAttributionV1::Unestablished
        );
    }

    #[test]
    fn provider_uuid_without_local_operation_binding_cannot_claim_causality() {
        let observation = PromotionEffectObservationV1 {
            source: PromotionEffectSourceV1::DirectProviderResult,
            causal_attribution: PromotionCausalAttributionV1::Established,
            local_promotion_operation_id: None,
            provider_operation_uuid: Some("uuid-1".into()),
            expected_pr_head_sha: "H1".into(),
            observed_merge_commit: "M1".into(),
        };
        assert_eq!(
            observation.validate().unwrap_err(),
            "direct provider result is missing local/provider operation binding"
        );
    }

    #[test]
    fn malformed_direct_result_cannot_claim_causality() {
        let observation = PromotionEffectObservationV1 {
            source: PromotionEffectSourceV1::DirectProviderResult,
            causal_attribution: PromotionCausalAttributionV1::Established,
            local_promotion_operation_id: None,
            provider_operation_uuid: Some("uuid-1".into()),
            expected_pr_head_sha: "H1".into(),
            observed_merge_commit: "M1".into(),
        };
        assert_eq!(
            observation.validate().unwrap_err(),
            "direct provider result is missing causal operation identity"
        );
    }
}
