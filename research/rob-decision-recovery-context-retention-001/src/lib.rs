//! Deterministic decision- and recovery-aware KV context eviction.
//!
//! The caller supplies qualified per-context evidence. This crate selects
//! eviction candidates but does not implement a KV allocator, model inference,
//! physical recovery solver or actuator authority.

#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BTreeSet;

/// Decision-impact evidence for losing a resident context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionImpact {
    /// Qualified upper bound on task-loss regret caused by eviction.
    Qualified {
        /// Versioned task-loss scale.
        loss_scale_id: String,
        /// Conservative eviction-regret upper bound in that scale.
        regret_upper_units: u64,
        /// Calibration receipt identity.
        calibration_id: String,
        /// Calibration receipt digest.
        calibration_digest: String,
        /// Last tick for which this decision-impact bound remains valid.
        valid_through_tick: u64,
    },
    /// Decision-impact evidence is missing, stale, incomplete or contradictory.
    Unknown,
    /// Evidence is explicitly invalid.
    Invalid,
}

/// One resident agent/robot context candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextEntry {
    /// Stable context identity.
    pub context_id: String,
    /// KV-cache footprint in tokens.
    pub token_count: u64,
    /// Last access tick.
    pub last_access_tick: u64,
    /// Hard pin reason: local control, RTA, interlock, safe-stop, source lineage, etc.
    pub hard_pin_reason: Option<String>,
    /// Context contains qualified evidence needed for a recovery path.
    pub recovery_critical: bool,
    /// Digest of the recovery evidence that makes this context critical.
    pub recovery_evidence_digest: Option<String>,
    /// Next-phase affinity in parts per million.
    pub phase_affinity_ppm: Option<u32>,
    /// Phase identity and manifest under which affinity was evaluated.
    pub phase_affinity_phase_id: Option<String>,
    /// Digest of the phase/transition manifest.
    pub phase_manifest_digest: Option<String>,
    /// Predicted probability of reuse before context expiry, in parts per million.
    pub future_reuse_probability_ppm: Option<u32>,
    /// Estimated cost to rebuild the context, in control-independent prefill ticks.
    pub prefill_latency_ticks: Option<u64>,
    /// Decision-loss impact if this context is evicted.
    pub decision_impact: DecisionImpact,
}

/// Deterministic context-retention policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionPolicy {
    /// Oldest last-access tick first.
    Lru,
    /// Lowest affinity to the next declared phase first.
    PhaseAffinity,
    /// Lowest transition-predicted next-use probability first.
    TransitionReuse,
    /// Recovery, decision-regret, phase, reuse and prefill lexicographic ordering.
    LuminousDecisionRecoveryAware,
}

/// Frozen request for additional KV capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetentionRequest {
    /// Total cache capacity in tokens.
    pub capacity_tokens: u64,
    /// Additional tokens requested by the incoming context/batch.
    pub requested_additional_tokens: u64,
    /// Current mission/control tick.
    pub current_tick: u64,
    /// Common task-loss scale expected for qualified context regret bounds.
    pub required_loss_scale_id: Option<String>,
    /// Next expected agent/mission phase.
    pub next_phase_id: String,
    /// Frozen next-phase manifest digest.
    pub phase_manifest_digest: String,
}

/// Reason a resident context was protected from eviction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtectionReason {
    /// Caller marked this context as non-evictable.
    HardPinned(String),
    /// Decision/regret evidence is Unknown.
    DecisionImpactUnknown,
    /// Decision/regret calibration has expired.
    DecisionImpactExpired,
    /// Recovery-critical evidence digest is missing.
    RecoveryEvidenceUnknown,
    /// Policy-specific phase-affinity evidence is missing or mismatched.
    PhaseAffinityUnknown,
    /// Policy-specific transition reuse evidence is missing.
    TransitionReuseUnknown,
    /// Policy-specific prefill-cost estimate is missing.
    PrefillCostUnknown,
}

/// Why the context was selected as a victim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvictionReason {
    /// Victim selected by the declared retention policy under capacity pressure.
    CapacityPressure,
}

/// One deterministic eviction receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictionReceipt {
    /// Evicted context identity.
    pub context_id: String,
    /// Number of tokens freed.
    pub freed_tokens: u64,
    /// Task-loss scale for the bound.
    pub loss_scale_id: String,
    /// Calibrated eviction regret upper bound.
    pub regret_upper_units: u64,
    /// Whether the context was recovery critical.
    pub recovery_critical: bool,
    /// Reason the context was evicted.
    pub reason: EvictionReason,
}

/// Final retention decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetentionDisposition {
    /// Requested KV growth fits after the stated evictions.
    Admitted,
    /// Protected contexts prevent the requested allocation from fitting.
    DeferredProtectedCapacity,
}

/// Complete context-retention plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetentionPlan {
    /// Policy used to rank candidates.
    pub policy: RetentionPolicy,
    /// Final admission/defer disposition.
    pub disposition: RetentionDisposition,
    /// KV tokens occupied before eviction.
    pub used_tokens_before: u64,
    /// Requested additional tokens.
    pub requested_additional_tokens: u64,
    /// KV tokens freed by this plan.
    pub freed_tokens: u64,
    /// KV tokens occupied after evictions and requested allocation.
    pub used_tokens_after: u64,
    /// Remaining cache capacity after the plan.
    pub remaining_capacity_tokens: u64,
    /// Ordered eviction receipts.
    pub evictions: Vec<EvictionReceipt>,
    /// Protected contexts and their reason.
    pub protected_contexts: Vec<(String, ProtectionReason)>,
    /// Explicitly false; retention planning grants no control/actuator authority.
    pub actuation_authorized: bool,
}

/// Invalid request or context-evidence input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetentionError {
    /// Context ID is empty.
    EmptyContextId,
    /// Context ID is duplicated.
    DuplicateContextId(String),
    /// A context has zero token footprint.
    ZeroTokenFootprint(String),
    /// PPM input lies outside [0, 1_000_000].
    InvalidProbability {
        /// Context identity.
        context_id: String,
        /// Field name.
        field: &'static str,
        /// Supplied value.
        value_ppm: u32,
    },
    /// Qualified decision-impact evidence is incomplete.
    InvalidDecisionEvidence(String),
    /// Recovery-critical context is missing its evidence digest.
    InvalidRecoveryEvidence(String),
    /// Required task-loss scales are not comparable.
    IncomparableLossScales {
        /// First observed task-loss scale.
        first: String,
        /// Incompatible task-loss scale.
        other: String,
    },
    /// Token arithmetic overflowed.
    TokenArithmeticOverflow,
    /// Requested next-phase identity or manifest is missing.
    EmptyPhaseScope,
}

/// Determine why a context is protected under this request and policy.
fn protection_reason(
    request: &RetentionRequest,
    policy: RetentionPolicy,
    context: &ContextEntry,
) -> Result<Option<ProtectionReason>, RetentionError> {
    if let Some(reason) = &context.hard_pin_reason {
        if reason.trim().is_empty() {
            return Err(RetentionError::InvalidDecisionEvidence(
                context.context_id.clone(),
            ));
        }
        return Ok(Some(ProtectionReason::HardPinned(reason.clone())));
    }

    match &context.decision_impact {
        DecisionImpact::Invalid => {
            return Err(RetentionError::InvalidDecisionEvidence(
                context.context_id.clone(),
            ));
        }
        DecisionImpact::Unknown => {
            return Ok(Some(ProtectionReason::DecisionImpactUnknown));
        }
        DecisionImpact::Qualified {
            loss_scale_id,
            calibration_id,
            calibration_digest,
            valid_through_tick,
            ..
        } => {
            if loss_scale_id.trim().is_empty()
                || calibration_id.trim().is_empty()
                || calibration_digest.trim().is_empty()
            {
                return Err(RetentionError::InvalidDecisionEvidence(
                    context.context_id.clone(),
                ));
            }
            if *valid_through_tick < request.current_tick {
                return Ok(Some(ProtectionReason::DecisionImpactExpired));
            }
        }
    }

    if context.recovery_critical
        && context
            .recovery_evidence_digest
            .as_ref()
            .is_none_or(|digest| digest.trim().is_empty())
    {
        return Ok(Some(ProtectionReason::RecoveryEvidenceUnknown));
    }

    match policy {
        RetentionPolicy::Lru => {}
        RetentionPolicy::PhaseAffinity | RetentionPolicy::LuminousDecisionRecoveryAware => {
            let phase_ok = context
                .phase_affinity_ppm
                .is_some_and(|value| value <= 1_000_000)
                && context.phase_affinity_phase_id.as_deref()
                    == Some(request.next_phase_id.as_str())
                && context.phase_manifest_digest.as_deref()
                    == Some(request.phase_manifest_digest.as_str());
            if !phase_ok {
                return Ok(Some(ProtectionReason::PhaseAffinityUnknown));
            }
        }
        RetentionPolicy::TransitionReuse => {
            if context
                .future_reuse_probability_ppm
                .is_none_or(|value| value > 1_000_000)
            {
                return Ok(Some(ProtectionReason::TransitionReuseUnknown));
            }
        }
    }

    if policy == RetentionPolicy::LuminousDecisionRecoveryAware
        && (context
            .future_reuse_probability_ppm
            .is_none_or(|value| value > 1_000_000)
            || context.prefill_latency_ticks.is_none())
    {
        return Ok(Some(
            if context
                .future_reuse_probability_ppm
                .is_none_or(|value| value > 1_000_000)
            {
                ProtectionReason::TransitionReuseUnknown
            } else {
                ProtectionReason::PrefillCostUnknown
            },
        ));
    }

    Ok(None)
}

fn compare_victims(policy: RetentionPolicy, left: &ContextEntry, right: &ContextEntry) -> Ordering {
    let left_impact = match &left.decision_impact {
        DecisionImpact::Qualified {
            regret_upper_units, ..
        } => *regret_upper_units,
        _ => u64::MAX,
    };
    let right_impact = match &right.decision_impact {
        DecisionImpact::Qualified {
            regret_upper_units, ..
        } => *regret_upper_units,
        _ => u64::MAX,
    };
    match policy {
        RetentionPolicy::Lru => left
            .last_access_tick
            .cmp(&right.last_access_tick)
            .then_with(|| left.context_id.cmp(&right.context_id)),
        RetentionPolicy::PhaseAffinity => left
            .phase_affinity_ppm
            .unwrap_or(u32::MAX)
            .cmp(&right.phase_affinity_ppm.unwrap_or(u32::MAX))
            .then_with(|| left.last_access_tick.cmp(&right.last_access_tick))
            .then_with(|| left.context_id.cmp(&right.context_id)),
        RetentionPolicy::TransitionReuse => left
            .future_reuse_probability_ppm
            .unwrap_or(u32::MAX)
            .cmp(&right.future_reuse_probability_ppm.unwrap_or(u32::MAX))
            .then_with(|| left.last_access_tick.cmp(&right.last_access_tick))
            .then_with(|| left.context_id.cmp(&right.context_id)),
        RetentionPolicy::LuminousDecisionRecoveryAware => left
            .recovery_critical
            .cmp(&right.recovery_critical)
            .then_with(|| left_impact.cmp(&right_impact))
            .then_with(|| {
                left.phase_affinity_ppm
                    .unwrap_or(u32::MAX)
                    .cmp(&right.phase_affinity_ppm.unwrap_or(u32::MAX))
            })
            .then_with(|| {
                left.future_reuse_probability_ppm
                    .unwrap_or(u32::MAX)
                    .cmp(&right.future_reuse_probability_ppm.unwrap_or(u32::MAX))
            })
            .then_with(|| {
                left.prefill_latency_ticks
                    .unwrap_or(u64::MAX)
                    .cmp(&right.prefill_latency_ticks.unwrap_or(u64::MAX))
            })
            .then_with(|| left.last_access_tick.cmp(&right.last_access_tick))
            .then_with(|| left.context_id.cmp(&right.context_id)),
    }
}

/// Plan which resident contexts to evict to satisfy an incremental KV request.
///
/// Hard-pinned and Unknown/expired contexts are protected for every policy.
/// Qualified regret comparisons require one common task-loss scale.
pub fn plan_retention(
    policy: RetentionPolicy,
    request: &RetentionRequest,
    contexts: &[ContextEntry],
) -> Result<RetentionPlan, RetentionError> {
    if request.next_phase_id.trim().is_empty() || request.phase_manifest_digest.trim().is_empty() {
        return Err(RetentionError::EmptyPhaseScope);
    }

    let mut ids = BTreeSet::new();
    let mut used = 0_u64;
    let mut qualified_scale: Option<&str> =
        if policy == RetentionPolicy::LuminousDecisionRecoveryAware {
            request.required_loss_scale_id.as_deref()
        } else {
            None
        };
    for context in contexts {
        if context.context_id.trim().is_empty() {
            return Err(RetentionError::EmptyContextId);
        }
        if !ids.insert(context.context_id.as_str()) {
            return Err(RetentionError::DuplicateContextId(
                context.context_id.clone(),
            ));
        }
        if context.token_count == 0 {
            return Err(RetentionError::ZeroTokenFootprint(
                context.context_id.clone(),
            ));
        }
        if context
            .phase_affinity_ppm
            .is_some_and(|value| value > 1_000_000)
        {
            return Err(RetentionError::InvalidProbability {
                context_id: context.context_id.clone(),
                field: "phase_affinity_ppm",
                value_ppm: context.phase_affinity_ppm.unwrap_or(u32::MAX),
            });
        }
        if context
            .future_reuse_probability_ppm
            .is_some_and(|value| value > 1_000_000)
        {
            return Err(RetentionError::InvalidProbability {
                context_id: context.context_id.clone(),
                field: "future_reuse_probability_ppm",
                value_ppm: context.future_reuse_probability_ppm.unwrap_or(u32::MAX),
            });
        }
        if context.recovery_critical
            && context
                .recovery_evidence_digest
                .as_ref()
                .is_some_and(|digest| digest.trim().is_empty())
        {
            return Err(RetentionError::InvalidRecoveryEvidence(
                context.context_id.clone(),
            ));
        }
        if let DecisionImpact::Qualified {
            loss_scale_id,
            calibration_id,
            calibration_digest,
            ..
        } = &context.decision_impact
        {
            if loss_scale_id.trim().is_empty()
                || calibration_id.trim().is_empty()
                || calibration_digest.trim().is_empty()
            {
                return Err(RetentionError::InvalidDecisionEvidence(
                    context.context_id.clone(),
                ));
            }
            if policy == RetentionPolicy::LuminousDecisionRecoveryAware {
                if let Some(first) = qualified_scale {
                    if first != loss_scale_id.as_str() {
                        return Err(RetentionError::IncomparableLossScales {
                            first: first.to_owned(),
                            other: loss_scale_id.clone(),
                        });
                    }
                } else {
                    qualified_scale = Some(loss_scale_id.as_str());
                }
            }
        }
        used = used
            .checked_add(context.token_count)
            .ok_or(RetentionError::TokenArithmeticOverflow)?;
    }
    if let Some(scale) = request.required_loss_scale_id.as_deref() {
        if scale.trim().is_empty() {
            return Err(RetentionError::InvalidDecisionEvidence(
                "<request-loss-scale>".to_owned(),
            ));
        }
    }

    let requested_total = used
        .checked_add(request.requested_additional_tokens)
        .ok_or(RetentionError::TokenArithmeticOverflow)?;
    if requested_total <= request.capacity_tokens {
        return Ok(RetentionPlan {
            policy,
            disposition: RetentionDisposition::Admitted,
            used_tokens_before: used,
            requested_additional_tokens: request.requested_additional_tokens,
            freed_tokens: 0,
            used_tokens_after: requested_total,
            remaining_capacity_tokens: request.capacity_tokens - requested_total,
            evictions: Vec::new(),
            protected_contexts: Vec::new(),
            actuation_authorized: false,
        });
    }

    let mut candidates: Vec<&ContextEntry> = Vec::new();
    let mut protected_contexts = Vec::new();
    for context in contexts {
        match protection_reason(request, policy, context)? {
            Some(reason) => protected_contexts.push((context.context_id.clone(), reason)),
            None => candidates.push(context),
        }
    }

    let protected_tokens = contexts
        .iter()
        .filter(|context| {
            protected_contexts
                .iter()
                .any(|(id, _)| id == &context.context_id)
        })
        .try_fold(0_u64, |sum, context| {
            sum.checked_add(context.token_count)
                .ok_or(RetentionError::TokenArithmeticOverflow)
        })?;
    let protected_total = protected_tokens
        .checked_add(request.requested_additional_tokens)
        .ok_or(RetentionError::TokenArithmeticOverflow)?;

    if protected_total > request.capacity_tokens {
        return Ok(RetentionPlan {
            policy,
            disposition: RetentionDisposition::DeferredProtectedCapacity,
            used_tokens_before: used,
            requested_additional_tokens: request.requested_additional_tokens,
            freed_tokens: 0,
            used_tokens_after: used,
            remaining_capacity_tokens: request.capacity_tokens.saturating_sub(used),
            evictions: Vec::new(),
            protected_contexts,
            actuation_authorized: false,
        });
    }

    candidates.sort_by(|left, right| compare_victims(policy, left, right));
    let required_free = requested_total - request.capacity_tokens;
    let mut freed = 0_u64;
    let mut evictions = Vec::new();
    for candidate in candidates {
        if freed >= required_free {
            break;
        }
        let (loss_scale_id, regret_upper_units) = match &candidate.decision_impact {
            DecisionImpact::Qualified {
                loss_scale_id,
                regret_upper_units,
                ..
            } => (loss_scale_id.clone(), *regret_upper_units),
            _ => continue,
        };
        freed = freed
            .checked_add(candidate.token_count)
            .ok_or(RetentionError::TokenArithmeticOverflow)?;
        evictions.push(EvictionReceipt {
            context_id: candidate.context_id.clone(),
            freed_tokens: candidate.token_count,
            loss_scale_id,
            regret_upper_units,
            recovery_critical: candidate.recovery_critical,
            reason: EvictionReason::CapacityPressure,
        });
    }

    if freed < required_free {
        return Ok(RetentionPlan {
            policy,
            disposition: RetentionDisposition::DeferredProtectedCapacity,
            used_tokens_before: used,
            requested_additional_tokens: request.requested_additional_tokens,
            freed_tokens: 0,
            used_tokens_after: used,
            remaining_capacity_tokens: request.capacity_tokens.saturating_sub(used),
            evictions: Vec::new(),
            protected_contexts,
            actuation_authorized: false,
        });
    }

    let used_after_eviction = used - freed;
    let used_after = used_after_eviction
        .checked_add(request.requested_additional_tokens)
        .ok_or(RetentionError::TokenArithmeticOverflow)?;

    Ok(RetentionPlan {
        policy,
        disposition: RetentionDisposition::Admitted,
        used_tokens_before: used,
        requested_additional_tokens: request.requested_additional_tokens,
        freed_tokens: freed,
        used_tokens_after: used_after,
        remaining_capacity_tokens: request.capacity_tokens - used_after,
        evictions,
        protected_contexts,
        actuation_authorized: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(
        id: &str,
        tokens: u64,
        last_access_tick: u64,
        regret: u64,
        recovery_critical: bool,
        phase: u32,
        reuse: u32,
        prefill: u64,
    ) -> ContextEntry {
        ContextEntry {
            context_id: id.to_owned(),
            token_count: tokens,
            last_access_tick,
            hard_pin_reason: None,
            recovery_critical,
            recovery_evidence_digest: recovery_critical.then(|| format!("recovery-{id}")),
            phase_affinity_ppm: Some(phase),
            phase_affinity_phase_id: Some("act".to_owned()),
            phase_manifest_digest: Some("phase-manifest-v1".to_owned()),
            future_reuse_probability_ppm: Some(reuse),
            prefill_latency_ticks: Some(prefill),
            decision_impact: DecisionImpact::Qualified {
                loss_scale_id: "mission-loss-v1".to_owned(),
                regret_upper_units: regret,
                calibration_id: format!("cal-{id}"),
                calibration_digest: format!("digest-{id}"),
                valid_through_tick: 100,
            },
        }
    }

    fn request(capacity: u64, additional: u64) -> RetentionRequest {
        RetentionRequest {
            capacity_tokens: capacity,
            requested_additional_tokens: additional,
            current_tick: 10,
            required_loss_scale_id: Some("mission-loss-v1".to_owned()),
            next_phase_id: "act".to_owned(),
            phase_manifest_digest: "phase-manifest-v1".to_owned(),
        }
    }

    fn ids(plan: &RetentionPlan) -> Vec<&str> {
        plan.evictions
            .iter()
            .map(|receipt| receipt.context_id.as_str())
            .collect()
    }

    #[test]
    fn hard_pinned_context_is_never_evicted() {
        let mut pinned = context("safety", 40, 1, 0, false, 0, 0, 1);
        pinned.hard_pin_reason = Some("local-rta-state".to_owned());
        let normal = context("normal", 40, 2, 1, false, 100, 100, 5);
        let plan = plan_retention(
            RetentionPolicy::LuminousDecisionRecoveryAware,
            &request(80, 40),
            &[pinned, normal],
        )
        .unwrap();
        assert_eq!(plan.disposition, RetentionDisposition::Admitted);
        assert_eq!(ids(&plan), vec!["normal"]);
        assert!(plan.protected_contexts.iter().any(|(id, _)| id == "safety"));
        assert!(!plan.actuation_authorized);
    }

    #[test]
    fn unknown_decision_impact_is_protected_not_zero() {
        let mut unknown = context("unknown", 40, 1, 0, false, 0, 0, 1);
        unknown.decision_impact = DecisionImpact::Unknown;
        let normal = context("normal", 40, 2, 2, false, 100, 100, 5);
        let plan = plan_retention(
            RetentionPolicy::LuminousDecisionRecoveryAware,
            &request(80, 40),
            &[unknown, normal],
        )
        .unwrap();
        assert_eq!(ids(&plan), vec!["normal"]);
        assert!(
            plan.protected_contexts
                .iter()
                .any(|(id, reason)| id == "unknown"
                    && *reason == ProtectionReason::DecisionImpactUnknown)
        );
    }

    #[test]
    fn expired_decision_impact_is_protected() {
        let mut expired = context("expired", 40, 1, 0, false, 0, 0, 1);
        if let DecisionImpact::Qualified {
            valid_through_tick, ..
        } = &mut expired.decision_impact
        {
            *valid_through_tick = 9;
        }
        let normal = context("normal", 40, 2, 2, false, 100, 100, 5);
        let plan = plan_retention(
            RetentionPolicy::LuminousDecisionRecoveryAware,
            &request(80, 40),
            &[expired, normal],
        )
        .unwrap();
        assert_eq!(ids(&plan), vec!["normal"]);
        assert!(
            plan.protected_contexts
                .iter()
                .any(|(id, reason)| id == "expired"
                    && *reason == ProtectionReason::DecisionImpactExpired)
        );
    }

    #[test]
    fn low_regret_non_recovery_context_precedes_high_regret_recovery_context() {
        let low = context("low", 40, 100, 1, false, 50, 200_000, 5);
        let recovery = context("recovery", 40, 1, 900, true, 900_000, 900_000, 500);
        let plan = plan_retention(
            RetentionPolicy::LuminousDecisionRecoveryAware,
            &request(80, 40),
            &[recovery, low],
        )
        .unwrap();
        assert_eq!(ids(&plan), vec!["low"]);
    }

    #[test]
    fn lru_control_can_choose_different_victim_than_luminous() {
        let low_regret_recent = context("low-regret-recent", 40, 90, 1, false, 50, 200_000, 5);
        let high_regret_old_recovery = context(
            "high-regret-old-recovery",
            40,
            1,
            800,
            true,
            900_000,
            900_000,
            500,
        );
        let contexts = vec![low_regret_recent, high_regret_old_recovery];
        let lru = plan_retention(RetentionPolicy::Lru, &request(80, 40), &contexts).unwrap();
        let luminous = plan_retention(
            RetentionPolicy::LuminousDecisionRecoveryAware,
            &request(80, 40),
            &contexts,
        )
        .unwrap();
        assert_eq!(ids(&lru), vec!["high-regret-old-recovery"]);
        assert_eq!(ids(&luminous), vec!["low-regret-recent"]);
    }

    #[test]
    fn phase_affinity_control_evicts_low_next_phase_affinity() {
        let high_affinity = context("high-phase", 40, 1, 0, false, 990_000, 800_000, 20);
        let low_affinity = context("low-phase", 40, 100, 0, false, 10_000, 800_000, 20);
        let plan = plan_retention(
            RetentionPolicy::PhaseAffinity,
            &request(80, 40),
            &[high_affinity, low_affinity],
        )
        .unwrap();
        assert_eq!(ids(&plan), vec!["low-phase"]);
    }

    #[test]
    fn transition_reuse_control_evicts_low_reuse_probability() {
        let high_reuse = context("high-reuse", 40, 1, 0, false, 500_000, 990_000, 20);
        let low_reuse = context("low-reuse", 40, 100, 0, false, 500_000, 10_000, 20);
        let plan = plan_retention(
            RetentionPolicy::TransitionReuse,
            &request(80, 40),
            &[high_reuse, low_reuse],
        )
        .unwrap();
        assert_eq!(ids(&plan), vec!["low-reuse"]);
    }

    #[test]
    fn recovery_critical_context_is_last_to_evict() {
        let normal = context("normal", 40, 100, 10, false, 10_000, 10_000, 1);
        let recovery = context("recovery", 40, 1, 10, true, 10_000, 10_000, 1);
        let plan = plan_retention(
            RetentionPolicy::LuminousDecisionRecoveryAware,
            &request(40, 40),
            &[recovery, normal],
        )
        .unwrap();
        assert_eq!(ids(&plan), vec!["normal"]);
    }

    #[test]
    fn incomparable_loss_scales_reject_global_regret_order() {
        let mut other = context("other", 40, 2, 1, false, 100, 100, 5);
        if let DecisionImpact::Qualified { loss_scale_id, .. } = &mut other.decision_impact {
            *loss_scale_id = "other-task-loss".to_owned();
        }
        assert!(matches!(
            plan_retention(
                RetentionPolicy::LuminousDecisionRecoveryAware,
                &request(80, 40),
                &[context("a", 40, 1, 1, false, 0, 0, 1), other],
            ),
            Err(RetentionError::IncomparableLossScales { .. })
        ));
    }

    #[test]
    fn baselines_do_not_compare_unrelated_loss_scales() {
        let a = context("a", 40, 1, 1, false, 100, 100, 1);
        let mut b = context("b", 40, 2, 2, false, 100, 100, 2);
        if let DecisionImpact::Qualified { loss_scale_id, .. } = &mut b.decision_impact {
            *loss_scale_id = "other-task-loss".to_owned();
        }
        let mut baseline_request = request(80, 40);
        baseline_request.required_loss_scale_id = None;
        let plan = plan_retention(RetentionPolicy::Lru, &baseline_request, &[a, b]).unwrap();
        assert_eq!(plan.disposition, RetentionDisposition::Admitted);
        assert_eq!(plan.evictions.len(), 1);
    }

    #[test]
    fn protected_capacity_excess_defers_without_partial_eviction() {
        let mut hard = context("hard", 60, 1, 0, false, 100, 100, 1);
        hard.hard_pin_reason = Some("local-rta".to_owned());
        let mut unknown = context("unknown", 30, 2, 0, false, 100, 100, 1);
        unknown.decision_impact = DecisionImpact::Unknown;
        let plan = plan_retention(
            RetentionPolicy::LuminousDecisionRecoveryAware,
            &request(100, 20),
            &[hard, unknown],
        )
        .unwrap();
        assert_eq!(
            plan.disposition,
            RetentionDisposition::DeferredProtectedCapacity
        );
        assert!(plan.evictions.is_empty());
        assert_eq!(plan.freed_tokens, 0);
    }

    #[test]
    fn enough_capacity_requires_no_eviction() {
        let contexts = vec![context("a", 20, 1, 1, false, 100, 100, 1)];
        let plan = plan_retention(
            RetentionPolicy::LuminousDecisionRecoveryAware,
            &request(100, 20),
            &contexts,
        )
        .unwrap();
        assert_eq!(plan.disposition, RetentionDisposition::Admitted);
        assert!(plan.evictions.is_empty());
        assert_eq!(plan.used_tokens_after, 40);
    }

    #[test]
    fn duplicate_context_ids_are_rejected() {
        let a = context("same", 20, 1, 1, false, 100, 100, 1);
        let b = context("same", 20, 2, 2, false, 100, 100, 1);
        assert_eq!(
            plan_retention(
                RetentionPolicy::LuminousDecisionRecoveryAware,
                &request(100, 20),
                &[a, b],
            ),
            Err(RetentionError::DuplicateContextId("same".to_owned()))
        );
    }

    #[test]
    fn invalid_ppm_probability_is_rejected() {
        let mut a = context("a", 40, 1, 1, false, 100, 100, 1);
        a.future_reuse_probability_ppm = Some(1_000_001);
        assert!(matches!(
            plan_retention(
                RetentionPolicy::LuminousDecisionRecoveryAware,
                &request(80, 40),
                &[a],
            ),
            Err(RetentionError::InvalidProbability { .. })
        ));
    }

    #[test]
    fn token_sum_overflow_is_rejected() {
        let a = context("a", u64::MAX, 1, 1, false, 100, 100, 1);
        let b = context("b", 1, 2, 1, false, 100, 100, 1);
        assert_eq!(
            plan_retention(RetentionPolicy::Lru, &request(u64::MAX, 1), &[a, b],),
            Err(RetentionError::TokenArithmeticOverflow)
        );
    }

    #[test]
    fn missing_phase_affinity_is_protected_for_phase_policy() {
        let mut a = context("a", 40, 1, 1, false, 100, 100, 1);
        a.phase_affinity_ppm = None;
        let b = context("b", 40, 2, 2, false, 100, 100, 1);
        let plan =
            plan_retention(RetentionPolicy::PhaseAffinity, &request(80, 40), &[a, b]).unwrap();
        assert_eq!(ids(&plan), vec!["b"]);
        assert!(
            plan.protected_contexts
                .iter()
                .any(|(id, reason)| id == "a" && *reason == ProtectionReason::PhaseAffinityUnknown)
        );
    }

    #[test]
    fn missing_prefill_cost_is_protected_for_luminous_policy() {
        let mut a = context("a", 40, 1, 1, false, 100, 100, 1);
        a.prefill_latency_ticks = None;
        let b = context("b", 40, 2, 2, false, 100, 100, 2);
        let plan = plan_retention(
            RetentionPolicy::LuminousDecisionRecoveryAware,
            &request(80, 40),
            &[a, b],
        )
        .unwrap();
        assert_eq!(ids(&plan), vec!["b"]);
        assert!(
            plan.protected_contexts
                .iter()
                .any(|(id, reason)| id == "a" && *reason == ProtectionReason::PrefillCostUnknown)
        );
    }

    #[test]
    fn invalid_decision_evidence_rejects_instead_of_becoming_zero() {
        let mut a = context("a", 40, 1, 0, false, 100, 100, 1);
        a.decision_impact = DecisionImpact::Invalid;
        assert_eq!(
            plan_retention(
                RetentionPolicy::LuminousDecisionRecoveryAware,
                &request(80, 40),
                &[a],
            ),
            Err(RetentionError::InvalidDecisionEvidence("a".to_owned()))
        );
    }

    #[test]
    fn recovery_critical_without_recovery_digest_is_protected() {
        let mut a = context("a", 40, 1, 0, true, 100, 100, 1);
        a.recovery_evidence_digest = None;
        let b = context("b", 40, 2, 1, false, 100, 100, 1);
        let plan = plan_retention(
            RetentionPolicy::LuminousDecisionRecoveryAware,
            &request(80, 40),
            &[a, b],
        )
        .unwrap();
        assert_eq!(ids(&plan), vec!["b"]);
        assert!(
            plan.protected_contexts
                .iter()
                .any(|(id, reason)| id == "a"
                    && *reason == ProtectionReason::RecoveryEvidenceUnknown)
        );
    }

    #[test]
    fn hard_pinned_contexts_alone_can_make_capacity_infeasible() {
        let mut a = context("a", 80, 1, 0, false, 100, 100, 1);
        a.hard_pin_reason = Some("safe-stop-state".to_owned());
        let plan = plan_retention(
            RetentionPolicy::LuminousDecisionRecoveryAware,
            &request(80, 1),
            &[a],
        )
        .unwrap();
        assert_eq!(
            plan.disposition,
            RetentionDisposition::DeferredProtectedCapacity
        );
        assert!(plan.evictions.is_empty());
    }
}
