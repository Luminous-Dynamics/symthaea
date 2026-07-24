// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, evidence-bound rollback triggers for activated upgrades.
//!
//! This module never dispatches a rollback by itself. It derives a bounded
//! trigger document and turns it into a short-lived rollback capability only
//! after a dedicated threshold ceremony.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::threshold::VerifiedThresholdCeremony;
use crate::upgrade_handoff::AuthorizedUpgradeHandoff;
use crate::upgrade_state::{FabricationUpgradeState, digest_upgrade_state};
use crate::upgrade_tracker::UpgradeStage;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const UPGRADE_HEALTH_SIGNAL_SCHEMA: &str = "symthaea.fabrication.upgrade-health-signal.v1";
pub const AUTOMATIC_ROLLBACK_TRIGGER_SCHEMA: &str =
    "symthaea.fabrication.automatic-rollback-trigger.v1";
pub const MAX_ROLLBACK_SIGNALS: usize = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RollbackTriggerKind {
    EmergencyStop,
    ContainmentEscalation,
    FailureRate,
    UncertainOutcome,
    TelemetryLoss,
    StateDivergence,
    ClockDiscontinuity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeHealthSignal {
    pub schema_version: String,
    pub handoff_digest: Sha256Digest,
    pub source_id: String,
    pub observed_at_unix_ms: u64,
    pub kind: RollbackTriggerKind,
    pub event_count: u64,
    pub attempted_jobs: u64,
    pub evidence_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutomaticRollbackPolicy {
    pub maximum_signal_age_ms: u64,
    pub maximum_authorization_duration_ms: u64,
    pub emergency_stop_threshold: u64,
    pub containment_threshold: u64,
    pub failure_rate_basis_points: u32,
    pub uncertain_rate_basis_points: u32,
    pub immediate_kinds: BTreeSet<RollbackTriggerKind>,
}

impl Default for AutomaticRollbackPolicy {
    fn default() -> Self {
        Self {
            maximum_signal_age_ms: 10 * 60 * 1_000,
            maximum_authorization_duration_ms: 5 * 60 * 1_000,
            emergency_stop_threshold: 1,
            containment_threshold: 1,
            failure_rate_basis_points: 500,
            uncertain_rate_basis_points: 100,
            immediate_kinds: [
                RollbackTriggerKind::StateDivergence,
                RollbackTriggerKind::ClockDiscontinuity,
            ]
            .into_iter()
            .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutomaticRollbackTrigger {
    pub schema_version: String,
    pub rollback_sequence: u64,
    pub handoff_digest: Sha256Digest,
    pub upgrade_state_digest: Sha256Digest,
    pub rollback_target_digest: Sha256Digest,
    pub triggering_kinds: BTreeSet<RollbackTriggerKind>,
    pub signal_digests: Vec<Sha256Digest>,
    pub evaluated_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutomaticRollbackError {
    UnsupportedSchema,
    InvalidPolicy,
    InvalidSourceId,
    ZeroDigest(&'static str),
    InvalidSignal,
    SequenceZero,
    UpgradeNotActivated,
    HandoffMismatch,
    StaleSignal,
    FutureSignal,
    TooManySignals,
    DuplicateSignal,
    NoTrigger,
    InvalidWindow,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedAutomaticRollback {
    trigger: AutomaticRollbackTrigger,
    trigger_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AuthorizedAutomaticRollback {
    pub fn trigger(&self) -> &AutomaticRollbackTrigger {
        &self.trigger
    }
    pub fn trigger_digest(&self) -> Sha256Digest {
        self.trigger_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }

    pub fn permits_rollback(&self, handoff_digest: Sha256Digest, unix_ms: u64) -> bool {
        self.trigger.handoff_digest == handoff_digest
            && unix_ms >= self.trigger.evaluated_at_unix_ms
            && unix_ms < self.trigger.expires_at_unix_ms
    }
}

impl UpgradeHealthSignal {
    pub fn validate(&self) -> Result<(), AutomaticRollbackError> {
        if self.schema_version != UPGRADE_HEALTH_SIGNAL_SCHEMA {
            return Err(AutomaticRollbackError::UnsupportedSchema);
        }
        if self.handoff_digest.0 == [0; 32] {
            return Err(AutomaticRollbackError::ZeroDigest("handoff_digest"));
        }
        if self.evidence_digest.0 == [0; 32] {
            return Err(AutomaticRollbackError::ZeroDigest("evidence_digest"));
        }
        if self.source_id.trim().is_empty()
            || self.source_id != self.source_id.trim()
            || self.source_id.len() > 256
            || self.source_id.chars().any(char::is_control)
            || self.event_count == 0
        {
            return Err(AutomaticRollbackError::InvalidSignal);
        }
        if matches!(
            self.kind,
            RollbackTriggerKind::FailureRate | RollbackTriggerKind::UncertainOutcome
        ) && self.attempted_jobs == 0
        {
            return Err(AutomaticRollbackError::InvalidSignal);
        }
        Ok(())
    }
}

pub fn digest_upgrade_health_signal(
    signal: &UpgradeHealthSignal,
) -> Result<Sha256Digest, AutomaticRollbackError> {
    signal.validate()?;
    let bytes = serde_json::to_vec(signal)
        .map_err(|error| AutomaticRollbackError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-health-signal-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn evaluate_automatic_rollback(
    rollback_sequence: u64,
    handoff: &AuthorizedUpgradeHandoff,
    upgrade_state: &FabricationUpgradeState,
    signals: &[UpgradeHealthSignal],
    evaluated_at_unix_ms: u64,
    expires_at_unix_ms: u64,
    policy: &AutomaticRollbackPolicy,
) -> Result<AutomaticRollbackTrigger, AutomaticRollbackError> {
    validate_policy(policy)?;
    if rollback_sequence == 0 {
        return Err(AutomaticRollbackError::SequenceZero);
    }
    if upgrade_state.active_stage != UpgradeStage::Activated {
        return Err(AutomaticRollbackError::UpgradeNotActivated);
    }
    if upgrade_state.evidence.handoff_digest != handoff.plan_digest {
        return Err(AutomaticRollbackError::HandoffMismatch);
    }
    if signals.is_empty() || signals.len() > MAX_ROLLBACK_SIGNALS {
        return Err(AutomaticRollbackError::TooManySignals);
    }
    let mut signal_digests = Vec::with_capacity(signals.len());
    let mut triggering_kinds = BTreeSet::new();
    for signal in signals {
        signal.validate()?;
        if signal.handoff_digest != handoff.plan_digest {
            return Err(AutomaticRollbackError::HandoffMismatch);
        }
        if signal.observed_at_unix_ms > evaluated_at_unix_ms {
            return Err(AutomaticRollbackError::FutureSignal);
        }
        if evaluated_at_unix_ms.saturating_sub(signal.observed_at_unix_ms)
            > policy.maximum_signal_age_ms
        {
            return Err(AutomaticRollbackError::StaleSignal);
        }
        if triggers(signal, policy) {
            triggering_kinds.insert(signal.kind);
        }
        signal_digests.push(digest_upgrade_health_signal(signal)?);
    }
    signal_digests.sort();
    if signal_digests.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(AutomaticRollbackError::DuplicateSignal);
    }
    if triggering_kinds.is_empty() {
        return Err(AutomaticRollbackError::NoTrigger);
    }
    if evaluated_at_unix_ms >= expires_at_unix_ms
        || expires_at_unix_ms.saturating_sub(evaluated_at_unix_ms)
            > policy.maximum_authorization_duration_ms
    {
        return Err(AutomaticRollbackError::InvalidWindow);
    }
    Ok(AutomaticRollbackTrigger {
        schema_version: AUTOMATIC_ROLLBACK_TRIGGER_SCHEMA.into(),
        rollback_sequence,
        handoff_digest: handoff.plan_digest,
        upgrade_state_digest: digest_upgrade_state(upgrade_state)
            .map_err(|_| AutomaticRollbackError::InvalidSignal)?,
        rollback_target_digest: handoff.plan.rollback_target_digest,
        triggering_kinds,
        signal_digests,
        evaluated_at_unix_ms,
        expires_at_unix_ms,
    })
}

pub fn digest_automatic_rollback_trigger(
    trigger: &AutomaticRollbackTrigger,
) -> Result<Sha256Digest, AutomaticRollbackError> {
    validate_trigger(trigger)?;
    let bytes = serde_json::to_vec(trigger)
        .map_err(|error| AutomaticRollbackError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.automatic-rollback-trigger-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_automatic_rollback(
    trigger: AutomaticRollbackTrigger,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedAutomaticRollback, AutomaticRollbackError> {
    let trigger_digest = digest_automatic_rollback_trigger(&trigger)?;
    if ceremony.purpose() != "automatic-upgrade-rollback" {
        return Err(AutomaticRollbackError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != trigger_digest {
        return Err(AutomaticRollbackError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedAutomaticRollback {
        trigger,
        trigger_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}

fn triggers(signal: &UpgradeHealthSignal, policy: &AutomaticRollbackPolicy) -> bool {
    if policy.immediate_kinds.contains(&signal.kind) {
        return true;
    }
    match signal.kind {
        RollbackTriggerKind::EmergencyStop => signal.event_count >= policy.emergency_stop_threshold,
        RollbackTriggerKind::ContainmentEscalation => {
            signal.event_count >= policy.containment_threshold
        }
        RollbackTriggerKind::FailureRate => {
            basis_points(signal.event_count, signal.attempted_jobs)
                >= u64::from(policy.failure_rate_basis_points)
        }
        RollbackTriggerKind::UncertainOutcome => {
            basis_points(signal.event_count, signal.attempted_jobs)
                >= u64::from(policy.uncertain_rate_basis_points)
        }
        RollbackTriggerKind::TelemetryLoss => signal.event_count > 0,
        RollbackTriggerKind::StateDivergence | RollbackTriggerKind::ClockDiscontinuity => true,
    }
}

fn validate_policy(policy: &AutomaticRollbackPolicy) -> Result<(), AutomaticRollbackError> {
    if policy.maximum_signal_age_ms == 0
        || policy.maximum_authorization_duration_ms == 0
        || policy.emergency_stop_threshold == 0
        || policy.containment_threshold == 0
        || policy.failure_rate_basis_points > 10_000
        || policy.uncertain_rate_basis_points > 10_000
    {
        return Err(AutomaticRollbackError::InvalidPolicy);
    }
    Ok(())
}

fn validate_trigger(trigger: &AutomaticRollbackTrigger) -> Result<(), AutomaticRollbackError> {
    if trigger.schema_version != AUTOMATIC_ROLLBACK_TRIGGER_SCHEMA {
        return Err(AutomaticRollbackError::UnsupportedSchema);
    }
    if trigger.rollback_sequence == 0 {
        return Err(AutomaticRollbackError::SequenceZero);
    }
    for (name, digest) in [
        ("handoff_digest", trigger.handoff_digest),
        ("upgrade_state_digest", trigger.upgrade_state_digest),
        ("rollback_target_digest", trigger.rollback_target_digest),
    ] {
        if digest.0 == [0; 32] {
            return Err(AutomaticRollbackError::ZeroDigest(name));
        }
    }
    if trigger.triggering_kinds.is_empty()
        || trigger.signal_digests.is_empty()
        || trigger
            .signal_digests
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || trigger.evaluated_at_unix_ms >= trigger.expires_at_unix_ms
    {
        return Err(AutomaticRollbackError::InvalidWindow);
    }
    Ok(())
}

fn basis_points(numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        10_000
    } else {
        numerator.saturating_mul(10_000) / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    #[test]
    fn failure_rate_threshold_is_deterministic() {
        let signal = UpgradeHealthSignal {
            schema_version: UPGRADE_HEALTH_SIGNAL_SCHEMA.into(),
            handoff_digest: sha256(b"handoff"),
            source_id: "gateway-a".into(),
            observed_at_unix_ms: 10,
            kind: RollbackTriggerKind::FailureRate,
            event_count: 5,
            attempted_jobs: 100,
            evidence_digest: sha256(b"evidence"),
        };
        assert!(triggers(&signal, &AutomaticRollbackPolicy::default()));
    }

    #[test]
    fn rate_signal_requires_attempted_jobs() {
        let signal = UpgradeHealthSignal {
            schema_version: UPGRADE_HEALTH_SIGNAL_SCHEMA.into(),
            handoff_digest: sha256(b"handoff"),
            source_id: "gateway-a".into(),
            observed_at_unix_ms: 10,
            kind: RollbackTriggerKind::FailureRate,
            event_count: 1,
            attempted_jobs: 0,
            evidence_digest: sha256(b"evidence"),
        };
        assert_eq!(
            signal.validate(),
            Err(AutomaticRollbackError::InvalidSignal)
        );
    }
}
