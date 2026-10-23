// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-011: anytime-valid harm monitor for a frozen policy.
//!
//! This consumes confirmed post-action outcomes for selected escalations. It is a
//! monitoring/certification reference, not a predictor of the next action's safety.
//! The likelihood-ratio mixture is an e-process under the conditional null that each
//! escalation's harmful-outcome probability is at least the declared risk limit.
//! Policy identity and the operating domain must remain frozen; distribution shift
//! revokes the monitor. The source is not a secure telemetry boundary.
//!
//! Run tests:
//!   rustc --edition=2024 --test tools/vision-dvpc-anytime-harm-monitor.rs -o /tmp/vis-dvpc-011-tests
//!   /tmp/vis-dvpc-011-tests

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

pub type EvidenceDigest = [u8; 32];
const ALTERNATIVE_FRACTIONS: [f64; 5] = [0.10, 0.25, 0.50, 0.75, 0.90];
pub const MAX_GROUPS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainStatusV1 {
    InDomain,
    OodSuspected,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupMonitorStatusV1 {
    InsufficientEvidence,
    Monitoring,
    Certified,
    RevokedShift,
}

impl GroupMonitorStatusV1 {
    pub fn label(self) -> &'static str {
        match self {
            Self::InsufficientEvidence => "INSUFFICIENT_EVIDENCE",
            Self::Monitoring => "MONITORING",
            Self::Certified => "CERTIFIED",
            Self::RevokedShift => "REVOKED_SHIFT",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorStatusV1 {
    InsufficientEvidence,
    Monitoring,
    Certified,
    RevokedShift,
}

impl MonitorStatusV1 {
    pub fn label(self) -> &'static str {
        match self {
            Self::InsufficientEvidence => "INSUFFICIENT_EVIDENCE",
            Self::Monitoring => "MONITORING",
            Self::Certified => "CERTIFIED",
            Self::RevokedShift => "REVOKED_SHIFT",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnytimeMonitorConfigV1 {
    pub policy_id: String,
    pub policy_digest: EvidenceDigest,
    pub loss_profile_id: String,
    pub domain_id: String,
    pub predeclared_group_ids: Vec<String>,
    pub policy_family_size: usize,
    pub family_wise_delta: f64,
    pub maximum_harm_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EscalationOutcomeV1 {
    /// One unique ID per independent evaluated escalation, not one per video frame.
    pub trace_id: String,
    pub source_group_id: String,
    pub policy_id: String,
    pub policy_digest: EvidenceDigest,
    pub loss_profile_id: String,
    pub domain_id: String,
    pub domain_status: DomainStatusV1,
    /// Confirmed post-action evaluator label for the already-selected escalation.
    pub harmful: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnytimeMonitorErrorV1 {
    InvalidConfig,
    InvalidIdentifier(String),
    InvalidDigest,
    DuplicateTraceId(String),
    UndeclaredGroup(String),
    PolicyBindingMismatch,
    MonitorAlreadyRevoked,
    CounterOverflow,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupMonitorSnapshotV1 {
    pub group_id: String,
    pub escalation_count: usize,
    pub harmful_count: usize,
    /// log of the mixture e-value; retained in log-space to avoid overflow.
    pub log_e_value: f64,
    pub log_certification_threshold: f64,
    pub status: GroupMonitorStatusV1,
}

#[derive(Debug, Clone, PartialEq)]
struct GroupStateV1 {
    escalation_count: usize,
    harmful_count: usize,
    log_e_value: f64,
    status: GroupMonitorStatusV1,
}

#[derive(Debug, Clone)]
pub struct AnytimeHarmMonitorV1 {
    config: AnytimeMonitorConfigV1,
    adjusted_delta: f64,
    log_certification_threshold: f64,
    groups: BTreeMap<String, GroupStateV1>,
    seen_trace_ids: BTreeSet<String>,
    revoked_shift: bool,
}

impl AnytimeHarmMonitorV1 {
    pub fn new(config: AnytimeMonitorConfigV1) -> Result<Self, AnytimeMonitorErrorV1> {
        validate_config(&config)?;
        let mut group_ids = config.predeclared_group_ids.clone();
        group_ids.sort();
        group_ids.dedup();
        if group_ids.len() != config.predeclared_group_ids.len() {
            return Err(AnytimeMonitorErrorV1::InvalidConfig);
        }
        let hypothesis_count = config
            .policy_family_size
            .checked_mul(group_ids.len())
            .ok_or(AnytimeMonitorErrorV1::InvalidConfig)?;
        if hypothesis_count == 0 {
            return Err(AnytimeMonitorErrorV1::InvalidConfig);
        }
        let adjusted_delta = config.family_wise_delta / hypothesis_count as f64;
        if !adjusted_delta.is_finite() || adjusted_delta <= 0.0 || adjusted_delta >= 1.0 {
            return Err(AnytimeMonitorErrorV1::InvalidConfig);
        }
        let log_certification_threshold = -adjusted_delta.ln();
        let groups = group_ids
            .into_iter()
            .map(|group_id| {
                (
                    group_id,
                    GroupStateV1 {
                        escalation_count: 0,
                        harmful_count: 0,
                        log_e_value: 0.0,
                        status: GroupMonitorStatusV1::InsufficientEvidence,
                    },
                )
            })
            .collect();
        Ok(Self {
            config,
            adjusted_delta,
            log_certification_threshold,
            groups,
            seen_trace_ids: BTreeSet::new(),
            revoked_shift: false,
        })
    }

    pub fn adjusted_delta(&self) -> f64 {
        self.adjusted_delta
    }

    pub fn log_certification_threshold(&self) -> f64 {
        self.log_certification_threshold
    }

    pub fn status(&self) -> MonitorStatusV1 {
        if self.revoked_shift {
            return MonitorStatusV1::RevokedShift;
        }
        if self
            .groups
            .values()
            .all(|group| group.status == GroupMonitorStatusV1::Certified)
        {
            return MonitorStatusV1::Certified;
        }
        if self
            .groups
            .values()
            .any(|group| group.escalation_count == 0)
        {
            return MonitorStatusV1::InsufficientEvidence;
        }
        MonitorStatusV1::Monitoring
    }

    pub fn group_snapshots(&self) -> Vec<GroupMonitorSnapshotV1> {
        self.groups
            .iter()
            .map(|(group_id, state)| GroupMonitorSnapshotV1 {
                group_id: group_id.clone(),
                escalation_count: state.escalation_count,
                harmful_count: state.harmful_count,
                log_e_value: state.log_e_value,
                log_certification_threshold: self.log_certification_threshold,
                status: if self.revoked_shift {
                    GroupMonitorStatusV1::RevokedShift
                } else {
                    state.status
                },
            })
            .collect()
    }

    /// Consume one confirmed post-action outcome for an escalation.
    ///
    /// The log-e value is recomputed from counts after every event, making receipts
    /// independent of floating-point accumulation order. If the input is OOD/unknown
    /// or its domain changes, revocation is absorbing. A current certificate can
    /// return to MONITORING after further harmful outcomes lower the current e-value;
    /// repeated crossing decisions remain anytime-valid by Ville's inequality.
    pub fn observe(
        &mut self,
        event: EscalationOutcomeV1,
    ) -> Result<MonitorStatusV1, AnytimeMonitorErrorV1> {
        if self.revoked_shift {
            return Err(AnytimeMonitorErrorV1::MonitorAlreadyRevoked);
        }
        if !valid_identifier(&event.trace_id)
            || !valid_identifier(&event.source_group_id)
            || !valid_identifier(&event.policy_id)
            || !valid_identifier(&event.loss_profile_id)
            || !valid_identifier(&event.domain_id)
        {
            return Err(AnytimeMonitorErrorV1::InvalidIdentifier(
                event.trace_id.clone(),
            ));
        }
        if self.seen_trace_ids.contains(&event.trace_id) {
            return Err(AnytimeMonitorErrorV1::DuplicateTraceId(event.trace_id));
        }

        if event.domain_status != DomainStatusV1::InDomain
            || event.domain_id != self.config.domain_id
        {
            self.seen_trace_ids.insert(event.trace_id);
            self.revoke_for_shift();
            return Ok(MonitorStatusV1::RevokedShift);
        }
        if event.policy_id != self.config.policy_id
            || event.policy_digest != self.config.policy_digest
            || event.loss_profile_id != self.config.loss_profile_id
        {
            self.revoke_for_shift();
            return Err(AnytimeMonitorErrorV1::PolicyBindingMismatch);
        }
        if !self.groups.contains_key(&event.source_group_id) {
            self.revoke_for_shift();
            return Err(AnytimeMonitorErrorV1::UndeclaredGroup(
                event.source_group_id,
            ));
        }

        let state = self
            .groups
            .get_mut(&event.source_group_id)
            .ok_or_else(|| AnytimeMonitorErrorV1::UndeclaredGroup(event.source_group_id.clone()))?;
        state.escalation_count = state
            .escalation_count
            .checked_add(1)
            .ok_or(AnytimeMonitorErrorV1::CounterOverflow)?;
        if event.harmful {
            state.harmful_count = state
                .harmful_count
                .checked_add(1)
                .ok_or(AnytimeMonitorErrorV1::CounterOverflow)?;
        }
        state.log_e_value = log_mixture_e_value(
            state.escalation_count,
            state.harmful_count,
            self.config.maximum_harm_rate,
        );
        state.status = if state.log_e_value >= self.log_certification_threshold {
            GroupMonitorStatusV1::Certified
        } else {
            GroupMonitorStatusV1::Monitoring
        };
        self.seen_trace_ids.insert(event.trace_id);
        Ok(self.status())
    }

    pub fn to_tsv(&self) -> String {
        let mut out = format!(
            "DVPC011\t{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{}\t{:.9}\n",
            self.status().label(),
            self.config.policy_id,
            digest_hex(&self.config.policy_digest),
            self.config.loss_profile_id,
            self.config.domain_id,
            self.config.policy_family_size,
            self.config.family_wise_delta,
            self.config.maximum_harm_rate,
            self.groups.len(),
            self.adjusted_delta,
        );
        for snapshot in self.group_snapshots() {
            out.push_str(&format!(
                "GROUP\t{}\t{}\t{}\t{}\t{:.12}\t{:.12}\n",
                snapshot.group_id,
                snapshot.status.label(),
                snapshot.escalation_count,
                snapshot.harmful_count,
                snapshot.log_e_value,
                snapshot.log_certification_threshold,
            ));
        }
        out
    }

    fn revoke_for_shift(&mut self) {
        self.revoked_shift = true;
        for group in self.groups.values_mut() {
            group.status = GroupMonitorStatusV1::RevokedShift;
        }
    }
}

fn validate_config(config: &AnytimeMonitorConfigV1) -> Result<(), AnytimeMonitorErrorV1> {
    if !valid_identifier(&config.policy_id)
        || !valid_identifier(&config.loss_profile_id)
        || !valid_identifier(&config.domain_id)
        || config.policy_digest.iter().all(|byte| *byte == 0)
        || config.predeclared_group_ids.is_empty()
        || config.predeclared_group_ids.len() > MAX_GROUPS
        || config.policy_family_size == 0
        || !config.family_wise_delta.is_finite()
        || config.family_wise_delta <= 0.0
        || config.family_wise_delta >= 1.0
        || !config.maximum_harm_rate.is_finite()
        || config.maximum_harm_rate <= 0.0
        || config.maximum_harm_rate >= 1.0
    {
        return Err(AnytimeMonitorErrorV1::InvalidConfig);
    }
    if config
        .predeclared_group_ids
        .iter()
        .any(|group| !valid_identifier(group))
    {
        return Err(AnytimeMonitorErrorV1::InvalidConfig);
    }
    Ok(())
}

/// log of the equal-weight mixture of fixed likelihood-ratio processes.
///
/// For each q < r, the likelihood ratio is a nonnegative supermartingale under
/// the conditional null p_t >= r. A fixed mixture remains a supermartingale;
/// by Ville's inequality, P(sup_t E_t >= 1/delta) <= delta.
fn log_mixture_e_value(escalations: usize, harmful: usize, risk_limit: f64) -> f64 {
    if escalations == 0 || harmful > escalations {
        return f64::NEG_INFINITY;
    }
    let clean = escalations - harmful;
    let logs: Vec<f64> = ALTERNATIVE_FRACTIONS
        .iter()
        .map(|fraction| {
            let q = risk_limit * fraction;
            harmful as f64 * (q / risk_limit).ln()
                + clean as f64 * ((1.0 - q) / (1.0 - risk_limit)).ln()
        })
        .collect();
    let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    maximum
        + logs
            .iter()
            .map(|value| (value - maximum).exp())
            .sum::<f64>()
            .ln()
        - (ALTERNATIVE_FRACTIONS.len() as f64).ln()
}

fn valid_identifier(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(|ch| matches!(ch, '\t' | '\r' | '\n'))
}

fn digest_hex(digest: &EvidenceDigest) -> String {
    let mut result = String::with_capacity(64);
    for byte in digest {
        result.push_str(&format!("{byte:02x}"));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICY: &str = "fixed-router-v1";
    const LOSS: &str = "collision-loss-v1";
    const DOMAIN: &str = "night-clear-v1";
    const DIGEST: EvidenceDigest = [0xA5; 32];

    fn config(groups: &[&str]) -> AnytimeMonitorConfigV1 {
        AnytimeMonitorConfigV1 {
            policy_id: POLICY.to_owned(),
            policy_digest: DIGEST,
            loss_profile_id: LOSS.to_owned(),
            domain_id: DOMAIN.to_owned(),
            predeclared_group_ids: groups.iter().map(|group| (*group).to_owned()).collect(),
            policy_family_size: 1,
            family_wise_delta: 0.05,
            maximum_harm_rate: 0.05,
        }
    }

    fn event(index: usize, group: &str, harmful: bool) -> EscalationOutcomeV1 {
        EscalationOutcomeV1 {
            trace_id: format!("trace-{index:05}"),
            source_group_id: group.to_owned(),
            policy_id: POLICY.to_owned(),
            policy_digest: DIGEST,
            loss_profile_id: LOSS.to_owned(),
            domain_id: DOMAIN.to_owned(),
            domain_status: DomainStatusV1::InDomain,
            harmful,
        }
    }

    #[test]
    fn no_events_is_insufficient_not_certified() {
        let monitor = AnytimeHarmMonitorV1::new(config(&["g1"])).unwrap();
        assert_eq!(monitor.status(), MonitorStatusV1::InsufficientEvidence);
        assert_eq!(
            monitor.group_snapshots()[0].status,
            GroupMonitorStatusV1::InsufficientEvidence
        );
    }

    #[test]
    fn clean_outcomes_cross_the_anytime_threshold() {
        let mut monitor = AnytimeHarmMonitorV1::new(config(&["g1"])).unwrap();
        for index in 0..150 {
            monitor.observe(event(index, "g1", false)).unwrap();
        }
        assert_eq!(monitor.status(), MonitorStatusV1::Certified);
        assert_eq!(
            monitor.group_snapshots()[0].status,
            GroupMonitorStatusV1::Certified
        );
        assert!(monitor.group_snapshots()[0].log_e_value >= monitor.log_certification_threshold());
    }

    #[test]
    fn all_harmful_outcomes_do_not_certify_low_harm_risk() {
        let mut monitor = AnytimeHarmMonitorV1::new(config(&["g1"])).unwrap();
        for index in 0..10 {
            monitor.observe(event(index, "g1", true)).unwrap();
        }
        assert_eq!(monitor.status(), MonitorStatusV1::Monitoring);
        assert!(monitor.group_snapshots()[0].log_e_value < monitor.log_certification_threshold());
    }

    #[test]
    fn new_harm_can_lower_current_certificate_back_to_monitoring() {
        let mut monitor = AnytimeHarmMonitorV1::new(config(&["g1"])).unwrap();
        for index in 0..150 {
            monitor.observe(event(index, "g1", false)).unwrap();
        }
        assert_eq!(monitor.status(), MonitorStatusV1::Certified);
        for index in 150..175 {
            monitor.observe(event(index, "g1", true)).unwrap();
        }
        assert_eq!(monitor.status(), MonitorStatusV1::Monitoring);
        assert_eq!(
            monitor.group_snapshots()[0].status,
            GroupMonitorStatusV1::Monitoring
        );
    }

    #[test]
    fn one_certified_group_does_not_certify_an_unseen_group() {
        let mut monitor = AnytimeHarmMonitorV1::new(config(&["common", "rare"])).unwrap();
        for index in 0..150 {
            monitor.observe(event(index, "common", false)).unwrap();
        }
        assert_eq!(
            monitor.group_snapshots()[0].status,
            GroupMonitorStatusV1::Certified
        );
        assert_eq!(
            monitor.group_snapshots()[1].status,
            GroupMonitorStatusV1::InsufficientEvidence
        );
        assert_eq!(monitor.status(), MonitorStatusV1::InsufficientEvidence);
    }

    #[test]
    fn global_status_waits_for_every_predeclared_group_to_cross() {
        let mut monitor = AnytimeHarmMonitorV1::new(config(&["g1", "g2"])).unwrap();
        for index in 0..150 {
            monitor.observe(event(index, "g1", false)).unwrap();
        }
        for index in 150..160 {
            monitor.observe(event(index, "g2", false)).unwrap();
        }
        assert_eq!(monitor.status(), MonitorStatusV1::Monitoring);
        for index in 160..299 {
            monitor.observe(event(index, "g2", false)).unwrap();
        }
        monitor.observe(event(299, "g2", false)).unwrap();
        assert_eq!(monitor.status(), MonitorStatusV1::Certified);
    }

    #[test]
    fn group_e_value_depends_on_counts_not_event_order() {
        let mut forward = AnytimeHarmMonitorV1::new(config(&["g1"])).unwrap();
        let mut reverse = AnytimeHarmMonitorV1::new(config(&["g1"])).unwrap();
        for index in 0..100 {
            forward
                .observe(event(index, "g1", index % 13 == 0))
                .unwrap();
        }
        for index in (0..100).rev() {
            reverse
                .observe(event(index, "g1", index % 13 == 0))
                .unwrap();
        }
        assert_eq!(forward.group_snapshots(), reverse.group_snapshots());
        assert_eq!(forward.to_tsv(), reverse.to_tsv());
    }

    #[test]
    fn shift_revocation_is_absorbing() {
        let mut monitor = AnytimeHarmMonitorV1::new(config(&["g1"])).unwrap();
        for index in 0..150 {
            monitor.observe(event(index, "g1", false)).unwrap();
        }
        let mut shifted = event(151, "g1", false);
        shifted.domain_status = DomainStatusV1::OodSuspected;
        assert_eq!(
            monitor.observe(shifted).unwrap(),
            MonitorStatusV1::RevokedShift
        );
        assert_eq!(monitor.status(), MonitorStatusV1::RevokedShift);
        assert!(matches!(
            monitor.observe(event(152, "g1", false)),
            Err(AnytimeMonitorErrorV1::MonitorAlreadyRevoked)
        ));
    }

    #[test]
    fn changed_policy_binding_revokes_monitor() {
        let mut monitor = AnytimeHarmMonitorV1::new(config(&["g1"])).unwrap();
        let mut stale = event(1, "g1", false);
        stale.policy_digest = [0xFE; 32];
        assert_eq!(
            monitor.observe(stale),
            Err(AnytimeMonitorErrorV1::PolicyBindingMismatch)
        );
        assert_eq!(monitor.status(), MonitorStatusV1::RevokedShift);
    }

    #[test]
    fn duplicate_trace_and_undeclared_group_fail_closed() {
        let mut duplicate_monitor = AnytimeHarmMonitorV1::new(config(&["g1"])).unwrap();
        duplicate_monitor.observe(event(1, "g1", false)).unwrap();
        assert!(matches!(
            duplicate_monitor.observe(event(1, "g1", false)),
            Err(AnytimeMonitorErrorV1::DuplicateTraceId(_))
        ));

        let mut undeclared_monitor = AnytimeHarmMonitorV1::new(config(&["g1"])).unwrap();
        assert!(matches!(
            undeclared_monitor.observe(event(1, "g2", false)),
            Err(AnytimeMonitorErrorV1::UndeclaredGroup(_))
        ));
        assert_eq!(undeclared_monitor.status(), MonitorStatusV1::RevokedShift);
    }

    #[test]
    fn policy_family_size_raises_certification_threshold() {
        let mut narrow = AnytimeHarmMonitorV1::new(config(&["g1"])).unwrap();
        let mut broad_config = config(&["g1"]);
        broad_config.policy_family_size = 10;
        let mut broad = AnytimeHarmMonitorV1::new(broad_config).unwrap();
        for index in 0..100 {
            narrow.observe(event(index, "g1", false)).unwrap();
            broad.observe(event(index, "g1", false)).unwrap();
        }
        assert_eq!(narrow.status(), MonitorStatusV1::Certified);
        assert_eq!(broad.status(), MonitorStatusV1::Monitoring);
        assert!(broad.log_certification_threshold() > narrow.log_certification_threshold());
    }

    #[test]
    fn config_rejects_duplicate_groups_and_bad_risk_values() {
        assert!(matches!(
            AnytimeHarmMonitorV1::new(config(&["g1", "g1"])),
            Err(AnytimeMonitorErrorV1::InvalidConfig)
        ));
        let mut bad = config(&["g1"]);
        bad.maximum_harm_rate = 1.0;
        assert!(matches!(
            AnytimeHarmMonitorV1::new(bad),
            Err(AnytimeMonitorErrorV1::InvalidConfig)
        ));
    }
}
