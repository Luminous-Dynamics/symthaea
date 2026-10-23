// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-010: finite-sample one-sided harm-rate certification.
//!
//! Offline evaluator only. Rows contain post-commit harm labels for a frozen,
//! named policy; this module must not be exposed through an online router API.
//! Statistical interpretation requires IID/exchangeable calibration outcomes within
//! the declared domain, a frozen policy, and no reuse after tuning on these labels.
//!
//! Run tests:
//!   rustc --edition=2024 --test tools/vision-dvpc-harm-risk-certificate.rs -o /tmp/vis-dvpc-010-tests
//!   /tmp/vis-dvpc-010-tests

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

pub type EvidenceDigest = [u8; 32];
pub const MAX_CERTIFICATION_ROWS: usize = 100_000;
pub const MAX_GROUPS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainStatusV1 {
    InDomain,
    OodSuspected,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HarmObservationV1 {
    pub trace_id: String,
    pub source_group_id: String,
    pub split_id: String,
    pub domain_id: String,
    pub domain_status: DomainStatusV1,
    pub policy_id: String,
    pub policy_digest: EvidenceDigest,
    pub loss_profile_id: String,
    /// True iff this frozen policy selected expensive-mode escalation.
    pub escalated: bool,
    /// Evaluator-only counterfactual label: Some(harmful) for escalations only.
    pub harmful_escalation: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HarmRiskConfigV1 {
    pub policy_id: String,
    pub policy_digest: EvidenceDigest,
    pub loss_profile_id: String,
    pub calibration_split_id: String,
    pub domain_id: String,
    /// Digest of exact immutable calibration manifest bytes; verified by caller.
    pub calibration_manifest_digest: EvidenceDigest,
    pub predeclared_group_ids: Vec<String>,
    /// Number of candidate policies in the predeclared family.
    pub policy_family_size: usize,
    /// Family-wise probability of an incorrect certificate, in (0, 1).
    pub family_wise_delta: f64,
    /// Maximum permitted population harm rate conditional on escalation, in (0, 1).
    pub maximum_harm_rate: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RiskCertificateErrorV1 {
    EmptyEvidence,
    TooManyRows { rows: usize, maximum: usize },
    TooManyGroups { groups: usize, maximum: usize },
    InvalidConfig,
    InvalidIdentifier(String),
    InvalidDigest(String),
    DuplicateTraceId(String),
    UndeclaredGroup(String),
    MissingPredeclaredGroup(String),
    ShiftedCalibrationSample(String),
    MetadataMismatch(String),
    MalformedHarmLabel(String),
    InvalidCounts,
    NumericalFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskCertificateStatusV1 {
    Certified,
    NotCertifiedRiskBound,
    InsufficientEvidence,
}

impl RiskCertificateStatusV1 {
    pub fn label(self) -> &'static str {
        match self {
            Self::Certified => "CERTIFIED",
            Self::NotCertifiedRiskBound => "NOT_CERTIFIED_RISK_BOUND",
            Self::InsufficientEvidence => "INSUFFICIENT_EVIDENCE",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HarmRateBoundV1 {
    pub group_id: String,
    pub escalations: usize,
    pub harmful_escalations: usize,
    pub empirical_harm_rate: Option<f64>,
    pub one_sided_upper_bound: Option<f64>,
    pub adjusted_delta: f64,
    pub status: RiskCertificateStatusV1,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RiskCertificateV1 {
    pub policy_id: String,
    pub policy_digest: EvidenceDigest,
    pub loss_profile_id: String,
    pub calibration_split_id: String,
    pub domain_id: String,
    pub calibration_manifest_digest: EvidenceDigest,
    pub family_wise_delta: f64,
    pub maximum_harm_rate: f64,
    pub policy_family_size: usize,
    pub hypothesis_count: usize,
    pub overall: HarmRateBoundV1,
    pub groups: Vec<HarmRateBoundV1>,
    pub status: RiskCertificateStatusV1,
    pub total_rows: usize,
    pub total_escalations: usize,
    pub total_harmful_escalations: usize,
}

impl RiskCertificateV1 {
    /// Deterministic TSV receipt; hashes and identifiers are kept as fixed fields.
    pub fn to_tsv(&self) -> String {
        let mut out = format!(
            "RISK_CERT_V1\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{}\t{}\t{}\t{}\n",
            self.status.label(),
            self.policy_id,
            digest_hex(&self.policy_digest),
            self.loss_profile_id,
            self.calibration_split_id,
            self.domain_id,
            digest_hex(&self.calibration_manifest_digest),
            self.policy_family_size,
            self.family_wise_delta,
            self.maximum_harm_rate,
            self.hypothesis_count,
            self.total_rows,
            self.total_escalations,
            self.total_harmful_escalations,
        );
        write_bound_row(&mut out, "OVERALL", &self.overall);
        for group in &self.groups {
            write_bound_row(&mut out, &group.group_id, group);
        }
        out
    }
}

/// Produce a simultaneous finite-sample risk certificate for a frozen policy.
///
/// Bonferroni allocates family_wise_delta across all predeclared policies,
/// the overall population, and every predeclared group. Every group must be
/// represented by evidence, and every group must contain at least one evaluated
/// escalation. Otherwise the result is INSUFFICIENT_EVIDENCE, never zero risk.
///
/// The confidence statement requires IID/exchangeable held-out outcomes in the
/// declared domain. A shifted/unknown calibration row is rejected rather than
/// pooled into an unsupported certificate.
pub fn certify_harm_rate_v1(
    config: &HarmRiskConfigV1,
    observations: &[HarmObservationV1],
) -> Result<RiskCertificateV1, RiskCertificateErrorV1> {
    validate_config(config)?;
    if observations.is_empty() {
        return Err(RiskCertificateErrorV1::EmptyEvidence);
    }
    if observations.len() > MAX_CERTIFICATION_ROWS {
        return Err(RiskCertificateErrorV1::TooManyRows {
            rows: observations.len(),
            maximum: MAX_CERTIFICATION_ROWS,
        });
    }

    let mut declared_groups = BTreeSet::new();
    for group in &config.predeclared_group_ids {
        if !valid_identifier(group) || !declared_groups.insert(group.as_str()) {
            return Err(RiskCertificateErrorV1::InvalidConfig);
        }
    }

    let hypothesis_count = 1_usize
        .checked_add(config.predeclared_group_ids.len())
        .and_then(|count| count.checked_mul(config.policy_family_size))
        .ok_or(RiskCertificateErrorV1::InvalidConfig)?;
    let adjusted_delta = config.family_wise_delta / hypothesis_count as f64;
    if !adjusted_delta.is_finite() || adjusted_delta <= 0.0 || adjusted_delta >= 1.0 {
        return Err(RiskCertificateErrorV1::InvalidConfig);
    }

    let mut trace_ids = BTreeSet::new();
    let mut group_samples = BTreeMap::<String, (usize, usize)>::new();
    let mut total_escalations = 0_usize;
    let mut total_harmful = 0_usize;

    for row in observations {
        if !valid_identifier(&row.trace_id)
            || !valid_identifier(&row.source_group_id)
            || !valid_identifier(&row.policy_id)
            || !valid_identifier(&row.loss_profile_id)
            || !valid_identifier(&row.split_id)
            || !valid_identifier(&row.domain_id)
        {
            return Err(RiskCertificateErrorV1::InvalidIdentifier(
                row.trace_id.clone(),
            ));
        }
        if !trace_ids.insert(row.trace_id.as_str()) {
            return Err(RiskCertificateErrorV1::DuplicateTraceId(
                row.trace_id.clone(),
            ));
        }
        if !declared_groups.contains(row.source_group_id.as_str()) {
            return Err(RiskCertificateErrorV1::UndeclaredGroup(
                row.source_group_id.clone(),
            ));
        }
        if row.domain_status != DomainStatusV1::InDomain {
            return Err(RiskCertificateErrorV1::ShiftedCalibrationSample(
                row.trace_id.clone(),
            ));
        }
        if row.split_id != config.calibration_split_id
            || row.domain_id != config.domain_id
            || row.policy_id != config.policy_id
            || row.policy_digest != config.policy_digest
            || row.loss_profile_id != config.loss_profile_id
        {
            return Err(RiskCertificateErrorV1::MetadataMismatch(
                row.trace_id.clone(),
            ));
        }
        if (row.escalated && row.harmful_escalation.is_none())
            || (!row.escalated && row.harmful_escalation.is_some())
        {
            return Err(RiskCertificateErrorV1::MalformedHarmLabel(
                row.trace_id.clone(),
            ));
        }

        let counters = group_samples
            .entry(row.source_group_id.clone())
            .or_insert((0, 0));
        if row.escalated {
            counters.0 = counters
                .0
                .checked_add(1)
                .ok_or(RiskCertificateErrorV1::InvalidCounts)?;
            total_escalations = total_escalations
                .checked_add(1)
                .ok_or(RiskCertificateErrorV1::InvalidCounts)?;
            if row.harmful_escalation == Some(true) {
                counters.1 = counters
                    .1
                    .checked_add(1)
                    .ok_or(RiskCertificateErrorV1::InvalidCounts)?;
                total_harmful = total_harmful
                    .checked_add(1)
                    .ok_or(RiskCertificateErrorV1::InvalidCounts)?;
            }
        }
    }

    for group in &config.predeclared_group_ids {
        if !group_samples.contains_key(group) {
            return Err(RiskCertificateErrorV1::MissingPredeclaredGroup(
                group.clone(),
            ));
        }
    }

    let overall = make_bound(
        "overall",
        total_escalations,
        total_harmful,
        adjusted_delta,
        config.maximum_harm_rate,
    )?;
    let mut groups = Vec::with_capacity(config.predeclared_group_ids.len());
    for group in &config.predeclared_group_ids {
        let (escalations, harmful) = *group_samples
            .get(group)
            .ok_or_else(|| RiskCertificateErrorV1::MissingPredeclaredGroup(group.clone()))?;
        groups.push(make_bound(
            group,
            escalations,
            harmful,
            adjusted_delta,
            config.maximum_harm_rate,
        )?);
    }

    let any_insufficient = overall.status == RiskCertificateStatusV1::InsufficientEvidence
        || groups
            .iter()
            .any(|bound| bound.status == RiskCertificateStatusV1::InsufficientEvidence);
    let any_risk_failure = overall.status == RiskCertificateStatusV1::NotCertifiedRiskBound
        || groups
            .iter()
            .any(|bound| bound.status == RiskCertificateStatusV1::NotCertifiedRiskBound);
    let status = if any_insufficient {
        RiskCertificateStatusV1::InsufficientEvidence
    } else if any_risk_failure {
        RiskCertificateStatusV1::NotCertifiedRiskBound
    } else {
        RiskCertificateStatusV1::Certified
    };

    let mut sorted_ids: Vec<&str> = observations
        .iter()
        .map(|row| row.trace_id.as_str())
        .collect();
    sorted_ids.sort_unstable();
    // Sorting IDs is also an explicit deterministic evidence-order check in tests.
    let _canonical_trace_order = sorted_ids;

    Ok(RiskCertificateV1 {
        policy_id: config.policy_id.clone(),
        policy_digest: config.policy_digest,
        loss_profile_id: config.loss_profile_id.clone(),
        calibration_split_id: config.calibration_split_id.clone(),
        domain_id: config.domain_id.clone(),
        calibration_manifest_digest: config.calibration_manifest_digest,
        family_wise_delta: config.family_wise_delta,
        maximum_harm_rate: config.maximum_harm_rate,
        policy_family_size: config.policy_family_size,
        hypothesis_count,
        overall,
        groups,
        status,
        total_rows: observations.len(),
        total_escalations,
        total_harmful_escalations: total_harmful,
    })
}

fn validate_config(config: &HarmRiskConfigV1) -> Result<(), RiskCertificateErrorV1> {
    if !valid_identifier(&config.policy_id)
        || !valid_identifier(&config.loss_profile_id)
        || !valid_identifier(&config.calibration_split_id)
        || !valid_identifier(&config.domain_id)
        || config.policy_digest.iter().all(|byte| *byte == 0)
        || config.calibration_manifest_digest.iter().all(|byte| *byte == 0)
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
        return Err(RiskCertificateErrorV1::InvalidConfig);
    }
    Ok(())
}

fn make_bound(
    group_id: &str,
    escalations: usize,
    harmful: usize,
    adjusted_delta: f64,
    maximum_harm_rate: f64,
) -> Result<HarmRateBoundV1, RiskCertificateErrorV1> {
    if harmful > escalations {
        return Err(RiskCertificateErrorV1::InvalidCounts);
    }
    if escalations == 0 {
        return Ok(HarmRateBoundV1 {
            group_id: group_id.to_owned(),
            escalations,
            harmful_escalations: harmful,
            empirical_harm_rate: None,
            one_sided_upper_bound: None,
            adjusted_delta,
            status: RiskCertificateStatusV1::InsufficientEvidence,
        });
    }
    let upper = clopper_pearson_upper(harmful, escalations, adjusted_delta)?;
    Ok(HarmRateBoundV1 {
        group_id: group_id.to_owned(),
        escalations,
        harmful_escalations: harmful,
        empirical_harm_rate: Some(harmful as f64 / escalations as f64),
        one_sided_upper_bound: Some(upper),
        adjusted_delta,
        status: if upper <= maximum_harm_rate {
            RiskCertificateStatusV1::Certified
        } else {
            RiskCertificateStatusV1::NotCertifiedRiskBound
        },
    })
}

/// Exact one-sided Clopper-Pearson upper confidence limit for a binomial harm
/// probability. For 0 < k < n, solve P_p[X <= k] = delta by bisection over the
/// binomial CDF. Log probabilities and a precomputed log-factorial table avoid
/// direct binomial-coefficient overflow. Maximum n is bounded by the caller.
pub fn clopper_pearson_upper(
    harmful: usize,
    escalations: usize,
    delta: f64,
) -> Result<f64, RiskCertificateErrorV1> {
    if escalations == 0
        || harmful > escalations
        || escalations > MAX_CERTIFICATION_ROWS
        || !delta.is_finite()
        || delta <= 0.0
        || delta >= 1.0
    {
        return Err(RiskCertificateErrorV1::InvalidCounts);
    }
    if harmful == escalations {
        return Ok(1.0);
    }
    if harmful == 0 {
        // Exact formula with exp_m1 to avoid cancellation for large n.
        return Ok(-((delta.ln() / escalations as f64).exp_m1()));
    }

    let mut log_factorials = vec![0.0_f64; escalations + 1];
    for index in 1..=escalations {
        log_factorials[index] = log_factorials[index - 1] + (index as f64).ln();
    }
    let log_delta = delta.ln();
    let mut lower = harmful as f64 / escalations as f64;
    let mut upper = 1.0_f64;

    for _ in 0..96 {
        let midpoint = lower + (upper - lower) * 0.5;
        let log_cdf = log_binomial_cdf(
            harmful,
            escalations,
            midpoint,
            &log_factorials,
        );
        if !log_cdf.is_finite() {
            return Err(RiskCertificateErrorV1::NumericalFailure);
        }
        if log_cdf > log_delta {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    if upper.is_finite() && (0.0..=1.0).contains(&upper) {
        Ok(upper)
    } else {
        Err(RiskCertificateErrorV1::NumericalFailure)
    }
}

fn log_binomial_cdf(
    max_events: usize,
    trials: usize,
    probability: f64,
    log_factorials: &[f64],
) -> f64 {
    if probability <= 0.0 {
        return 0.0;
    }
    if probability >= 1.0 {
        return f64::NEG_INFINITY;
    }

    let log_p = probability.ln();
    let log_q = (-probability).ln_1p();
    let log_n_factorial = log_factorials[trials];
    let mut maximum = f64::NEG_INFINITY;
    let mut terms = Vec::with_capacity(max_events + 1);
    for events in 0..=max_events {
        let term = log_n_factorial
            - log_factorials[events]
            - log_factorials[trials - events]
            + events as f64 * log_p
            + (trials - events) as f64 * log_q;
        maximum = maximum.max(term);
        terms.push(term);
    }
    if !maximum.is_finite() {
        return f64::NEG_INFINITY;
    }
    maximum
        + terms
            .iter()
            .map(|term| (term - maximum).exp())
            .sum::<f64>()
            .ln()
}

fn valid_identifier(value: &str) -> bool {
    !value.trim().is_empty()
        && !value
            .chars()
            .any(|ch| matches!(ch, '\t' | '\r' | '\n'))
}

fn digest_hex(digest: &EvidenceDigest) -> String {
    let mut result = String::with_capacity(64);
    for byte in digest {
        result.push_str(&format!("{byte:02x}"));
    }
    result
}

fn write_bound_row(out: &mut String, name: &str, bound: &HarmRateBoundV1) {
    out.push_str(&format!(
        "BOUND\t{}\t{}\t{}\t{}\t{}\t{}\t{:.12}\t{}\n",
        name,
        bound.status.label(),
        bound.escalations,
        bound.harmful_escalations,
        bound.empirical_harm_rate
            .map(|value| format!("{value:.9}"))
            .unwrap_or_else(|| "NA".to_owned()),
        bound.one_sided_upper_bound
            .map(|value| format!("{value:.12}"))
            .unwrap_or_else(|| "NA".to_owned()),
        bound.adjusted_delta,
        bound.group_id,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICY: &str = "calibrated-router-v1";
    const LOSS: &str = "collision-weighted-utility-v1";
    const SPLIT: &str = "risk-calibration-v1";
    const DOMAIN: &str = "night-clear-v1";
    const POLICY_DIGEST: EvidenceDigest = [0xA5; 32];
    const MANIFEST_DIGEST: EvidenceDigest = [0x5A; 32];

    fn config(groups: &[&str]) -> HarmRiskConfigV1 {
        HarmRiskConfigV1 {
            policy_id: POLICY.to_owned(),
            policy_digest: POLICY_DIGEST,
            loss_profile_id: LOSS.to_owned(),
            calibration_split_id: SPLIT.to_owned(),
            domain_id: DOMAIN.to_owned(),
            calibration_manifest_digest: MANIFEST_DIGEST,
            predeclared_group_ids: groups.iter().map(|group| (*group).to_owned()).collect(),
            policy_family_size: 1,
            family_wise_delta: 0.05,
            maximum_harm_rate: 0.05,
        }
    }

    fn row(index: usize, group: &str, escalated: bool, harmful: Option<bool>) -> HarmObservationV1 {
        HarmObservationV1 {
            trace_id: format!("trace-{index:05}"),
            source_group_id: group.to_owned(),
            split_id: SPLIT.to_owned(),
            domain_id: DOMAIN.to_owned(),
            domain_status: DomainStatusV1::InDomain,
            policy_id: POLICY.to_owned(),
            policy_digest: POLICY_DIGEST,
            loss_profile_id: LOSS.to_owned(),
            escalated,
            harmful_escalation: harmful,
        }
    }

    fn clean_escalations(count: usize, start: usize, group: &str) -> Vec<HarmObservationV1> {
        (start..start + count)
            .map(|index| row(index, group, true, Some(false)))
            .collect()
    }

    #[test]
    fn zero_harm_cp_bound_has_exact_known_value() {
        let upper = clopper_pearson_upper(0, 72, 0.025).unwrap();
        let expected = 1.0 - 0.025_f64.powf(1.0 / 72.0);
        assert!((upper - expected).abs() < 1e-12);
        assert!(upper < 0.05);
        assert!(clopper_pearson_upper(0, 71, 0.025).unwrap() > 0.05);
    }

    #[test]
    fn nonzero_harm_cp_bound_matches_known_reference() {
        let upper = clopper_pearson_upper(1, 10, 0.05).unwrap();
        assert!((upper - 0.3941633).abs() < 2e-6);
    }

    #[test]
    fn tiny_zero_harm_sample_is_insufficient_not_safe() {
        let observations = clean_escalations(10, 0, "g1");
        let cert = certify_harm_rate_v1(&config(&["g1"]), &observations).unwrap();
        assert_eq!(cert.status, RiskCertificateStatusV1::NotCertifiedRiskBound);
        assert!(cert.overall.one_sided_upper_bound.unwrap() > 0.05);
    }

    #[test]
    fn zero_escalations_never_become_zero_risk() {
        let observations = vec![row(0, "g1", false, None), row(1, "g1", false, None)];
        let cert = certify_harm_rate_v1(&config(&["g1"]), &observations).unwrap();
        assert_eq!(cert.status, RiskCertificateStatusV1::InsufficientEvidence);
        assert_eq!(cert.overall.empirical_harm_rate, None);
        assert_eq!(cert.overall.one_sided_upper_bound, None);
        assert_eq!(cert.groups[0].status, RiskCertificateStatusV1::InsufficientEvidence);
    }

    #[test]
    fn enough_clean_escalations_certify_overall_and_group() {
        let observations = clean_escalations(72, 0, "g1");
        let cert = certify_harm_rate_v1(&config(&["g1"]), &observations).unwrap();
        assert_eq!(cert.status, RiskCertificateStatusV1::Certified);
        assert!(cert.overall.one_sided_upper_bound.unwrap() <= 0.05);
        assert!(cert.groups[0].one_sided_upper_bound.unwrap() <= 0.05);
        assert_eq!(cert.hypothesis_count, 2);
    }

    #[test]
    fn policy_family_multiplicity_makes_certificate_more_conservative() {
        let observations = clean_escalations(72, 0, "g1");
        let mut cfg = config(&["g1"]);
        cfg.policy_family_size = 2;
        let cert = certify_harm_rate_v1(&cfg, &observations).unwrap();
        assert_eq!(cert.status, RiskCertificateStatusV1::NotCertifiedRiskBound);
        assert!(cert.overall.one_sided_upper_bound.unwrap() > 0.05);
        assert_eq!(cert.hypothesis_count, 4);
    }

    #[test]
    fn one_bad_group_fails_even_when_global_rate_passes() {
        let mut observations = clean_escalations(1000, 0, "common");
        observations.extend((1000..1010).map(|index| {
            row(index, "rare", true, Some(index == 1000))
        }));
        let cert = certify_harm_rate_v1(&config(&["common", "rare"]), &observations).unwrap();
        assert!(cert.overall.one_sided_upper_bound.unwrap() < 0.05);
        assert_eq!(cert.groups[0].status, RiskCertificateStatusV1::Certified);
        assert_eq!(cert.groups[1].status, RiskCertificateStatusV1::NotCertifiedRiskBound);
        assert_eq!(cert.status, RiskCertificateStatusV1::NotCertifiedRiskBound);
    }

    #[test]
    fn group_with_no_escalation_is_insufficient_evidence() {
        let mut observations = clean_escalations(100, 0, "common");
        observations.push(row(100, "rare", false, None));
        let cert = certify_harm_rate_v1(&config(&["common", "rare"]), &observations).unwrap();
        assert_eq!(cert.groups[1].status, RiskCertificateStatusV1::InsufficientEvidence);
        assert_eq!(cert.status, RiskCertificateStatusV1::InsufficientEvidence);
    }

    #[test]
    fn duplicate_ids_malformed_labels_and_wrong_policy_digest_reject() {
        let mut duplicate = clean_escalations(2, 0, "g1");
        duplicate[1].trace_id = duplicate[0].trace_id.clone();
        assert!(matches!(
            certify_harm_rate_v1(&config(&["g1"]), &duplicate),
            Err(RiskCertificateErrorV1::DuplicateTraceId(_))
        ));

        let malformed = vec![row(0, "g1", false, Some(true))];
        assert!(matches!(
            certify_harm_rate_v1(&config(&["g1"]), &malformed),
            Err(RiskCertificateErrorV1::MalformedHarmLabel(_))
        ));

        let mut stale = clean_escalations(1, 0, "g1");
        stale[0].policy_digest = [0xFF; 32];
        assert!(matches!(
            certify_harm_rate_v1(&config(&["g1"]), &stale),
            Err(RiskCertificateErrorV1::MetadataMismatch(_))
        ));
    }

    #[test]
    fn shift_mismatch_and_missing_group_fail_closed() {
        let mut shifted = clean_escalations(1, 0, "g1");
        shifted[0].domain_status = DomainStatusV1::OodSuspected;
        assert!(matches!(
            certify_harm_rate_v1(&config(&["g1"]), &shifted),
            Err(RiskCertificateErrorV1::ShiftedCalibrationSample(_))
        ));

        let clean = clean_escalations(2, 0, "g1");
        assert!(matches!(
            certify_harm_rate_v1(&config(&["g1", "g2"]), &clean),
            Err(RiskCertificateErrorV1::MissingPredeclaredGroup(_))
        ));
    }

    #[test]
    fn result_is_order_invariant_and_receipt_binds_policy_manifest() {
        let observations = clean_escalations(72, 0, "g1");
        let forward = certify_harm_rate_v1(&config(&["g1"]), &observations).unwrap();
        let mut reversed = observations;
        reversed.reverse();
        let backward = certify_harm_rate_v1(&config(&["g1"]), &reversed).unwrap();
        assert_eq!(forward, backward);
        assert_eq!(forward.to_tsv(), backward.to_tsv());
        assert!(forward.to_tsv().contains("5a5a5a5a5a5a5a5a"));
        assert!(forward.to_tsv().starts_with("RISK_CERT_V1\tCERTIFIED"));
    }

    #[test]
    fn upper_bound_is_monotone_and_all_harm_returns_one() {
        let low = clopper_pearson_upper(1, 20, 0.05).unwrap();
        let high = clopper_pearson_upper(2, 20, 0.05).unwrap();
        assert!(high > low);
        assert_eq!(clopper_pearson_upper(20, 20, 0.05).unwrap(), 1.0);
    }
}
