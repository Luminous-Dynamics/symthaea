//! VIS-DVPC-043: support/identification gate for off-policy evaluation.
//!
//! The gate is estimator-agnostic. It checks the action-support contract before
//! any downstream value estimator is allowed to publish an interval or authorize
//! a decision. No external crates are required.
#![deny(unsafe_code)]

use std::env;
use std::fmt::Write as _;

const PROB_SUM_TOL: f64 = 1e-9;
const SUPPORT_MASS_TOL: f64 = 1e-12;
const NUMERIC_TOL: f64 = 1e-9;
const WEAK_MAX_RATIO: f64 = 20.0;
const WEAK_MIN_POPULATION_ESS_FRACTION: f64 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SupportStatus {
    Identified,
    WeakSupport,
    NotIdentified,
    InvalidInput,
}
impl SupportStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Identified => "IDENTIFIED",
            Self::WeakSupport => "WEAK_SUPPORT",
            Self::NotIdentified => "NOT_IDENTIFIED",
            Self::InvalidInput => "INVALID_INPUT",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecisionStatus {
    EligibleForEvidenceReview,
    InsufficientEvidence,
    SuppressValueAndInterval,
    InvalidInputNoDecision,
}
impl DecisionStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::EligibleForEvidenceReview => "ELIGIBLE_FOR_EVIDENCE_REVIEW",
            Self::InsufficientEvidence => "INSUFFICIENT_EVIDENCE",
            Self::SuppressValueAndInterval => "SUPPRESS_VALUE_AND_INTERVAL",
            Self::InvalidInputNoDecision => "INVALID_INPUT_NO_DECISION",
        }
    }
}

#[derive(Clone, Debug)]
struct SupportAudit {
    status: SupportStatus,
    decision: DecisionStatus,
    unsupported_target_mass: Option<f64>,
    max_importance_ratio: Option<f64>,
    importance_weight_second_moment: Option<f64>,
    population_ess_fraction: Option<f64>,
    numeric_value_disclosure: &'static str,
    reason: &'static str,
    action_count: usize,
    behavior_probabilities: Vec<f64>,
    target_probabilities: Vec<f64>,
}

fn sum_to_one(values: &[f64]) -> bool {
    let sum: f64 = values.iter().sum();
    sum.is_finite() && (sum - 1.0).abs() <= PROB_SUM_TOL
}
fn valid_probability_vector(values: &[f64]) -> bool {
    !values.is_empty()
        && values.iter().all(|x| x.is_finite() && *x >= 0.0 && *x <= 1.0)
        && sum_to_one(values)
}

impl SupportAudit {
    fn invalid(reason: &'static str, action_count: usize) -> Self {
        Self {
            status: SupportStatus::InvalidInput,
            decision: DecisionStatus::InvalidInputNoDecision,
            unsupported_target_mass: None,
            max_importance_ratio: None,
            importance_weight_second_moment: None,
            population_ess_fraction: None,
            numeric_value_disclosure: "SUPPRESSED_INVALID_INPUT",
            reason,
            action_count,
            behavior_probabilities: Vec::new(),
            target_probabilities: Vec::new(),
        }
    }

    fn new(
        behavior: &[f64],
        target: &[f64],
        weak_max_ratio: f64,
        weak_min_ess_fraction: f64,
    ) -> Self {
        if behavior.len() != target.len() || behavior.is_empty() {
            return Self::invalid("probability_vector_length_mismatch_or_empty", behavior.len());
        }
        if !valid_probability_vector(behavior) || !valid_probability_vector(target) {
            return Self::invalid("probabilities_must_be_finite_in_[0,1]_and_sum_to_one", behavior.len());
        }
        if !weak_max_ratio.is_finite()
            || weak_max_ratio <= 0.0
            || !weak_min_ess_fraction.is_finite()
            || !(0.0..=1.0).contains(&weak_min_ess_fraction)
            || weak_min_ess_fraction == 0.0
        {
            return Self::invalid("invalid_weak_support_thresholds", behavior.len());
        }

        let unsupported_target_mass: f64 = behavior.iter().zip(target.iter())
            .filter_map(|(&h, &pi)| if h == 0.0 { Some(pi) } else { None })
            .sum();

        if unsupported_target_mass > SUPPORT_MASS_TOL {
            return Self {
                status: SupportStatus::NotIdentified,
                decision: DecisionStatus::SuppressValueAndInterval,
                unsupported_target_mass: Some(unsupported_target_mass),
                max_importance_ratio: None,
                importance_weight_second_moment: None,
                population_ess_fraction: None,
                numeric_value_disclosure: "SUPPRESSED_UNSUPPORTED_TARGET_MASS",
                reason: "target_policy_assigns_positive_mass_to_unlogged_action",
                action_count: behavior.len(),
                behavior_probabilities: behavior.to_vec(),
                target_probabilities: target.to_vec(),
            };
        }

        let mut max_ratio: f64 = 0.0;
        let mut second_moment: f64 = 0.0;
        for (&h, &pi) in behavior.iter().zip(target.iter()) {
            if h == 0.0 {
                // This can only be a target-zero action after the support test.
                continue;
            }
            let ratio = pi / h;
            if !ratio.is_finite() {
                return Self::invalid("non_finite_importance_ratio", behavior.len());
            }
            max_ratio = max_ratio.max(ratio);
            second_moment += pi * pi / h;
        }
        if !second_moment.is_finite() || second_moment < 1.0 - 1e-8 {
            return Self::invalid("invalid_importance_weight_second_moment", behavior.len());
        }
        let ess_fraction = (1.0 / second_moment).clamp(0.0, 1.0);
        let weak = max_ratio > weak_max_ratio || ess_fraction < weak_min_ess_fraction;
        Self {
            status: if weak { SupportStatus::WeakSupport } else { SupportStatus::Identified },
            decision: if weak {
                DecisionStatus::InsufficientEvidence
            } else {
                DecisionStatus::EligibleForEvidenceReview
            },
            unsupported_target_mass: Some(0.0),
            max_importance_ratio: Some(max_ratio),
            importance_weight_second_moment: Some(second_moment),
            population_ess_fraction: Some(ess_fraction),
            numeric_value_disclosure: if weak {
                "ESTIMATOR_OUTPUT_MAY_BE_COMPUTED_BUT_NOT_DECISION_ELIGIBLE"
            } else {
                "ELIGIBLE_FOR_SEPARATE_ESTIMATOR"
            },
            reason: if weak {
                "importance_weight_concentration_or_population_ess_below_gate"
            } else {
                "target_support_is_covered_and_registered_overlap_gate_passes"
            },
            action_count: behavior.len(),
            behavior_probabilities: behavior.to_vec(),
            target_probabilities: target.to_vec(),
        }
    }

    fn expected_weight(&self, action: usize) -> Option<f64> {
        // Use the exact immutable distributions that were audited; never accept
        // caller-supplied replacement propensities after the gate has run.
        if matches!(self.status, SupportStatus::InvalidInput | SupportStatus::NotIdentified)
            || action >= self.behavior_probabilities.len()
            || action >= self.target_probabilities.len()
        {
            return None;
        }
        let h = self.behavior_probabilities[action];
        let pi = self.target_probabilities[action];
        if !h.is_finite() || !pi.is_finite() || h <= 0.0 {
            return None;
        }
        let w = pi / h;
        if w.is_finite() { Some(w) } else { None }
    }

    fn verify_logged_weight(&self, action: usize, claimed_weight: f64) -> bool {
        match self.expected_weight(action) {
            Some(expected) => claimed_weight.is_finite() && claimed_weight >= 0.0
                && (claimed_weight - expected).abs() <= NUMERIC_TOL * (1.0 + expected.abs()),
            None => false,
        }
    }
}

fn empirical_ess(weights: &[f64]) -> Option<f64> {
    if weights.is_empty() || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return None;
    }
    let sum: f64 = weights.iter().sum();
    let sum_sq: f64 = weights.iter().map(|w| w*w).sum();
    if !sum.is_finite() || !sum_sq.is_finite() || sum_sq <= 0.0 {
        return None;
    }
    Some((sum*sum/sum_sq).clamp(0.0, weights.len() as f64))
}

fn fmt_option(value: Option<f64>) -> String {
    value.map(|x| format!("{x:.12}")).unwrap_or_else(|| "SUPPRESSED".to_owned())
}

fn push_audit(out: &mut String, name: &str, audit: &SupportAudit) {
    let _ = writeln!(out, "SUPPORT_CASE\t{name}\tstatus\t{}", audit.status.as_str());
    let _ = writeln!(out, "SUPPORT_CASE\t{name}\tdecision\t{}", audit.decision.as_str());
    let _ = writeln!(out, "SUPPORT_CASE\t{name}\tunsupported_target_mass\t{}", fmt_option(audit.unsupported_target_mass));
    let _ = writeln!(out, "SUPPORT_CASE\t{name}\tmax_importance_ratio\t{}", fmt_option(audit.max_importance_ratio));
    let _ = writeln!(out, "SUPPORT_CASE\t{name}\timportance_weight_second_moment\t{}", fmt_option(audit.importance_weight_second_moment));
    let _ = writeln!(out, "SUPPORT_CASE\t{name}\tpopulation_ess_fraction\t{}", fmt_option(audit.population_ess_fraction));
    let _ = writeln!(out, "SUPPORT_CASE\t{name}\tnumeric_value_disclosure\t{}", audit.numeric_value_disclosure);
    let _ = writeln!(out, "SUPPORT_CASE\t{name}\treason\t{}", audit.reason);
    let _ = writeln!(out, "SUPPORT_CASE\t{name}\taction_count\t{}", audit.action_count);
}

fn audit_fixtures() -> Result<String, String> {
    let cases: [(&str, &[f64], &[f64]); 6] = [
        ("balanced", &[0.5,0.5], &[0.5,0.5]),
        ("weak_overlap", &[0.01,0.99], &[0.5,0.5]),
        ("near_zero_overlap", &[0.000001,0.999999], &[0.5,0.5]),
        ("zero_support", &[0.0,1.0], &[0.5,0.5]),
        ("zero_target_mass_on_unlogged", &[0.0,1.0], &[0.0,1.0]),
        ("invalid_probability_sum", &[0.4,0.4], &[0.5,0.5]),
    ];
    let mut out = String::from("record_type\tcase_id\tfield\tvalue\n");
    for (name, h, pi) in cases {
        let audit = SupportAudit::new(h, pi, WEAK_MAX_RATIO, WEAK_MIN_POPULATION_ESS_FRACTION);
        push_audit(&mut out, name, &audit);
        match name {
            "balanced" => {
                if audit.status != SupportStatus::Identified
                    || !close_required(audit.max_importance_ratio, 1.0)
                    || !close_required(audit.population_ess_fraction, 1.0) {
                    return Err("balanced known-answer control failed".to_owned());
                }
            }
            "weak_overlap" => {
                if audit.status != SupportStatus::WeakSupport
                    || !close_required(audit.max_importance_ratio, 50.0)
                    || !close_required(audit.importance_weight_second_moment, 25.2525252525) {
                    return Err("weak-overlap known-answer control failed".to_owned());
                }
            }
            "near_zero_overlap" => {
                if audit.status != SupportStatus::WeakSupport
                    || !audit.population_ess_fraction.is_some_and(|x| x < 1e-5) {
                    return Err("near-zero-overlap known-answer control failed".to_owned());
                }
            }
            "zero_support" => {
                if audit.status != SupportStatus::NotIdentified
                    || audit.numeric_value_disclosure != "SUPPRESSED_UNSUPPORTED_TARGET_MASS"
                    || audit.max_importance_ratio.is_some() || audit.population_ess_fraction.is_some() {
                    return Err("zero-support fail-closed control failed".to_owned());
                }
            }
            "zero_target_mass_on_unlogged" => {
                if audit.status != SupportStatus::Identified
                    || !close_required(audit.max_importance_ratio, 1.0)
                    || !close_required(audit.population_ess_fraction, 1.0) {
                    return Err("target-zero unlogged-action control failed".to_owned());
                }
            }
            "invalid_probability_sum" => {
                if audit.status != SupportStatus::InvalidInput
                    || audit.decision != DecisionStatus::InvalidInputNoDecision
                    || audit.max_importance_ratio.is_some() {
                    return Err("invalid-input fail-closed control failed".to_owned());
                }
            }
            _ => unreachable!("frozen fixture"),
        }
        let _ = writeln!(out, "CASE\t{name}\tstatus\tOK");
    }

    let balanced=SupportAudit::new(&[0.5,0.5],&[0.5,0.5],WEAK_MAX_RATIO,WEAK_MIN_POPULATION_ESS_FRACTION);
    if !balanced.verify_logged_weight(1,1.0)
        || balanced.verify_logged_weight(1,0.9) {
        return Err("logged propensity-ratio verification control failed".to_owned());
    }
    let weak=SupportAudit::new(&[0.01,0.99],&[0.5,0.5],WEAK_MAX_RATIO,WEAK_MIN_POPULATION_ESS_FRACTION);
    if weak.verify_logged_weight(0,49.0)
        || !weak.verify_logged_weight(0,50.0) {
        return Err("mismatched logged-weight fail-closed control failed".to_owned());
    }
    let empirical=empirical_ess(&[1.0,1.0,1.0,1.0]).ok_or("empirical ESS failed")?;
    if !close(empirical,4.0,1e-12) { return Err("empirical ESS known-answer failed".to_owned()); }
    let _=writeln!(out,"SUPPORT_CONTROL\tlogged_ratio_match\tstatus\tOK");
    let _=writeln!(out,"SUPPORT_CONTROL\tlogged_ratio_mismatch\tstatus\tOK");
    let _=writeln!(out,"SUPPORT_CONTROL\tempirical_ess\tvalue\t{empirical:.12}");
    Ok(out)
}

fn close_required(value: Option<f64>, expected: f64) -> bool {
    value.is_some_and(|x| (x-expected).abs() <= 1e-8*(1.0+expected.abs()))
}

fn main() {
    let result=match env::args().nth(1).as_deref() {
        Some("--audit") => audit_fixtures(),
        _=>Err("usage: vision-dvpc-043-support-identification-gate --audit".to_owned()),
    };
    match result {
        Ok(out)=>print!("{out}"),
        Err(err)=>{eprintln!("VIS_DVPC_043_ERROR\t{err}");std::process::exit(2);}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_support_suppresses_all_value_and_interval_disclosure() {
        let a=SupportAudit::new(&[0.0,1.0],&[0.5,0.5],WEAK_MAX_RATIO,WEAK_MIN_POPULATION_ESS_FRACTION);
        assert_eq!(a.status,SupportStatus::NotIdentified);
        assert_eq!(a.decision,DecisionStatus::SuppressValueAndInterval);
        assert_eq!(a.numeric_value_disclosure,"SUPPRESSED_UNSUPPORTED_TARGET_MASS");
        assert!(a.max_importance_ratio.is_none());
        assert!(a.importance_weight_second_moment.is_none());
        assert!(a.population_ess_fraction.is_none());
    }

    #[test]
    fn positive_logger_support_with_zero_target_mass_is_not_false_alarm() {
        let a=SupportAudit::new(&[0.0,1.0],&[0.0,1.0],WEAK_MAX_RATIO,WEAK_MIN_POPULATION_ESS_FRACTION);
        assert_eq!(a.status,SupportStatus::Identified);
        assert_eq!(a.unsupported_target_mass,Some(0.0));
        assert_eq!(a.max_importance_ratio,Some(1.0));
    }

    #[test]
    fn weak_and_near_zero_overlap_are_explicitly_non_decision_eligible() {
        let weak=SupportAudit::new(&[0.01,0.99],&[0.5,0.5],WEAK_MAX_RATIO,WEAK_MIN_POPULATION_ESS_FRACTION);
        let near=SupportAudit::new(&[0.000001,0.999999],&[0.5,0.5],WEAK_MAX_RATIO,WEAK_MIN_POPULATION_ESS_FRACTION);
        assert_eq!(weak.status,SupportStatus::WeakSupport);
        assert_eq!(weak.decision,DecisionStatus::InsufficientEvidence);
        assert!(close_required(weak.population_ess_fraction,1.0/25.2525252525));
        assert_eq!(near.status,SupportStatus::WeakSupport);
        assert!(near.population_ess_fraction.unwrap()<1e-5);
    }

    #[test]
    fn invalid_probabilities_thresholds_or_shape_fail_closed() {
        assert_eq!(SupportAudit::new(&[0.4,0.4],&[0.5,0.5],WEAK_MAX_RATIO,WEAK_MIN_POPULATION_ESS_FRACTION).status,SupportStatus::InvalidInput);
        assert_eq!(SupportAudit::new(&[f64::NAN,1.0],&[0.5,0.5],WEAK_MAX_RATIO,WEAK_MIN_POPULATION_ESS_FRACTION).status,SupportStatus::InvalidInput);
        assert_eq!(SupportAudit::new(&[0.5,0.5],&[0.5,0.5],0.0,WEAK_MIN_POPULATION_ESS_FRACTION).status,SupportStatus::InvalidInput);
        assert_eq!(SupportAudit::new(&[0.5],&[0.5,0.5],WEAK_MAX_RATIO,WEAK_MIN_POPULATION_ESS_FRACTION).status,SupportStatus::InvalidInput);
    }

    #[test]
    fn mismatched_logged_propensity_weight_fails_closed() {
        let a=SupportAudit::new(&[0.01,0.99],&[0.5,0.5],WEAK_MAX_RATIO,WEAK_MIN_POPULATION_ESS_FRACTION);
        assert!(a.verify_logged_weight(&[0.01,0.99],&[0.5,0.5],0,50.0));
        assert!(!a.verify_logged_weight(0,49.0));
        let unsupported=SupportAudit::new(&[0.0,1.0],&[0.5,0.5],WEAK_MAX_RATIO,WEAK_MIN_POPULATION_ESS_FRACTION);
        assert!(!unsupported.verify_logged_weight(1,0.5));
    }

    #[test]
    fn empirical_ess_is_not_the_population_support_proof() {
        assert_eq!(empirical_ess(&[1.0,1.0,1.0,1.0]),Some(4.0));
        assert!(empirical_ess(&[1.0,2.0,4.0]).unwrap()<3.0);
        assert_eq!(empirical_ess(&[0.0,0.0]),None);
        assert_eq!(empirical_ess(&[1.0,f64::NAN]),None);
    }

    #[test]
    fn every_fixture_emits_explicit_status_and_suppressed_fields_when_required() {
        let out=audit_fixtures().unwrap();
        assert_eq!(out.matches("CASE\t").count(),6);
        assert!(out.contains("zero_support\tstatus\tNOT_IDENTIFIED"));
        assert!(out.contains("zero_support\tmax_importance_ratio\tSUPPRESSED"));
        assert!(out.contains("zero_support\tpopulation_ess_fraction\tSUPPRESSED"));
        assert!(out.contains("weak_overlap\tdecision\tINSUFFICIENT_EVIDENCE"));
    }
}
