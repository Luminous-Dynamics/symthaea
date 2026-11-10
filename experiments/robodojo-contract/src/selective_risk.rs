//! Finite-sample selective-risk certificate for a pre-registered robotics gate.
//!
//! The certificate bounds failure and unsafe-event rates only among calibration
//! samples accepted by one frozen score-threshold rule. Calibration and
//! deployment samples must be independent and exchangeable within the exact
//! task/policy/environment/evaluator context. Distribution shift invalidates
//! the claim unless a separately qualified condition-specific certificate is
//! supplied. This is not a physical safety guarantee or actuator authorization.

use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiskContext {
    pub task_id: String,
    /// State/motion stratum, e.g. free_motion, contact_onset, sustained_contact,
    /// constrained_insertion or release. The regime is evaluator-defined.
    pub state_regime_id: String,
    /// Digest of the exact predicate/partition that assigns a state to this regime.
    pub state_regime_definition_sha256: String,
    pub policy_artifact_sha256: String,
    /// Digest of the forecast/risk estimator and output post-processing path.
    pub forecast_pipeline_sha256: String,
    pub environment_condition_sha256: String,
    pub evaluator_sha256: String,
}

impl RiskContext {
    pub fn validate(&self) -> Result<(), RiskCertificateError> {
        if self.task_id.trim().is_empty() || self.state_regime_id.trim().is_empty() {
            return Err(RiskCertificateError::EmptyIdentity);
        }
        for digest in [
            self.state_regime_definition_sha256.as_str(),
            self.policy_artifact_sha256.as_str(),
            self.forecast_pipeline_sha256.as_str(),
            self.environment_condition_sha256.as_str(),
            self.evaluator_sha256.as_str(),
        ] {
            if !is_sha256_hex(digest) {
                return Err(RiskCertificateError::InvalidDigest);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectiveRiskSample {
    pub sample_id: String,
    /// Pre-registered independent unit, e.g. one held-out scene/seed/episode
    /// cluster. Repeated episodes from one unit must share this identity.
    pub independence_unit_id: String,
    pub context: RiskContext,
    /// Forecast emitted before the episode outcome is known.
    pub predicted_success_probability: f64,
    /// Forecast emitted before the episode outcome is known.
    pub predicted_unsafe_event_probability: f64,
    /// Official full-task outcome. Invalid/interrupted runs must be false.
    /// None means the official task outcome was not obtained for a valid run.
    /// An invalid run is still counted as a task failure.
    pub observed_success: Option<bool>,
    /// None means the unsafe-event outcome was not completely observed.
    pub observed_unsafe_event: Option<bool>,
    /// False runs remain in the accepted set and count as task failures.
    pub valid_run: bool,
}

impl SelectiveRiskSample {
    pub fn validate(&self) -> Result<(), RiskCertificateError> {
        if self.sample_id.trim().is_empty() || self.independence_unit_id.trim().is_empty() {
            return Err(RiskCertificateError::EmptyIdentity);
        }
        self.context.validate()?;
        if !valid_probability(self.predicted_success_probability)
            || !valid_probability(self.predicted_unsafe_event_probability)
        {
            return Err(RiskCertificateError::InvalidProbability);
        }
        if self.observed_success == Some(true) && !self.valid_run {
            return Err(RiskCertificateError::SuccessOnInvalidRun);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectiveRiskPolicy {
    /// Fixed before calibration; only predictions at or above this score enter
    /// the accepted cohort.
    pub minimum_predicted_success_probability: f64,
    /// Fixed before calibration; only predictions at or below this score enter
    /// the accepted cohort.
    pub maximum_predicted_unsafe_event_probability: f64,
    /// Maximum tolerated task-failure rate among accepted episodes.
    pub maximum_accepted_failure_rate: f64,
    /// Maximum tolerated unsafe-event rate among accepted episodes.
    pub maximum_accepted_unsafe_event_rate: f64,
    /// Family-wise error probability for the two one-sided bounds.
    pub confidence_delta: f64,
    pub minimum_accepted_samples: usize,
}

impl SelectiveRiskPolicy {
    pub fn validate(self) -> Result<(), RiskCertificateError> {
        if !valid_probability(self.minimum_predicted_success_probability)
            || !valid_probability(self.maximum_predicted_unsafe_event_probability)
            || !valid_probability(self.maximum_accepted_failure_rate)
            || !valid_probability(self.maximum_accepted_unsafe_event_rate)
        {
            return Err(RiskCertificateError::InvalidProbability);
        }
        if !self.confidence_delta.is_finite()
            || !(0.0..1.0).contains(&self.confidence_delta)
            || self.minimum_accepted_samples == 0
        {
            return Err(RiskCertificateError::InvalidPolicy);
        }
        Ok(())
    }

    fn accepts(self, sample: &SelectiveRiskSample) -> bool {
        sample.predicted_success_probability >= self.minimum_predicted_success_probability
            && sample.predicted_unsafe_event_probability
                <= self.maximum_predicted_unsafe_event_probability
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskCertificateError {
    EmptySamples,
    EmptyIdentity,
    InvalidDigest,
    InvalidProbability,
    InvalidPolicy,
    SuccessOnInvalidRun,
    DuplicateSample,
    DuplicateIndependenceUnit,
    MixedContext,
    InvalidRuntimeContext,
    RuntimeContextMismatch,
    ScoreOutsideAcceptedRegion,
    RiskNotQualified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskQualificationStatus {
    InsufficientAcceptedSamples,
    IncompleteOutcomeEvidence,
    FailureRiskExceedsLimit,
    UnsafeRiskExceedsLimit,
    BothRiskLimitsExceed,
    QualifiedForDeclaredContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectiveRiskCertificate {
    context: RiskContext,
    /// Digest of the immutable calibration dataset/manifest. The caller
    /// supplies this; this module does not hash or authenticate the file.
    calibration_dataset_sha256: String,
    policy: SelectiveRiskPolicy,
    total_samples: usize,
    accepted_samples: usize,
    rejected_samples: usize,
    accepted_valid_runs: usize,
    accepted_task_failures: usize,
    accepted_unsafe_events: usize,
    accepted_task_outcomes_unmeasured: usize,
    accepted_unsafe_event_outcomes_unmeasured: usize,
    empirical_failure_rate: Option<f64>,
    empirical_unsafe_event_rate: Option<f64>,
    /// Exact one-sided Clopper-Pearson upper bounds. Each uses delta/2 so the
    /// pair of claims has at least 1-delta simultaneous confidence.
    failure_rate_upper_bound: Option<f64>,
    unsafe_event_rate_upper_bound: Option<f64>,
    failure_bound_delta: f64,
    unsafe_bound_delta: f64,
    status: RiskQualificationStatus,
}

impl SelectiveRiskCertificate {
    pub fn context(&self) -> &RiskContext { &self.context }
    pub fn calibration_dataset_sha256(&self) -> &str { &self.calibration_dataset_sha256 }
    pub fn policy(&self) -> SelectiveRiskPolicy { self.policy }
    pub fn status(&self) -> RiskQualificationStatus { self.status }
    pub fn total_samples(&self) -> usize { self.total_samples }
    pub fn accepted_samples(&self) -> usize { self.accepted_samples }
    pub fn rejected_samples(&self) -> usize { self.rejected_samples }
    pub fn accepted_valid_runs(&self) -> usize { self.accepted_valid_runs }
    pub fn accepted_task_failures(&self) -> usize { self.accepted_task_failures }
    pub fn accepted_unsafe_events(&self) -> usize { self.accepted_unsafe_events }
    pub fn accepted_task_outcomes_unmeasured(&self) -> usize { self.accepted_task_outcomes_unmeasured }
    pub fn accepted_unsafe_event_outcomes_unmeasured(&self) -> usize { self.accepted_unsafe_event_outcomes_unmeasured }
    pub fn empirical_failure_rate(&self) -> Option<f64> { self.empirical_failure_rate }
    pub fn empirical_unsafe_event_rate(&self) -> Option<f64> { self.empirical_unsafe_event_rate }
    pub fn failure_rate_upper_bound(&self) -> Option<f64> { self.failure_rate_upper_bound }
    pub fn unsafe_event_rate_upper_bound(&self) -> Option<f64> { self.unsafe_event_rate_upper_bound }
    pub fn failure_bound_delta(&self) -> f64 { self.failure_bound_delta }
    pub fn unsafe_bound_delta(&self) -> f64 { self.unsafe_bound_delta }

    /// This checks whether a new score belongs to the certified selective
    /// region. It does not authorize an actuator command or establish safety.
    pub fn check_runtime_score(
        &self,
        runtime_context: &RiskContext,
        predicted_success_probability: f64,
        predicted_unsafe_event_probability: f64,
    ) -> Result<(), RiskCertificateError> {
        runtime_context.validate()?;
        if runtime_context != &self.context {
            return Err(RiskCertificateError::RuntimeContextMismatch);
        }
        if !valid_probability(predicted_success_probability)
            || !valid_probability(predicted_unsafe_event_probability)
        {
            return Err(RiskCertificateError::InvalidProbability);
        }
        if predicted_success_probability
            < self.policy.minimum_predicted_success_probability
            || predicted_unsafe_event_probability
                > self.policy.maximum_predicted_unsafe_event_probability
        {
            return Err(RiskCertificateError::ScoreOutsideAcceptedRegion);
        }
        if self.status != RiskQualificationStatus::QualifiedForDeclaredContext {
            return Err(RiskCertificateError::RiskNotQualified);
        }
        Ok(())
    }
}

/// Build one certificate for one frozen context and one pre-registered
/// selective rule. Samples outside the rule's accepted score region are
/// counted but excluded from the accepted-cohort risk estimates.
///
/// The two one-sided Clopper-Pearson upper bounds use alpha=delta/2 each.
/// Under independent, exchangeable calibration/deployment episodes within the
/// declared context, the union bound gives simultaneous confidence >= 1-delta.
/// This function cannot establish independence, exchangeability, held-out
/// status, data provenance, or correctness of the supplied context digests.
pub fn certify_selective_risk(
    samples: &[SelectiveRiskSample],
    policy: SelectiveRiskPolicy,
    calibration_dataset_sha256: impl Into<String>,
) -> Result<SelectiveRiskCertificate, RiskCertificateError> {
    if samples.is_empty() {
        return Err(RiskCertificateError::EmptySamples);
    }
    policy.validate()?;
    let calibration_dataset_sha256 = calibration_dataset_sha256.into();
    if !is_sha256_hex(&calibration_dataset_sha256) {
        return Err(RiskCertificateError::InvalidDigest);
    }

    let context = samples[0].context.clone();
    context.validate()?;
    let mut seen = BTreeSet::new();
    let mut seen_independence_units = BTreeSet::new();
    let mut accepted_samples = 0usize;
    let mut accepted_valid_runs = 0usize;
    let mut accepted_task_failures = 0usize;
    let mut accepted_unsafe_events = 0usize;
    let mut accepted_task_outcomes_unmeasured = 0usize;
    let mut accepted_unsafe_event_outcomes_unmeasured = 0usize;

    for sample in samples {
        sample.validate()?;
        if sample.context != context {
            return Err(RiskCertificateError::MixedContext);
        }
        if !seen.insert(sample.sample_id.as_str()) {
            return Err(RiskCertificateError::DuplicateSample);
        }
        if !seen_independence_units.insert(sample.independence_unit_id.as_str()) {
            return Err(RiskCertificateError::DuplicateIndependenceUnit);
        }
        if !policy.accepts(sample) {
            continue;
        }
        accepted_samples += 1;
        if sample.valid_run {
            accepted_valid_runs += 1;
        }
        if !sample.valid_run {
            accepted_task_failures += 1;
        } else if let Some(success) = sample.observed_success {
            if !success { accepted_task_failures += 1; }
        } else {
            accepted_task_outcomes_unmeasured += 1;
        }
        match sample.observed_unsafe_event {
            Some(true) => accepted_unsafe_events += 1,
            Some(false) => {}
            None => accepted_unsafe_event_outcomes_unmeasured += 1,
        }
    }

    let rejected_samples = samples.len() - accepted_samples;
    let alpha_per_bound = policy.confidence_delta / 2.0;
    let (empirical_failure_rate, failure_rate_upper_bound) =
        if accepted_samples == 0 || accepted_task_outcomes_unmeasured > 0 {
            (None, None)
        } else {
        let empirical = accepted_task_failures as f64 / accepted_samples as f64;
        (
            Some(empirical),
            Some(clopper_pearson_upper(
                accepted_task_failures,
                accepted_samples,
                alpha_per_bound,
            )),
        )
    };
    let (empirical_unsafe_event_rate, unsafe_event_rate_upper_bound) =
        if accepted_samples == 0 || accepted_unsafe_event_outcomes_unmeasured > 0 {
            (None, None)
        } else {
            let empirical = accepted_unsafe_events as f64 / accepted_samples as f64;
            (
                Some(empirical),
                Some(clopper_pearson_upper(
                    accepted_unsafe_events,
                    accepted_samples,
                    alpha_per_bound,
                )),
            )
        };

    let enough_samples = accepted_samples >= policy.minimum_accepted_samples;
    let failure_exceeds = failure_rate_upper_bound
        .is_some_and(|upper| upper > policy.maximum_accepted_failure_rate);
    let unsafe_exceeds = unsafe_event_rate_upper_bound
        .is_some_and(|upper| upper > policy.maximum_accepted_unsafe_event_rate);
    let has_unmeasured_outcomes = accepted_task_outcomes_unmeasured > 0
        || accepted_unsafe_event_outcomes_unmeasured > 0;
    let status = if !enough_samples {
        RiskQualificationStatus::InsufficientAcceptedSamples
    } else if has_unmeasured_outcomes {
        RiskQualificationStatus::IncompleteOutcomeEvidence
    } else {
        match (failure_exceeds, unsafe_exceeds) {
            (true, true) => RiskQualificationStatus::BothRiskLimitsExceed,
            (true, false) => RiskQualificationStatus::FailureRiskExceedsLimit,
            (false, true) => RiskQualificationStatus::UnsafeRiskExceedsLimit,
            (false, false) => RiskQualificationStatus::QualifiedForDeclaredContext,
        }
    };

    Ok(SelectiveRiskCertificate {
        context,
        calibration_dataset_sha256,
        policy,
        total_samples: samples.len(),
        accepted_samples,
        rejected_samples,
        accepted_valid_runs,
        accepted_task_failures,
        accepted_unsafe_events,
        accepted_task_outcomes_unmeasured,
        accepted_unsafe_event_outcomes_unmeasured,
        empirical_failure_rate,
        empirical_unsafe_event_rate,
        failure_rate_upper_bound,
        unsafe_event_rate_upper_bound,
        failure_bound_delta: alpha_per_bound,
        unsafe_bound_delta: alpha_per_bound,
        status,
    })
}

/// One-sided exact Clopper-Pearson upper confidence bound for a binomial rate.
/// For k < n, solve P[X <= k | n,p] = alpha. The k=0 form is closed-form.
fn clopper_pearson_upper(events: usize, samples: usize, alpha: f64) -> f64 {
    if samples == 0 || events >= samples {
        return 1.0;
    }
    if events == 0 {
        return 1.0 - alpha.powf(1.0 / samples as f64);
    }

    let target_log_cdf = alpha.ln();
    let mut low = events as f64 / samples as f64;
    let mut high = 1.0;
    for _ in 0..80 {
        let midpoint = (low + high) / 2.0;
        let log_cdf = log_binomial_lower_tail(events, samples, midpoint);
        if log_cdf > target_log_cdf {
            low = midpoint;
        } else {
            high = midpoint;
        }
    }
    high
}

/// Stable log(P[X <= k]) for X~Binomial(n,p). Uses a recurrence for adjacent
/// probability masses and log-sum-exp; the bound routine calls this only for
/// 0 < p < 1 and k < n.
fn log_binomial_lower_tail(k: usize, n: usize, p: f64) -> f64 {
    if k >= n {
        return 0.0;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::NEG_INFINITY;
    }

    let log_p = p.ln();
    let log_q = (-p).ln_1p();
    let mut log_mass = n as f64 * log_q;
    let mut log_total = log_mass;

    for index in 0..k {
        log_mass += ((n - index) as f64).ln()
            - ((index + 1) as f64).ln()
            + log_p
            - log_q;
        log_total = log_add_exp(log_total, log_mass);
    }
    log_total.min(0.0)
}

fn log_add_exp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(condition: &str) -> RiskContext {
        RiskContext {
            task_id: "wash_lettuce".to_owned(),
            state_regime_id: "contactless".to_owned(),
            state_regime_definition_sha256: "6".repeat(64),
            policy_artifact_sha256: "1".repeat(64),
            forecast_pipeline_sha256: "5".repeat(64),
            environment_condition_sha256: condition.repeat(64),
            evaluator_sha256: "3".repeat(64),
        }
    }

    fn policy() -> SelectiveRiskPolicy {
        SelectiveRiskPolicy {
            minimum_predicted_success_probability: 0.8,
            maximum_predicted_unsafe_event_probability: 0.05,
            maximum_accepted_failure_rate: 0.08,
            maximum_accepted_unsafe_event_rate: 0.08,
            confidence_delta: 0.05,
            minimum_accepted_samples: 50,
        }
    }

    fn sample(id: &str, success: bool, unsafe_event: bool) -> SelectiveRiskSample {
        SelectiveRiskSample {
            sample_id: id.to_owned(),
            independence_unit_id: id.to_owned(),
            context: context("2"),
            predicted_success_probability: 0.95,
            predicted_unsafe_event_probability: 0.01,
            observed_success: Some(success),
            observed_unsafe_event: Some(unsafe_event),
            valid_run: true,
        }
    }

    fn safe_samples(count: usize) -> Vec<SelectiveRiskSample> {
        (0..count)
            .map(|index| sample(&format!("episode-{index:04}"), true, false))
            .collect()
    }

    #[test]
    fn zero_event_exact_bound_matches_closed_form_and_qualifies_when_powered() {
        let samples = safe_samples(100);
        let certificate =
            certify_selective_risk(&samples, policy(), "4".repeat(64)).unwrap();
        let expected = 1.0 - 0.025_f64.powf(1.0 / 100.0);
        assert!((certificate.failure_rate_upper_bound.unwrap() - expected).abs() < 1e-12);
        assert!((certificate.unsafe_event_rate_upper_bound.unwrap() - expected).abs() < 1e-12);
        assert_eq!(certificate.status(), RiskQualificationStatus::QualifiedForDeclaredContext);
        assert_eq!(certificate.accepted_samples(), 100);
        assert_eq!(certificate.rejected_samples(), 0);
    }

    #[test]
    fn invalid_runs_count_as_task_failures_in_the_accepted_denominator() {
        let mut samples = safe_samples(100);
        samples[0].valid_run = false;
        samples[0].observed_success = Some(false);
        let certificate =
            certify_selective_risk(&samples, policy(), "4".repeat(64)).unwrap();
        assert_eq!(certificate.accepted_samples(), 100);
        assert_eq!(certificate.accepted_valid_runs(), 99);
        assert_eq!(certificate.accepted_task_failures(), 1);
        assert_eq!(certificate.empirical_failure_rate(), Some(0.01));
    }

    #[test]
    fn one_unsafe_event_is_not_hidden_by_successful_task_completion() {
        let mut samples = safe_samples(100);
        samples[0].observed_unsafe_event = Some(true);
        let certificate =
            certify_selective_risk(&samples, policy(), "4".repeat(64)).unwrap();
        assert_eq!(certificate.accepted_task_failures(), 0);
        assert_eq!(certificate.accepted_unsafe_events(), 1);
        assert_eq!(certificate.empirical_unsafe_event_rate(), Some(0.01));
        assert_eq!(certificate.status(), RiskQualificationStatus::QualifiedForDeclaredContext);
    }

    #[test]
    fn insufficient_accepted_samples_never_qualify() {
        let samples = safe_samples(49);
        let certificate =
            certify_selective_risk(&samples, policy(), "4".repeat(64)).unwrap();
        assert_eq!(
            certificate.status(),
            RiskQualificationStatus::InsufficientAcceptedSamples
        );
        assert!(certificate.check_runtime_score(&context("2"), 0.95, 0.01).is_err());
    }

    #[test]
    fn observations_outside_the_frozen_score_region_do_not_enter_risk_estimates() {
        let mut samples = safe_samples(100);
        samples[0].predicted_success_probability = 0.79;
        samples[1].predicted_unsafe_event_probability = 0.06;
        let certificate =
            certify_selective_risk(&samples, policy(), "4".repeat(64)).unwrap();
        assert_eq!(certificate.accepted_samples(), 98);
        assert_eq!(certificate.rejected_samples(), 2);
    }

    #[test]
    fn runtime_must_match_exact_policy_environment_and_evaluator_context() {
        let certificate =
            certify_selective_risk(&safe_samples(100), policy(), "4".repeat(64)).unwrap();
        assert_eq!(
            certificate.check_runtime_score(&context("2"), 0.95, 0.01),
            Ok(())
        );
        assert_eq!(
            certificate.check_runtime_score(&context("5"), 0.95, 0.01),
            Err(RiskCertificateError::RuntimeContextMismatch)
        );
        let mut other_policy = context("2");
        other_policy.policy_artifact_sha256 = "9".repeat(64);
        assert_eq!(
            certificate.check_runtime_score(&other_policy, 0.95, 0.01),
            Err(RiskCertificateError::RuntimeContextMismatch)
        );

        let mut other_forecast_pipeline = context("2");
        other_forecast_pipeline.forecast_pipeline_sha256 = "8".repeat(64);
        assert_eq!(
            certificate.check_runtime_score(&other_forecast_pipeline, 0.95, 0.01),
            Err(RiskCertificateError::RuntimeContextMismatch)
        );
    }

    #[test]
    fn runtime_must_match_the_exact_state_regime_and_its_definition() {
        let certificate =
            certify_selective_risk(&safe_samples(100), policy(), "4".repeat(64)).unwrap();

        let mut other_regime = context("2");
        other_regime.state_regime_id = "sustained_contact".to_owned();
        assert_eq!(
            certificate.check_runtime_score(&other_regime, 0.95, 0.01),
            Err(RiskCertificateError::RuntimeContextMismatch)
        );

        let mut changed_partition = context("2");
        changed_partition.state_regime_definition_sha256 = "7".repeat(64);
        assert_eq!(
            certificate.check_runtime_score(&changed_partition, 0.95, 0.01),
            Err(RiskCertificateError::RuntimeContextMismatch)
        );
    }

    #[test]
    fn a_runtime_score_outside_the_certified_acceptance_region_is_rejected() {
        let certificate =
            certify_selective_risk(&safe_samples(100), policy(), "4".repeat(64)).unwrap();
        assert_eq!(
            certificate.check_runtime_score(&context("2"), 0.75, 0.01),
            Err(RiskCertificateError::ScoreOutsideAcceptedRegion)
        );
        assert_eq!(
            certificate.check_runtime_score(&context("2"), 0.95, 0.06),
            Err(RiskCertificateError::ScoreOutsideAcceptedRegion)
        );
    }

    #[test]
    fn observed_risk_above_the_pre_registered_limit_blocks_qualification() {
        let mut samples = safe_samples(100);
        for index in 0..20 {
            samples[index].observed_success = Some(false);
            samples[index].observed_unsafe_event = Some(true);
        }
        let certificate =
            certify_selective_risk(&samples, policy(), "4".repeat(64)).unwrap();
        assert_eq!(
            certificate.status(),
            RiskQualificationStatus::BothRiskLimitsExceed
        );
        assert!(certificate.check_runtime_score(&context("2"), 0.95, 0.01).is_err());
    }

    #[test]
    fn mixed_contexts_duplicate_ids_and_invalid_digests_reject() {
        let mut mixed = safe_samples(100);
        mixed[1].context.environment_condition_sha256 = "5".repeat(64);
        assert_eq!(
            certify_selective_risk(&mixed, policy(), "4".repeat(64)),
            Err(RiskCertificateError::MixedContext)
        );

        let mut duplicate = safe_samples(100);
        duplicate[1].sample_id = duplicate[0].sample_id.clone();
        assert_eq!(
            certify_selective_risk(&duplicate, policy(), "4".repeat(64)),
            Err(RiskCertificateError::DuplicateSample)
        );

        let mut repeated_unit = safe_samples(100);
        repeated_unit[1].independence_unit_id = repeated_unit[0].independence_unit_id.clone();
        assert_eq!(
            certify_selective_risk(&repeated_unit, policy(), "4".repeat(64)),
            Err(RiskCertificateError::DuplicateIndependenceUnit)
        );

        let mut invalid = safe_samples(100);
        invalid[0].context.evaluator_sha256 = "not-a-digest".to_owned();
        assert_eq!(
            certify_selective_risk(&invalid, policy(), "4".repeat(64)),
            Err(RiskCertificateError::InvalidDigest)
        );
    }

    #[test]
    fn missing_success_or_safety_labels_never_become_zero_risk() {
        let mut samples = safe_samples(100);
        samples[0].observed_success = None;
        samples[1].observed_unsafe_event = None;
        let certificate =
            certify_selective_risk(&samples, policy(), "4".repeat(64)).unwrap();
        assert_eq!(
            certificate.status(),
            RiskQualificationStatus::IncompleteOutcomeEvidence
        );
        assert_eq!(certificate.accepted_task_outcomes_unmeasured(), 1);
        assert_eq!(certificate.accepted_unsafe_event_outcomes_unmeasured(), 1);
        assert!(certificate.failure_rate_upper_bound().is_none());
        assert!(certificate.unsafe_event_rate_upper_bound().is_none());
        assert!(certificate
            .check_runtime_score(&context("2"), 0.95, 0.01)
            .is_err());
    }

    #[test]
    fn sample_count_monotonically_tightens_zero_event_upper_bound() {
        let small = clopper_pearson_upper(0, 50, 0.025);
        let large = clopper_pearson_upper(0, 200, 0.025);
        assert!(large < small);
    }

    #[test]
    fn binomial_upper_bound_increases_with_observed_events() {
        let no_events = clopper_pearson_upper(0, 100, 0.025);
        let one_event = clopper_pearson_upper(1, 100, 0.025);
        let five_events = clopper_pearson_upper(5, 100, 0.025);
        assert!(no_events < one_event);
        assert!(one_event < five_events);
    }

    #[test]
    fn nonzero_event_bounds_match_exact_binomial_quantiles() {
        assert!(
            (clopper_pearson_upper(1, 100, 0.025) - 0.054459385392080645).abs() < 1e-12
        );
        assert!(
            (clopper_pearson_upper(5, 100, 0.025) - 0.11283491110546275).abs() < 1e-12
        );
    }

    #[test]
    fn failure_and_unsafe_event_limits_fail_independently() {
        let mut failures_only = safe_samples(100);
        for sample in failures_only.iter_mut().take(20) {
            sample.observed_success = Some(false);
        }
        let failure_certificate =
            certify_selective_risk(&failures_only, policy(), "4".repeat(64)).unwrap();
        assert_eq!(
            failure_certificate.status(),
            RiskQualificationStatus::FailureRiskExceedsLimit
        );

        let mut unsafe_only = safe_samples(100);
        for sample in unsafe_only.iter_mut().take(20) {
            sample.observed_unsafe_event = Some(true);
        }
        let unsafe_certificate =
            certify_selective_risk(&unsafe_only, policy(), "4".repeat(64)).unwrap();
        assert_eq!(
            unsafe_certificate.status(),
            RiskQualificationStatus::UnsafeRiskExceedsLimit
        );
    }

    #[test]
    fn empty_and_underpowered_calibration_samples_are_not_a_certificate() {
        assert_eq!(
            certify_selective_risk(&[], policy(), "4".repeat(64)),
            Err(RiskCertificateError::EmptySamples)
        );
        let all_rejected: Vec<_> = (0..20)
            .map(|index| {
                let mut row = sample(&format!("low-{index}"), true, false);
                row.predicted_success_probability = 0.2;
                row
            })
            .collect();
        let certificate =
            certify_selective_risk(&all_rejected, policy(), "4".repeat(64)).unwrap();
        assert_eq!(certificate.accepted_samples(), 0);
        assert_eq!(certificate.status(), RiskQualificationStatus::InsufficientAcceptedSamples);
        assert!(certificate.failure_rate_upper_bound().is_none());
    }

    #[test]
    fn invalid_probability_policy_and_dataset_digest_reject() {
        let mut invalid = safe_samples(100);
        invalid[0].predicted_success_probability = f64::NAN;
        assert_eq!(
            certify_selective_risk(&invalid, policy(), "4".repeat(64)),
            Err(RiskCertificateError::InvalidProbability)
        );

        let mut invalid_policy = policy();
        invalid_policy.confidence_delta = 0.0;
        assert_eq!(
            certify_selective_risk(&safe_samples(100), invalid_policy, "4".repeat(64)),
            Err(RiskCertificateError::InvalidPolicy)
        );
        assert_eq!(
            certify_selective_risk(&safe_samples(100), policy(), "bad"),
            Err(RiskCertificateError::InvalidDigest)
        );
    }
}
