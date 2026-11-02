//! Two-sample finite-support TV transport certificate.
//!
//! Given independent IID complete-path samples from a calibration law P and
//! a target law Q on the same frozen finite categorical support, this module
//! bounds TV(P, Q) by the empirical two-sample TV plus two Weissman-style
//! concentration radii. For every measurable functional g with range [a, b],
//! |E_Q[g] - E_P[g]| <= (b-a) * TV(P, Q).
//!
//! This is a distribution-level certificate: it covers every bounded
//! functional on the declared support, not merely a predeclared list. It is
//! intentionally conservative and can become vacuous as support dimension
//! grows. Declared IID/independence/support semantics are contract assertions;
//! digests bind bytes but do not authenticate those assertions.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{error::Error, fmt};

const REQUEST_SCHEMA: &str =
    "symthaea.architecture-engineering.finite-tv-two-sample-transport-request";
const REQUEST_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str =
    "symthaea.architecture-engineering.finite-tv-two-sample-transport-report";
const REPORT_VERSION: &str = "0.1.0";
const MAX_SUPPORT_SIZE: usize = 100_000;
const MAX_EXACT_F64_INTEGER: u64 = 9_007_199_254_740_992;
const COUNT_SUM_TOLERANCE: f64 = 1e-10;
const LN_2: f64 = std::f64::consts::LN_2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteTvTransportError {
    pub path: String,
    pub message: String,
}

impl FiniteTvTransportError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for FiniteTvTransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for FiniteTvTransportError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteTvLawSample {
    /// Digest of the canonical finite support and its exact category ordering.
    pub path_support_sha256: String,
    /// Digest of the immutable artifact from which the category counts came.
    pub sample_data_sha256: String,
    /// Counts over the frozen ordered finite support.
    pub observed_path_counts: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteTvTwoSampleTransportRequest {
    pub schema: String,
    pub version: String,
    pub transport_id: String,
    /// Binds the registered functional family to which the certificate applies.
    pub functional_contract_sha256: String,
    /// Design identities are explicit and may differ under transport.
    pub calibration_design_sha256: String,
    pub target_design_sha256: String,
    pub calibration_sample: FiniteTvLawSample,
    pub target_sample: FiniteTvLawSample,
    /// Weissman failure allocation for the calibration empirical law.
    pub calibration_failure_probability: f64,
    /// Weissman failure allocation for the target empirical law.
    pub target_failure_probability: f64,
    /// Both samples must be IID draws of complete units from their respective laws.
    pub iid_calibration_complete_path_draws: bool,
    pub iid_target_complete_path_draws: bool,
    /// Required by this v1 independent-split protocol.
    pub independent_calibration_target_samples: bool,
    /// Both reference samples must be disjoint from the confirmatory stream.
    /// This is a contract declaration, not authenticated by this module.
    pub reference_samples_independent_of_confirmatory_stream: bool,
    /// Target counts must be pre-outcome covariates/path categories, not confirmatory outcomes.
    pub target_sample_pre_outcome_only: bool,
}

impl FiniteTvTwoSampleTransportRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        transport_id: impl Into<String>,
        functional_contract_sha256: impl Into<String>,
        calibration_design_sha256: impl Into<String>,
        target_design_sha256: impl Into<String>,
        calibration_sample: FiniteTvLawSample,
        target_sample: FiniteTvLawSample,
        calibration_failure_probability: f64,
        target_failure_probability: f64,
        iid_calibration_complete_path_draws: bool,
        iid_target_complete_path_draws: bool,
        independent_calibration_target_samples: bool,
        reference_samples_independent_of_confirmatory_stream: bool,
        target_sample_pre_outcome_only: bool,
    ) -> Result<Self, FiniteTvTransportError> {
        let request = Self {
            schema: REQUEST_SCHEMA.into(),
            version: REQUEST_VERSION.into(),
            transport_id: transport_id.into(),
            functional_contract_sha256: functional_contract_sha256.into(),
            calibration_design_sha256: calibration_design_sha256.into(),
            target_design_sha256: target_design_sha256.into(),
            calibration_sample,
            target_sample,
            calibration_failure_probability,
            target_failure_probability,
            iid_calibration_complete_path_draws,
            iid_target_complete_path_draws,
            independent_calibration_target_samples,
            reference_samples_independent_of_confirmatory_stream,
            target_sample_pre_outcome_only,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), FiniteTvTransportError> {
        if self.schema != REQUEST_SCHEMA || self.version != REQUEST_VERSION {
            return Err(FiniteTvTransportError::new(
                "request.schema",
                "unsupported finite-TV two-sample transport schema/version",
            ));
        }
        if self.transport_id.trim().is_empty() {
            return Err(FiniteTvTransportError::new(
                "request.transport_id",
                "transport ID must be non-empty",
            ));
        }
        validate_hash(
            &self.functional_contract_sha256,
            "request.functional_contract_sha256",
        )?;
        validate_hash(
            &self.calibration_design_sha256,
            "request.calibration_design_sha256",
        )?;
        validate_hash(
            &self.target_design_sha256,
            "request.target_design_sha256",
        )?;
        validate_sample(&self.calibration_sample, "request.calibration_sample")?;
        validate_sample(&self.target_sample, "request.target_sample")?;
        if self.calibration_sample.path_support_sha256
            != self.target_sample.path_support_sha256
            || self.calibration_sample.observed_path_counts.len()
                != self.target_sample.observed_path_counts.len()
        {
            return Err(FiniteTvTransportError::new(
                "request.path_support",
                "calibration and target samples must use the same frozen support digest, size, and category order",
            ));
        }
        validate_open_probability(
            self.calibration_failure_probability,
            "request.calibration_failure_probability",
        )?;
        validate_open_probability(
            self.target_failure_probability,
            "request.target_failure_probability",
        )?;
        if self.calibration_failure_probability + self.target_failure_probability >= 1.0 {
            return Err(FiniteTvTransportError::new(
                "request.failure_probability",
                "the sum of calibration and target failure allocations must be below one",
            ));
        }
        if !self.iid_calibration_complete_path_draws {
            return Err(FiniteTvTransportError::new(
                "request.iid_calibration_complete_path_draws",
                "calibration-side Weissman bound requires IID complete-path draws",
            ));
        }
        if !self.iid_target_complete_path_draws {
            return Err(FiniteTvTransportError::new(
                "request.iid_target_complete_path_draws",
                "target-side Weissman bound requires IID complete-path draws",
            ));
        }
        if !self.independent_calibration_target_samples {
            return Err(FiniteTvTransportError::new(
                "request.independent_calibration_target_samples",
                "v1 requires independently sampled calibration and target reference streams",
            ));
        }
        if !self.reference_samples_independent_of_confirmatory_stream {
            return Err(FiniteTvTransportError::new(
                "request.reference_samples_independent_of_confirmatory_stream",
                "calibration and target reference samples must be independent of the confirmatory stream for the v1 alpha-composition guarantee",
            ));
        }
        if !self.target_sample_pre_outcome_only {
            return Err(FiniteTvTransportError::new(
                "request.target_sample_pre_outcome_only",
                "target reference sample must not reuse confirmatory outcomes",
            ));
        }
        Ok(())
    }

    pub fn request_sha256(&self) -> Result<String, FiniteTvTransportError> {
        self.validate()?;
        digest(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FiniteTvTransportStatus {
    Informative,
    Vacuous,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteTvTwoSampleTransportReport {
    pub schema: String,
    pub version: String,
    pub transport_id: String,
    pub request_sha256: String,
    pub functional_contract_sha256: String,
    pub calibration_design_sha256: String,
    pub target_design_sha256: String,
    pub path_support_sha256: String,
    pub support_size: usize,
    pub calibration_sample_data_sha256: String,
    pub target_sample_data_sha256: String,
    pub calibration_sample_size: u64,
    pub target_sample_size: u64,
    pub empirical_total_variation_distance: f64,
    pub calibration_concentration_slack: f64,
    pub target_concentration_slack: f64,
    pub calibration_failure_probability: f64,
    pub target_failure_probability: f64,
    pub independent_calibration_target_samples: bool,
    pub reference_samples_independent_of_confirmatory_stream: bool,
    /// Union-bound failure probability for the two-sample TV transport event.
    pub transport_failure_probability: f64,
    /// min(1, empirical TV + calibration slack + target slack).
    pub total_variation_radius: f64,
    pub status: FiniteTvTransportStatus,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    transport_id: &'a str,
    request_sha256: &'a str,
    functional_contract_sha256: &'a str,
    calibration_design_sha256: &'a str,
    target_design_sha256: &'a str,
    path_support_sha256: &'a str,
    support_size: usize,
    calibration_sample_data_sha256: &'a str,
    target_sample_data_sha256: &'a str,
    calibration_sample_size: u64,
    target_sample_size: u64,
    empirical_total_variation_distance: f64,
    calibration_concentration_slack: f64,
    target_concentration_slack: f64,
    calibration_failure_probability: f64,
    target_failure_probability: f64,
    independent_calibration_target_samples: bool,
    reference_samples_independent_of_confirmatory_stream: bool,
    transport_failure_probability: f64,
    total_variation_radius: f64,
    status: FiniteTvTransportStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportedExpectationInterval {
    pub functional_lower_bound: f64,
    pub functional_upper_bound: f64,
    pub calibration_lower_confidence_bound: f64,
    pub calibration_upper_confidence_bound: f64,
    pub maximum_absolute_expectation_shift: f64,
    pub target_lower_confidence_bound: f64,
    pub target_upper_confidence_bound: f64,
    pub transport_failure_probability: f64,
    pub transport_report_sha256: String,
}

fn validate_open_probability(value: f64, path: &str) -> Result<(), FiniteTvTransportError> {
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(FiniteTvTransportError::new(
            path,
            "failure probability must be finite and strictly between zero and one",
        ));
    }
    Ok(())
}

fn validate_hash(value: &str, path: &str) -> Result<(), FiniteTvTransportError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(FiniteTvTransportError::new(
            path,
            "expected a 64-character hexadecimal SHA-256 digest",
        ));
    }
    Ok(())
}

fn validate_sample(sample: &FiniteTvLawSample, path: &str) -> Result<(), FiniteTvTransportError> {
    validate_hash(
        &sample.path_support_sha256,
        &format!("{path}.path_support_sha256"),
    )?;
    validate_hash(
        &sample.sample_data_sha256,
        &format!("{path}.sample_data_sha256"),
    )?;
    if sample.observed_path_counts.is_empty()
        || sample.observed_path_counts.len() > MAX_SUPPORT_SIZE
    {
        return Err(FiniteTvTransportError::new(
            format!("{path}.observed_path_counts"),
            format!("support must contain 1..={MAX_SUPPORT_SIZE} categories"),
        ));
    }
    let total = checked_sample_size(&sample.observed_path_counts, path)?;
    if total == 0 {
        return Err(FiniteTvTransportError::new(
            format!("{path}.observed_path_counts"),
            "sample size must be positive",
        ));
    }
    Ok(())
}

fn checked_sample_size(counts: &[u64], path: &str) -> Result<u64, FiniteTvTransportError> {
    let total = counts.iter().try_fold(0_u64, |sum, count| {
        sum.checked_add(*count).ok_or_else(|| {
            FiniteTvTransportError::new(
                format!("{path}.observed_path_counts"),
                "sample count overflow",
            )
        })
    })?;
    if total > MAX_EXACT_F64_INTEGER {
        return Err(FiniteTvTransportError::new(
            format!("{path}.observed_path_counts"),
            "sample size exceeds exact f64 integer conversion supported by this reference implementation",
        ));
    }
    Ok(total)
}

fn weissman_tv_slack(
    support_size: usize,
    sample_size: u64,
    failure_probability: f64,
) -> Result<f64, FiniteTvTransportError> {
    if support_size == 1 {
        return Ok(0.0);
    }
    let numerator = (support_size as f64) * LN_2 - failure_probability.ln();
    let slack = (numerator / (2.0 * sample_size as f64)).sqrt().min(1.0);
    if !slack.is_finite() {
        return Err(FiniteTvTransportError::new(
            "report.concentration_slack",
            "non-finite Weissman concentration slack",
        ));
    }
    Ok(slack)
}

fn empirical_tv(
    calibration_counts: &[u64],
    target_counts: &[u64],
    calibration_sample_size: u64,
    target_sample_size: u64,
) -> f64 {
    let mut l1 = 0.0;
    for (calibration_count, target_count) in
        calibration_counts.iter().zip(target_counts)
    {
        let calibration_probability =
            (*calibration_count as f64) / (calibration_sample_size as f64);
        let target_probability = (*target_count as f64) / (target_sample_size as f64);
        l1 += (calibration_probability - target_probability).abs();
    }
    (0.5 * l1).min(1.0)
}

pub fn calibrate_finite_tv_two_sample_transport(
    input: &FiniteTvTwoSampleTransportRequest,
) -> Result<FiniteTvTwoSampleTransportReport, FiniteTvTransportError> {
    let request = input.clone();
    request.validate()?;
    let calibration_sample_size = checked_sample_size(
        &request.calibration_sample.observed_path_counts,
        "request.calibration_sample",
    )?;
    let target_sample_size = checked_sample_size(
        &request.target_sample.observed_path_counts,
        "request.target_sample",
    )?;
    let support_size = request.calibration_sample.observed_path_counts.len();
    let empirical_total_variation_distance = empirical_tv(
        &request.calibration_sample.observed_path_counts,
        &request.target_sample.observed_path_counts,
        calibration_sample_size,
        target_sample_size,
    );
    let calibration_concentration_slack = weissman_tv_slack(
        support_size,
        calibration_sample_size,
        request.calibration_failure_probability,
    )?;
    let target_concentration_slack = weissman_tv_slack(
        support_size,
        target_sample_size,
        request.target_failure_probability,
    )?;
    let total_variation_radius = (empirical_total_variation_distance
        + calibration_concentration_slack
        + target_concentration_slack)
        .min(1.0);
    let transport_failure_probability =
        request.calibration_failure_probability + request.target_failure_probability;
    let status = if total_variation_radius >= 1.0 {
        FiniteTvTransportStatus::Vacuous
    } else {
        FiniteTvTransportStatus::Informative
    };
    let request_sha256 = request.request_sha256()?;
    let mut report = FiniteTvTwoSampleTransportReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        transport_id: request.transport_id,
        request_sha256,
        functional_contract_sha256: request.functional_contract_sha256,
        calibration_design_sha256: request.calibration_design_sha256,
        target_design_sha256: request.target_design_sha256,
        path_support_sha256: request.calibration_sample.path_support_sha256,
        support_size,
        calibration_sample_data_sha256: request.calibration_sample.sample_data_sha256,
        target_sample_data_sha256: request.target_sample.sample_data_sha256,
        calibration_sample_size,
        target_sample_size,
        empirical_total_variation_distance,
        calibration_concentration_slack,
        target_concentration_slack,
        calibration_failure_probability: request.calibration_failure_probability,
        target_failure_probability: request.target_failure_probability,
        independent_calibration_target_samples: request.independent_calibration_target_samples,
        reference_samples_independent_of_confirmatory_stream: request
            .reference_samples_independent_of_confirmatory_stream,
        transport_failure_probability,
        total_variation_radius,
        status,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

pub fn verify_finite_tv_two_sample_transport_report(
    request: &FiniteTvTwoSampleTransportRequest,
    report: &FiniteTvTwoSampleTransportReport,
) -> Result<(), FiniteTvTransportError> {
    let expected = calibrate_finite_tv_two_sample_transport(request)?;
    if expected != *report {
        return Err(FiniteTvTransportError::new(
            "report",
            "stored two-sample transport report differs from deterministic reconstruction",
        ));
    }
    let expected_digest = report_digest(report)?;
    if expected_digest != report.report_sha256 {
        return Err(FiniteTvTransportError::new(
            "report.report_sha256",
            "transport report digest does not match canonical payload",
        ));
    }
    Ok(())
}

/// Expand a calibration-law confidence interval to the target law.
///
/// If g has known range [a, b], this adds (b-a) * TV(P_cal, P_target) to
/// both sides and clips back to [a, b]. The returned failure probability is
/// only the transport event; callers must add the functional calibration
/// failure probability and any other relevant events by union bound.
pub fn transport_bounded_expectation_interval(
    request: &FiniteTvTwoSampleTransportRequest,
    report: &FiniteTvTwoSampleTransportReport,
    functional_lower_bound: f64,
    functional_upper_bound: f64,
    calibration_lower_confidence_bound: f64,
    calibration_upper_confidence_bound: f64,
) -> Result<TransportedExpectationInterval, FiniteTvTransportError> {
    verify_finite_tv_two_sample_transport_report(request, report)?;
    if !functional_lower_bound.is_finite()
        || !functional_upper_bound.is_finite()
        || functional_lower_bound > functional_upper_bound
    {
        return Err(FiniteTvTransportError::new(
            "functional_range",
            "functional range must be finite and ordered",
        ));
    }
    if !calibration_lower_confidence_bound.is_finite()
        || !calibration_upper_confidence_bound.is_finite()
        || calibration_lower_confidence_bound > calibration_upper_confidence_bound
        || calibration_lower_confidence_bound
            < functional_lower_bound - COUNT_SUM_TOLERANCE
        || calibration_upper_confidence_bound
            > functional_upper_bound + COUNT_SUM_TOLERANCE
    {
        return Err(FiniteTvTransportError::new(
            "calibration_interval",
            "calibration interval must be finite, ordered, and contained in the registered functional range",
        ));
    }
    let functional_range_width = functional_upper_bound - functional_lower_bound;
    if !functional_range_width.is_finite() {
        return Err(FiniteTvTransportError::new(
            "functional_range",
            "functional range width overflowed finite f64 arithmetic",
        ));
    }
    let maximum_absolute_expectation_shift =
        functional_range_width * report.total_variation_radius;
    if !maximum_absolute_expectation_shift.is_finite() {
        return Err(FiniteTvTransportError::new(
            "transported_interval",
            "transport shift overflowed finite f64 arithmetic",
        ));
    }
    let target_lower_confidence_bound = (calibration_lower_confidence_bound
        - maximum_absolute_expectation_shift)
        .max(functional_lower_bound);
    let target_upper_confidence_bound = (calibration_upper_confidence_bound
        + maximum_absolute_expectation_shift)
        .min(functional_upper_bound);
    if !target_lower_confidence_bound.is_finite()
        || !target_upper_confidence_bound.is_finite()
        || target_lower_confidence_bound > target_upper_confidence_bound
    {
        return Err(FiniteTvTransportError::new(
            "transported_interval",
            "transported confidence interval is non-finite or inverted",
        ));
    }
    Ok(TransportedExpectationInterval {
        functional_lower_bound,
        functional_upper_bound,
        calibration_lower_confidence_bound,
        calibration_upper_confidence_bound,
        maximum_absolute_expectation_shift,
        target_lower_confidence_bound,
        target_upper_confidence_bound,
        transport_failure_probability: report.transport_failure_probability,
        transport_report_sha256: report.report_sha256.clone(),
    })
}

fn report_digest(report: &FiniteTvTwoSampleTransportReport) -> Result<String, FiniteTvTransportError> {
    let payload = ReportPayload {
        schema: &report.schema,
        version: &report.version,
        transport_id: &report.transport_id,
        request_sha256: &report.request_sha256,
        functional_contract_sha256: &report.functional_contract_sha256,
        calibration_design_sha256: &report.calibration_design_sha256,
        target_design_sha256: &report.target_design_sha256,
        path_support_sha256: &report.path_support_sha256,
        support_size: report.support_size,
        calibration_sample_data_sha256: &report.calibration_sample_data_sha256,
        target_sample_data_sha256: &report.target_sample_data_sha256,
        calibration_sample_size: report.calibration_sample_size,
        target_sample_size: report.target_sample_size,
        empirical_total_variation_distance: report.empirical_total_variation_distance,
        calibration_concentration_slack: report.calibration_concentration_slack,
        target_concentration_slack: report.target_concentration_slack,
        calibration_failure_probability: report.calibration_failure_probability,
        target_failure_probability: report.target_failure_probability,
        independent_calibration_target_samples: report.independent_calibration_target_samples,
        reference_samples_independent_of_confirmatory_stream: report
            .reference_samples_independent_of_confirmatory_stream,
        transport_failure_probability: report.transport_failure_probability,
        total_variation_radius: report.total_variation_radius,
        status: report.status,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| FiniteTvTransportError::new("serialization", error.to_string()))?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUPPORT_HASH: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const FUNCTION_HASH: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const CALIBRATION_DATA_HASH: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const TARGET_DATA_HASH: &str =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const CALIBRATION_DESIGN_HASH: &str =
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const TARGET_DESIGN_HASH: &str =
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn sample(counts: Vec<u64>, data_hash: &str) -> FiniteTvLawSample {
        FiniteTvLawSample {
            path_support_sha256: SUPPORT_HASH.into(),
            sample_data_sha256: data_hash.into(),
            observed_path_counts: counts,
        }
    }

    fn request(
        calibration_counts: Vec<u64>,
        target_counts: Vec<u64>,
    ) -> FiniteTvTwoSampleTransportRequest {
        FiniteTvTwoSampleTransportRequest::new(
            "two-sample-transport-001",
            FUNCTION_HASH,
            CALIBRATION_DESIGN_HASH,
            TARGET_DESIGN_HASH,
            sample(calibration_counts, CALIBRATION_DATA_HASH),
            sample(target_counts, TARGET_DATA_HASH),
            0.005,
            0.005,
            true,
            true,
            true,
            true,
            true,
        )
        .unwrap()
    }

    #[test]
    fn two_sample_radius_adds_empirical_tv_and_both_weissman_slacks() {
        let request = request(vec![80, 20], vec![50, 50]);
        let report = calibrate_finite_tv_two_sample_transport(&request).unwrap();
        assert!((report.empirical_total_variation_distance - 0.30).abs() < 1e-12);
        let expected_calibration_slack =
            ((2.0 * LN_2 - 0.005_f64.ln()) / (2.0 * 100.0)).sqrt();
        assert!((report.calibration_concentration_slack - expected_calibration_slack).abs() < 1e-12);
        assert_eq!(report.transport_failure_probability, 0.01);
        assert!(report.total_variation_radius < 1.0);
        assert_eq!(report.status, FiniteTvTransportStatus::Informative);
        verify_finite_tv_two_sample_transport_report(&request, &report).unwrap();
    }

    #[test]
    fn one_category_support_has_exact_zero_transport_radius() {
        let request = request(vec![100], vec![17]);
        let report = calibrate_finite_tv_two_sample_transport(&request).unwrap();
        assert_eq!(report.empirical_total_variation_distance, 0.0);
        assert_eq!(report.calibration_concentration_slack, 0.0);
        assert_eq!(report.target_concentration_slack, 0.0);
        assert_eq!(report.total_variation_radius, 0.0);
    }

    #[test]
    fn bounded_expectation_interval_expands_by_range_width_times_tv() {
        let request = request(vec![80, 20], vec![50, 50]);
        let report = calibrate_finite_tv_two_sample_transport(&request).unwrap();
        let transported = transport_bounded_expectation_interval(
            &request,
            &report,
            2.0,
            6.0,
            3.0,
            4.0,
        )
        .unwrap();
        let expected_shift = 4.0 * report.total_variation_radius;
        assert!((transported.maximum_absolute_expectation_shift - expected_shift).abs() < 1e-12);
        assert!((transported.target_lower_confidence_bound - (3.0 - expected_shift).max(2.0)).abs() < 1e-12);
        assert!((transported.target_upper_confidence_bound - (4.0 + expected_shift).min(6.0)).abs() < 1e-12);
        assert_eq!(transported.transport_failure_probability, 0.01);
    }

    #[test]
    fn rejects_support_mismatch_and_sampling_contract_failures() {
        let mut mismatch = request(vec![80, 20], vec![50, 50]);
        mismatch.target_sample.path_support_sha256 =
            "1111111111111111111111111111111111111111111111111111111111111111".into();
        assert!(mismatch.validate().is_err());

        let mut serial = request(vec![80, 20], vec![50, 50]);
        serial.iid_target_complete_path_draws = false;
        assert!(serial.validate().is_err());

        let mut reused = request(vec![80, 20], vec![50, 50]);
        reused.target_sample_pre_outcome_only = false;
        assert!(reused.validate().is_err());
    }

    #[test]
    fn rejects_nonindependent_calibration_target_samples() {
        let mut request = request(vec![80, 20], vec![50, 50]);
        request.independent_calibration_target_samples = false;
        assert!(request.validate().is_err());
    }

    #[test]
    fn rejects_non_iid_calibration_sample() {
        let mut request = request(vec![80, 20], vec![50, 50]);
        request.iid_calibration_complete_path_draws = false;
        assert!(request.validate().is_err());
    }

    #[test]
    fn rejects_overflowing_functional_range_instead_of_emitting_infinite_bounds() {
        let request = request(vec![80, 20], vec![50, 50]);
        let report = calibrate_finite_tv_two_sample_transport(&request).unwrap();
        assert!(transport_bounded_expectation_interval(
            &request,
            &report,
            -f64::MAX,
            f64::MAX,
            -1.0,
            1.0,
        )
        .is_err());
    }

    #[test]
    fn rejects_invalid_error_allocations_and_empty_samples() {
        let mut bad_delta = request(vec![80, 20], vec![50, 50]);
        bad_delta.calibration_failure_probability = 0.7;
        bad_delta.target_failure_probability = 0.4;
        assert!(bad_delta.validate().is_err());

        let mut empty = request(vec![80, 20], vec![50, 50]);
        empty.target_sample.observed_path_counts = vec![0, 0];
        assert!(empty.validate().is_err());
    }

    #[test]
    fn caps_vacuous_high_dimensional_radius_and_marks_it_vacuous() {
        let support_size = 64;
        let request = request(
            std::iter::repeat_n(1_u64, support_size).collect(),
            std::iter::repeat_n(1_u64, support_size).collect(),
        );
        let report = calibrate_finite_tv_two_sample_transport(&request).unwrap();
        assert_eq!(report.total_variation_radius, 1.0);
        assert_eq!(report.status, FiniteTvTransportStatus::Vacuous);
    }

    #[test]
    fn rejects_tampered_report_and_out_of_range_source_interval() {
        let request = request(vec![80, 20], vec![50, 50]);
        let report = calibrate_finite_tv_two_sample_transport(&request).unwrap();
        let mut tampered = report.clone();
        tampered.total_variation_radius *= 0.5;
        assert!(verify_finite_tv_two_sample_transport_report(&request, &tampered).is_err());
        assert!(transport_bounded_expectation_interval(
            &request,
            &report,
            0.0,
            1.0,
            -0.1,
            0.7,
        )
        .is_err());
    }
}
