//! Direct simultaneous calibration-to-target transport for a frozen family
//! of bounded decision functionals on a shared finite support.
//!
//! Unlike full-distribution TV transport, this module covers only the exact
//! registered functional roster. For each function, two-sided Hoeffding bounds
//! are allocated across both independent samples and all registered functions.
//! The resulting shift interval is simultaneous over that frozen family.
//!
//! This can be far tighter than a finite-support TV radius when K is large and
//! the functional family is small. It does not cover omitted functions and
//! must never be described as a full-distribution transport certificate.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, error::Error, fmt};

const REQUEST_SCHEMA: &str =
    "symthaea.architecture-engineering.direct-functional-transport-request";
const REQUEST_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str =
    "symthaea.architecture-engineering.direct-functional-transport-report";
const REPORT_VERSION: &str = "0.1.0";
const MAX_SUPPORT_SIZE: usize = 100_000;
const MAX_FUNCTIONAL_COUNT: usize = 10_000;
const MAX_FUNCTIONAL_VALUE_CELLS: usize = 10_000_000;
const MAX_EXACT_F64_INTEGER: u64 = 9_007_199_254_740_992;
const LN_2: f64 = std::f64::consts::LN_2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectFunctionalTransportError {
    pub path: String,
    pub message: String,
}

impl DirectFunctionalTransportError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for DirectFunctionalTransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for DirectFunctionalTransportError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectFunctionalLawSample {
    pub path_support_sha256: String,
    pub sample_data_sha256: String,
    pub observed_path_counts: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredBoundedFunctional {
    pub functional_id: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
    /// Frozen g(path_category) values in the exact support category order.
    pub category_values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectFunctionalTransportRequest {
    pub schema: String,
    pub version: String,
    pub transport_id: String,
    /// Hash of the exact support digest and sorted registered function roster.
    pub functional_contract_sha256: String,
    pub calibration_design_sha256: String,
    pub target_design_sha256: String,
    pub calibration_sample: DirectFunctionalLawSample,
    pub target_sample: DirectFunctionalLawSample,
    /// Family-wide failure allocation covering both samples and all functions.
    pub family_failure_probability: f64,
    pub iid_calibration_complete_path_draws: bool,
    pub iid_target_complete_path_draws: bool,
    pub independent_calibration_target_samples: bool,
    pub reference_samples_independent_of_confirmatory_stream: bool,
    pub target_sample_pre_outcome_only: bool,
    pub registered_functionals: Vec<RegisteredBoundedFunctional>,
}

impl DirectFunctionalTransportRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        transport_id: impl Into<String>,
        calibration_design_sha256: impl Into<String>,
        target_design_sha256: impl Into<String>,
        calibration_sample: DirectFunctionalLawSample,
        target_sample: DirectFunctionalLawSample,
        family_failure_probability: f64,
        iid_calibration_complete_path_draws: bool,
        iid_target_complete_path_draws: bool,
        independent_calibration_target_samples: bool,
        reference_samples_independent_of_confirmatory_stream: bool,
        target_sample_pre_outcome_only: bool,
        mut registered_functionals: Vec<RegisteredBoundedFunctional>,
    ) -> Result<Self, DirectFunctionalTransportError> {
        registered_functionals.sort_by(|a, b| a.functional_id.cmp(&b.functional_id));
        let functional_contract_sha256 = functional_family_sha256(
            &calibration_sample.path_support_sha256,
            &registered_functionals,
        )?;
        let request = Self {
            schema: REQUEST_SCHEMA.into(),
            version: REQUEST_VERSION.into(),
            transport_id: transport_id.into(),
            functional_contract_sha256,
            calibration_design_sha256: calibration_design_sha256.into(),
            target_design_sha256: target_design_sha256.into(),
            calibration_sample,
            target_sample,
            family_failure_probability,
            iid_calibration_complete_path_draws,
            iid_target_complete_path_draws,
            independent_calibration_target_samples,
            reference_samples_independent_of_confirmatory_stream,
            target_sample_pre_outcome_only,
            registered_functionals,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), DirectFunctionalTransportError> {
        if self.schema != REQUEST_SCHEMA || self.version != REQUEST_VERSION {
            return Err(DirectFunctionalTransportError::new(
                "request.schema",
                "unsupported direct-functional transport schema/version",
            ));
        }
        if self.transport_id.trim().is_empty() {
            return Err(DirectFunctionalTransportError::new(
                "request.transport_id",
                "transport ID must be non-empty",
            ));
        }
        validate_hash(
            &self.calibration_design_sha256,
            "request.calibration_design_sha256",
        )?;
        validate_hash(&self.target_design_sha256, "request.target_design_sha256")?;
        validate_sample(&self.calibration_sample, "request.calibration_sample")?;
        validate_sample(&self.target_sample, "request.target_sample")?;
        if self.calibration_sample.path_support_sha256
            != self.target_sample.path_support_sha256
            || self.calibration_sample.observed_path_counts.len()
                != self.target_sample.observed_path_counts.len()
        {
            return Err(DirectFunctionalTransportError::new(
                "request.path_support",
                "calibration and target counts must use the same frozen support digest, size, and category order",
            ));
        }
        validate_open_probability(
            self.family_failure_probability,
            "request.family_failure_probability",
        )?;
        if !self.iid_calibration_complete_path_draws {
            return Err(DirectFunctionalTransportError::new(
                "request.iid_calibration_complete_path_draws",
                "calibration Hoeffding bounds require IID complete-path draws",
            ));
        }
        if !self.iid_target_complete_path_draws {
            return Err(DirectFunctionalTransportError::new(
                "request.iid_target_complete_path_draws",
                "target Hoeffding bounds require IID complete-path draws",
            ));
        }
        if !self.independent_calibration_target_samples {
            return Err(DirectFunctionalTransportError::new(
                "request.independent_calibration_target_samples",
                "v1 requires independent calibration and target reference samples",
            ));
        }
        if !self.reference_samples_independent_of_confirmatory_stream {
            return Err(DirectFunctionalTransportError::new(
                "request.reference_samples_independent_of_confirmatory_stream",
                "both reference samples must be independent of the confirmatory stream",
            ));
        }
        if !self.target_sample_pre_outcome_only {
            return Err(DirectFunctionalTransportError::new(
                "request.target_sample_pre_outcome_only",
                "target reference counts must not reuse confirmatory outcomes",
            ));
        }
        if self.registered_functionals.is_empty()
            || self.registered_functionals.len() > MAX_FUNCTIONAL_COUNT
        {
            return Err(DirectFunctionalTransportError::new(
                "request.registered_functionals",
                format!("functional roster must contain 1..={MAX_FUNCTIONAL_COUNT} entries"),
            ));
        }
        let support_size = self.calibration_sample.observed_path_counts.len();
        let value_cells = support_size
            .checked_mul(self.registered_functionals.len())
            .ok_or_else(|| DirectFunctionalTransportError::new(
                "request.registered_functionals",
                "support-by-functional value matrix size overflowed",
            ))?;
        if value_cells > MAX_FUNCTIONAL_VALUE_CELLS {
            return Err(DirectFunctionalTransportError::new(
                "request.registered_functionals",
                format!("support-by-functional value matrix exceeds {MAX_FUNCTIONAL_VALUE_CELLS} cells"),
            ));
        }
        if self.registered_functionals.windows(2).any(|pair| {
            pair[0].functional_id.as_str() >= pair[1].functional_id.as_str()
        }) {
            return Err(DirectFunctionalTransportError::new(
                "request.registered_functionals",
                "functional roster must be strictly sorted by unique functional ID",
            ));
        }
        let mut ids = BTreeSet::new();
        for (index, functional) in self.registered_functionals.iter().enumerate() {
            let path = format!("request.registered_functionals[{index}]");
            if functional.functional_id.trim().is_empty()
                || !ids.insert(functional.functional_id.as_str())
            {
                return Err(DirectFunctionalTransportError::new(
                    format!("{path}.functional_id"),
                    "functional IDs must be non-empty and unique",
                ));
            }
            if !functional.lower_bound.is_finite()
                || !functional.upper_bound.is_finite()
                || functional.lower_bound > functional.upper_bound
            {
                return Err(DirectFunctionalTransportError::new(
                    format!("{path}.range"),
                    "functional range must be finite and ordered",
                ));
            }
            let width = functional.upper_bound - functional.lower_bound;
            if !width.is_finite() {
                return Err(DirectFunctionalTransportError::new(
                    format!("{path}.range"),
                    "functional range width must be finite",
                ));
            }
            if functional.category_values.len() != support_size {
                return Err(DirectFunctionalTransportError::new(
                    format!("{path}.category_values"),
                    "one function value is required for every frozen support category",
                ));
            }
            for (category, value) in functional.category_values.iter().enumerate() {
                if !value.is_finite()
                    || *value < functional.lower_bound
                    || *value > functional.upper_bound
                {
                    return Err(DirectFunctionalTransportError::new(
                        format!("{path}.category_values[{category}]"),
                        "category value must be finite and lie inside the registered range",
                    ));
                }
            }
        }
        let expected_contract = functional_family_sha256(
            &self.calibration_sample.path_support_sha256,
            &self.registered_functionals,
        )?;
        if expected_contract != self.functional_contract_sha256 {
            return Err(DirectFunctionalTransportError::new(
                "request.functional_contract_sha256",
                "functional-family digest does not match the exact support, IDs, ranges, and category values",
            ));
        }
        Ok(())
    }

    pub fn request_sha256(&self) -> Result<String, DirectFunctionalTransportError> {
        self.validate()?;
        digest(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectFunctionalStatus {
    Informative,
    Vacuous,
    ExactConstant,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectFunctionalShiftBound {
    pub functional_id: String,
    pub functional_lower_bound: f64,
    pub functional_upper_bound: f64,
    pub calibration_empirical_mean: f64,
    pub target_empirical_mean: f64,
    pub empirical_target_minus_calibration_shift: f64,
    pub calibration_hoeffding_radius: f64,
    pub target_hoeffding_radius: f64,
    pub simultaneous_row_failure_probability: f64,
    pub shift_lower_confidence_bound: f64,
    pub shift_upper_confidence_bound: f64,
    pub status: DirectFunctionalStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectFunctionalTransportReport {
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
    pub family_failure_probability: f64,
    pub functional_bounds: Vec<DirectFunctionalShiftBound>,
    pub all_functionals_informative: bool,
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
    family_failure_probability: f64,
    functional_bounds: &'a [DirectFunctionalShiftBound],
    all_functionals_informative: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectTransportedExpectationInterval {
    pub functional_id: String,
    pub functional_contract_sha256: String,
    pub calibration_interval_source_sha256: String,
    pub calibration_lower_confidence_bound: f64,
    pub calibration_upper_confidence_bound: f64,
    pub shift_lower_confidence_bound: f64,
    pub shift_upper_confidence_bound: f64,
    pub target_lower_confidence_bound: f64,
    pub target_upper_confidence_bound: f64,
    pub family_failure_probability: f64,
    pub transport_report_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectTransportAdjustedNullErrorBudget {
    pub nominal_alpha: f64,
    pub calibration_null_failure_probability: f64,
    pub transport_family_failure_probability: f64,
    pub other_null_failure_probability: f64,
    pub total_null_failure_probability: f64,
    pub conditional_alpha: f64,
    pub transport_report_sha256: String,
    pub functional_contract_sha256: String,
}

fn validate_open_probability(value: f64, path: &str) -> Result<(), DirectFunctionalTransportError> {
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(DirectFunctionalTransportError::new(
            path,
            "failure probability must be finite and strictly between zero and one",
        ));
    }
    Ok(())
}

fn validate_hash(value: &str, path: &str) -> Result<(), DirectFunctionalTransportError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DirectFunctionalTransportError::new(
            path,
            "expected a 64-character hexadecimal SHA-256 digest",
        ));
    }
    Ok(())
}

fn validate_sample(
    sample: &DirectFunctionalLawSample,
    path: &str,
) -> Result<(), DirectFunctionalTransportError> {
    validate_hash(&sample.path_support_sha256, &format!("{path}.path_support_sha256"))?;
    validate_hash(&sample.sample_data_sha256, &format!("{path}.sample_data_sha256"))?;
    if sample.observed_path_counts.is_empty()
        || sample.observed_path_counts.len() > MAX_SUPPORT_SIZE
    {
        return Err(DirectFunctionalTransportError::new(
            format!("{path}.observed_path_counts"),
            format!("support must contain 1..={MAX_SUPPORT_SIZE} categories"),
        ));
    }
    if checked_sample_size(&sample.observed_path_counts, path)? == 0 {
        return Err(DirectFunctionalTransportError::new(
            format!("{path}.observed_path_counts"),
            "sample size must be positive",
        ));
    }
    Ok(())
}

fn checked_sample_size(
    counts: &[u64],
    path: &str,
) -> Result<u64, DirectFunctionalTransportError> {
    let total = counts.iter().try_fold(0_u64, |sum, count| {
        sum.checked_add(*count).ok_or_else(|| {
            DirectFunctionalTransportError::new(
                format!("{path}.observed_path_counts"),
                "sample count overflow",
            )
        })
    })?;
    if total > MAX_EXACT_F64_INTEGER {
        return Err(DirectFunctionalTransportError::new(
            format!("{path}.observed_path_counts"),
            "sample size exceeds exact f64 integer conversion supported by this reference implementation",
        ));
    }
    Ok(total)
}

fn functional_family_sha256(
    support_sha256: &str,
    functionals: &[RegisteredBoundedFunctional],
) -> Result<String, DirectFunctionalTransportError> {
    validate_hash(support_sha256, "functional_family.path_support_sha256")?;
    let mut canonical = functionals.to_vec();
    canonical.sort_by(|a, b| a.functional_id.cmp(&b.functional_id));
    digest(&(support_sha256, canonical))
}

fn empirical_mean(
    counts: &[u64],
    sample_size: u64,
    values: &[f64],
    path: &str,
) -> Result<f64, DirectFunctionalTransportError> {
    let mut total = 0.0;
    for (count, value) in counts.iter().zip(values) {
        let probability = (*count as f64) / (sample_size as f64);
        total += probability * *value;
    }
    if !total.is_finite() {
        return Err(DirectFunctionalTransportError::new(
            path,
            "empirical functional mean overflowed finite f64 arithmetic",
        ));
    }
    Ok(total)
}

fn report_digest(
    report: &DirectFunctionalTransportReport,
) -> Result<String, DirectFunctionalTransportError> {
    digest(&ReportPayload {
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
        family_failure_probability: report.family_failure_probability,
        functional_bounds: &report.functional_bounds,
        all_functionals_informative: report.all_functionals_informative,
    })
}

fn digest<T: Serialize>(value: &T) -> Result<String, DirectFunctionalTransportError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| DirectFunctionalTransportError::new("serialization", error.to_string()))?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

pub fn calibrate_direct_functional_transport(
    input: &DirectFunctionalTransportRequest,
) -> Result<DirectFunctionalTransportReport, DirectFunctionalTransportError> {
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
    let functional_count = request.registered_functionals.len();
    let row_delta = request.family_failure_probability / (2.0 * functional_count as f64);
    if !row_delta.is_finite() || row_delta <= 0.0 {
        return Err(DirectFunctionalTransportError::new(
            "request.family_failure_probability",
            "per-sample/per-functional failure allocation underflowed to zero",
        ));
    }
    let mut bounds = Vec::with_capacity(functional_count);
    let mut all_functionals_informative = true;
    for functional in &request.registered_functionals {
        let width = functional.upper_bound - functional.lower_bound;
        let calibration_mean = empirical_mean(
            &request.calibration_sample.observed_path_counts,
            calibration_sample_size,
            &functional.category_values,
            "calibration_empirical_mean",
        )?;
        let target_mean = empirical_mean(
            &request.target_sample.observed_path_counts,
            target_sample_size,
            &functional.category_values,
            "target_empirical_mean",
        )?;
        let empirical_shift = target_mean - calibration_mean;
        if !empirical_shift.is_finite() {
            return Err(DirectFunctionalTransportError::new(
                "empirical_shift",
                "target-minus-calibration empirical shift overflowed",
            ));
        }
        let calibration_radius = if width == 0.0 {
            0.0
        } else {
            width * ((2.0 / row_delta).ln() / (2.0 * calibration_sample_size as f64)).sqrt()
        };
        let target_radius = if width == 0.0 {
            0.0
        } else {
            width * ((2.0 / row_delta).ln() / (2.0 * target_sample_size as f64)).sqrt()
        };
        if !calibration_radius.is_finite() || !target_radius.is_finite() {
            return Err(DirectFunctionalTransportError::new(
                "hoeffding_radius",
                "Hoeffding radius overflowed finite f64 arithmetic",
            ));
        }
        let combined_radius = (calibration_radius + target_radius).min(width);
        let raw_lower = empirical_shift - combined_radius;
        let raw_upper = empirical_shift + combined_radius;
        if !raw_lower.is_finite() || !raw_upper.is_finite() {
            return Err(DirectFunctionalTransportError::new(
                "shift_interval",
                "simultaneous shift interval overflowed finite f64 arithmetic",
            ));
        }
        let shift_lower = raw_lower.max(-width);
        let shift_upper = raw_upper.min(width);
        if shift_lower > shift_upper {
            return Err(DirectFunctionalTransportError::new(
                "shift_interval",
                "simultaneous shift interval is inverted",
            ));
        }
        let status = if width == 0.0 {
            DirectFunctionalStatus::ExactConstant
        } else if shift_lower <= -width && shift_upper >= width {
            all_functionals_informative = false;
            DirectFunctionalStatus::Vacuous
        } else {
            DirectFunctionalStatus::Informative
        };
        bounds.push(DirectFunctionalShiftBound {
            functional_id: functional.functional_id.clone(),
            functional_lower_bound: functional.lower_bound,
            functional_upper_bound: functional.upper_bound,
            calibration_empirical_mean: calibration_mean,
            target_empirical_mean: target_mean,
            empirical_target_minus_calibration_shift: empirical_shift,
            calibration_hoeffding_radius: calibration_radius,
            target_hoeffding_radius: target_radius,
            simultaneous_row_failure_probability: row_delta,
            shift_lower_confidence_bound: shift_lower,
            shift_upper_confidence_bound: shift_upper,
            status,
        });
    }
    let request_sha256 = request.request_sha256()?;
    let mut report = DirectFunctionalTransportReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        transport_id: request.transport_id,
        request_sha256,
        functional_contract_sha256: request.functional_contract_sha256,
        calibration_design_sha256: request.calibration_design_sha256,
        target_design_sha256: request.target_design_sha256,
        path_support_sha256: request.calibration_sample.path_support_sha256,
        support_size: request.calibration_sample.observed_path_counts.len(),
        calibration_sample_data_sha256: request.calibration_sample.sample_data_sha256,
        target_sample_data_sha256: request.target_sample.sample_data_sha256,
        calibration_sample_size,
        target_sample_size,
        family_failure_probability: request.family_failure_probability,
        functional_bounds: bounds,
        all_functionals_informative,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

pub fn verify_direct_functional_transport_report(
    request: &DirectFunctionalTransportRequest,
    report: &DirectFunctionalTransportReport,
) -> Result<(), DirectFunctionalTransportError> {
    let expected = calibrate_direct_functional_transport(request)?;
    if expected != *report {
        return Err(DirectFunctionalTransportError::new(
            "report",
            "stored direct-functional transport report differs from deterministic reconstruction",
        ));
    }
    if report_digest(report)? != report.report_sha256 {
        return Err(DirectFunctionalTransportError::new(
            "report.report_sha256",
            "direct-functional report digest does not match canonical payload",
        ));
    }
    Ok(())
}

pub fn transport_registered_functional_interval(
    request: &DirectFunctionalTransportRequest,
    report: &DirectFunctionalTransportReport,
    functional_id: &str,
    functional_contract_sha256: &str,
    calibration_interval_source_sha256: &str,
    calibration_lower_confidence_bound: f64,
    calibration_upper_confidence_bound: f64,
) -> Result<DirectTransportedExpectationInterval, DirectFunctionalTransportError> {
    verify_direct_functional_transport_report(request, report)?;
    validate_hash(functional_contract_sha256, "functional_contract_sha256")?;
    validate_hash(
        calibration_interval_source_sha256,
        "calibration_interval_source_sha256",
    )?;
    if functional_contract_sha256 != request.functional_contract_sha256 {
        return Err(DirectFunctionalTransportError::new(
            "functional_contract_sha256",
            "requested interval functional family does not match the frozen transport contract",
        ));
    }
    let bound = report.functional_bounds.iter()
        .find(|row| row.functional_id == functional_id)
        .ok_or_else(|| DirectFunctionalTransportError::new(
            "functional_id",
            "functional ID is not in the frozen registered family",
        ))?;
    if !calibration_lower_confidence_bound.is_finite()
        || !calibration_upper_confidence_bound.is_finite()
        || calibration_lower_confidence_bound > calibration_upper_confidence_bound
        || calibration_lower_confidence_bound < bound.functional_lower_bound
        || calibration_upper_confidence_bound > bound.functional_upper_bound
    {
        return Err(DirectFunctionalTransportError::new(
            "calibration_interval",
            "calibration interval must be finite, ordered, and inside the registered functional range",
        ));
    }
    let target_lower = (calibration_lower_confidence_bound
        + bound.shift_lower_confidence_bound)
        .max(bound.functional_lower_bound);
    let target_upper = (calibration_upper_confidence_bound
        + bound.shift_upper_confidence_bound)
        .min(bound.functional_upper_bound);
    if !target_lower.is_finite() || !target_upper.is_finite() || target_lower > target_upper {
        return Err(DirectFunctionalTransportError::new(
            "target_interval",
            "transported calibration interval is non-finite or empty",
        ));
    }
    Ok(DirectTransportedExpectationInterval {
        functional_id: functional_id.into(),
        functional_contract_sha256: request.functional_contract_sha256.clone(),
        calibration_interval_source_sha256: calibration_interval_source_sha256.into(),
        calibration_lower_confidence_bound,
        calibration_upper_confidence_bound,
        shift_lower_confidence_bound: bound.shift_lower_confidence_bound,
        shift_upper_confidence_bound: bound.shift_upper_confidence_bound,
        target_lower_confidence_bound: target_lower,
        target_upper_confidence_bound: target_upper,
        family_failure_probability: report.family_failure_probability,
        transport_report_sha256: report.report_sha256.clone(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn compose_direct_functional_null_error_budget(
    request: &DirectFunctionalTransportRequest,
    report: &DirectFunctionalTransportReport,
    functional_contract_sha256: &str,
    nominal_alpha: f64,
    calibration_null_failure_probability: f64,
    calibration_independent_of_confirmatory_stream: bool,
    other_null_failure_probability: f64,
    other_null_independent_of_confirmatory_stream: bool,
) -> Result<DirectTransportAdjustedNullErrorBudget, DirectFunctionalTransportError> {
    verify_direct_functional_transport_report(request, report)?;
    validate_hash(functional_contract_sha256, "budget.functional_contract_sha256")?;
    if functional_contract_sha256 != request.functional_contract_sha256 {
        return Err(DirectFunctionalTransportError::new(
            "budget.functional_contract_sha256",
            "calibration failure event does not match the frozen functional-family contract",
        ));
    }
    validate_open_probability(nominal_alpha, "budget.nominal_alpha")?;
    if !calibration_null_failure_probability.is_finite()
        || calibration_null_failure_probability < 0.0
        || calibration_null_failure_probability >= 1.0
        || !other_null_failure_probability.is_finite()
        || other_null_failure_probability < 0.0
        || other_null_failure_probability >= 1.0
    {
        return Err(DirectFunctionalTransportError::new(
            "budget.failure_probabilities",
            "null-side failure probabilities must be finite and lie in [0, 1)",
        ));
    }
    if !calibration_independent_of_confirmatory_stream {
        return Err(DirectFunctionalTransportError::new(
            "budget.calibration_independent_of_confirmatory_stream",
            "upstream calibration evidence must be independent of confirmation",
        ));
    }
    if other_null_failure_probability > 0.0 && !other_null_independent_of_confirmatory_stream {
        return Err(DirectFunctionalTransportError::new(
            "budget.other_null_independent_of_confirmatory_stream",
            "additional null-side evidence must be independent of confirmation",
        ));
    }
    let total = calibration_null_failure_probability
        + report.family_failure_probability
        + other_null_failure_probability;
    if !total.is_finite() || total >= nominal_alpha {
        return Err(DirectFunctionalTransportError::new(
            "budget.total_null_failure_probability",
            "combined null-side failure probability must be strictly below nominal alpha",
        ));
    }
    let conditional_alpha = (nominal_alpha - total) / (1.0 - total);
    if !conditional_alpha.is_finite() || !(0.0..1.0).contains(&conditional_alpha) {
        return Err(DirectFunctionalTransportError::new(
            "budget.conditional_alpha",
            "conditional alpha is non-finite or outside (0, 1)",
        ));
    }
    Ok(DirectTransportAdjustedNullErrorBudget {
        nominal_alpha,
        calibration_null_failure_probability,
        transport_family_failure_probability: report.family_failure_probability,
        other_null_failure_probability,
        total_null_failure_probability: total,
        conditional_alpha,
        transport_report_sha256: report.report_sha256.clone(),
        functional_contract_sha256: request.functional_contract_sha256.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUPPORT: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const CAL_DATA: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const TARGET_DATA: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const CAL_DESIGN: &str =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const TARGET_DESIGN: &str =
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const CAL_INTERVAL: &str =
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn sample(counts: Vec<u64>, source: &str) -> DirectFunctionalLawSample {
        DirectFunctionalLawSample {
            path_support_sha256: SUPPORT.into(),
            sample_data_sha256: source.into(),
            observed_path_counts: counts,
        }
    }

    fn functional(id: &str, values: Vec<f64>, lower: f64, upper: f64) -> RegisteredBoundedFunctional {
        RegisteredBoundedFunctional {
            functional_id: id.into(),
            lower_bound: lower,
            upper_bound: upper,
            category_values: values,
        }
    }

    fn request(
        calibration: Vec<u64>,
        target: Vec<u64>,
        functionals: Vec<RegisteredBoundedFunctional>,
    ) -> DirectFunctionalTransportRequest {
        DirectFunctionalTransportRequest::new(
            "direct-functional-transport-001",
            CAL_DESIGN,
            TARGET_DESIGN,
            sample(calibration, CAL_DATA),
            sample(target, TARGET_DATA),
            0.01,
            true,
            true,
            true,
            true,
            true,
            functionals,
        )
        .unwrap()
    }

    #[test]
    fn simultaneous_hoeffding_shift_bound_matches_analytic_formula() {
        let request = request(
            vec![80, 20],
            vec![50, 50],
            vec![functional("crash", vec![0.0, 1.0], 0.0, 1.0)],
        );
        let report = calibrate_direct_functional_transport(&request).unwrap();
        let row = &report.functional_bounds[0];
        assert!((row.calibration_empirical_mean - 0.20).abs() < 1e-12);
        assert!((row.target_empirical_mean - 0.50).abs() < 1e-12);
        assert!((row.empirical_target_minus_calibration_shift - 0.30).abs() < 1e-12);
        let expected_radius = ((4.0 / 0.01_f64).ln() / (2.0 * 100.0)).sqrt();
        assert!((row.calibration_hoeffding_radius - expected_radius).abs() < 1e-12);
        assert!((row.target_hoeffding_radius - expected_radius).abs() < 1e-12);
        assert!((row.shift_lower_confidence_bound
            - (0.30 - 2.0 * expected_radius).max(-1.0)).abs() < 1e-12);
        verify_direct_functional_transport_report(&request, &report).unwrap();
    }

    #[test]
    fn large_support_small_functional_family_does_not_pay_tv_dimension_penalty() {
        let k = 512;
        let counts = vec![100_u64; k];
        let mut values = vec![0.0; k];
        values[k - 1] = 1.0;
        let request = request(
            counts.clone(),
            counts,
            vec![functional("rare-event", values, 0.0, 1.0)],
        );
        let report = calibrate_direct_functional_transport(&request).unwrap();
        let row = &report.functional_bounds[0];
        assert!(row.calibration_hoeffding_radius < 0.02);
        assert!(row.shift_lower_confidence_bound > -0.04);
        assert!(row.shift_upper_confidence_bound < 0.04);
        let tv_slack_per_side =
            ((k as f64 * LN_2 + (1.0_f64 / 0.005).ln()) / (2.0 * 51_200.0)).sqrt();
        let full_tv_shift_interval_width = 2.0 * (2.0 * tv_slack_per_side);
        assert!(row.shift_upper_confidence_bound - row.shift_lower_confidence_bound
            < 0.3 * full_tv_shift_interval_width);
        assert_eq!(row.status, DirectFunctionalStatus::Informative);
    }

    #[test]
    fn constant_function_has_exact_zero_shift_and_zero_radius() {
        let request = request(
            vec![40, 60, 20],
            vec![10, 30, 70],
            vec![functional("constant", vec![7.0, 7.0, 7.0], 7.0, 7.0)],
        );
        let report = calibrate_direct_functional_transport(&request).unwrap();
        let row = &report.functional_bounds[0];
        assert_eq!(row.calibration_empirical_mean, 7.0);
        assert_eq!(row.target_empirical_mean, 7.0);
        assert_eq!(row.calibration_hoeffding_radius, 0.0);
        assert_eq!(row.target_hoeffding_radius, 0.0);
        assert_eq!(row.shift_lower_confidence_bound, 0.0);
        assert_eq!(row.shift_upper_confidence_bound, 0.0);
        assert_eq!(row.status, DirectFunctionalStatus::ExactConstant);
    }

    #[test]
    fn multiple_functionals_share_one_frozen_simultaneous_family_budget() {
        let request = request(
            vec![50, 50, 50],
            vec![45, 55, 50],
            vec![
                functional("access", vec![1.0, 0.0, 0.0], 0.0, 1.0),
                functional("crash", vec![0.0, 1.0, 0.0], 0.0, 1.0),
                functional("cost", vec![0.0, 0.5, 1.0], 0.0, 1.0),
            ],
        );
        let report = calibrate_direct_functional_transport(&request).unwrap();
        assert!(report.functional_bounds.iter().all(|row| {
            (row.simultaneous_row_failure_probability - (0.01 / 6.0)).abs() < 1e-12
        }));
        assert_eq!(report.functional_bounds.len(), 3);
    }

    #[test]
    fn orthogonal_unregistered_functional_is_not_claimed_as_covered() {
        let registered = request(
            vec![25, 25, 25, 25],
            vec![40, 10, 20, 30],
            vec![functional("registered-first-half", vec![1.0, 1.0, 0.0, 0.0], 0.0, 1.0)],
        );
        let report = calibrate_direct_functional_transport(&registered).unwrap();
        assert_eq!(report.functional_bounds.len(), 1);
        assert_eq!(report.functional_bounds[0].functional_id, "registered-first-half");
        assert_eq!(report.functional_contract_sha256, registered.functional_contract_sha256);
        // The registered first-half functional is unchanged (mean 0.5),
        // while the omitted category-zero indicator shifts from 0.25 to 0.40.
        assert_eq!(report.functional_bounds[0].calibration_empirical_mean, 0.5);
        assert_eq!(report.functional_bounds[0].target_empirical_mean, 0.5);
        assert!(transport_registered_functional_interval(
            &registered,
            &report,
            "omitted-category-zero",
            &registered.functional_contract_sha256,
            CAL_INTERVAL,
            0.0,
            1.0,
        )
        .is_err());
    }

    #[test]
    fn target_interval_uses_directional_shift_and_binds_source_receipt() {
        let request = request(
            vec![80, 20],
            vec![50, 50],
            vec![functional("crash", vec![0.0, 1.0], 0.0, 1.0)],
        );
        let report = calibrate_direct_functional_transport(&request).unwrap();
        let target = transport_registered_functional_interval(
            &request,
            &report,
            "crash",
            &request.functional_contract_sha256,
            CAL_INTERVAL,
            0.10,
            0.30,
        )
        .unwrap();
        let bound = &report.functional_bounds[0];
        assert!((target.target_lower_confidence_bound
            - (0.10 + bound.shift_lower_confidence_bound).max(0.0)).abs() < 1e-12);
        assert!((target.target_upper_confidence_bound
            - (0.30 + bound.shift_upper_confidence_bound).min(1.0)).abs() < 1e-12);
        assert_eq!(target.calibration_interval_source_sha256, CAL_INTERVAL);
    }

    #[test]
    fn rejects_family_roster_tampering_and_unknown_function_ids() {
        let request = request(
            vec![50, 50],
            vec![50, 50],
            vec![functional("crash", vec![0.0, 1.0], 0.0, 1.0)],
        );
        let report = calibrate_direct_functional_transport(&request).unwrap();
        let mut tampered = request.clone();
        tampered.registered_functionals.push(functional("late-added", vec![1.0, 0.0], 0.0, 1.0));
        assert!(tampered.validate().is_err());
        assert!(transport_registered_functional_interval(
            &request,
            &report,
            "late-added",
            &request.functional_contract_sha256,
            CAL_INTERVAL,
            0.0,
            1.0,
        )
        .is_err());
        let mut bad_report = report.clone();
        bad_report.functional_bounds[0].shift_lower_confidence_bound = 0.0;
        assert!(verify_direct_functional_transport_report(&request, &bad_report).is_err());
    }

    #[test]
    fn rejects_iid_support_and_confirmatory_independence_failures() {
        let base = request(
            vec![50, 50],
            vec![50, 50],
            vec![functional("crash", vec![0.0, 1.0], 0.0, 1.0)],
        );
        let mut serial = base.clone();
        serial.iid_target_complete_path_draws = false;
        assert!(serial.validate().is_err());
        let mut coupled = base.clone();
        coupled.independent_calibration_target_samples = false;
        assert!(coupled.validate().is_err());
        let mut confirmation_dependent = base;
        confirmation_dependent.reference_samples_independent_of_confirmatory_stream = false;
        assert!(confirmation_dependent.validate().is_err());
    }

    #[test]
    fn rejects_missing_duplicate_or_out_of_range_functional_values() {
        let duplicate = DirectFunctionalTransportRequest::new(
            "duplicate-functional",
            CAL_DESIGN,
            TARGET_DESIGN,
            sample(vec![50, 50], CAL_DATA),
            sample(vec![50, 50], TARGET_DATA),
            0.01,
            true,
            true,
            true,
            true,
            true,
            vec![
                functional("duplicate", vec![0.0, 1.0], 0.0, 1.0),
                functional("duplicate", vec![1.0, 0.0], 0.0, 1.0),
            ],
        );
        assert!(duplicate.is_err());

        let mut out_of_range = request(
            vec![50, 50],
            vec![50, 50],
            vec![functional("range", vec![0.0, 1.0], 0.0, 1.0)],
        );
        out_of_range.registered_functionals[0].category_values[1] = 1.5;
        assert!(out_of_range.validate().is_err());

        let mut missing_value = request(
            vec![50, 50],
            vec![50, 50],
            vec![functional("missing", vec![0.0, 1.0], 0.0, 1.0)],
        );
        missing_value.registered_functionals[0].category_values.pop();
        assert!(missing_value.validate().is_err());
    }

    #[test]
    fn composes_transport_failure_with_calibration_and_other_null_events() {
        let request = request(
            vec![50, 50],
            vec![50, 50],
            vec![functional("crash", vec![0.0, 1.0], 0.0, 1.0)],
        );
        let report = calibrate_direct_functional_transport(&request).unwrap();
        let budget = compose_direct_functional_null_error_budget(
            &request,
            &report,
            &request.functional_contract_sha256,
            0.05,
            0.01,
            true,
            0.002,
            true,
        )
        .unwrap();
        assert!((budget.total_null_failure_probability - 0.022).abs() < 1e-12);
        assert!((budget.conditional_alpha - (0.028 / 0.978)).abs() < 1e-12);
    }

    #[test]
    fn rejects_wrong_contract_and_over_budget_null_composition() {
        let request = request(
            vec![50, 50],
            vec![50, 50],
            vec![functional("crash", vec![0.0, 1.0], 0.0, 1.0)],
        );
        let report = calibrate_direct_functional_transport(&request).unwrap();
        assert!(compose_direct_functional_null_error_budget(
            &request, &report, TARGET_DESIGN, 0.05, 0.01, true, 0.0, true
        ).is_err());
        assert!(compose_direct_functional_null_error_budget(
            &request, &report, &request.functional_contract_sha256, 0.05, 0.03, true, 0.03, true
        ).is_err());
    }

    #[test]
    fn report_sha_and_request_digest_bind_every_functional_and_sample_field() {
        let request = request(
            vec![60, 40],
            vec![40, 60],
            vec![functional("crash", vec![0.0, 1.0], 0.0, 1.0)],
        );
        let report = calibrate_direct_functional_transport(&request).unwrap();
        assert_eq!(report.functional_contract_sha256, request.functional_contract_sha256);
        assert_ne!(report.request_sha256, "");
        assert_ne!(report.report_sha256, "");
        verify_direct_functional_transport_report(&request, &report).unwrap();
        let mut altered_sample = request.clone();
        altered_sample.target_sample.observed_path_counts[0] += 1;
        assert!(altered_sample.validate().is_ok());
        assert!(verify_direct_functional_transport_report(&altered_sample, &report).is_err());
    }
}
