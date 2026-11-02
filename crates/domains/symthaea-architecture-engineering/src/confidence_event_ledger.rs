//! Shared, typed union-bound ledger for null-side confidence failure events.
//!
//! This ledger does not assume that failure events are independent of one
//! another. It composes their failure probabilities by a union bound. Each
//! evidence stream must, however, be declared independent of the confirmatory
//! outcome stream so the resulting qualification event can be conditioned on
//! before deriving a conditional Type-I threshold.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, error::Error, fmt};

const REQUEST_SCHEMA: &str =
    "symthaea.architecture-engineering.confidence-event-ledger-request";
const REQUEST_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str =
    "symthaea.architecture-engineering.confidence-event-ledger-report";
const REPORT_VERSION: &str = "0.1.0";
const MAX_EVENT_COUNT: usize = 10_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfidenceEventLedgerError {
    pub path: String,
    pub message: String,
}

impl ConfidenceEventLedgerError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ConfidenceEventLedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for ConfidenceEventLedgerError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NullSideConfidenceEventRole {
    CalibrationNullConfidence,
    CalibrationToTargetTransport,
    OtherNullSideConfidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NullSideConfidenceEvent {
    pub event_id: String,
    pub role: NullSideConfidenceEventRole,
    /// Failure probability of this single registered confidence event.
    /// Zero is allowed only when the upstream method provides an exact event.
    pub failure_probability: f64,
    /// Digest of the source evidence artifact, not proof of its truth.
    pub source_receipt_sha256: String,
    /// Digest of the functional/estimand/model contract this event qualifies.
    pub contract_sha256: String,
    pub target_design_sha256: String,
    /// Each event's source stream must be independent of confirmatory outcomes.
    pub evidence_independent_of_confirmatory_stream: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidenceEventLedgerRequest {
    pub schema: String,
    pub version: String,
    pub ledger_id: String,
    pub nominal_alpha: f64,
    pub target_design_sha256: String,
    /// Sorted by stable event_id before hashing and report generation.
    pub events: Vec<NullSideConfidenceEvent>,
}

impl ConfidenceEventLedgerRequest {
    pub fn new(
        ledger_id: impl Into<String>,
        nominal_alpha: f64,
        target_design_sha256: impl Into<String>,
        mut events: Vec<NullSideConfidenceEvent>,
    ) -> Result<Self, ConfidenceEventLedgerError> {
        events.sort_by(|a, b| a.event_id.cmp(&b.event_id));
        let request = Self {
            schema: REQUEST_SCHEMA.into(),
            version: REQUEST_VERSION.into(),
            ledger_id: ledger_id.into(),
            nominal_alpha,
            target_design_sha256: target_design_sha256.into(),
            events,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), ConfidenceEventLedgerError> {
        if self.schema != REQUEST_SCHEMA || self.version != REQUEST_VERSION {
            return Err(ConfidenceEventLedgerError::new(
                "request.schema",
                "unsupported confidence-event ledger schema/version",
            ));
        }
        if self.ledger_id.trim().is_empty() {
            return Err(ConfidenceEventLedgerError::new(
                "request.ledger_id",
                "ledger ID must be non-empty",
            ));
        }
        validate_open_probability(self.nominal_alpha, "request.nominal_alpha")?;
        validate_hash(&self.target_design_sha256, "request.target_design_sha256")?;
        if self.events.is_empty() || self.events.len() > MAX_EVENT_COUNT {
            return Err(ConfidenceEventLedgerError::new(
                "request.events",
                format!("event roster must contain 1..={MAX_EVENT_COUNT} events"),
            ));
        }
        if self.events.windows(2).any(|pair| {
            pair[0].event_id.as_str() >= pair[1].event_id.as_str()
        }) {
            return Err(ConfidenceEventLedgerError::new(
                "request.events",
                "event roster must be strictly sorted by unique event ID",
            ));
        }
        let mut ids = BTreeSet::new();
        let mut total_failure_probability = 0.0_f64;
        for (index, event) in self.events.iter().enumerate() {
            let path = format!("request.events[{index}]");
            if event.event_id.trim().is_empty() || !ids.insert(event.event_id.as_str()) {
                return Err(ConfidenceEventLedgerError::new(
                    format!("{path}.event_id"),
                    "event IDs must be non-empty and unique",
                ));
            }
            validate_closed_probability(
                event.failure_probability,
                &format!("{path}.failure_probability"),
            )?;
            validate_hash(
                &event.source_receipt_sha256,
                &format!("{path}.source_receipt_sha256"),
            )?;
            validate_hash(&event.contract_sha256, &format!("{path}.contract_sha256"))?;
            validate_hash(
                &event.target_design_sha256,
                &format!("{path}.target_design_sha256"),
            )?;
            if event.target_design_sha256 != self.target_design_sha256 {
                return Err(ConfidenceEventLedgerError::new(
                    format!("{path}.target_design_sha256"),
                    "event target design differs from the ledger confirmatory design",
                ));
            }
            if !event.evidence_independent_of_confirmatory_stream {
                return Err(ConfidenceEventLedgerError::new(
                    format!("{path}.evidence_independent_of_confirmatory_stream"),
                    "null-side confidence evidence must be independent of confirmatory outcomes",
                ));
            }
            total_failure_probability += event.failure_probability;
            if !total_failure_probability.is_finite() {
                return Err(ConfidenceEventLedgerError::new(
                    "request.events",
                    "sum of null-side failure probabilities overflowed",
                ));
            }
        }
        if total_failure_probability >= self.nominal_alpha {
            return Err(ConfidenceEventLedgerError::new(
                "request.events",
                "union-bound null-side confidence failure probability must be strictly below nominal alpha",
            ));
        }
        Ok(())
    }

    pub fn request_sha256(&self) -> Result<String, ConfidenceEventLedgerError> {
        self.validate()?;
        digest(self)
    }

    pub fn event_roster_sha256(&self) -> Result<String, ConfidenceEventLedgerError> {
        self.validate()?;
        digest(&self.events)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidenceEventLedgerReport {
    pub schema: String,
    pub version: String,
    pub ledger_id: String,
    pub request_sha256: String,
    pub event_roster_sha256: String,
    pub target_design_sha256: String,
    pub nominal_alpha: f64,
    pub events: Vec<NullSideConfidenceEvent>,
    pub event_count: usize,
    pub total_null_failure_probability: f64,
    pub conditional_alpha: f64,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    ledger_id: &'a str,
    request_sha256: &'a str,
    event_roster_sha256: &'a str,
    target_design_sha256: &'a str,
    nominal_alpha: f64,
    events: &'a [NullSideConfidenceEvent],
    event_count: usize,
    total_null_failure_probability: f64,
    conditional_alpha: f64,
}

fn validate_open_probability(
    value: f64,
    path: &str,
) -> Result<(), ConfidenceEventLedgerError> {
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(ConfidenceEventLedgerError::new(
            path,
            "probability must be finite and strictly between zero and one",
        ));
    }
    Ok(())
}

fn validate_closed_probability(
    value: f64,
    path: &str,
) -> Result<(), ConfidenceEventLedgerError> {
    if !value.is_finite() || value < 0.0 || value >= 1.0 {
        return Err(ConfidenceEventLedgerError::new(
            path,
            "failure probability must be finite and lie in [0, 1)",
        ));
    }
    Ok(())
}

fn validate_hash(value: &str, path: &str) -> Result<(), ConfidenceEventLedgerError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ConfidenceEventLedgerError::new(
            path,
            "expected a 64-character hexadecimal SHA-256 digest",
        ));
    }
    Ok(())
}

fn digest<T: Serialize>(value: &T) -> Result<String, ConfidenceEventLedgerError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ConfidenceEventLedgerError::new("serialization", error.to_string()))?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn report_digest(report: &ConfidenceEventLedgerReport) -> Result<String, ConfidenceEventLedgerError> {
    digest(&ReportPayload {
        schema: &report.schema,
        version: &report.version,
        ledger_id: &report.ledger_id,
        request_sha256: &report.request_sha256,
        event_roster_sha256: &report.event_roster_sha256,
        target_design_sha256: &report.target_design_sha256,
        nominal_alpha: report.nominal_alpha,
        events: &report.events,
        event_count: report.event_count,
        total_null_failure_probability: report.total_null_failure_probability,
        conditional_alpha: report.conditional_alpha,
    })
}

pub fn compose_confidence_event_ledger(
    input: &ConfidenceEventLedgerRequest,
) -> Result<ConfidenceEventLedgerReport, ConfidenceEventLedgerError> {
    input.validate()?;
    let total_null_failure_probability = input.events
        .iter()
        .map(|event| event.failure_probability)
        .sum::<f64>();
    if !total_null_failure_probability.is_finite()
        || total_null_failure_probability >= input.nominal_alpha
    {
        return Err(ConfidenceEventLedgerError::new(
            "report.total_null_failure_probability",
            "union-bound failure probability must be finite and strictly below nominal alpha",
        ));
    }
    let conditional_alpha =
        (input.nominal_alpha - total_null_failure_probability)
            / (1.0 - total_null_failure_probability);
    if !conditional_alpha.is_finite() || !(0.0..1.0).contains(&conditional_alpha) {
        return Err(ConfidenceEventLedgerError::new(
            "report.conditional_alpha",
            "derived conditional alpha is non-finite or outside (0, 1)",
        ));
    }
    let mut report = ConfidenceEventLedgerReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        ledger_id: input.ledger_id.clone(),
        request_sha256: input.request_sha256()?,
        event_roster_sha256: input.event_roster_sha256()?,
        target_design_sha256: input.target_design_sha256.clone(),
        nominal_alpha: input.nominal_alpha,
        events: input.events.clone(),
        event_count: input.events.len(),
        total_null_failure_probability,
        conditional_alpha,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

/// Reconstruct the report from the complete frozen request; this does not
/// trust report totals, event counts, ordering, or caller-supplied digests.
pub fn verify_confidence_event_ledger_report(
    request: &ConfidenceEventLedgerRequest,
    report: &ConfidenceEventLedgerReport,
) -> Result<(), ConfidenceEventLedgerError> {
    let expected = compose_confidence_event_ledger(request)?;
    if expected != *report {
        return Err(ConfidenceEventLedgerError::new(
            "report",
            "stored ledger report differs from deterministic reconstruction",
        ));
    }
    if report_digest(report)? != report.report_sha256 {
        return Err(ConfidenceEventLedgerError::new(
            "report.report_sha256",
            "ledger report digest does not match canonical payload",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TARGET: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const CONTRACT_A: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const CONTRACT_B: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SOURCE_A: &str =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const SOURCE_B: &str =
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const SOURCE_C: &str =
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn event(
        id: &str,
        role: NullSideConfidenceEventRole,
        delta: f64,
        source: &str,
        contract: &str,
    ) -> NullSideConfidenceEvent {
        NullSideConfidenceEvent {
            event_id: id.into(),
            role,
            failure_probability: delta,
            source_receipt_sha256: source.into(),
            contract_sha256: contract.into(),
            target_design_sha256: TARGET.into(),
            evidence_independent_of_confirmatory_stream: true,
        }
    }

    fn request() -> ConfidenceEventLedgerRequest {
        ConfidenceEventLedgerRequest::new(
            "null-qualification-ledger-001",
            0.05,
            TARGET,
            vec![
                event("transport", NullSideConfidenceEventRole::CalibrationToTargetTransport, 0.01, SOURCE_B, CONTRACT_A),
                event("other", NullSideConfidenceEventRole::OtherNullSideConfidence, 0.002, SOURCE_C, CONTRACT_B),
                event("calibration", NullSideConfidenceEventRole::CalibrationNullConfidence, 0.01, SOURCE_A, CONTRACT_A),
            ],
        )
        .unwrap()
    }

    #[test]
    fn three_event_union_bound_and_conditional_alpha_match_analytical_fixture() {
        let request = request();
        let report = compose_confidence_event_ledger(&request).unwrap();
        assert_eq!(report.event_count, 3);
        assert!((report.total_null_failure_probability - 0.022).abs() < 1e-12);
        assert!((report.conditional_alpha - (0.028 / 0.978)).abs() < 1e-12);
        assert!(report.conditional_alpha < report.nominal_alpha);
        verify_confidence_event_ledger_report(&request, &report).unwrap();
    }

    #[test]
    fn event_order_is_canonical_and_replayable() {
        let first = request();
        let mut reversed = first.clone();
        reversed.events.reverse();
        // A manually reordered request is not canonical and therefore rejects.
        assert!(reversed.validate().is_err());

        let rebuilt = ConfidenceEventLedgerRequest::new(
            "null-qualification-ledger-001",
            0.05,
            TARGET,
            reversed.events,
        )
        .unwrap();
        assert_eq!(first.request_sha256().unwrap(), rebuilt.request_sha256().unwrap());
        assert_eq!(
            compose_confidence_event_ledger(&first).unwrap(),
            compose_confidence_event_ledger(&rebuilt).unwrap()
        );
    }

    #[test]
    fn rejects_duplicate_event_ids() {
        let duplicate = ConfidenceEventLedgerRequest::new(
            "duplicate-event",
            0.05,
            TARGET,
            vec![
                event("same", NullSideConfidenceEventRole::CalibrationNullConfidence, 0.01, SOURCE_A, CONTRACT_A),
                event("same", NullSideConfidenceEventRole::OtherNullSideConfidence, 0.01, SOURCE_B, CONTRACT_B),
            ],
        );
        assert!(duplicate.is_err());
    }

    #[test]
    fn rejects_target_mismatch_and_confirmation_dependent_evidence() {
        let mut mismatch = request();
        mismatch.events[0].target_design_sha256 = CONTRACT_B.into();
        assert!(mismatch.validate().is_err());

        let mut dependent = request();
        dependent.events[0].evidence_independent_of_confirmatory_stream = false;
        assert!(dependent.validate().is_err());
    }

    #[test]
    fn rejects_empty_roster_invalid_probabilities_and_over_budget_ledger() {
        let empty = ConfidenceEventLedgerRequest::new(
            "empty",
            0.05,
            TARGET,
            vec![],
        );
        assert!(empty.is_err());

        let mut invalid = request();
        invalid.events[0].failure_probability = f64::NAN;
        assert!(invalid.validate().is_err());

        let excessive = ConfidenceEventLedgerRequest::new(
            "too-much-alpha",
            0.05,
            TARGET,
            vec![
                event("a", NullSideConfidenceEventRole::CalibrationNullConfidence, 0.03, SOURCE_A, CONTRACT_A),
                event("b", NullSideConfidenceEventRole::OtherNullSideConfidence, 0.02, SOURCE_B, CONTRACT_B),
            ],
        );
        assert!(excessive.is_err());
    }

    #[test]
    fn zero_failure_event_is_allowed_and_still_receipt_bound() {
        let request = ConfidenceEventLedgerRequest::new(
            "zero-event",
            0.05,
            TARGET,
            vec![event(
                "exact-event",
                NullSideConfidenceEventRole::CalibrationNullConfidence,
                0.0,
                SOURCE_A,
                CONTRACT_A,
            )],
        )
        .unwrap();
        let report = compose_confidence_event_ledger(&request).unwrap();
        assert_eq!(report.total_null_failure_probability, 0.0);
        assert_eq!(report.conditional_alpha, 0.05);
        assert_eq!(report.events[0].source_receipt_sha256, SOURCE_A);
    }

    #[test]
    fn tampering_with_receipts_roles_roster_or_report_digest_is_rejected() {
        let request = request();
        let report = compose_confidence_event_ledger(&request).unwrap();

        let mut tampered_report = report.clone();
        tampered_report.events[0].role = NullSideConfidenceEventRole::OtherNullSideConfidence;
        assert!(verify_confidence_event_ledger_report(&request, &tampered_report).is_err());

        let mut bad_digest = report.clone();
        bad_digest.report_sha256 = SOURCE_A.into();
        assert!(verify_confidence_event_ledger_report(&request, &bad_digest).is_err());

        let mut late_event_request = request.clone();
        late_event_request.events.push(event(
            "z-late",
            NullSideConfidenceEventRole::OtherNullSideConfidence,
            0.001,
            SOURCE_C,
            CONTRACT_B,
        ));
        assert!(verify_confidence_event_ledger_report(&late_event_request, &report).is_err());
    }

    #[test]
    fn report_reconstructs_every_total_and_contract_field() {
        let request = request();
        let report = compose_confidence_event_ledger(&request).unwrap();
        let mut wrong_total = report.clone();
        wrong_total.total_null_failure_probability = 0.001;
        assert!(verify_confidence_event_ledger_report(&request, &wrong_total).is_err());

        let mut wrong_target = report.clone();
        wrong_target.target_design_sha256 = CONTRACT_A.into();
        assert!(verify_confidence_event_ledger_report(&request, &wrong_target).is_err());

        let mut wrong_roster = report.clone();
        wrong_roster.event_roster_sha256 = CONTRACT_B.into();
        assert!(verify_confidence_event_ledger_report(&request, &wrong_roster).is_err());
    }
}
