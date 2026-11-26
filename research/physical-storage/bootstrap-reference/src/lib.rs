//! Fail-closed bootstrap/control-plane qualification semantics.
//!
//! This crate is a dependency-free decision kernel. It is not an optical reader,
//! calibration solver, signature verifier, JSON Schema validator, or payload ECC.
//! It keeps bootstrap and payload outcomes separate and requires both before a
//! full archive recovery can pass.

use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapObjectKind {
    FormatIdentity,
    CodeDefinition,
    PhysicalLayout,
    Calibration,
    IndexAddress,
    KnownAnswerVector,
    ReaderBootstrap,
    CapsuleReference,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapObjectState {
    Current,
    Recovered,
    Missing,
    Corrupt,
    Stale,
    WrongButValid,
    Unknown,
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapRecord {
    pub object_id: String,
    pub kind: BootstrapObjectKind,
    pub required: bool,
    pub state: BootstrapObjectState,
    pub expected_digest_sha256: String,
    pub observed_digest_sha256: Option<String>,
    /// Required for calibration records; identifies the reader/firmware scope.
    pub expected_reader_id: Option<String>,
    pub observed_reader_id: Option<String>,
    /// None means calibration validity was not established.
    pub calibration_current: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapDisposition {
    Passed,
    Recovered,
    Failed,
    Incomplete,
    NotEvaluated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadDisposition {
    Exact,
    Recovered,
    Failed,
    Uncorrectable,
    NotEvaluated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullArchiveDisposition {
    Pass,
    Fail,
    Incomplete,
    NotEvaluated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractError {
    EmptyObjectId,
    DuplicateObjectId,
    InvalidExpectedDigest,
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Qualify the required bootstrap records without accepting stale, unknown,
/// wrong-but-valid, or digest-mismatched metadata.
pub fn qualify_bootstrap(
    records: &[BootstrapRecord],
) -> Result<BootstrapDisposition, ContractError> {
    let mut seen = HashSet::with_capacity(records.len());
    for record in records {
        if record.object_id.trim().is_empty() {
            return Err(ContractError::EmptyObjectId);
        }
        if !seen.insert(record.object_id.as_str()) {
            return Err(ContractError::DuplicateObjectId);
        }
        if record.required && !is_sha256_hex(&record.expected_digest_sha256) {
            return Err(ContractError::InvalidExpectedDigest);
        }
    }

    let mut required_count = 0usize;
    let mut has_recovered = false;
    let mut incomplete = false;
    let mut failed = false;

    for record in records.iter().filter(|record| record.required) {
        required_count += 1;
        match record.state {
            BootstrapObjectState::Current | BootstrapObjectState::Recovered => {
                match record.observed_digest_sha256.as_deref() {
                    Some(observed)
                        if observed.eq_ignore_ascii_case(&record.expected_digest_sha256) => {}
                    Some(_) => {
                        failed = true;
                        continue;
                    }
                    None => {
                        incomplete = true;
                        continue;
                    }
                }

                if record.kind == BootstrapObjectKind::Calibration {
                    match (
                        record.expected_reader_id.as_deref(),
                        record.observed_reader_id.as_deref(),
                        record.calibration_current,
                    ) {
                        (Some(expected), Some(observed), Some(true))
                            if expected == observed => {}
                        (Some(_), Some(_), Some(false)) => {
                            failed = true;
                            continue;
                        }
                        (Some(expected), Some(observed), Some(true))
                            if expected != observed =>
                        {
                            failed = true;
                            continue;
                        }
                        _ => {
                            incomplete = true;
                            continue;
                        }
                    }
                }

                if record.state == BootstrapObjectState::Recovered {
                    has_recovered = true;
                }
            }
            BootstrapObjectState::Missing | BootstrapObjectState::Unknown => {
                incomplete = true;
            }
            BootstrapObjectState::Corrupt
            | BootstrapObjectState::Stale
            | BootstrapObjectState::WrongButValid
            | BootstrapObjectState::NotApplicable => {
                failed = true;
            }
        }
    }

    if required_count == 0 {
        return Ok(BootstrapDisposition::Incomplete);
    }
    if failed {
        Ok(BootstrapDisposition::Failed)
    } else if incomplete {
        Ok(BootstrapDisposition::Incomplete)
    } else if has_recovered {
        Ok(BootstrapDisposition::Recovered)
    } else {
        Ok(BootstrapDisposition::Passed)
    }
}

/// Combine independent bootstrap and payload dispositions.
///
/// A full archive pass requires both planes to be exact/recovered. A known
/// failure in either plane blocks the pass; missing/unknown evidence is
/// incomplete, never converted into success.
pub fn evaluate_full_archive(
    bootstrap: BootstrapDisposition,
    payload: PayloadDisposition,
) -> FullArchiveDisposition {
    match (bootstrap, payload) {
        (BootstrapDisposition::Failed, _) => FullArchiveDisposition::Fail,
        (_, PayloadDisposition::Failed | PayloadDisposition::Uncorrectable) => {
            FullArchiveDisposition::Fail
        }
        (BootstrapDisposition::Passed | BootstrapDisposition::Recovered,
         PayloadDisposition::Exact | PayloadDisposition::Recovered) => {
            FullArchiveDisposition::Pass
        }
        (BootstrapDisposition::NotEvaluated, PayloadDisposition::NotEvaluated)
        | (BootstrapDisposition::Passed | BootstrapDisposition::Recovered,
           PayloadDisposition::NotEvaluated) => FullArchiveDisposition::NotEvaluated,
        (BootstrapDisposition::Incomplete | BootstrapDisposition::NotEvaluated, _) => {
            FullArchiveDisposition::Incomplete
        }
        (_, PayloadDisposition::NotEvaluated) => FullArchiveDisposition::NotEvaluated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(
        object_id: &str,
        kind: BootstrapObjectKind,
        state: BootstrapObjectState,
    ) -> BootstrapRecord {
        BootstrapRecord {
            object_id: object_id.to_string(),
            kind,
            required: true,
            state,
            expected_digest_sha256: "a".repeat(64),
            observed_digest_sha256: Some("a".repeat(64)),
            expected_reader_id: None,
            observed_reader_id: None,
            calibration_current: None,
        }
    }

    fn valid_calibration() -> BootstrapRecord {
        BootstrapRecord {
            expected_reader_id: Some("reader-A-fw-1".into()),
            observed_reader_id: Some("reader-A-fw-1".into()),
            calibration_current: Some(true),
            ..record(
                "CAL-A",
                BootstrapObjectKind::Calibration,
                BootstrapObjectState::Current,
            )
        }
    }

    #[test]
    fn current_required_records_pass() {
        let records = vec![
            record("FMT", BootstrapObjectKind::FormatIdentity, BootstrapObjectState::Current),
            record("CODE", BootstrapObjectKind::CodeDefinition, BootstrapObjectState::Current),
            valid_calibration(),
        ];
        assert_eq!(qualify_bootstrap(&records), Ok(BootstrapDisposition::Passed));
    }

    #[test]
    fn recovered_control_record_is_reported_separately() {
        let records = vec![
            record("FMT", BootstrapObjectKind::FormatIdentity, BootstrapObjectState::Current),
            record("CODE", BootstrapObjectKind::CodeDefinition, BootstrapObjectState::Recovered),
        ];
        assert_eq!(qualify_bootstrap(&records), Ok(BootstrapDisposition::Recovered));
    }

    #[test]
    fn unknown_or_missing_required_control_is_incomplete() {
        for state in [BootstrapObjectState::Unknown, BootstrapObjectState::Missing] {
            let records = vec![record("FMT", BootstrapObjectKind::FormatIdentity, state)];
            assert_eq!(qualify_bootstrap(&records), Ok(BootstrapDisposition::Incomplete));
        }
    }

    #[test]
    fn stale_and_wrong_but_valid_descriptors_fail_closed() {
        for state in [BootstrapObjectState::Stale, BootstrapObjectState::WrongButValid] {
            let records = vec![record("FMT", BootstrapObjectKind::FormatIdentity, state)];
            assert_eq!(qualify_bootstrap(&records), Ok(BootstrapDisposition::Failed));
        }
    }

    #[test]
    fn a_digest_mismatch_fails_even_when_descriptor_parses() {
        let mut item = record(
            "FMT",
            BootstrapObjectKind::FormatIdentity,
            BootstrapObjectState::Current,
        );
        item.observed_digest_sha256 = Some("b".repeat(64));
        assert_eq!(qualify_bootstrap(&[item]), Ok(BootstrapDisposition::Failed));
    }

    #[test]
    fn calibration_must_match_reader_and_be_current() {
        let mut wrong_reader = valid_calibration();
        wrong_reader.observed_reader_id = Some("reader-B-fw-1".into());
        assert_eq!(qualify_bootstrap(&[wrong_reader]), Ok(BootstrapDisposition::Failed));

        let mut stale = valid_calibration();
        stale.calibration_current = Some(false);
        assert_eq!(qualify_bootstrap(&[stale]), Ok(BootstrapDisposition::Failed));

        let mut unknown = valid_calibration();
        unknown.calibration_current = None;
        assert_eq!(qualify_bootstrap(&[unknown]), Ok(BootstrapDisposition::Incomplete));
    }

    #[test]
    fn duplicate_bootstrap_ids_are_rejected() {
        let first = record("FMT", BootstrapObjectKind::FormatIdentity, BootstrapObjectState::Current);
        let second = record("FMT", BootstrapObjectKind::CodeDefinition, BootstrapObjectState::Current);
        assert_eq!(qualify_bootstrap(&[first, second]), Err(ContractError::DuplicateObjectId));
    }

    #[test]
    fn full_archive_requires_both_planes() {
        assert_eq!(
            evaluate_full_archive(BootstrapDisposition::Passed, PayloadDisposition::Exact),
            FullArchiveDisposition::Pass
        );
        assert_eq!(
            evaluate_full_archive(BootstrapDisposition::Recovered, PayloadDisposition::Recovered),
            FullArchiveDisposition::Pass
        );
        assert_eq!(
            evaluate_full_archive(BootstrapDisposition::Incomplete, PayloadDisposition::Exact),
            FullArchiveDisposition::Incomplete
        );
        assert_eq!(
            evaluate_full_archive(BootstrapDisposition::Failed, PayloadDisposition::Exact),
            FullArchiveDisposition::Fail
        );
        assert_eq!(
            evaluate_full_archive(BootstrapDisposition::Passed, PayloadDisposition::Uncorrectable),
            FullArchiveDisposition::Fail
        );
    }

    #[test]
    fn no_required_bootstrap_records_is_not_a_pass() {
        assert_eq!(qualify_bootstrap(&[]), Ok(BootstrapDisposition::Incomplete));
    }
}
