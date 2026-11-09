//! Shared bitemporal admissibility rules for financial research and quantitative runs.
//!
//! This crate does not fetch sources, authenticate signatures, verify artifact bytes, or decide
//! whether a source is truthful. Callers must pass availability proofs only after an independent
//! provenance/byte-verification layer has checked them.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

/// UTC instant with nanosecond precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp {
    unix_seconds: i64,
    nanoseconds: u32,
}

impl Timestamp {
    pub const fn unix_seconds(self) -> i64 { self.unix_seconds }
    pub const fn nanoseconds(self) -> u32 { self.nanoseconds }
    pub fn parse_rfc3339(input: &str) -> Result<Self, TimestampError> { input.parse() }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimestampError {
    pub input: String,
    pub reason: &'static str,
}

impl fmt::Display for TimestampError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid RFC3339 timestamp {:?}: {}", self.input, self.reason)
    }
}
impl std::error::Error for TimestampError {}

fn digits(input: &[u8], start: usize, len: usize) -> Option<u32> {
    let end = start.checked_add(len)?;
    let slice = input.get(start..end)?;
    if !slice.iter().all(u8::is_ascii_digit) { return None; }
    slice.iter().try_fold(0u32, |value, byte| {
        value.checked_mul(10)?.checked_add(u32::from(*byte - b'0'))
    })
}

fn leap_year(year: u32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}
fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Days since 1970-01-01, using the proleptic Gregorian calendar.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let adjusted_year = year - if month <= 2 { 1 } else { 0 };
    let era = if adjusted_year >= 0 { adjusted_year } else { adjusted_year - 399 } / 400;
    let year_of_era = adjusted_year - era * 400;
    let shifted_month = i64::from(month) + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

impl FromStr for Timestamp {
    type Err = TimestampError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let fail = |reason| TimestampError { input: input.to_owned(), reason };
        let bytes = input.as_bytes();
        if bytes.len() < 20 || !bytes.is_ascii() {
            return Err(fail("expected an ASCII RFC3339 timestamp with an explicit offset"));
        }
        if bytes.get(4) != Some(&b'-')
            || bytes.get(7) != Some(&b'-')
            || !matches!(bytes.get(10), Some(&b'T' | &b't'))
            || bytes.get(13) != Some(&b':')
            || bytes.get(16) != Some(&b':')
        {
            return Err(fail("invalid date/time separators"));
        }

        let year = digits(bytes, 0, 4).ok_or_else(|| fail("invalid year"))?;
        let month = digits(bytes, 5, 2).ok_or_else(|| fail("invalid month"))?;
        let day = digits(bytes, 8, 2).ok_or_else(|| fail("invalid day"))?;
        let hour = digits(bytes, 11, 2).ok_or_else(|| fail("invalid hour"))?;
        let minute = digits(bytes, 14, 2).ok_or_else(|| fail("invalid minute"))?;
        let second = digits(bytes, 17, 2).ok_or_else(|| fail("invalid second"))?;

        if year == 0 || !(1..=12).contains(&month) || !(1..=days_in_month(year, month)).contains(&day) {
            return Err(fail("date is outside the supported Gregorian calendar"));
        }
        if hour > 23 || minute > 59 || second > 59 {
            return Err(fail("time is out of range; leap seconds are not accepted"));
        }

        let mut index = 19;
        let mut nanoseconds = 0u32;
        if bytes.get(index) == Some(&b'.') {
            index += 1;
            let fraction_start = index;
            while bytes.get(index).is_some_and(u8::is_ascii_digit) { index += 1; }
            let fraction_len = index - fraction_start;
            if !(1..=9).contains(&fraction_len) {
                return Err(fail("fractional seconds must contain 1 to 9 digits"));
            }
            let fraction = digits(bytes, fraction_start, fraction_len)
                .ok_or_else(|| fail("invalid fractional seconds"))?;
            nanoseconds = fraction * 10u32.pow((9 - fraction_len) as u32);
        }

        let (offset_sign, offset_seconds) = match bytes.get(index) {
            Some(&b'Z' | &b'z') if index + 1 == bytes.len() => (0i64, 0i64),
            Some(&b'+' | &b'-') => {
                if index + 6 != bytes.len() || bytes.get(index + 3) != Some(&b':') {
                    return Err(fail("offset must use ±HH:MM"));
                }
                let offset_hour = digits(bytes, index + 1, 2)
                    .ok_or_else(|| fail("invalid offset hour"))?;
                let offset_minute = digits(bytes, index + 4, 2)
                    .ok_or_else(|| fail("invalid offset minute"))?;
                if offset_hour > 23 || offset_minute > 59 {
                    return Err(fail("UTC offset is out of range"));
                }
                let sign = if bytes[index] == b'+' { 1i64 } else { -1i64 };
                (sign, i64::from(offset_hour * 3_600 + offset_minute * 60))
            }
            _ => return Err(fail("explicit Z or ±HH:MM offset is required")),
        };

        let days = days_from_civil(i64::from(year), month, day);
        let local_seconds = days.checked_mul(86_400)
            .and_then(|v| v.checked_add(i64::from(hour) * 3_600))
            .and_then(|v| v.checked_add(i64::from(minute) * 60))
            .and_then(|v| v.checked_add(i64::from(second)))
            .ok_or_else(|| fail("timestamp arithmetic overflow"))?;
        let unix_seconds = local_seconds.checked_sub(offset_sign * offset_seconds)
            .ok_or_else(|| fail("UTC offset arithmetic overflow"))?;

        Ok(Self { unix_seconds, nanoseconds })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationMode {
    /// Later retrieval is allowed only with independently verified historical availability.
    HistoricalReconstruction,
    /// Source availability and local ingestion must both precede the decision frontier.
    ProspectiveStreaming,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RightsDisposition { Authorized, NotAuthorized, Unknown }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AvailabilityProofKind {
    PrimaryReleaseRecord,
    ProviderAvailabilityReceipt,
    SignedHistoricalArchive,
    EntitlementRecord,
    IngestionLog,
}

/// Attested result from the caller's independent proof verifier, not a self-asserted boolean.
/// The proof artifact must already have passed byte and provenance checks before construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AvailabilityProof {
    Verified {
        proof_ref: String,
        kind: AvailabilityProofKind,
        evidence_id: String,
        principal_id: String,
        available_at: Timestamp,
    },
    Missing,
    Unverifiable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceVersion {
    pub evidence_id: String,
    pub source_id: String,
    pub source_record_id: String,
    /// Stable identity for the logical record across restatements/revisions.
    pub revision_group_id: String,
    pub provenance_family_id: String,
    pub valid_from: Option<Timestamp>,
    /// Exclusive upper bound: validity intervals are [valid_from, valid_until).
    pub valid_until: Option<Timestamp>,
    /// Public publication time, when applicable; it does not override actor-scoped availability.
    pub published_at: Option<Timestamp>,
    /// Actor-scoped earliest availability of this exact source version.
    pub known_at: Option<Timestamp>,
    /// When this evaluation environment actually retrieved/ingested the bytes.
    pub retrieved_at: Timestamp,
    pub rights_profile_id: String,
    pub rights: RightsDisposition,
    pub availability_proof: AvailabilityProof,
    /// Evidence IDs of prior versions replaced by this version.
    pub supersedes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationContext {
    pub mode: EvaluationMode,
    pub principal_id: String,
    pub rights_profile_id: String,
    pub decision_frontier: Timestamp,
    /// The instant/period against which valid-time relevance is tested. It is not the
    /// decision frontier and may be earlier or later than it.
    pub target_time: Option<Timestamp>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Disposition { Pass, Fail, Indeterminate }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidityState {
    InForceAtTarget,
    NotYetEffectiveAtTarget,
    ExpiredAtTarget,
    Unspecified,
    NotEvaluated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReasonCode {
    EmptyIdentity,
    InvalidEvaluationContext,
    RightsProfileMismatch,
    RightsDenied,
    RightsUnknown,
    KnownAtMissing,
    AvailabilityProofMissing,
    AvailabilityProofUnverifiable,
    AvailabilityProofIdentityMismatch,
    AvailabilityProofTimeMismatch,
    RetrievedBeforeKnownAt,
    FutureKnownEvidence,
    LateIngestion,
    InvalidValidityInterval,
    ValidityNotYetEffective,
    ValidityExpired,
    ValidityUnspecified,
    TargetTimeNotProvided,
}

impl ReasonCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EmptyIdentity => "EMPTY_IDENTITY",
            Self::InvalidEvaluationContext => "INVALID_EVALUATION_CONTEXT",
            Self::RightsProfileMismatch => "RIGHTS_PROFILE_MISMATCH",
            Self::RightsDenied => "RIGHTS_DENIED",
            Self::RightsUnknown => "RIGHTS_UNKNOWN",
            Self::KnownAtMissing => "KNOWN_AT_MISSING",
            Self::AvailabilityProofMissing => "AVAILABILITY_PROOF_MISSING",
            Self::AvailabilityProofUnverifiable => "AVAILABILITY_PROOF_UNVERIFIABLE",
            Self::AvailabilityProofIdentityMismatch => "AVAILABILITY_PROOF_IDENTITY_MISMATCH",
            Self::AvailabilityProofTimeMismatch => "AVAILABILITY_PROOF_TIME_MISMATCH",
            Self::RetrievedBeforeKnownAt => "RETRIEVED_BEFORE_KNOWN_AT",
            Self::FutureKnownEvidence => "FUTURE_KNOWN_EVIDENCE",
            Self::LateIngestion => "LATE_INGESTION",
            Self::InvalidValidityInterval => "INVALID_VALIDITY_INTERVAL",
            Self::ValidityNotYetEffective => "VALIDITY_NOT_YET_EFFECTIVE",
            Self::ValidityExpired => "VALIDITY_EXPIRED",
            Self::ValidityUnspecified => "VALIDITY_UNSPECIFIED",
            Self::TargetTimeNotProvided => "TARGET_TIME_NOT_PROVIDED",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceEvaluation {
    pub evidence_id: String,
    pub disposition: Disposition,
    pub validity: ValidityState,
    pub reasons: Vec<ReasonCode>,
    /// Retain prior versions; list later admissible versions that supersede this one.
    pub superseded_by: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleEvaluation {
    pub evidence: Vec<EvidenceEvaluation>,
    /// Unique provenance families among admissible evidence, not number of documents.
    pub independent_source_families: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleError {
    pub reason: &'static str,
    pub evidence_id: Option<String>,
}
impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.evidence_id {
            Some(id) => write!(f, "{}: {}", self.reason, id),
            None => f.write_str(self.reason),
        }
    }
}
impl std::error::Error for BundleError {}

fn evaluate_validity(
    record: &EvidenceVersion,
    target: Option<Timestamp>,
    reasons: &mut Vec<ReasonCode>,
) -> (ValidityState, bool) {
    if let (Some(start), Some(end)) = (record.valid_from, record.valid_until) {
        if end <= start {
            reasons.push(ReasonCode::InvalidValidityInterval);
            return (ValidityState::Unspecified, true);
        }
    }
    let Some(target) = target else {
        reasons.push(ReasonCode::TargetTimeNotProvided);
        return (ValidityState::NotEvaluated, false);
    };
    if record.valid_from.is_some_and(|start| target < start) {
        reasons.push(ReasonCode::ValidityNotYetEffective);
        return (ValidityState::NotYetEffectiveAtTarget, false);
    }
    if record.valid_until.is_some_and(|end| target >= end) {
        reasons.push(ReasonCode::ValidityExpired);
        return (ValidityState::ExpiredAtTarget, false);
    }
    if record.valid_from.is_none() && record.valid_until.is_none() {
        reasons.push(ReasonCode::ValidityUnspecified);
        return (ValidityState::Unspecified, false);
    }
    (ValidityState::InForceAtTarget, false)
}

/// Evaluate one exact evidence version against one immutable information frontier.
///
/// A future-effective fact can be admissible knowledge; callers must preserve
/// NotYetEffectiveAtTarget so it cannot be represented as a realized current fact.
pub fn evaluate_evidence(record: &EvidenceVersion, context: &EvaluationContext) -> EvidenceEvaluation {
    let mut reasons = Vec::new();
    let mut failed = false;
    let mut indeterminate = false;

    if context.principal_id.trim().is_empty() || context.rights_profile_id.trim().is_empty() {
        reasons.push(ReasonCode::InvalidEvaluationContext);
        failed = true;
    }

    if record.evidence_id.trim().is_empty()
        || record.source_id.trim().is_empty()
        || record.source_record_id.trim().is_empty()
        || record.revision_group_id.trim().is_empty()
        || record.provenance_family_id.trim().is_empty()
    {
        reasons.push(ReasonCode::EmptyIdentity);
        failed = true;
    }
    if record.rights_profile_id != context.rights_profile_id {
        reasons.push(ReasonCode::RightsProfileMismatch);
        failed = true;
    }
    match record.rights {
        RightsDisposition::Authorized => {}
        RightsDisposition::NotAuthorized => {
            reasons.push(ReasonCode::RightsDenied);
            failed = true;
        }
        RightsDisposition::Unknown => {
            reasons.push(ReasonCode::RightsUnknown);
            indeterminate = true;
        }
    }

    if record.known_at.is_none() {
        reasons.push(ReasonCode::KnownAtMissing);
        indeterminate = true;
    }
    match &record.availability_proof {
        AvailabilityProof::Verified {
            proof_ref,
            evidence_id,
            principal_id,
            available_at,
            ..
        } => {
            if proof_ref.trim().is_empty()
                || evidence_id != &record.evidence_id
                || principal_id != &context.principal_id
            {
                reasons.push(ReasonCode::AvailabilityProofIdentityMismatch);
                failed = true;
            }
            if record.known_at.is_some_and(|known_at| known_at != *available_at) {
                reasons.push(ReasonCode::AvailabilityProofTimeMismatch);
                failed = true;
            }
        }
        AvailabilityProof::Missing => {
            reasons.push(ReasonCode::AvailabilityProofMissing);
            indeterminate = true;
        }
        AvailabilityProof::Unverifiable => {
            reasons.push(ReasonCode::AvailabilityProofUnverifiable);
            indeterminate = true;
        }
    }

    if let Some(known_at) = record.known_at {
        if record.retrieved_at < known_at {
            reasons.push(ReasonCode::RetrievedBeforeKnownAt);
            failed = true;
        }
        if known_at > context.decision_frontier {
            reasons.push(ReasonCode::FutureKnownEvidence);
            failed = true;
        }
        if context.mode == EvaluationMode::ProspectiveStreaming
            && record.retrieved_at > context.decision_frontier
        {
            reasons.push(ReasonCode::LateIngestion);
            failed = true;
        }
    }

    let (validity, invalid_interval) = evaluate_validity(record, context.target_time, &mut reasons);
    if invalid_interval { failed = true; }
    if context.target_time.is_none() { indeterminate = true; }
    let disposition = if failed {
        Disposition::Fail
    } else if indeterminate {
        Disposition::Indeterminate
    } else {
        Disposition::Pass
    };
    reasons.sort();
    reasons.dedup();

    EvidenceEvaluation {
        evidence_id: record.evidence_id.clone(),
        disposition,
        validity,
        reasons,
        superseded_by: Vec::new(),
    }
}

fn is_superseded_by(
    candidate: &EvidenceVersion,
    ancestor_id: &str,
    by_id: &str,
    records: &BTreeMap<&str, &EvidenceVersion>,
    visited: &mut BTreeSet<String>,
) -> bool {
    if !visited.insert(by_id.to_owned()) { return false; }
    let Some(by_record) = records.get(by_id) else { return false; };
    if by_record.supersedes.iter().any(|id| id == ancestor_id) { return true; }
    by_record.supersedes.iter().any(|id| {
        records.get(id.as_str()).is_some_and(|parent| {
            parent.revision_group_id == candidate.revision_group_id
                && is_superseded_by(candidate, ancestor_id, id, records, visited)
        })
    })
}

/// Validate revision ancestry and evaluate a bundle. All versions remain in the output;
/// an admissible later revision only marks the earlier version as superseded at this frontier.
pub fn evaluate_bundle(
    records: &[EvidenceVersion],
    context: &EvaluationContext,
) -> Result<BundleEvaluation, BundleError> {
    let mut by_id = BTreeMap::new();
    for record in records {
        if record.evidence_id.trim().is_empty()
            || by_id.insert(record.evidence_id.as_str(), record).is_some()
        {
            return Err(BundleError {
                reason: "evidence IDs must be non-empty and unique",
                evidence_id: Some(record.evidence_id.clone()),
            });
        }
    }

    for record in records {
        for parent_id in &record.supersedes {
            let Some(parent) = by_id.get(parent_id.as_str()) else {
                return Err(BundleError {
                    reason: "revision ancestry references an unknown evidence ID",
                    evidence_id: Some(record.evidence_id.clone()),
                });
            };
            if parent.evidence_id == record.evidence_id
                || parent.source_id != record.source_id
                || parent.source_record_id != record.source_record_id
                || parent.revision_group_id != record.revision_group_id
            {
                return Err(BundleError {
                    reason: "revision ancestry crosses source identity or contains a self-edge",
                    evidence_id: Some(record.evidence_id.clone()),
                });
            }
        }
    }

    fn visit<'a>(
        id: &'a str,
        records: &BTreeMap<&'a str, &'a EvidenceVersion>,
        state: &mut BTreeMap<&'a str, u8>,
    ) -> bool {
        match state.get(id).copied().unwrap_or(0) {
            1 => return false,
            2 => return true,
            _ => {}
        }
        state.insert(id, 1);
        if let Some(record) = records.get(id) {
            for parent in &record.supersedes {
                if !visit(parent, records, state) { return false; }
            }
        }
        state.insert(id, 2);
        true
    }
    let mut graph_state = BTreeMap::new();
    for id in by_id.keys() {
        if !visit(id, &by_id, &mut graph_state) {
            return Err(BundleError {
                reason: "revision ancestry contains a cycle",
                evidence_id: Some((*id).to_owned()),
            });
        }
    }

    let mut evaluations: Vec<_> = records
        .iter()
        .map(|record| evaluate_evidence(record, context))
        .collect();
    let passing: BTreeSet<String> = evaluations
        .iter()
        .filter(|item| item.disposition == Disposition::Pass)
        .map(|item| item.evidence_id.clone())
        .collect();

    for evaluation in &mut evaluations {
        if evaluation.disposition != Disposition::Pass { continue; }
        let Some(record) = by_id.get(evaluation.evidence_id.as_str()) else { continue; };
        for other_id in &passing {
            if other_id == &evaluation.evidence_id { continue; }
            let mut visited = BTreeSet::new();
            if is_superseded_by(record, &evaluation.evidence_id, other_id, &by_id, &mut visited) {
                evaluation.superseded_by.push(other_id.clone());
            }
        }
        evaluation.superseded_by.sort();
    }

    let independent_source_families = records
        .iter()
        .filter(|record| passing.contains(&record.evidence_id))
        .map(|record| record.provenance_family_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();

    evaluations.sort_by(|left, right| left.evidence_id.cmp(&right.evidence_id));
    Ok(BundleEvaluation { evidence: evaluations, independent_source_families })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(value: &str) -> Timestamp {
        Timestamp::parse_rfc3339(value).expect("valid test timestamp")
    }
    fn context(mode: EvaluationMode, frontier: &str) -> EvaluationContext {
        EvaluationContext {
            mode,
            principal_id: "researcher:public-profile".into(),
            rights_profile_id: "public-research-v1".into(),
            decision_frontier: ts(frontier),
            target_time: Some(ts("2026-11-09T00:00:00Z")),
        }
    }
    fn evidence(id: &str, known: &str, retrieved: &str) -> EvidenceVersion {
        let known_at = ts(known);
        EvidenceVersion {
            evidence_id: id.into(),
            source_id: "source:filing".into(),
            source_record_id: "record:issuer-quarterly".into(),
            revision_group_id: "revision-group:issuer-quarterly".into(),
            provenance_family_id: "family:issuer-filing".into(),
            valid_from: Some(ts("2026-11-01T00:00:00Z")),
            valid_until: None,
            published_at: Some(known_at),
            known_at: Some(known_at),
            retrieved_at: ts(retrieved),
            rights_profile_id: "public-research-v1".into(),
            rights: RightsDisposition::Authorized,
            availability_proof: AvailabilityProof::Verified {
                proof_ref: format!("proof:{id}"),
                kind: AvailabilityProofKind::PrimaryReleaseRecord,
                evidence_id: id.into(),
                principal_id: "researcher:public-profile".into(),
                available_at: known_at,
            },
            supersedes: Vec::new(),
        }
    }

    #[test]
    fn offset_equivalent_instants_compare_equal() {
        assert_eq!(ts("2026-11-09T12:00:00+02:00"), ts("2026-11-09T10:00:00Z"));
        assert_eq!(ts("2026-11-09T05:00:00.1-05:00"), ts("2026-11-09T10:00:00.100000000Z"));
    }

    #[test]
    fn malformed_and_ambiguous_timestamps_reject() {
        for value in [
            "2026-02-30T00:00:00Z",
            "2026-11-09T24:00:00Z",
            "2026-11-09T10:00:60Z",
            "2026-11-09T10:00:00",
            "2026-11-09T10:00:00.1234567890Z",
            "2026-11-09 10:00:00Z",
        ] {
            assert!(Timestamp::parse_rfc3339(value).is_err(), "{value}");
        }
    }

    #[test]
    fn backdated_but_late_known_filing_is_rejected() {
        let mut record = evidence("evidence:backdated-filing", "2026-11-10T09:00:00Z", "2026-11-10T09:01:00Z");
        record.valid_from = Some(ts("2026-10-01T00:00:00Z"));
        let result = evaluate_evidence(&record, &context(EvaluationMode::HistoricalReconstruction, "2026-11-09T12:00:00Z"));
        assert_eq!(result.disposition, Disposition::Fail);
        assert!(result.reasons.contains(&ReasonCode::FutureKnownEvidence));
    }

    #[test]
    fn historical_reconstruction_can_retrieve_later_with_proof() {
        let record = evidence("evidence:archived-filing", "2026-11-08T09:00:00Z", "2026-11-10T09:00:00Z");
        let result = evaluate_evidence(&record, &context(EvaluationMode::HistoricalReconstruction, "2026-11-09T12:00:00Z"));
        assert_eq!(result.disposition, Disposition::Pass);
    }

    #[test]
    fn prospective_streaming_rejects_late_ingestion() {
        let record = evidence("evidence:delayed-feed", "2026-11-09T08:00:00Z", "2026-11-09T13:00:00Z");
        let result = evaluate_evidence(&record, &context(EvaluationMode::ProspectiveStreaming, "2026-11-09T12:00:00Z"));
        assert_eq!(result.disposition, Disposition::Fail);
        assert!(result.reasons.contains(&ReasonCode::LateIngestion));
    }

    #[test]
    fn missing_availability_proof_is_indeterminate_not_a_pass() {
        let mut record = evidence("evidence:unproven-archive", "2026-11-08T09:00:00Z", "2026-11-10T09:00:00Z");
        record.availability_proof = AvailabilityProof::Missing;
        let result = evaluate_evidence(&record, &context(EvaluationMode::HistoricalReconstruction, "2026-11-09T12:00:00Z"));
        assert_eq!(result.disposition, Disposition::Indeterminate);
        assert!(result.reasons.contains(&ReasonCode::AvailabilityProofMissing));
    }

    #[test]
    fn known_future_effective_contract_is_admissible_but_not_current() {
        let mut record = evidence("evidence:announced-contract", "2026-11-08T09:00:00Z", "2026-11-08T09:05:00Z");
        record.valid_from = Some(ts("2026-12-01T00:00:00Z"));
        let result = evaluate_evidence(&record, &context(EvaluationMode::HistoricalReconstruction, "2026-11-09T12:00:00Z"));
        assert_eq!(result.disposition, Disposition::Pass);
        assert_eq!(result.validity, ValidityState::NotYetEffectiveAtTarget);
        assert!(result.reasons.contains(&ReasonCode::ValidityNotYetEffective));
    }

    #[test]
    fn revision_selection_respects_the_decision_frontier_and_retains_ancestry() {
        let v1 = evidence("evidence:original", "2026-11-08T09:00:00Z", "2026-11-08T09:05:00Z");
        let mut v2 = evidence("evidence:restatement", "2026-11-10T09:00:00Z", "2026-11-10T09:05:00Z");
        v2.supersedes = vec!["evidence:original".into()];

        let historical = evaluate_bundle(&[v1.clone(), v2.clone()], &context(EvaluationMode::HistoricalReconstruction, "2026-11-09T12:00:00Z"))
            .expect("valid version chain");
        let original = historical.evidence.iter().find(|item| item.evidence_id == "evidence:original").unwrap();
        let restatement = historical.evidence.iter().find(|item| item.evidence_id == "evidence:restatement").unwrap();
        assert_eq!(original.disposition, Disposition::Pass);
        assert!(original.superseded_by.is_empty());
        assert_eq!(restatement.disposition, Disposition::Fail);

        let later = evaluate_bundle(&[v1, v2], &context(EvaluationMode::HistoricalReconstruction, "2026-11-11T12:00:00Z"))
            .expect("valid version chain");
        let original = later.evidence.iter().find(|item| item.evidence_id == "evidence:original").unwrap();
        assert_eq!(original.disposition, Disposition::Pass);
        assert_eq!(original.superseded_by, vec!["evidence:restatement".to_owned()]);
    }

    #[test]
    fn mirrored_documents_do_not_inflate_independent_source_count() {
        let mut first = evidence("evidence:copy-a", "2026-11-08T09:00:00Z", "2026-11-08T09:05:00Z");
        let mut second = evidence("evidence:copy-b", "2026-11-08T09:00:00Z", "2026-11-08T09:06:00Z");
        first.provenance_family_id = "family:upstream-filing".into();
        second.provenance_family_id = "family:upstream-filing".into();
        second.source_record_id = "mirror:copy-b".into();
        second.revision_group_id = "revision-group:mirror-copy".into();
        let bundle = evaluate_bundle(&[first, second], &context(EvaluationMode::HistoricalReconstruction, "2026-11-09T12:00:00Z"))
            .expect("valid mirror records");
        assert_eq!(bundle.independent_source_families, 1);
    }

    #[test]
    fn historical_replay_is_invariant_to_later_retrieval_time() {
        let first = evidence("evidence:retrieval-a", "2026-11-08T09:00:00Z", "2026-11-09T13:00:00Z");
        let mut second = first.clone();
        second.retrieved_at = ts("2026-11-12T18:00:00Z");
        let ctx = context(EvaluationMode::HistoricalReconstruction, "2026-11-09T12:00:00Z");
        assert_eq!(evaluate_evidence(&first, &ctx), evaluate_evidence(&second, &ctx));
    }

    #[test]
    fn denied_rights_fail_and_unknown_rights_are_indeterminate() {
        let mut denied = evidence("evidence:denied", "2026-11-08T09:00:00Z", "2026-11-08T09:05:00Z");
        denied.rights = RightsDisposition::NotAuthorized;
        assert_eq!(evaluate_evidence(&denied, &context(EvaluationMode::HistoricalReconstruction, "2026-11-09T12:00:00Z")).disposition, Disposition::Fail);

        let mut unknown = denied;
        unknown.evidence_id = "evidence:rights-unknown".into();
        unknown.rights = RightsDisposition::Unknown;
        if let AvailabilityProof::Verified { evidence_id, proof_ref, .. } = &mut unknown.availability_proof {
            *evidence_id = unknown.evidence_id.clone();
            *proof_ref = "proof:rights-unknown".into();
        }
        assert_eq!(evaluate_evidence(&unknown, &context(EvaluationMode::HistoricalReconstruction, "2026-11-09T12:00:00Z")).disposition, Disposition::Indeterminate);
    }


    #[test]
    fn empty_principal_fails_and_missing_target_time_is_indeterminate() {
        let record = evidence("evidence:context-test", "2026-11-08T09:00:00Z", "2026-11-08T09:05:00Z");
        let mut empty_principal = context(EvaluationMode::HistoricalReconstruction, "2026-11-09T12:00:00Z");
        empty_principal.principal_id.clear();
        let failed = evaluate_evidence(&record, &empty_principal);
        assert_eq!(failed.disposition, Disposition::Fail);
        assert!(failed.reasons.contains(&ReasonCode::InvalidEvaluationContext));

        let mut missing_target = context(EvaluationMode::HistoricalReconstruction, "2026-11-09T12:00:00Z");
        missing_target.target_time = None;
        let unresolved = evaluate_evidence(&record, &missing_target);
        assert_eq!(unresolved.disposition, Disposition::Indeterminate);
        assert!(unresolved.reasons.contains(&ReasonCode::TargetTimeNotProvided));
    }

    #[test]
    fn exact_frontier_is_inclusive_and_valid_until_is_exclusive() {
        let mut record = evidence("evidence:boundary", "2026-11-09T12:00:00Z", "2026-11-09T12:00:00Z");
        record.valid_until = Some(ts("2026-11-09T00:00:00Z"));
        let mut ctx = context(EvaluationMode::HistoricalReconstruction, "2026-11-09T12:00:00Z");
        ctx.target_time = Some(ts("2026-11-09T00:00:00Z"));
        let result = evaluate_evidence(&record, &ctx);
        assert_eq!(result.disposition, Disposition::Pass);
        assert_eq!(result.validity, ValidityState::ExpiredAtTarget);
        assert!(!result.reasons.contains(&ReasonCode::FutureKnownEvidence));
    }

    #[test]
    fn actor_scoped_availability_is_not_overridden_by_publication_time() {
        let mut record = evidence("evidence:embargo-access", "2026-11-08T08:00:00Z", "2026-11-08T08:05:00Z");
        record.published_at = Some(ts("2026-11-10T08:00:00Z"));
        let result = evaluate_evidence(&record, &context(EvaluationMode::HistoricalReconstruction, "2026-11-09T12:00:00Z"));
        assert_eq!(result.disposition, Disposition::Pass);
    }

    #[test]
    fn revision_cycles_reject() {
        let mut first = evidence("evidence:cycle-a", "2026-11-08T09:00:00Z", "2026-11-08T09:05:00Z");
        let mut second = evidence("evidence:cycle-b", "2026-11-08T10:00:00Z", "2026-11-08T10:05:00Z");
        first.supersedes = vec![second.evidence_id.clone()];
        second.supersedes = vec![first.evidence_id.clone()];
        assert!(evaluate_bundle(&[first, second], &context(EvaluationMode::HistoricalReconstruction, "2026-11-09T12:00:00Z")).is_err());
    }
}
