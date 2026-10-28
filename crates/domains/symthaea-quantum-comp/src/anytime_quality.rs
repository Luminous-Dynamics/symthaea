//! Verifier-backed anytime-quality traces for unweighted maximum independent set.
//!
//! Actual graph, candidate, and receipt bytes are checked before they contribute to the
//! score. Hash-shaped caller fields and claimed dispositions are not treated as evidence.
//! Artifact paths are checked lexically here; the ingestion layer must also resolve them
//! under the evidence bundle and reject symlink escapes before materializing the bytes.
//! Digests alone do not authenticate the process or prove that a timestamp is truthful.

use super::{
    CandidateVerification, MisEvidenceContext, MisEvidenceVerdict, VerifiedMisCandidate,
    canonical_mis_graph_vertex_count, sha256_hex, verify_mis_candidate_for_trial_with_receipt,
};
use std::{collections::HashSet, error::Error, fmt};

/// Terminal state for one solver attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnytimeTerminalOutcome {
    /// An independently verified feasible set met the frozen target.
    TargetHit,
    /// At least one verified feasible set exists, but none met the target.
    FeasibleBelowTarget,
    /// Events exist, but none is a verified feasible independent set.
    NoFeasibleCandidate,
    /// The execution budget expired.
    TimedOut,
    /// The solver or runner crashed.
    Crashed,
    /// The attempt was explicitly aborted.
    Aborted,
    /// No candidate event was emitted.
    MissingOutput,
}

/// Candidate checkpoint and its materialized artifact bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedAnytimeQualityEvent {
    /// Unique checkpoint ID within this trace.
    pub event_id: String,
    /// Direct deployment elapsed time through persistence of this event's evidence.
    pub elapsed_deployment_ms: f64,
    /// SHA-256 of the exact candidate artifact bytes.
    pub candidate_sha256: String,
    /// Relative candidate-artifact path within the evidence bundle.
    pub candidate_artifact_path: String,
    /// Exact candidate bytes loaded from candidate_artifact_path.
    pub candidate_bytes: Vec<u8>,
    /// Claimed disposition; verified claims are recomputed below.
    pub verification: CandidateVerification,
    /// Caller-reported cardinality, diagnostic only if the event is not verified.
    pub reported_cardinality: Option<usize>,
    /// SHA-256 of the exact receipt bytes, if present.
    pub receipt_sha256: Option<String>,
    /// Relative receipt-artifact path, if present.
    pub receipt_artifact_path: Option<String>,
    /// Exact receipt bytes loaded from receipt_artifact_path, if present.
    pub receipt_bytes: Option<Vec<u8>>,
}

/// All frozen identities and ordered events for one solver trial.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedAnytimeQualityTrace {
    /// Trace schema version, currently exactly 1.0.
    pub schema_version: String,
    /// Direct deployment-time clock boundary required by PHOT-EVAL-005A.
    pub clock_boundary: String,
    /// Protocol identifier.
    pub protocol_id: String,
    /// Lowercase SHA-256 of the frozen experiment protocol.
    pub protocol_sha256: String,
    /// Globally unique trial ID across comparison arms.
    pub trial_id: String,
    /// Pairing key shared by the compared attempts.
    pub pair_id: String,
    /// Canonical base-10 u64 seed identity, without leading zeroes except "0".
    pub seed_id: String,
    /// Frozen solver-budget identifier.
    pub budget_id: String,
    /// SHA-256 of the exact canonical graph bytes supplied to the evaluator.
    pub graph_sha256: String,
    /// SHA-256 identifying the verifier source or build.
    pub verifier_sha256: String,
    /// SHA-256 identifying the frozen solver source.
    pub solver_source_sha256: String,
    /// SHA-256 identifying the frozen solver configuration.
    pub solver_configuration_sha256: String,
    /// SHA-256 identifying the frozen execution environment.
    pub environment_sha256: String,
    /// Vertex count declared by the canonical graph.
    pub graph_vertices: usize,
    /// Frozen target cardinality; not necessarily a proven optimum.
    pub target_cardinality: usize,
    /// Fixed integration horizon in deployment milliseconds.
    pub horizon_ms: f64,
    /// Terminal outcome for the attempt.
    pub terminal_outcome: AnytimeTerminalOutcome,
    /// Ordered checkpoints, including unverified and rejected events.
    pub events: Vec<VerifiedAnytimeQualityEvent>,
}

/// Recomputed summary of the verified incumbent-quality curve.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedAnytimeQualitySummary {
    /// Trial identity from the validated trace.
    pub trial_id: String,
    /// Frozen integration horizon in milliseconds.
    pub horizon_ms: f64,
    /// Frozen graph vertex count.
    pub graph_vertices: usize,
    /// Normalized area under best verified cardinality divided by vertex count.
    pub normalized_area_under_quality_curve: f64,
    /// Best verified feasible cardinality by the horizon, inclusive.
    pub best_verified_cardinality_within_horizon: usize,
    /// Best verified feasible cardinality over all recorded event times.
    pub best_verified_cardinality_all_events: usize,
    /// Number of independently reverified feasible checkpoints.
    pub verified_feasible_event_count: usize,
    /// Number of independently reverified conflicting checkpoints.
    pub verified_infeasible_event_count: usize,
    /// Unverified and verifier-failure checkpoints, which receive no quality credit.
    pub rejected_event_count: usize,
    /// Events strictly after the horizon.
    pub post_horizon_event_count: usize,
    /// Total candidate event count.
    pub event_count: usize,
}

/// Validation error for a verified anytime-quality trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifiedAnytimeQualityError {
    /// Trace schema version is unsupported.
    InvalidSchemaVersion,
    /// Trace clock boundary differs from the frozen protocol definition.
    InvalidClockBoundary,
    /// Empty or noncanonical identifier.
    InvalidIdentifier(&'static str),
    /// Not 64 lowercase hexadecimal characters.
    InvalidDigest(&'static str),
    /// Seed is not canonical base-10 u64.
    InvalidSeedIdentity,
    /// Graph vertex count is invalid or disagrees with graph bytes.
    InvalidGraphDimensions,
    /// Target is zero or larger than the graph.
    InvalidTarget,
    /// Non-finite or non-positive horizon.
    InvalidHorizon,
    /// Invalid event timestamp.
    InvalidTimestamp(String),
    /// Timestamps are not nondecreasing.
    DecreasingTimestamps,
    /// Duplicate event ID.
    DuplicateEventId(String),
    /// Unsafe relative artifact path.
    UnsafeArtifactPath(&'static str, String),
    /// Candidate bytes do not match the declared digest.
    CandidateDigestMismatch(String),
    /// Receipt bytes do not match the declared digest.
    ReceiptDigestMismatch(String),
    /// Receipt path/digest/bytes are only partially present.
    IncompleteReceiptArtifact(String),
    /// Candidate event claims that no candidate was returned.
    UnexpectedNoCandidate(String),
    /// A verified event lacks receipt bytes.
    MissingVerifiedReceipt(String),
    /// Reverification of receipt/candidate/context failed.
    ReceiptVerificationFailed(String),
    /// Declared verification disposition disagrees with recomputation.
    VerificationDispositionMismatch(String),
    /// Reported cardinality differs from parsed candidate bytes.
    ReportedCardinalityMismatch(String),
    /// Cardinality is outside the graph bounds.
    CardinalityOutOfRange(String),
    /// Terminal outcome conflicts with verified events.
    TerminalOutcomeContradictsEvidence,
}

impl fmt::Display for VerifiedAnytimeQualityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion => write!(f, "unsupported anytime trace schema version"),
            Self::InvalidClockBoundary => write!(f, "trace clock boundary does not match protocol"),
            Self::InvalidIdentifier(field) => write!(f, "invalid identifier: {field}"),
            Self::InvalidDigest(field) => write!(f, "invalid SHA-256 digest: {field}"),
            Self::InvalidSeedIdentity => write!(f, "seed identity is not canonical base-10 u64"),
            Self::InvalidGraphDimensions => write!(f, "invalid or inconsistent graph dimensions"),
            Self::InvalidTarget => write!(f, "target cardinality is invalid"),
            Self::InvalidHorizon => write!(f, "horizon must be finite and positive"),
            Self::InvalidTimestamp(event) => write!(f, "invalid timestamp for event {event}"),
            Self::DecreasingTimestamps => write!(f, "event timestamps are not nondecreasing"),
            Self::DuplicateEventId(event) => write!(f, "duplicate event ID: {event}"),
            Self::UnsafeArtifactPath(field, path) => write!(f, "unsafe {field} path: {path}"),
            Self::CandidateDigestMismatch(event) => write!(f, "candidate digest mismatch: {event}"),
            Self::ReceiptDigestMismatch(event) => write!(f, "receipt digest mismatch: {event}"),
            Self::IncompleteReceiptArtifact(event) => write!(f, "incomplete receipt artifact: {event}"),
            Self::UnexpectedNoCandidate(event) => write!(f, "no-candidate disposition in event {event}"),
            Self::MissingVerifiedReceipt(event) => write!(f, "missing receipt for verified event {event}"),
            Self::ReceiptVerificationFailed(event) => write!(f, "reverification failed for event {event}"),
            Self::VerificationDispositionMismatch(event) => write!(f, "disposition mismatch for event {event}"),
            Self::ReportedCardinalityMismatch(event) => write!(f, "reported cardinality mismatch for {event}"),
            Self::CardinalityOutOfRange(event) => write!(f, "cardinality exceeds graph size at {event}"),
            Self::TerminalOutcomeContradictsEvidence => write!(f, "terminal outcome contradicts verified evidence"),
        }
    }
}
impl Error for VerifiedAnytimeQualityError {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_sha256(value: &str, field: &'static str) -> Result<(), VerifiedAnytimeQualityError> {
    if valid_sha256(value) {
        Ok(())
    } else {
        Err(VerifiedAnytimeQualityError::InvalidDigest(field))
    }
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), VerifiedAnytimeQualityError> {
    if value.is_empty()
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-')
        })
    {
        Err(VerifiedAnytimeQualityError::InvalidIdentifier(field))
    } else {
        Ok(())
    }
}

fn validate_relative_artifact_path(
    path: &str,
    field: &'static str,
) -> Result<(), VerifiedAnytimeQualityError> {
    if path.is_empty()
        || path.starts_with('/')
        || path.starts_with('\\')
        || path.contains('\\')
        || path.contains(':')
        || path.contains('\0')
        || path.split('/').any(|part| part.is_empty() || part == "." || part == "..")
    {
        Err(VerifiedAnytimeQualityError::UnsafeArtifactPath(field, path.to_owned()))
    } else {
        Ok(())
    }
}

/// Recompute an anytime-quality score from canonical graph and exact event artifact bytes.
///
/// The caller must materialize bytes from the paths in the trace. Every event claiming a
/// verified disposition is rechecked against the graph and a receipt bound to the trace's
/// protocol, trial, pair, and seed identity. Any failed claimed verification invalidates
/// the trace instead of silently downgrading a claimed verified event.
pub fn compare_verified_anytime_quality(
    graph_bytes: &[u8],
    trace: &VerifiedAnytimeQualityTrace,
) -> Result<VerifiedAnytimeQualitySummary, VerifiedAnytimeQualityError> {
    if trace.schema_version != "1.0" {
        return Err(VerifiedAnytimeQualityError::InvalidSchemaVersion);
    }
    if trace.clock_boundary != "request-entry-to-persisted-candidate-and-verification-evidence" {
        return Err(VerifiedAnytimeQualityError::InvalidClockBoundary);
    }
    validate_identifier(&trace.protocol_id, "protocol_id")?;
    validate_identifier(&trace.trial_id, "trial_id")?;
    validate_identifier(&trace.pair_id, "pair_id")?;
    validate_identifier(&trace.budget_id, "budget_id")?;
    for (value, field) in [
        (&trace.protocol_sha256, "protocol_sha256"),
        (&trace.graph_sha256, "graph_sha256"),
        (&trace.verifier_sha256, "verifier_sha256"),
        (&trace.solver_source_sha256, "solver_source_sha256"),
        (&trace.solver_configuration_sha256, "solver_configuration_sha256"),
        (&trace.environment_sha256, "environment_sha256"),
    ] {
        require_sha256(value, field)?;
    }
    if sha256_hex(graph_bytes) != trace.graph_sha256 {
        return Err(VerifiedAnytimeQualityError::InvalidDigest("graph_bytes"));
    }
    if trace.seed_id.is_empty()
        || !trace.seed_id.bytes().all(|byte| byte.is_ascii_digit())
        || (trace.seed_id.len() > 1 && trace.seed_id.starts_with('0'))
        || trace.seed_id.parse::<u64>().map(|seed| seed.to_string() != trace.seed_id).unwrap_or(true)
    {
        return Err(VerifiedAnytimeQualityError::InvalidSeedIdentity);
    }
    if !trace.horizon_ms.is_finite() || trace.horizon_ms <= 0.0 {
        return Err(VerifiedAnytimeQualityError::InvalidHorizon);
    }
    if trace.graph_vertices == 0
        || canonical_mis_graph_vertex_count(graph_bytes).ok() != Some(trace.graph_vertices)
    {
        return Err(VerifiedAnytimeQualityError::InvalidGraphDimensions);
    }
    if trace.target_cardinality == 0 || trace.target_cardinality > trace.graph_vertices {
        return Err(VerifiedAnytimeQualityError::InvalidTarget);
    }

    let seed = trace.seed_id.parse::<u64>()
        .map_err(|_| VerifiedAnytimeQualityError::InvalidSeedIdentity)?;
    let mut seen_ids = HashSet::with_capacity(trace.events.len());
    let mut previous_time = 0.0f64;
    let mut have_previous_time = false;
    let mut feasible_count = 0usize;
    let mut infeasible_count = 0usize;
    let mut rejected_count = 0usize;
    let mut post_horizon_count = 0usize;
    let mut best_within_horizon = 0usize;
    let mut best_all_events = 0usize;
    let mut target_hit = false;
    let mut feasible_events: Vec<(f64, usize)> = Vec::new();

    for event in &trace.events {
        validate_identifier(&event.event_id, "event_id")?;
        if !seen_ids.insert(event.event_id.as_str()) {
            return Err(VerifiedAnytimeQualityError::DuplicateEventId(event.event_id.clone()));
        }
        if !event.elapsed_deployment_ms.is_finite() || event.elapsed_deployment_ms < 0.0 {
            return Err(VerifiedAnytimeQualityError::InvalidTimestamp(event.event_id.clone()));
        }
        if have_previous_time && event.elapsed_deployment_ms < previous_time {
            return Err(VerifiedAnytimeQualityError::DecreasingTimestamps);
        }
        previous_time = event.elapsed_deployment_ms;
        have_previous_time = true;

        validate_relative_artifact_path(&event.candidate_artifact_path, "candidate")?;
        require_sha256(&event.candidate_sha256, "candidate_sha256")?;
        if sha256_hex(&event.candidate_bytes) != event.candidate_sha256 {
            return Err(VerifiedAnytimeQualityError::CandidateDigestMismatch(event.event_id.clone()));
        }
        if event.reported_cardinality.is_some_and(|count| count > trace.graph_vertices) {
            return Err(VerifiedAnytimeQualityError::CardinalityOutOfRange(event.event_id.clone()));
        }

        let receipt_presence = [
            event.receipt_sha256.is_some(),
            event.receipt_artifact_path.is_some(),
            event.receipt_bytes.is_some(),
        ];
        if receipt_presence.iter().any(|value| *value) && !receipt_presence.iter().all(|value| *value) {
            return Err(VerifiedAnytimeQualityError::IncompleteReceiptArtifact(event.event_id.clone()));
        }
        if let (Some(digest), Some(path), Some(bytes)) = (
            event.receipt_sha256.as_deref(),
            event.receipt_artifact_path.as_deref(),
            event.receipt_bytes.as_deref(),
        ) {
            validate_relative_artifact_path(path, "receipt")?;
            require_sha256(digest, "receipt_sha256")?;
            if sha256_hex(bytes) != digest {
                return Err(VerifiedAnytimeQualityError::ReceiptDigestMismatch(event.event_id.clone()));
            }
        }

        let verified_evidence: Option<VerifiedMisCandidate> = match event.verification {
            CandidateVerification::VerifiedIndependentSet | CandidateVerification::VerifiedConflict => {
                let (receipt_bytes, receipt_digest) = match (
                    event.receipt_bytes.as_deref(),
                    event.receipt_sha256.as_deref(),
                ) {
                    (Some(bytes), Some(digest)) => (bytes, digest),
                    _ => return Err(VerifiedAnytimeQualityError::MissingVerifiedReceipt(event.event_id.clone())),
                };
                let context = MisEvidenceContext::new(
                    &trace.protocol_sha256, &trace.trial_id, &trace.pair_id, seed,
                ).map_err(|_| VerifiedAnytimeQualityError::ReceiptVerificationFailed(event.event_id.clone()))?;
                let evidence = verify_mis_candidate_for_trial_with_receipt(
                    graph_bytes,
                    &event.candidate_bytes,
                    trace.target_cardinality,
                    &trace.verifier_sha256,
                    context,
                    receipt_bytes,
                ).map_err(|_| VerifiedAnytimeQualityError::ReceiptVerificationFailed(event.event_id.clone()))?;
                if evidence.candidate_sha256() != event.candidate_sha256
                    || evidence.receipt_sha256() != receipt_digest
                {
                    return Err(VerifiedAnytimeQualityError::ReceiptVerificationFailed(event.event_id.clone()));
                }
                if event.reported_cardinality.is_some_and(|count| count != evidence.candidate_size()) {
                    return Err(VerifiedAnytimeQualityError::ReportedCardinalityMismatch(event.event_id.clone()));
                }
                let disposition_matches = match event.verification {
                    CandidateVerification::VerifiedIndependentSet =>
                        evidence.verdict() == MisEvidenceVerdict::VerifiedIndependentSet
                            && evidence.conflict_count() == 0,
                    CandidateVerification::VerifiedConflict =>
                        evidence.verdict() == MisEvidenceVerdict::VerifiedConflict
                            && evidence.conflict_count() > 0,
                    _ => false,
                };
                if !disposition_matches {
                    return Err(VerifiedAnytimeQualityError::VerificationDispositionMismatch(event.event_id.clone()));
                }
                Some(evidence)
            }
            CandidateVerification::Unverified | CandidateVerification::VerificationFailed => {
                // Receipt metadata on a rejected event is never used as authority.
                rejected_count += 1;
                None
            }
            CandidateVerification::NotReturned => {
                return Err(VerifiedAnytimeQualityError::UnexpectedNoCandidate(event.event_id.clone()));
            }
        };

        if let Some(evidence) = verified_evidence {
            match evidence.verdict() {
                MisEvidenceVerdict::VerifiedIndependentSet => {
                    let cardinality = evidence.candidate_size();
                    feasible_count += 1;
                    best_all_events = best_all_events.max(cardinality);
                    if event.elapsed_deployment_ms <= trace.horizon_ms {
                        best_within_horizon = best_within_horizon.max(cardinality);
                        feasible_events.push((event.elapsed_deployment_ms, cardinality));
                    }
                    if cardinality >= trace.target_cardinality {
                        target_hit = true;
                    }
                }
                MisEvidenceVerdict::VerifiedConflict => infeasible_count += 1,
            }
        }
        if event.elapsed_deployment_ms > trace.horizon_ms {
            post_horizon_count += 1;
        }
    }

    match trace.terminal_outcome {
        AnytimeTerminalOutcome::TargetHit if !target_hit =>
            return Err(VerifiedAnytimeQualityError::TerminalOutcomeContradictsEvidence),
        AnytimeTerminalOutcome::FeasibleBelowTarget if feasible_count == 0 || target_hit =>
            return Err(VerifiedAnytimeQualityError::TerminalOutcomeContradictsEvidence),
        AnytimeTerminalOutcome::NoFeasibleCandidate if trace.events.is_empty() || feasible_count > 0 =>
            return Err(VerifiedAnytimeQualityError::TerminalOutcomeContradictsEvidence),
        AnytimeTerminalOutcome::MissingOutput if !trace.events.is_empty() =>
            return Err(VerifiedAnytimeQualityError::TerminalOutcomeContradictsEvidence),
        _ => {}
    }

    // Timestamps are already monotonic. Group ties and update the incumbent with the
    // maximum verified cardinality at that timestamp, so event order cannot change AUC.
    let mut integrated_quality_ms = 0.0f64;
    let mut incumbent = 0usize;
    let mut cursor = 0.0f64;
    let mut index = 0usize;
    while index < feasible_events.len() {
        let time = feasible_events[index].0;
        integrated_quality_ms += (time - cursor) * (incumbent as f64 / trace.graph_vertices as f64);
        cursor = time;
        while index < feasible_events.len() && feasible_events[index].0 == time {
            incumbent = incumbent.max(feasible_events[index].1);
            index += 1;
        }
    }
    integrated_quality_ms +=
        (trace.horizon_ms - cursor) * (incumbent as f64 / trace.graph_vertices as f64);
    let normalized_area_under_quality_curve =
        (integrated_quality_ms / trace.horizon_ms).clamp(0.0, 1.0);

    Ok(VerifiedAnytimeQualitySummary {
        trial_id: trace.trial_id.clone(),
        horizon_ms: trace.horizon_ms,
        graph_vertices: trace.graph_vertices,
        normalized_area_under_quality_curve,
        best_verified_cardinality_within_horizon: best_within_horizon,
        best_verified_cardinality_all_events: best_all_events,
        verified_feasible_event_count: feasible_count,
        verified_infeasible_event_count: infeasible_count,
        rejected_event_count: rejected_count,
        post_horizon_event_count: post_horizon_count,
        event_count: trace.events.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::verify_mis_candidate_for_trial;

    const GRAPH: &[u8] = b"SYMTHEA-MIS-GRAPH-V1\nvertices 10\nedges 0\n";
    const PROTOCOL: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const VERIFIER: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SOURCE: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const CONFIG: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const ENVIRONMENT: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn verified_event(
        event_id: &str, elapsed_ms: f64, vertices: &[usize], trial_id: &str, pair_id: &str, seed: u64,
    ) -> VerifiedAnytimeQualityEvent {
        let candidate_bytes = super::super::encode_canonical_mis_candidate(10, vertices).unwrap();
        let context = MisEvidenceContext::new(PROTOCOL, trial_id, pair_id, seed).unwrap();
        let evidence = verify_mis_candidate_for_trial(
            GRAPH, &candidate_bytes, 5, VERIFIER, context.clone(),
        ).unwrap();
        let receipt_bytes = evidence.receipt_bytes().to_vec();
        let checked = verify_mis_candidate_for_trial_with_receipt(
            GRAPH, &candidate_bytes, 5, VERIFIER, context, &receipt_bytes,
        ).unwrap();
        VerifiedAnytimeQualityEvent {
            event_id: event_id.to_owned(),
            elapsed_deployment_ms: elapsed_ms,
            candidate_sha256: sha256_hex(&candidate_bytes),
            candidate_artifact_path: format!("candidates/{event_id}.bin"),
            candidate_bytes,
            verification: if checked.verdict() == MisEvidenceVerdict::VerifiedIndependentSet {
                CandidateVerification::VerifiedIndependentSet
            } else { CandidateVerification::VerifiedConflict },
            reported_cardinality: Some(checked.candidate_size()),
            receipt_sha256: Some(sha256_hex(&receipt_bytes)),
            receipt_artifact_path: Some(format!("receipts/{event_id}.json")),
            receipt_bytes: Some(receipt_bytes),
        }
    }

    fn unverified_event(event_id: &str, elapsed_ms: f64, vertices: &[usize]) -> VerifiedAnytimeQualityEvent {
        let candidate_bytes = super::super::encode_canonical_mis_candidate(10, vertices).unwrap();
        VerifiedAnytimeQualityEvent {
            event_id: event_id.to_owned(),
            elapsed_deployment_ms: elapsed_ms,
            candidate_sha256: sha256_hex(&candidate_bytes),
            candidate_artifact_path: format!("candidates/{event_id}.bin"),
            candidate_bytes,
            verification: CandidateVerification::Unverified,
            reported_cardinality: Some(vertices.len()),
            receipt_sha256: None,
            receipt_artifact_path: None,
            receipt_bytes: None,
        }
    }

    fn trace(
        terminal_outcome: AnytimeTerminalOutcome, events: Vec<VerifiedAnytimeQualityEvent>,
    ) -> VerifiedAnytimeQualityTrace {
        VerifiedAnytimeQualityTrace {
            schema_version: "1.0".to_owned(),
            clock_boundary: "request-entry-to-persisted-candidate-and-verification-evidence".to_owned(),
            protocol_id: "PHOT-EVAL-005A".to_owned(),
            protocol_sha256: PROTOCOL.to_owned(),
            trial_id: "trial-1".to_owned(),
            pair_id: "pair-1".to_owned(),
            seed_id: "7".to_owned(),
            budget_id: "budget-1".to_owned(),
            graph_sha256: sha256_hex(GRAPH),
            verifier_sha256: VERIFIER.to_owned(),
            solver_source_sha256: SOURCE.to_owned(),
            solver_configuration_sha256: CONFIG.to_owned(),
            environment_sha256: ENVIRONMENT.to_owned(),
            graph_vertices: 10,
            target_cardinality: 5,
            horizon_ms: 1000.0,
            terminal_outcome,
            events,
        }
    }

    #[test]
    fn ignores_unverified_large_claim_and_computes_hand_calculated_auc() {
        let trace = trace(AnytimeTerminalOutcome::FeasibleBelowTarget, vec![
            unverified_event("unverified-8", 20.0, &[0, 1, 2, 3, 4, 5, 6, 7]),
            verified_event("verified-2", 100.0, &[0, 1], "trial-1", "pair-1", 7),
            verified_event("verified-4", 350.0, &[0, 1, 2, 3], "trial-1", "pair-1", 7),
        ]);
        let result = compare_verified_anytime_quality(GRAPH, &trace).unwrap();
        assert!((result.normalized_area_under_quality_curve - 0.31).abs() < 1e-12);
        assert_eq!(result.best_verified_cardinality_within_horizon, 4);
        assert_eq!(result.best_verified_cardinality_all_events, 4);
        assert_eq!(result.verified_feasible_event_count, 2);
        assert_eq!(result.rejected_event_count, 1);
    }

    #[test]
    fn empty_trace_scores_zero_and_requires_missing_output_terminal() {
        let result = compare_verified_anytime_quality(
            GRAPH, &trace(AnytimeTerminalOutcome::MissingOutput, vec![]),
        ).unwrap();
        assert_eq!(result.normalized_area_under_quality_curve, 0.0);
        assert_eq!(result.best_verified_cardinality_within_horizon, 0);
        assert_eq!(result.event_count, 0);
        assert!(matches!(
            compare_verified_anytime_quality(GRAPH, &trace(AnytimeTerminalOutcome::NoFeasibleCandidate, vec![])),
            Err(VerifiedAnytimeQualityError::TerminalOutcomeContradictsEvidence)
        ));
    }

    #[test]
    fn equal_timestamp_events_are_order_independent() {
        let card2 = verified_event("two", 100.0, &[0, 1], "trial-1", "pair-1", 7);
        let card4 = verified_event("four", 100.0, &[0, 1, 2, 3], "trial-1", "pair-1", 7);
        let forward = compare_verified_anytime_quality(GRAPH, &trace(
            AnytimeTerminalOutcome::FeasibleBelowTarget, vec![card2.clone(), card4.clone()],
        )).unwrap();
        let reverse = compare_verified_anytime_quality(GRAPH, &trace(
            AnytimeTerminalOutcome::FeasibleBelowTarget, vec![card4, card2],
        )).unwrap();
        assert_eq!(forward.normalized_area_under_quality_curve, reverse.normalized_area_under_quality_curve);
        assert!((forward.normalized_area_under_quality_curve - 0.36).abs() < 1e-12);
    }

    #[test]
    fn checkpoint_exactly_at_horizon_has_no_positive_area() {
        let result = compare_verified_anytime_quality(GRAPH, &trace(
            AnytimeTerminalOutcome::FeasibleBelowTarget,
            vec![verified_event("at-horizon", 1000.0, &[0, 1, 2, 3], "trial-1", "pair-1", 7)],
        )).unwrap();
        assert_eq!(result.normalized_area_under_quality_curve, 0.0);
        assert_eq!(result.best_verified_cardinality_within_horizon, 4);
    }

    #[test]
    fn post_horizon_solution_is_retained_but_does_not_change_auc() {
        let result = compare_verified_anytime_quality(GRAPH, &trace(
            AnytimeTerminalOutcome::TargetHit,
            vec![verified_event("late-target", 1200.0, &[0, 1, 2, 3, 4], "trial-1", "pair-1", 7)],
        )).unwrap();
        assert_eq!(result.normalized_area_under_quality_curve, 0.0);
        assert_eq!(result.best_verified_cardinality_within_horizon, 0);
        assert_eq!(result.best_verified_cardinality_all_events, 5);
        assert_eq!(result.post_horizon_event_count, 1);
    }

    #[test]
    fn tampered_receipt_bytes_fail_even_when_digest_is_updated() {
        let mut event = verified_event("valid", 100.0, &[0, 1, 2], "trial-1", "pair-1", 7);
        let mut receipt = event.receipt_bytes.clone().unwrap();
        receipt.push(b'X');
        event.receipt_sha256 = Some(sha256_hex(&receipt));
        event.receipt_bytes = Some(receipt);
        assert!(matches!(
            compare_verified_anytime_quality(GRAPH, &trace(
                AnytimeTerminalOutcome::FeasibleBelowTarget, vec![event],
            )),
            Err(VerifiedAnytimeQualityError::ReceiptVerificationFailed(_))
        ));
    }

    #[test]
    fn rejects_cross_trial_receipt_context() {
        let event = verified_event("valid", 100.0, &[0, 1, 2], "trial-other", "pair-1", 7);
        assert!(matches!(
            compare_verified_anytime_quality(GRAPH, &trace(
                AnytimeTerminalOutcome::FeasibleBelowTarget, vec![event],
            )),
            Err(VerifiedAnytimeQualityError::ReceiptVerificationFailed(_))
        ));
    }

    #[test]
    fn rejects_duplicate_ids_decreasing_times_unsafe_paths_and_bad_cards() {
        let first = unverified_event("one", 100.0, &[0, 1]);
        let duplicate = unverified_event("one", 200.0, &[0, 1]);
        assert!(matches!(
            compare_verified_anytime_quality(GRAPH, &trace(
                AnytimeTerminalOutcome::NoFeasibleCandidate, vec![first.clone(), duplicate],
            )),
            Err(VerifiedAnytimeQualityError::DuplicateEventId(_))
        ));
        assert!(matches!(
            compare_verified_anytime_quality(GRAPH, &trace(
                AnytimeTerminalOutcome::NoFeasibleCandidate,
                vec![unverified_event("late", 200.0, &[0]), unverified_event("early", 100.0, &[1])],
            )),
            Err(VerifiedAnytimeQualityError::DecreasingTimestamps)
        ));
        let mut unsafe_path = unverified_event("unsafe", 10.0, &[0]);
        unsafe_path.candidate_artifact_path = "../escape.bin".to_owned();
        assert!(matches!(
            compare_verified_anytime_quality(GRAPH, &trace(
                AnytimeTerminalOutcome::NoFeasibleCandidate, vec![unsafe_path],
            )),
            Err(VerifiedAnytimeQualityError::UnsafeArtifactPath(_, _))
        ));
        let mut too_large = unverified_event("too-large", 10.0, &[0]);
        too_large.reported_cardinality = Some(11);
        assert!(matches!(
            compare_verified_anytime_quality(GRAPH, &trace(
                AnytimeTerminalOutcome::NoFeasibleCandidate, vec![too_large],
            )),
            Err(VerifiedAnytimeQualityError::CardinalityOutOfRange(_))
        ));
    }


    #[test]
    fn rejects_unsupported_schema_and_clock_boundary() {
        let valid = trace(AnytimeTerminalOutcome::MissingOutput, vec![]);
        let mut wrong_version = valid.clone();
        wrong_version.schema_version = "2.0".to_owned();
        assert!(matches!(
            compare_verified_anytime_quality(GRAPH, &wrong_version),
            Err(VerifiedAnytimeQualityError::InvalidSchemaVersion)
        ));
        let mut wrong_clock = valid;
        wrong_clock.clock_boundary = "solver-discovery-time".to_owned();
        assert!(matches!(
            compare_verified_anytime_quality(GRAPH, &wrong_clock),
            Err(VerifiedAnytimeQualityError::InvalidClockBoundary)
        ));
    }

    #[test]
    fn rejects_bad_graph_identity_and_nonfinite_timestamps() {
        let mut wrong_graph = trace(AnytimeTerminalOutcome::MissingOutput, vec![]);
        wrong_graph.graph_sha256 = SOURCE.to_owned();
        assert!(matches!(
            compare_verified_anytime_quality(GRAPH, &wrong_graph),
            Err(VerifiedAnytimeQualityError::InvalidDigest("graph_bytes"))
        ));
        let mut invalid_time = unverified_event("not-finite", f64::INFINITY, &[0]);
        assert!(matches!(
            compare_verified_anytime_quality(GRAPH, &trace(
                AnytimeTerminalOutcome::NoFeasibleCandidate, vec![invalid_time],
            )),
            Err(VerifiedAnytimeQualityError::InvalidTimestamp(_))
        ));
    }
}
