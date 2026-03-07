//! Protocol violation tracking and reporting for DKG ceremonies
//!
//! Records and classifies DKG protocol violations for integration with
//! external reputation/slashing systems (e.g., Mycelix consciousness gating).

use serde::{Deserialize, Serialize};

use crate::participant::ParticipantId;

/// Severity of a protocol violation
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Minor: late submission, slow response (informational)
    Minor = 1,
    /// Moderate: invalid share that could be accidental
    Moderate = 2,
    /// Severe: equivocation, sending different shares to different participants
    Severe = 3,
    /// Critical: attempted key extraction, providing invalid commitments
    Critical = 4,
}

/// Type of protocol violation
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationType {
    /// Failed to submit deal within timeout
    DealTimeout,
    /// Submitted an invalid share (verification failed)
    InvalidShare {
        /// The affected recipient
        recipient: u32,
    },
    /// Submitted invalid commitments
    InvalidCommitment,
    /// Sent different shares to different verification paths (equivocation)
    Equivocation {
        /// Evidence: hash of the two conflicting shares
        evidence_hash: [u8; 32],
    },
    /// Hash commitment mismatch during reveal phase
    CommitRevealMismatch,
    /// Submitted non-zero-sharing polynomial during refresh
    InvalidRefreshPolynomial,
    /// Attempted to submit a deal in wrong phase
    WrongPhaseSubmission,
    /// Duplicate deal submission
    DuplicateSubmission,
}

impl ViolationType {
    /// Get the default severity for this violation type
    pub fn default_severity(&self) -> ViolationSeverity {
        match self {
            Self::DealTimeout => ViolationSeverity::Minor,
            Self::WrongPhaseSubmission => ViolationSeverity::Minor,
            Self::DuplicateSubmission => ViolationSeverity::Minor,
            Self::InvalidShare { .. } => ViolationSeverity::Moderate,
            Self::CommitRevealMismatch => ViolationSeverity::Severe,
            Self::InvalidCommitment => ViolationSeverity::Severe,
            Self::InvalidRefreshPolynomial => ViolationSeverity::Severe,
            Self::Equivocation { .. } => ViolationSeverity::Critical,
        }
    }
}

/// A recorded protocol violation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Violation {
    /// The violating participant
    pub participant: ParticipantId,
    /// Type of violation
    pub violation_type: ViolationType,
    /// Severity level
    pub severity: ViolationSeverity,
    /// The epoch/ceremony in which this occurred
    pub epoch: u64,
    /// Unix timestamp when the violation was recorded
    pub timestamp_secs: u64,
}

impl Violation {
    /// Create a new violation record
    pub fn new(
        participant: ParticipantId,
        violation_type: ViolationType,
        epoch: u64,
        timestamp_secs: u64,
    ) -> Self {
        let severity = violation_type.default_severity();
        Self {
            participant,
            violation_type,
            severity,
            epoch,
            timestamp_secs,
        }
    }

    /// Create with custom severity override
    pub fn with_severity(
        participant: ParticipantId,
        violation_type: ViolationType,
        severity: ViolationSeverity,
        epoch: u64,
        timestamp_secs: u64,
    ) -> Self {
        Self {
            participant,
            violation_type,
            severity,
            epoch,
            timestamp_secs,
        }
    }
}

/// Collects violations during a ceremony and provides summary statistics
#[derive(Debug, Default)]
pub struct ViolationTracker {
    violations: Vec<Violation>,
}

impl ViolationTracker {
    /// Create a new empty tracker
    pub fn new() -> Self {
        Self {
            violations: Vec::new(),
        }
    }

    /// Record a violation
    pub fn record(&mut self, violation: Violation) {
        self.violations.push(violation);
    }

    /// Record a violation from components
    pub fn record_violation(
        &mut self,
        participant: ParticipantId,
        violation_type: ViolationType,
        epoch: u64,
        timestamp_secs: u64,
    ) {
        self.violations.push(Violation::new(
            participant,
            violation_type,
            epoch,
            timestamp_secs,
        ));
    }

    /// Get all violations
    pub fn violations(&self) -> &[Violation] {
        &self.violations
    }

    /// Get violations for a specific participant
    pub fn violations_for(&self, participant: ParticipantId) -> Vec<&Violation> {
        self.violations
            .iter()
            .filter(|v| v.participant == participant)
            .collect()
    }

    /// Get violations at or above a severity level
    pub fn violations_at_severity(&self, min_severity: ViolationSeverity) -> Vec<&Violation> {
        self.violations
            .iter()
            .filter(|v| v.severity >= min_severity)
            .collect()
    }

    /// Count total violations
    pub fn total_count(&self) -> usize {
        self.violations.len()
    }

    /// Count violations for a specific participant
    pub fn count_for(&self, participant: ParticipantId) -> usize {
        self.violations
            .iter()
            .filter(|v| v.participant == participant)
            .count()
    }

    /// Get the maximum severity violation for a participant
    pub fn max_severity_for(&self, participant: ParticipantId) -> Option<ViolationSeverity> {
        self.violations
            .iter()
            .filter(|v| v.participant == participant)
            .map(|v| v.severity)
            .max()
    }

    /// Compute a reputation penalty score for a participant (0.0 = no penalty, 1.0 = maximum)
    ///
    /// Weighted by severity: Minor=0.05, Moderate=0.15, Severe=0.40, Critical=1.0
    pub fn penalty_score(&self, participant: ParticipantId) -> f64 {
        let mut score = 0.0f64;
        for v in self.violations.iter().filter(|v| v.participant == participant) {
            score += match v.severity {
                ViolationSeverity::Minor => 0.05,
                ViolationSeverity::Moderate => 0.15,
                ViolationSeverity::Severe => 0.40,
                ViolationSeverity::Critical => 1.0,
            };
        }
        score.min(1.0)
    }

    /// Get all participants who have violations
    pub fn violating_participants(&self) -> Vec<ParticipantId> {
        let mut seen = Vec::new();
        for v in &self.violations {
            if !seen.contains(&v.participant) {
                seen.push(v.participant);
            }
        }
        seen
    }

    /// Drain all violations (transfers ownership, clears the tracker)
    pub fn drain(&mut self) -> Vec<Violation> {
        std::mem::take(&mut self.violations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_violation_severity_ordering() {
        assert!(ViolationSeverity::Minor < ViolationSeverity::Moderate);
        assert!(ViolationSeverity::Moderate < ViolationSeverity::Severe);
        assert!(ViolationSeverity::Severe < ViolationSeverity::Critical);
    }

    #[test]
    fn test_violation_type_default_severity() {
        assert_eq!(ViolationType::DealTimeout.default_severity(), ViolationSeverity::Minor);
        assert_eq!(ViolationType::InvalidShare { recipient: 1 }.default_severity(), ViolationSeverity::Moderate);
        assert_eq!(ViolationType::InvalidCommitment.default_severity(), ViolationSeverity::Severe);
        assert_eq!(ViolationType::Equivocation { evidence_hash: [0u8; 32] }.default_severity(), ViolationSeverity::Critical);
    }

    #[test]
    fn test_tracker_record_and_query() {
        let mut tracker = ViolationTracker::new();

        tracker.record_violation(ParticipantId(1), ViolationType::DealTimeout, 0, 100);
        tracker.record_violation(ParticipantId(1), ViolationType::InvalidShare { recipient: 2 }, 0, 101);
        tracker.record_violation(ParticipantId(2), ViolationType::InvalidCommitment, 0, 102);

        assert_eq!(tracker.total_count(), 3);
        assert_eq!(tracker.count_for(ParticipantId(1)), 2);
        assert_eq!(tracker.count_for(ParticipantId(2)), 1);
        assert_eq!(tracker.count_for(ParticipantId(3)), 0);
    }

    #[test]
    fn test_tracker_severity_filter() {
        let mut tracker = ViolationTracker::new();

        tracker.record_violation(ParticipantId(1), ViolationType::DealTimeout, 0, 100);
        tracker.record_violation(ParticipantId(1), ViolationType::InvalidShare { recipient: 2 }, 0, 101);
        tracker.record_violation(ParticipantId(2), ViolationType::Equivocation { evidence_hash: [0; 32] }, 0, 102);

        let severe_plus = tracker.violations_at_severity(ViolationSeverity::Severe);
        assert_eq!(severe_plus.len(), 1); // Only equivocation

        let moderate_plus = tracker.violations_at_severity(ViolationSeverity::Moderate);
        assert_eq!(moderate_plus.len(), 2); // InvalidShare + Equivocation
    }

    #[test]
    fn test_tracker_max_severity() {
        let mut tracker = ViolationTracker::new();

        tracker.record_violation(ParticipantId(1), ViolationType::DealTimeout, 0, 100);
        tracker.record_violation(ParticipantId(1), ViolationType::InvalidShare { recipient: 2 }, 0, 101);

        assert_eq!(tracker.max_severity_for(ParticipantId(1)), Some(ViolationSeverity::Moderate));
        assert_eq!(tracker.max_severity_for(ParticipantId(99)), None);
    }

    #[test]
    fn test_penalty_score() {
        let mut tracker = ViolationTracker::new();

        // Single minor violation = 0.05
        tracker.record_violation(ParticipantId(1), ViolationType::DealTimeout, 0, 100);
        assert!((tracker.penalty_score(ParticipantId(1)) - 0.05).abs() < 1e-10);

        // Add moderate = 0.05 + 0.15 = 0.20
        tracker.record_violation(ParticipantId(1), ViolationType::InvalidShare { recipient: 2 }, 0, 101);
        assert!((tracker.penalty_score(ParticipantId(1)) - 0.20).abs() < 1e-10);

        // Critical alone = 1.0
        tracker.record_violation(ParticipantId(2), ViolationType::Equivocation { evidence_hash: [0; 32] }, 0, 102);
        assert!((tracker.penalty_score(ParticipantId(2)) - 1.0).abs() < 1e-10);

        // No violations = 0.0
        assert!((tracker.penalty_score(ParticipantId(3)) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_penalty_score_capped_at_1() {
        let mut tracker = ViolationTracker::new();

        // Many violations should cap at 1.0
        for i in 0..100 {
            tracker.record_violation(ParticipantId(1), ViolationType::InvalidShare { recipient: i }, 0, i as u64);
        }
        assert!((tracker.penalty_score(ParticipantId(1)) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_violating_participants() {
        let mut tracker = ViolationTracker::new();

        tracker.record_violation(ParticipantId(3), ViolationType::DealTimeout, 0, 100);
        tracker.record_violation(ParticipantId(1), ViolationType::DealTimeout, 0, 101);
        tracker.record_violation(ParticipantId(3), ViolationType::InvalidCommitment, 0, 102);

        let violators = tracker.violating_participants();
        assert_eq!(violators.len(), 2);
        assert!(violators.contains(&ParticipantId(1)));
        assert!(violators.contains(&ParticipantId(3)));
    }

    #[test]
    fn test_tracker_drain() {
        let mut tracker = ViolationTracker::new();

        tracker.record_violation(ParticipantId(1), ViolationType::DealTimeout, 0, 100);
        tracker.record_violation(ParticipantId(2), ViolationType::InvalidCommitment, 0, 101);

        let drained = tracker.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(tracker.total_count(), 0);
    }

    #[test]
    fn test_violation_with_custom_severity() {
        let v = Violation::with_severity(
            ParticipantId(1),
            ViolationType::DealTimeout,
            ViolationSeverity::Severe, // override from Minor
            0,
            100,
        );
        assert_eq!(v.severity, ViolationSeverity::Severe);
    }
}
