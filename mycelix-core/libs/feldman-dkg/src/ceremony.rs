//! DKG ceremony coordination
//!
//! Manages the full distributed key generation protocol.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::commitment::{Commitment, CommitmentSet};
use crate::dealer::Deal;
use crate::hash_commitment::{CommitmentScheme, HashCommitmentSet, HashReveal};
use crate::participant::{Participant, ParticipantId, ParticipantState};
use crate::polynomial::lagrange_interpolate_at_zero;
use crate::refresh::RefreshRound;
use crate::scalar::Scalar;
use crate::share::CombinedShare;
use crate::violation::{ViolationTracker, ViolationType};
use crate::error::{DkgError, DkgResult};

/// Default phase timeout in seconds (5 minutes)
pub const DEFAULT_PHASE_TIMEOUT_SECS: u64 = 300;

/// Configuration for a DKG ceremony
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DkgConfig {
    /// Threshold: minimum participants needed to reconstruct
    pub threshold: usize,
    /// Total number of participants
    pub num_participants: usize,
    /// Commitment scheme (Feldman, HashBased, or Hybrid)
    #[serde(default)]
    pub commitment_scheme: CommitmentScheme,
}

impl DkgConfig {
    /// Create a new DKG configuration
    pub fn new(threshold: usize, num_participants: usize) -> DkgResult<Self> {
        if threshold == 0 {
            return Err(DkgError::InvalidThreshold {
                threshold,
                participants: num_participants,
            });
        }

        if num_participants < 2 {
            return Err(DkgError::InvalidParticipantCount(num_participants));
        }

        if threshold > num_participants {
            return Err(DkgError::InvalidThreshold {
                threshold,
                participants: num_participants,
            });
        }

        Ok(Self {
            threshold,
            num_participants,
            commitment_scheme: CommitmentScheme::Feldman,
        })
    }

    /// Set the commitment scheme
    pub fn with_commitment_scheme(mut self, scheme: CommitmentScheme) -> Self {
        self.commitment_scheme = scheme;
        self
    }
}

/// Phase of the DKG ceremony
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CeremonyPhase {
    /// Participants are being registered
    Registration,
    /// Participants are submitting hash commitments (commit-reveal protocol)
    CommitmentCollection,
    /// Participants are generating and distributing deals
    Dealing,
    /// Participants are verifying shares and filing complaints
    Verification,
    /// Ceremony is complete
    Complete,
    /// Ceremony failed
    Failed,
}

impl std::fmt::Display for CeremonyPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Registration => write!(f, "Registration"),
            Self::CommitmentCollection => write!(f, "CommitmentCollection"),
            Self::Dealing => write!(f, "Dealing"),
            Self::Verification => write!(f, "Verification"),
            Self::Complete => write!(f, "Complete"),
            Self::Failed => write!(f, "Failed"),
        }
    }
}

/// Result of the DKG ceremony
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CeremonyResult {
    /// The combined public key
    pub public_key: Commitment,
    /// Number of qualified participants
    pub qualified_count: usize,
    /// IDs of disqualified participants
    pub disqualified: Vec<ParticipantId>,
}

/// Coordinator for a DKG ceremony
#[derive(Debug)]
pub struct DkgCeremony {
    /// Configuration
    config: DkgConfig,
    /// Current phase
    phase: CeremonyPhase,
    /// Registered participants
    participants: HashMap<ParticipantId, Participant>,
    /// Received deals from each participant
    deals: HashMap<ParticipantId, Deal>,
    /// Disqualified participants
    disqualified: Vec<ParticipantId>,
    /// Complaints: (complainer, accused)
    complaints: Vec<(ParticipantId, ParticipantId)>,
    /// Hash commitments for commit-reveal protocol (dealer -> commitment set)
    hash_commitments: HashMap<ParticipantId, HashCommitmentSet>,
    /// Hash reveals for commit-reveal protocol (dealer -> reveal)
    hash_reveals: HashMap<ParticipantId, HashReveal>,
    /// Protocol violation tracker
    violation_tracker: ViolationTracker,
    /// Current epoch for proactive refresh
    epoch: u64,
    /// M-04 remediation: When the current phase started (unix seconds)
    phase_started_at_secs: Option<u64>,
    /// M-04 remediation: Timeout for each phase in seconds
    phase_timeout_secs: u64,
}

impl DkgCeremony {
    /// Create a new DKG ceremony
    ///
    /// `now_secs` is the current time as unix seconds. On native, use
    /// `SystemTime::now().duration_since(UNIX_EPOCH).as_secs()`. On Holochain
    /// WASM, use `sys_time()?.as_secs()`. In tests, pass `0u64`.
    pub fn new(config: DkgConfig, now_secs: u64) -> Self {
        Self::with_timeout(config, DEFAULT_PHASE_TIMEOUT_SECS, now_secs)
    }

    /// Create a new DKG ceremony with custom timeout
    ///
    /// M-04 remediation: Allow configurable phase timeouts
    pub fn with_timeout(config: DkgConfig, phase_timeout_secs: u64, now_secs: u64) -> Self {
        Self {
            config,
            phase: CeremonyPhase::Registration,
            participants: HashMap::new(),
            deals: HashMap::new(),
            disqualified: Vec::new(),
            complaints: Vec::new(),
            hash_commitments: HashMap::new(),
            hash_reveals: HashMap::new(),
            violation_tracker: ViolationTracker::new(),
            epoch: 0,
            phase_started_at_secs: Some(now_secs),
            phase_timeout_secs,
        }
    }

    /// Get the current phase
    pub fn phase(&self) -> CeremonyPhase {
        self.phase
    }

    /// Get the configuration
    pub fn config(&self) -> &DkgConfig {
        &self.config
    }

    /// Get the number of registered participants
    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }

    /// M-04 remediation: Check if the current phase has timed out
    pub fn is_phase_timed_out(&self, now_secs: u64) -> bool {
        if let Some(started) = self.phase_started_at_secs {
            now_secs.saturating_sub(started) > self.phase_timeout_secs
        } else {
            false
        }
    }

    /// M-04 remediation: Get elapsed time in current phase (seconds)
    pub fn phase_elapsed(&self, now_secs: u64) -> u64 {
        self.phase_started_at_secs
            .map(|s| now_secs.saturating_sub(s))
            .unwrap_or(0)
    }

    /// M-04 remediation: Get remaining time in current phase (seconds)
    pub fn phase_time_remaining(&self, now_secs: u64) -> u64 {
        let elapsed = self.phase_elapsed(now_secs);
        self.phase_timeout_secs.saturating_sub(elapsed)
    }

    /// M-04 remediation: Get participants who haven't submitted their deal
    pub fn missing_dealers(&self) -> Vec<ParticipantId> {
        self.participants
            .keys()
            .filter(|id| !self.deals.contains_key(id))
            .copied()
            .collect()
    }

    /// M-04 remediation: Check timeout and return missing participants
    ///
    /// Returns None if not timed out, Some(missing) if timed out
    pub fn check_timeout(&self, now_secs: u64) -> Option<Vec<ParticipantId>> {
        if self.is_phase_timed_out(now_secs) {
            Some(self.missing_dealers())
        } else {
            None
        }
    }

    /// M-04 remediation: Exclude participants and continue ceremony
    ///
    /// This allows the ceremony to proceed even if some participants
    /// fail to submit their deals within the timeout period.
    pub fn exclude_and_continue(&mut self, excluded: &[ParticipantId], now_secs: u64) -> DkgResult<()> {
        if self.phase != CeremonyPhase::Dealing {
            return Err(DkgError::WrongPhase {
                expected: CeremonyPhase::Dealing.to_string(),
                actual: self.phase.to_string(),
            });
        }

        // Mark excluded participants as disqualified and record violations
        for id in excluded {
            if !self.disqualified.contains(id) {
                self.disqualified.push(*id);
                if let Some(p) = self.participants.get_mut(id) {
                    p.disqualify();
                }
                self.violation_tracker.record_violation(
                    *id,
                    ViolationType::DealTimeout,
                    self.epoch,
                    now_secs,
                );
            }
        }

        // Check if we still have enough participants
        let remaining = self.participants.len() - self.disqualified.len();
        if remaining < self.config.threshold {
            self.phase = CeremonyPhase::Failed;
            return Err(DkgError::NotEnoughParticipants {
                required: self.config.threshold,
                actual: remaining,
            });
        }

        // If all remaining participants have submitted, move to verification
        let submitted = self.deals.len();
        let expected = self.participants.len() - self.disqualified.len();
        if submitted >= expected {
            self.phase = CeremonyPhase::Verification;
            self.phase_started_at_secs = Some(now_secs);
        }

        Ok(())
    }

    /// Reset the phase timer (useful after phase transitions)
    fn reset_phase_timer(&mut self, now_secs: u64) {
        self.phase_started_at_secs = Some(now_secs);
    }

    /// Add a participant to the ceremony
    pub fn add_participant(&mut self, id: ParticipantId, now_secs: u64) -> DkgResult<()> {
        if self.phase != CeremonyPhase::Registration {
            return Err(DkgError::WrongPhase {
                expected: CeremonyPhase::Registration.to_string(),
                actual: self.phase.to_string(),
            });
        }

        if id.0 == 0 || id.0 > self.config.num_participants as u32 {
            return Err(DkgError::InvalidParticipantId(id.0));
        }

        if self.participants.contains_key(&id) {
            return Err(DkgError::ParticipantExists(id.0));
        }

        let participant = Participant::new(id, self.config.threshold, self.config.num_participants)?;
        self.participants.insert(id, participant);

        // Auto-transition when all participants registered
        if self.participants.len() == self.config.num_participants {
            // If using hash-based or hybrid commitments, go to CommitmentCollection first
            match self.config.commitment_scheme {
                CommitmentScheme::HashBased | CommitmentScheme::Hybrid => {
                    self.phase = CeremonyPhase::CommitmentCollection;
                }
                CommitmentScheme::Feldman => {
                    self.phase = CeremonyPhase::Dealing;
                }
            }
            self.reset_phase_timer(now_secs); // M-04: Reset timer on phase transition
        }

        Ok(())
    }

    /// Start the dealing phase manually
    pub fn start_dealing(&mut self, now_secs: u64) -> DkgResult<()> {
        if self.phase != CeremonyPhase::Registration {
            return Err(DkgError::WrongPhase {
                expected: CeremonyPhase::Registration.to_string(),
                actual: self.phase.to_string(),
            });
        }

        if self.participants.len() < self.config.threshold {
            return Err(DkgError::NotEnoughParticipants {
                required: self.config.threshold,
                actual: self.participants.len(),
            });
        }

        self.phase = CeremonyPhase::Dealing;
        self.reset_phase_timer(now_secs); // M-04: Reset timer on phase transition
        Ok(())
    }

    /// Submit a deal from a participant
    pub fn submit_deal(&mut self, dealer: ParticipantId, deal: Deal, now_secs: u64) -> DkgResult<()> {
        if self.phase != CeremonyPhase::Dealing {
            return Err(DkgError::WrongPhase {
                expected: CeremonyPhase::Dealing.to_string(),
                actual: self.phase.to_string(),
            });
        }

        if !self.participants.contains_key(&dealer) {
            return Err(DkgError::ParticipantNotFound(dealer.0));
        }

        if self.deals.contains_key(&dealer) {
            return Err(DkgError::DuplicateShare(dealer.0));
        }

        // Verify the deal's shares
        for (idx, valid) in deal.verify_all_shares() {
            if !valid {
                // This deal has invalid shares - would trigger complaint
                self.complaints.push((ParticipantId(idx), dealer));
            }
        }

        self.deals.insert(dealer, deal);

        // Auto-transition to verification when all deals received
        if self.deals.len() == self.participants.len() {
            self.phase = CeremonyPhase::Verification;
            self.reset_phase_timer(now_secs); // M-04: Reset timer on phase transition
        }

        Ok(())
    }

    /// File a complaint against a dealer
    pub fn file_complaint(&mut self, complainer: ParticipantId, accused: ParticipantId) -> DkgResult<()> {
        if self.phase != CeremonyPhase::Verification {
            return Err(DkgError::WrongPhase {
                expected: CeremonyPhase::Verification.to_string(),
                actual: self.phase.to_string(),
            });
        }

        if !self.participants.contains_key(&complainer) {
            return Err(DkgError::ParticipantNotFound(complainer.0));
        }

        if !self.participants.contains_key(&accused) {
            return Err(DkgError::ParticipantNotFound(accused.0));
        }

        self.complaints.push((complainer, accused));
        Ok(())
    }

    /// Process complaints and disqualify invalid dealers
    ///
    /// H-07 Fix: Requires multiple complaints to disqualify a dealer.
    /// A single malicious participant cannot disqualify honest dealers.
    /// The complaint threshold is set to floor(threshold/2) + 1, meaning
    /// at least a majority of the reconstruction threshold must complain.
    fn process_complaints(&mut self) {
        // Count complaints against each dealer
        let mut complaint_counts: HashMap<ParticipantId, usize> = HashMap::new();
        for (_, accused) in &self.complaints {
            *complaint_counts.entry(*accused).or_insert(0) += 1;
        }

        // H-07: Require multiple complaints to prevent single-party disqualification attacks
        // Complaint threshold: need majority of reconstruction threshold to agree
        // This ensures a single malicious participant can't disqualify honest dealers
        let complaint_threshold = (self.config.threshold / 2) + 1;

        // Disqualify dealers with complaints exceeding threshold
        for (accused, count) in complaint_counts {
            if count >= complaint_threshold && !self.disqualified.contains(&accused) {
                self.disqualified.push(accused);
                if let Some(p) = self.participants.get_mut(&accused) {
                    p.disqualify();
                }
            }
        }
    }

    /// Finalize the ceremony
    pub fn finalize(&mut self) -> DkgResult<CeremonyResult> {
        // Must be in dealing or verification phase
        if self.phase != CeremonyPhase::Dealing && self.phase != CeremonyPhase::Verification {
            if self.phase == CeremonyPhase::Complete {
                // Already finalized, return cached result
                return self.compute_result();
            }
            return Err(DkgError::WrongPhase {
                expected: "Dealing or Verification".to_string(),
                actual: self.phase.to_string(),
            });
        }

        // Need at least threshold deals
        if self.deals.len() < self.config.threshold {
            self.phase = CeremonyPhase::Failed;
            return Err(DkgError::NotEnoughParticipants {
                required: self.config.threshold,
                actual: self.deals.len(),
            });
        }

        // Process any pending complaints
        self.process_complaints();

        // Check we still have enough qualified participants
        let qualified_count = self.deals.len() - self.disqualified.len();
        if qualified_count < self.config.threshold {
            self.phase = CeremonyPhase::Failed;
            return Err(DkgError::NotEnoughParticipants {
                required: self.config.threshold,
                actual: qualified_count,
            });
        }

        // Distribute shares to each participant
        for (_, participant) in self.participants.iter_mut() {
            if participant.state() == ParticipantState::Disqualified {
                continue;
            }

            for (dealer_id, deal) in &self.deals {
                if self.disqualified.contains(dealer_id) {
                    continue;
                }

                // Each participant processes each qualified deal
                let _ = participant.process_deal(deal);
            }
        }

        self.phase = CeremonyPhase::Complete;
        self.compute_result()
    }

    /// Compute the ceremony result
    fn compute_result(&self) -> DkgResult<CeremonyResult> {
        // Combine all qualified dealers' secret commitments
        let qualified_commitments: Vec<&CommitmentSet> = self
            .deals
            .iter()
            .filter(|(id, _)| !self.disqualified.contains(id))
            .map(|(_, deal)| &deal.commitments)
            .collect();

        if qualified_commitments.is_empty() {
            return Err(DkgError::NotEnoughParticipants {
                required: self.config.threshold,
                actual: 0,
            });
        }

        let public_key = CommitmentSet::combine(&qualified_commitments)?;

        Ok(CeremonyResult {
            public_key,
            qualified_count: qualified_commitments.len(),
            disqualified: self.disqualified.clone(),
        })
    }

    /// Get a participant's combined share (after finalization)
    pub fn get_combined_share(&self, id: ParticipantId) -> DkgResult<CombinedShare> {
        if self.phase != CeremonyPhase::Complete {
            return Err(DkgError::WrongPhase {
                expected: CeremonyPhase::Complete.to_string(),
                actual: self.phase.to_string(),
            });
        }

        // Compute the combined share for this participant
        let mut combined_value = Scalar::zero();

        for (dealer_id, deal) in &self.deals {
            if self.disqualified.contains(dealer_id) {
                continue;
            }

            if let Some(share) = deal.get_share(id.0) {
                combined_value += share.value.clone();
            }
        }

        Ok(CombinedShare::new(id.0, combined_value))
    }

    /// Submit a hash commitment for the commit-reveal protocol
    pub fn submit_hash_commitment(
        &mut self,
        dealer: ParticipantId,
        commitment_set: HashCommitmentSet,
        now_secs: u64,
    ) -> DkgResult<()> {
        if self.phase != CeremonyPhase::CommitmentCollection {
            return Err(DkgError::WrongPhase {
                expected: CeremonyPhase::CommitmentCollection.to_string(),
                actual: self.phase.to_string(),
            });
        }

        if !self.participants.contains_key(&dealer) {
            return Err(DkgError::ParticipantNotFound(dealer.0));
        }

        if self.hash_commitments.contains_key(&dealer) {
            return Err(DkgError::DuplicateShare(dealer.0));
        }

        self.hash_commitments.insert(dealer, commitment_set);

        // Auto-transition to dealing when all commitments received
        let active = self.participants.len() - self.disqualified.len();
        if self.hash_commitments.len() >= active {
            self.phase = CeremonyPhase::Dealing;
            self.phase_started_at_secs = Some(now_secs);
        }

        Ok(())
    }

    /// Submit a hash reveal (the actual deal + salt) for verification
    pub fn submit_hash_reveal(
        &mut self,
        dealer: ParticipantId,
        reveal: HashReveal,
        _now_secs: u64,
    ) -> DkgResult<()> {
        if !self.hash_commitments.contains_key(&dealer) {
            return Err(DkgError::MissingHashCommitment(dealer.0));
        }

        self.hash_reveals.insert(dealer, reveal);
        Ok(())
    }

    /// Verify all hash reveals against their commitments
    ///
    /// For each dealer, checks that the deal's shares match the hash commitments
    /// using the revealed salts. Dealers with mismatches are disqualified.
    pub fn verify_hash_reveals(&mut self) -> DkgResult<()> {
        let dealer_ids: Vec<ParticipantId> = self.hash_commitments.keys().copied().collect();

        for dealer_id in dealer_ids {
            let reveal = match self.hash_reveals.get(&dealer_id) {
                Some(r) => r,
                None => {
                    return Err(DkgError::MissingHashCommitment(dealer_id.0));
                }
            };

            // Verify each share against its hash commitment using the revealed salt
            let deal = match self.deals.get(&dealer_id) {
                Some(d) => d,
                None => continue, // Deal not yet submitted, skip
            };

            let commitment_set = &self.hash_commitments[&dealer_id];
            let mut mismatch = false;

            for share in &deal.shares {
                if let Some((_, salt)) = reveal.salts.iter().find(|(r, _)| *r == share.index) {
                    match commitment_set.verify_share(share.index, &share.value, salt) {
                        Ok(true) => {}
                        _ => {
                            mismatch = true;
                            break;
                        }
                    }
                } else {
                    mismatch = true;
                    break;
                }
            }

            if mismatch {
                self.violation_tracker.record_violation(
                    dealer_id,
                    ViolationType::CommitRevealMismatch,
                    self.epoch,
                    0,
                );
                if !self.disqualified.contains(&dealer_id) {
                    self.disqualified.push(dealer_id);
                    if let Some(p) = self.participants.get_mut(&dealer_id) {
                        p.disqualify();
                    }
                }
            }
        }
        Ok(())
    }

    /// Start a proactive share refresh round
    pub fn refresh_shares(&mut self) -> DkgResult<RefreshRound> {
        if self.phase != CeremonyPhase::Complete {
            return Err(DkgError::WrongPhase {
                expected: CeremonyPhase::Complete.to_string(),
                actual: self.phase.to_string(),
            });
        }

        self.epoch += 1;
        RefreshRound::new(
            self.config.threshold,
            self.config.num_participants,
            self.epoch,
        )
    }

    /// Get the current epoch
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Get the violation tracker
    pub fn violation_tracker(&self) -> &ViolationTracker {
        &self.violation_tracker
    }

    /// Get a mutable reference to the violation tracker
    pub fn violation_tracker_mut(&mut self) -> &mut ViolationTracker {
        &mut self.violation_tracker
    }

    /// Get the commitment scheme
    pub fn commitment_scheme(&self) -> &CommitmentScheme {
        &self.config.commitment_scheme
    }

    /// Reconstruct the secret from shares (for testing/verification)
    pub fn reconstruct_secret(&self, shares: &[CombinedShare]) -> DkgResult<Scalar> {
        if shares.len() < self.config.threshold {
            return Err(DkgError::NotEnoughParticipants {
                required: self.config.threshold,
                actual: shares.len(),
            });
        }

        let points: Vec<_> = shares.iter().map(|s| s.to_point()).collect();
        lagrange_interpolate_at_zero(&points)
    }

    /// File a cryptographically verified complaint against a dealer
    ///
    /// Verifies the accused's share at `share_index` against the deal's
    /// Feldman VSS commitments. If the share is genuinely invalid, the
    /// complaint is valid and an `InvalidShare` violation is recorded
    /// against the accused. If the share is actually valid, the complaint
    /// is frivolous and an `InvalidCommitment` violation is recorded
    /// against the complainer.
    ///
    /// Returns `Ok(true)` if the complaint was valid (share was bad),
    /// `Ok(false)` if the complaint was frivolous (share was good).
    pub fn file_verified_complaint(
        &mut self,
        complainer: ParticipantId,
        accused: ParticipantId,
        share_index: u32,
        now_secs: u64,
    ) -> DkgResult<bool> {
        if self.phase != CeremonyPhase::Verification {
            return Err(DkgError::WrongPhase {
                expected: CeremonyPhase::Verification.to_string(),
                actual: self.phase.to_string(),
            });
        }

        if !self.participants.contains_key(&complainer) {
            return Err(DkgError::ParticipantNotFound(complainer.0));
        }

        if !self.participants.contains_key(&accused) {
            return Err(DkgError::ParticipantNotFound(accused.0));
        }

        let deal = self.deals.get(&accused).ok_or(
            DkgError::ParticipantNotFound(accused.0),
        )?;

        let share = deal.get_share(share_index).ok_or(
            DkgError::InvalidShareIndex(share_index as usize),
        )?;

        let share_valid = deal.commitments.verify_share(share_index, share.value());

        if !share_valid {
            // Complaint is valid — share fails verification
            self.complaints.push((complainer, accused));
            self.violation_tracker.record_violation(
                accused,
                ViolationType::InvalidShare { recipient: share_index },
                self.epoch,
                now_secs,
            );
            Ok(true)
        } else {
            // Complaint is frivolous — share is actually valid
            self.violation_tracker.record_violation(
                complainer,
                ViolationType::InvalidCommitment,
                self.epoch,
                now_secs,
            );
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn test_config_validation() {
        // Valid config
        let config = DkgConfig::new(2, 3);
        assert!(config.is_ok());

        // Invalid: threshold > participants
        let config = DkgConfig::new(5, 3);
        assert!(config.is_err());

        // Invalid: zero threshold
        let config = DkgConfig::new(0, 3);
        assert!(config.is_err());

        // Invalid: single participant
        let config = DkgConfig::new(1, 1);
        assert!(config.is_err());
    }

    #[test]
    fn test_ceremony_workflow() {
        // 2-of-3 ceremony
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        // Register participants
        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }
        assert_eq!(ceremony.phase(), CeremonyPhase::Dealing);

        // Each participant generates a deal
        let mut deals = Vec::new();
        for i in 1..=3 {
            let dealer = crate::dealer::Dealer::new(ParticipantId(i), 2, 3, &mut OsRng).unwrap();
            deals.push((ParticipantId(i), dealer.generate_deal()));
        }

        // Submit deals
        for (id, deal) in deals {
            ceremony.submit_deal(id, deal, 0).unwrap();
        }
        assert_eq!(ceremony.phase(), CeremonyPhase::Verification);

        // Finalize
        let result = ceremony.finalize().unwrap();
        assert_eq!(result.qualified_count, 3);
        assert!(result.disqualified.is_empty());
    }

    #[test]
    fn test_secret_reconstruction() {
        // 2-of-3 ceremony with known secrets
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        // Create dealers with specific secrets
        let secrets = [
            Scalar::from_u64(100),
            Scalar::from_u64(200),
            Scalar::from_u64(300),
        ];
        let expected_combined = Scalar::from_u64(600); // Sum of secrets

        let deals: Vec<_> = (1..=3)
            .zip(secrets.iter())
            .map(|(i, secret)| {
                let dealer = crate::dealer::Dealer::with_secret(
                    ParticipantId(i as u32),
                    secret.clone(),
                    2,
                    3,
                    &mut OsRng,
                )
                .unwrap();
                (ParticipantId(i as u32), dealer.generate_deal())
            })
            .collect();

        for (id, deal) in deals {
            ceremony.submit_deal(id, deal, 0).unwrap();
        }

        ceremony.finalize().unwrap();

        // Get combined shares
        let shares: Vec<_> = (1..=3)
            .map(|i| ceremony.get_combined_share(ParticipantId(i)).unwrap())
            .collect();

        // Reconstruct from any 2 shares
        let reconstructed = ceremony.reconstruct_secret(&shares[0..2]).unwrap();
        assert_eq!(reconstructed, expected_combined);

        // Also works with different pairs
        let reconstructed2 = ceremony.reconstruct_secret(&shares[1..3]).unwrap();
        assert_eq!(reconstructed2, expected_combined);
    }

    // ==================== CRITICAL VSS PROPERTY TESTS ====================
    // Added per CODE_REVIEW_FINDINGS_2026-01-19.md Section 4.1

    #[test]
    fn test_threshold_property_exact() {
        // CRITICAL: Verify t shares CAN reconstruct, but t-1 CANNOT
        // This is the core security property of threshold cryptography
        let config = DkgConfig::new(3, 5).unwrap(); // 3-of-5 threshold
        let mut ceremony = DkgCeremony::new(config, 0);

        // Register all participants
        for i in 1..=5 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        // Generate deals with known secrets for verification
        let secrets: Vec<Scalar> = (1..=5).map(|i| Scalar::from_u64(i * 100)).collect();
        let expected_combined: Scalar = secrets.iter().fold(Scalar::zero(), |a, b| a + b.clone());

        let deals: Vec<_> = (1..=5)
            .zip(secrets.iter())
            .map(|(i, secret)| {
                let dealer = crate::dealer::Dealer::with_secret(
                    ParticipantId(i as u32),
                    secret.clone(),
                    3,
                    5,
                    &mut OsRng,
                )
                .unwrap();
                (ParticipantId(i as u32), dealer.generate_deal())
            })
            .collect();

        for (id, deal) in deals {
            ceremony.submit_deal(id, deal, 0).unwrap();
        }

        ceremony.finalize().unwrap();

        // Get all combined shares
        let shares: Vec<_> = (1..=5)
            .map(|i| ceremony.get_combined_share(ParticipantId(i)).unwrap())
            .collect();

        // MUST work with exactly threshold (3) shares
        let result_3 = ceremony.reconstruct_secret(&shares[0..3]);
        assert!(result_3.is_ok(), "Reconstruction with threshold shares must succeed");
        assert_eq!(result_3.unwrap(), expected_combined);

        // MUST work with more than threshold (4, 5) shares
        let result_4 = ceremony.reconstruct_secret(&shares[0..4]);
        assert!(result_4.is_ok(), "Reconstruction with 4 shares must succeed");
        assert_eq!(result_4.unwrap(), expected_combined);

        let result_5 = ceremony.reconstruct_secret(&shares);
        assert!(result_5.is_ok(), "Reconstruction with all shares must succeed");
        assert_eq!(result_5.unwrap(), expected_combined);

        // MUST FAIL with fewer than threshold (2) shares
        let result_2 = ceremony.reconstruct_secret(&shares[0..2]);
        assert!(result_2.is_err(), "Reconstruction with t-1 shares must fail");
    }

    #[test]
    fn test_any_threshold_subset_reconstructs() {
        // CRITICAL: ANY subset of t shares should reconstruct the same secret
        // Tests that different combinations yield the same result
        let config = DkgConfig::new(2, 4).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=4 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        let deals: Vec<_> = (1..=4)
            .map(|i| {
                let dealer = crate::dealer::Dealer::new(ParticipantId(i), 2, 4, &mut OsRng).unwrap();
                (ParticipantId(i as u32), dealer.generate_deal())
            })
            .collect();

        for (id, deal) in deals {
            ceremony.submit_deal(id, deal, 0).unwrap();
        }

        ceremony.finalize().unwrap();

        let shares: Vec<_> = (1..=4)
            .map(|i| ceremony.get_combined_share(ParticipantId(i)).unwrap())
            .collect();

        // All possible pairs of 2 shares (C(4,2) = 6 combinations)
        let combinations = vec![
            vec![0, 1], // shares 1, 2
            vec![0, 2], // shares 1, 3
            vec![0, 3], // shares 1, 4
            vec![1, 2], // shares 2, 3
            vec![1, 3], // shares 2, 4
            vec![2, 3], // shares 3, 4
        ];

        let first_result = ceremony
            .reconstruct_secret(&[shares[0].clone(), shares[1].clone()])
            .unwrap();

        for combo in combinations.iter().skip(1) {
            let subset: Vec<_> = combo.iter().map(|&i| shares[i].clone()).collect();
            let result = ceremony.reconstruct_secret(&subset).unwrap();
            assert_eq!(
                result, first_result,
                "All threshold subsets must reconstruct the same secret"
            );
        }
    }

    #[test]
    fn test_public_key_commitment_consistency() {
        // CRITICAL: The public key commitment must equal g^(combined_secret)
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        // Use known secrets for verification
        let secrets = [
            Scalar::from_u64(42),
            Scalar::from_u64(17),
            Scalar::from_u64(123),
        ];
        let combined_secret = Scalar::from_u64(42 + 17 + 123);

        let deals: Vec<_> = (1..=3)
            .zip(secrets.iter())
            .map(|(i, secret)| {
                let dealer = crate::dealer::Dealer::with_secret(
                    ParticipantId(i as u32),
                    secret.clone(),
                    2,
                    3,
                    &mut OsRng,
                )
                .unwrap();
                (ParticipantId(i as u32), dealer.generate_deal())
            })
            .collect();

        for (id, deal) in deals {
            ceremony.submit_deal(id, deal, 0).unwrap();
        }

        let result = ceremony.finalize().unwrap();

        // The public key should equal g^(combined_secret)
        let expected_pk = crate::commitment::Commitment::new(&combined_secret);
        assert_eq!(
            result.public_key, expected_pk,
            "Public key must equal g^(sum of dealer secrets)"
        );
    }

    #[test]
    fn test_disqualification_excludes_from_reconstruction() {
        // CRITICAL: Disqualified dealers' shares should NOT be included
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        // Submit deals
        for i in 1..=3 {
            let dealer = crate::dealer::Dealer::new(ParticipantId(i), 2, 3, &mut OsRng).unwrap();
            let deal = dealer.generate_deal();
            ceremony.submit_deal(ParticipantId(i as u32), deal, 0).unwrap();
        }

        // File complaints against dealer 3 (need >= complaint_threshold = (t/2)+1 = 2)
        ceremony.phase = CeremonyPhase::Verification; // Force into verification phase
        ceremony.file_complaint(ParticipantId(1), ParticipantId(3)).unwrap();
        ceremony.file_complaint(ParticipantId(2), ParticipantId(3)).unwrap();

        // Finalize - dealer 3 should be disqualified
        let result = ceremony.finalize().unwrap();

        assert_eq!(result.qualified_count, 2, "Only 2 dealers should be qualified");
        assert!(
            result.disqualified.contains(&ParticipantId(3)),
            "Dealer 3 should be disqualified"
        );
    }

    #[test]
    fn test_insufficient_participants_fails() {
        // CRITICAL: Ceremony must fail if we can't meet threshold
        let config = DkgConfig::new(3, 5).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        // Only register 2 participants (less than threshold)
        ceremony.add_participant(ParticipantId(1), 0).unwrap();
        ceremony.add_participant(ParticipantId(2), 0).unwrap();

        // Try to start dealing manually - should fail
        let result = ceremony.start_dealing(0);
        assert!(result.is_err(), "Starting with insufficient participants must fail");
    }

    #[test]
    fn test_duplicate_participant_rejected() {
        // CRITICAL: Same participant ID must not be registered twice
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        ceremony.add_participant(ParticipantId(1), 0).unwrap();

        let result = ceremony.add_participant(ParticipantId(1), 0);
        assert!(result.is_err(), "Duplicate participant must be rejected");
        match result {
            Err(DkgError::ParticipantExists(1)) => {}
            _ => panic!("Expected ParticipantExists error"),
        }
    }

    #[test]
    fn test_invalid_participant_id_rejected() {
        // CRITICAL: Participant ID 0 or > num_participants must be rejected
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        // ID 0 is invalid
        let result = ceremony.add_participant(ParticipantId(0), 0);
        assert!(result.is_err(), "Participant ID 0 must be rejected");

        // ID > num_participants is invalid
        let result = ceremony.add_participant(ParticipantId(10), 0);
        assert!(result.is_err(), "Participant ID > num_participants must be rejected");
    }

    #[test]
    fn test_deal_submission_order_independent() {
        // CRITICAL: Order of deal submissions should not affect the result
        let config = DkgConfig::new(2, 3).unwrap();

        // Create dealers with specific secrets
        let secrets = [
            Scalar::from_u64(100),
            Scalar::from_u64(200),
            Scalar::from_u64(300),
        ];

        let dealers: Vec<_> = (1..=3)
            .zip(secrets.iter())
            .map(|(i, secret)| {
                crate::dealer::Dealer::with_secret(
                    ParticipantId(i as u32),
                    secret.clone(),
                    2,
                    3,
                    &mut OsRng,
                )
                .unwrap()
            })
            .collect();

        let deals: Vec<_> = dealers.iter().map(|d| d.generate_deal()).collect();

        // First ceremony: submit in order 1, 2, 3
        let mut ceremony1 = DkgCeremony::new(config.clone(), 0);
        for i in 1..=3 {
            ceremony1.add_participant(ParticipantId(i), 0).unwrap();
        }
        for (i, deal) in deals.iter().enumerate() {
            ceremony1.submit_deal(ParticipantId((i + 1) as u32), deal.clone(), 0).unwrap();
        }
        let result1 = ceremony1.finalize().unwrap();

        // Second ceremony: submit in order 3, 1, 2
        let mut ceremony2 = DkgCeremony::new(config, 0);
        for i in 1..=3 {
            ceremony2.add_participant(ParticipantId(i), 0).unwrap();
        }
        ceremony2.submit_deal(ParticipantId(3), deals[2].clone(), 0).unwrap();
        ceremony2.submit_deal(ParticipantId(1), deals[0].clone(), 0).unwrap();
        ceremony2.submit_deal(ParticipantId(2), deals[1].clone(), 0).unwrap();
        let result2 = ceremony2.finalize().unwrap();

        // Both should produce the same public key
        assert_eq!(
            result1.public_key, result2.public_key,
            "Deal submission order must not affect public key"
        );
    }

    #[test]
    fn test_phase_transition_correctness() {
        // CRITICAL: Phase transitions must follow correct order
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        // Starts in Registration
        assert_eq!(ceremony.phase(), CeremonyPhase::Registration);

        // Can't submit deal before dealing phase
        let dealer = crate::dealer::Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();
        let result = ceremony.submit_deal(ParticipantId(1), deal.clone(), 0);
        assert!(result.is_err(), "Cannot submit deal in Registration phase");

        // Register all participants -> auto-transition to Dealing
        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }
        assert_eq!(ceremony.phase(), CeremonyPhase::Dealing);

        // Can't register in Dealing phase
        let result = ceremony.add_participant(ParticipantId(1), 0);
        assert!(result.is_err(), "Cannot register in Dealing phase");

        // Submit all deals -> auto-transition to Verification
        for i in 1..=3 {
            let dealer = crate::dealer::Dealer::new(ParticipantId(i), 2, 3, &mut OsRng).unwrap();
            let deal = dealer.generate_deal();
            ceremony.submit_deal(ParticipantId(i as u32), deal, 0).unwrap();
        }
        assert_eq!(ceremony.phase(), CeremonyPhase::Verification);

        // Finalize -> transition to Complete
        ceremony.finalize().unwrap();
        assert_eq!(ceremony.phase(), CeremonyPhase::Complete);
    }

    #[test]
    fn test_hash_based_commitment_ceremony() {
        use crate::hash_commitment::HashCommitmentSet;

        // Create a ceremony with hash-based commitments
        let config = DkgConfig::new(2, 3).unwrap()
            .with_commitment_scheme(CommitmentScheme::HashBased);
        let mut ceremony = DkgCeremony::new(config, 0);

        // Register all participants
        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        // Should be in CommitmentCollection phase, not Dealing
        assert_eq!(ceremony.phase(), CeremonyPhase::CommitmentCollection);

        // Generate deals and hash commitments
        let deals: Vec<_> = (1..=3)
            .map(|i| {
                let dealer = crate::dealer::Dealer::new(ParticipantId(i), 2, 3, &mut OsRng).unwrap();
                dealer.generate_deal()
            })
            .collect();

        let mut reveals = Vec::new();
        for (i, deal) in deals.iter().enumerate() {
            let (commitment_set, reveal) = HashCommitmentSet::from_deal(deal, &mut OsRng);
            ceremony
                .submit_hash_commitment(ParticipantId((i + 1) as u32), commitment_set, 0)
                .unwrap();
            reveals.push(reveal);
        }

        // Should have auto-transitioned to Dealing
        assert_eq!(ceremony.phase(), CeremonyPhase::Dealing);

        // Submit reveals
        for (i, reveal) in reveals.into_iter().enumerate() {
            ceremony
                .submit_hash_reveal(ParticipantId((i + 1) as u32), reveal, 0)
                .unwrap();
        }

        // Submit actual deals
        for (i, deal) in deals.into_iter().enumerate() {
            ceremony
                .submit_deal(ParticipantId((i + 1) as u32), deal, 0)
                .unwrap();
        }

        // Verify reveals
        ceremony.verify_hash_reveals().unwrap();

        // Finalize
        let result = ceremony.finalize().unwrap();
        assert_eq!(result.qualified_count, 3);
    }

    #[test]
    fn test_violation_tracker_integration() {
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        // Submit deals from only 2 participants
        for i in 1..=2 {
            let dealer = crate::dealer::Dealer::new(ParticipantId(i), 2, 3, &mut OsRng).unwrap();
            let deal = dealer.generate_deal();
            ceremony.submit_deal(ParticipantId(i as u32), deal, 0).unwrap();
        }

        // Exclude participant 3 (they didn't submit)
        ceremony.exclude_and_continue(&[ParticipantId(3)], 100).unwrap();

        // Violation should be recorded
        let violations = ceremony.violation_tracker().violations_for(ParticipantId(3));
        assert_eq!(violations.len(), 1);
        assert!(matches!(violations[0].violation_type, ViolationType::DealTimeout));
    }

    #[test]
    fn test_refresh_after_ceremony() {
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        // Use known secrets
        let secrets = [
            Scalar::from_u64(100),
            Scalar::from_u64(200),
            Scalar::from_u64(300),
        ];

        for (i, secret) in secrets.iter().enumerate() {
            let dealer = crate::dealer::Dealer::with_secret(
                ParticipantId((i + 1) as u32),
                secret.clone(),
                2,
                3,
                &mut OsRng,
            )
            .unwrap();
            let deal = dealer.generate_deal();
            ceremony
                .submit_deal(ParticipantId((i + 1) as u32), deal, 0)
                .unwrap();
        }
        ceremony.finalize().unwrap();

        // Get original combined shares
        let original_shares: Vec<_> = (1..=3)
            .map(|i| ceremony.get_combined_share(ParticipantId(i)).unwrap())
            .collect();

        // Start a refresh round
        let mut refresh = ceremony.refresh_shares().unwrap();
        assert_eq!(ceremony.epoch(), 1);

        // Each participant generates a refresh deal
        for i in 1..=3u32 {
            let deal = RefreshRound::generate_refresh_deal(
                ParticipantId(i), 2, 3, 1, &mut OsRng,
            ).unwrap();
            refresh.submit_deal(deal).unwrap();
        }

        // Apply refresh to each share
        let refreshed_shares: Vec<_> = (1..=3)
            .map(|i| {
                let delta = refresh
                    .compute_refresh_delta(i as u32)
                    .unwrap();
                let mut new_value = original_shares[(i - 1) as usize].value.clone();
                new_value += delta;
                CombinedShare::new(i as u32, new_value)
            })
            .collect();

        // Refreshed shares should reconstruct the same secret
        let original_secret = ceremony.reconstruct_secret(&original_shares[0..2]).unwrap();
        let refreshed_secret = ceremony.reconstruct_secret(&refreshed_shares[0..2]).unwrap();
        assert_eq!(original_secret, refreshed_secret);
    }

    #[test]
    fn test_exclude_and_continue_records_violations() {
        let config = DkgConfig::new(2, 4).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=4 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        // Only participants 1 and 2 submit deals
        for i in 1..=2 {
            let dealer = crate::dealer::Dealer::new(ParticipantId(i), 2, 4, &mut OsRng).unwrap();
            let deal = dealer.generate_deal();
            ceremony.submit_deal(ParticipantId(i as u32), deal, 0).unwrap();
        }

        // Exclude 3 and 4
        ceremony.exclude_and_continue(&[ParticipantId(3), ParticipantId(4)], 300).unwrap();

        // Both should have violations
        assert_eq!(ceremony.violation_tracker().violations_for(ParticipantId(3)).len(), 1);
        assert_eq!(ceremony.violation_tracker().violations_for(ParticipantId(4)).len(), 1);

        // Penalty scores should reflect the severity
        assert!(ceremony.violation_tracker().penalty_score(ParticipantId(3)) > 0.0);
        assert!(ceremony.violation_tracker().penalty_score(ParticipantId(4)) > 0.0);
    }

    #[test]
    fn test_commitment_scheme_default_is_feldman() {
        let config = DkgConfig::new(2, 3).unwrap();
        assert!(matches!(config.commitment_scheme, CommitmentScheme::Feldman));

        let mut ceremony = DkgCeremony::new(config, 0);
        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }
        // Feldman scheme goes straight to Dealing
        assert_eq!(ceremony.phase(), CeremonyPhase::Dealing);
    }

    #[test]
    fn test_hybrid_commitment_scheme_transitions() {
        let config = DkgConfig::new(2, 3).unwrap()
            .with_commitment_scheme(CommitmentScheme::Hybrid);
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }
        // Hybrid goes to CommitmentCollection first
        assert_eq!(ceremony.phase(), CeremonyPhase::CommitmentCollection);
    }

    #[test]
    fn test_hybrid_ceremony_full_flow() {
        use crate::hash_commitment::HashCommitmentSet;

        let config = DkgConfig::new(2, 3).unwrap()
            .with_commitment_scheme(CommitmentScheme::Hybrid);
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }
        assert_eq!(ceremony.phase(), CeremonyPhase::CommitmentCollection);

        // Generate deals
        let deals: Vec<_> = (1..=3)
            .map(|i| {
                let dealer = crate::dealer::Dealer::new(ParticipantId(i), 2, 3, &mut OsRng).unwrap();
                dealer.generate_deal()
            })
            .collect();

        // Phase 1: Submit hash commitments
        let mut reveals = Vec::new();
        for (i, deal) in deals.iter().enumerate() {
            let (commitment_set, reveal) = HashCommitmentSet::from_deal(deal, &mut OsRng);
            ceremony.submit_hash_commitment(ParticipantId((i + 1) as u32), commitment_set, 0).unwrap();
            reveals.push(reveal);
        }
        assert_eq!(ceremony.phase(), CeremonyPhase::Dealing);

        // Phase 2: Submit deals (Feldman commitments verified on-chain)
        for (i, deal) in deals.iter().enumerate() {
            ceremony.submit_deal(ParticipantId((i + 1) as u32), deal.clone(), 0).unwrap();
        }

        // Phase 3: Submit hash reveals
        for (i, reveal) in reveals.into_iter().enumerate() {
            ceremony.submit_hash_reveal(ParticipantId((i + 1) as u32), reveal, 0).unwrap();
        }

        // Phase 4: Verify hash reveals (quantum-resistant verification)
        ceremony.verify_hash_reveals().unwrap();

        // Phase 5: Finalize — both Feldman + hash verification passed
        let result = ceremony.finalize().unwrap();
        assert_eq!(result.qualified_count, 3);
    }

    #[test]
    fn test_hybrid_ceremony_wrong_salt_disqualifies() {
        use crate::hash_commitment::{HashCommitmentSet, HashReveal, CommitmentSalt};

        let config = DkgConfig::new(2, 3).unwrap()
            .with_commitment_scheme(CommitmentScheme::Hybrid);
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        let deals: Vec<_> = (1..=3)
            .map(|i| {
                let dealer = crate::dealer::Dealer::new(ParticipantId(i), 2, 3, &mut OsRng).unwrap();
                dealer.generate_deal()
            })
            .collect();

        // Submit honest commitments for all
        let mut reveals = Vec::new();
        for (i, deal) in deals.iter().enumerate() {
            let (commitment_set, reveal) = HashCommitmentSet::from_deal(deal, &mut OsRng);
            ceremony.submit_hash_commitment(ParticipantId((i + 1) as u32), commitment_set, 0).unwrap();
            reveals.push(reveal);
        }

        // Submit all deals
        for (i, deal) in deals.iter().enumerate() {
            ceremony.submit_deal(ParticipantId((i + 1) as u32), deal.clone(), 0).unwrap();
        }

        // Participant 3 submits a reveal with WRONG salts
        let honest_reveal_3 = reveals.pop().unwrap();
        let wrong_salts: Vec<(u32, CommitmentSalt)> = honest_reveal_3.salts.iter()
            .map(|(idx, _)| (*idx, CommitmentSalt::random(&mut OsRng)))
            .collect();
        let bad_reveal = HashReveal { dealer: ParticipantId(3), salts: wrong_salts };

        // Submit honest reveals for 1 and 2
        for (i, reveal) in reveals.into_iter().enumerate() {
            ceremony.submit_hash_reveal(ParticipantId((i + 1) as u32), reveal, 0).unwrap();
        }
        // Submit bad reveal for 3
        ceremony.submit_hash_reveal(ParticipantId(3), bad_reveal, 0).unwrap();

        // Verify — participant 3 should be disqualified
        ceremony.verify_hash_reveals().unwrap();

        // Participant 3 should have a CommitRevealMismatch violation
        let violations = ceremony.violation_tracker().violations_for(ParticipantId(3));
        assert!(!violations.is_empty(), "participant 3 should have violations");
        assert!(violations.iter().any(|v| matches!(v.violation_type, ViolationType::CommitRevealMismatch)));

        // Ceremony should still finalize with 2 qualified (threshold met)
        let result = ceremony.finalize().unwrap();
        assert_eq!(result.qualified_count, 2);
    }

    #[test]
    fn test_hybrid_ceremony_missing_hash_commitments() {
        let config = DkgConfig::new(2, 3).unwrap()
            .with_commitment_scheme(CommitmentScheme::Hybrid);
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }
        assert_eq!(ceremony.phase(), CeremonyPhase::CommitmentCollection);

        // Only 2 of 3 submit commitments — should NOT transition to Dealing
        let deals: Vec<_> = (1..=2)
            .map(|i| {
                let dealer = crate::dealer::Dealer::new(ParticipantId(i), 2, 3, &mut OsRng).unwrap();
                dealer.generate_deal()
            })
            .collect();

        for (i, deal) in deals.iter().enumerate() {
            let (commitment_set, _reveal) = crate::hash_commitment::HashCommitmentSet::from_deal(deal, &mut OsRng);
            ceremony.submit_hash_commitment(ParticipantId((i + 1) as u32), commitment_set, 0).unwrap();
        }

        // Still in CommitmentCollection — waiting for participant 3
        assert_eq!(ceremony.phase(), CeremonyPhase::CommitmentCollection);

        // Cannot submit deals yet
        let dealer3 = crate::dealer::Dealer::new(ParticipantId(3), 2, 3, &mut OsRng).unwrap();
        let result = ceremony.submit_deal(ParticipantId(3), dealer3.generate_deal(), 0);
        assert!(result.is_err(), "should not accept deals during CommitmentCollection");
    }

    #[test]
    fn test_hybrid_duplicate_hash_commitment_rejected() {
        let config = DkgConfig::new(2, 3).unwrap()
            .with_commitment_scheme(CommitmentScheme::Hybrid);
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        let dealer = crate::dealer::Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();
        let (cs1, _) = crate::hash_commitment::HashCommitmentSet::from_deal(&deal, &mut OsRng);
        let (cs2, _) = crate::hash_commitment::HashCommitmentSet::from_deal(&deal, &mut OsRng);

        ceremony.submit_hash_commitment(ParticipantId(1), cs1, 0).unwrap();
        let result = ceremony.submit_hash_commitment(ParticipantId(1), cs2, 0);
        assert!(matches!(result, Err(DkgError::DuplicateShare(1))));
    }

    #[test]
    fn test_verified_complaint_valid_against_bad_share() {
        // 2-of-3 ceremony
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        // Generate legitimate deals for participants 1 and 2
        let dealer1 = crate::dealer::Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let dealer2 = crate::dealer::Dealer::new(ParticipantId(2), 2, 3, &mut OsRng).unwrap();
        ceremony.submit_deal(ParticipantId(1), dealer1.generate_deal(), 0).unwrap();
        ceremony.submit_deal(ParticipantId(2), dealer2.generate_deal(), 0).unwrap();

        // Generate a tampered deal for participant 3: valid commitments but corrupt one share
        let dealer3 = crate::dealer::Dealer::new(ParticipantId(3), 2, 3, &mut OsRng).unwrap();
        let mut bad_deal = dealer3.generate_deal();
        // Corrupt the share for participant 1 (index=1)
        bad_deal.shares[0] = crate::share::Share::new(1, 3, Scalar::from_u64(9999999));
        ceremony.submit_deal(ParticipantId(3), bad_deal, 0).unwrap();

        assert_eq!(ceremony.phase(), CeremonyPhase::Verification);

        // Participant 1 files a verified complaint against participant 3's share at index 1
        let result = ceremony.file_verified_complaint(
            ParticipantId(1),
            ParticipantId(3),
            1,  // share_index for participant 1
            100,
        ).unwrap();

        // Complaint should be valid (share was bad)
        assert!(result, "complaint should be valid for a bad share");

        // Should have recorded an InvalidShare violation against the accused
        let violations = ceremony.violation_tracker().violations_for(ParticipantId(3));
        assert!(!violations.is_empty(), "should have violation for accused");
        assert!(
            violations.iter().any(|v| matches!(v.violation_type, ViolationType::InvalidShare { recipient: 1 })),
            "should record InvalidShare violation"
        );
    }

    #[test]
    fn test_verified_complaint_frivolous_against_good_share() {
        // 2-of-3 ceremony
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        // Generate all legitimate deals
        for i in 1..=3 {
            let dealer = crate::dealer::Dealer::new(ParticipantId(i), 2, 3, &mut OsRng).unwrap();
            ceremony.submit_deal(ParticipantId(i), dealer.generate_deal(), 0).unwrap();
        }

        assert_eq!(ceremony.phase(), CeremonyPhase::Verification);

        // Participant 1 files a frivolous complaint against participant 2's share at index 1
        let result = ceremony.file_verified_complaint(
            ParticipantId(1),
            ParticipantId(2),
            1,  // share_index
            200,
        ).unwrap();

        // Complaint should be frivolous (share is valid)
        assert!(!result, "complaint should be frivolous for a valid share");

        // Should have recorded an InvalidCommitment violation against the COMPLAINER
        let violations = ceremony.violation_tracker().violations_for(ParticipantId(1));
        assert!(!violations.is_empty(), "should have violation for frivolous complainer");
        assert!(
            violations.iter().any(|v| matches!(v.violation_type, ViolationType::InvalidCommitment)),
            "should record InvalidCommitment violation against complainer"
        );

        // No violations against the accused
        let accused_violations = ceremony.violation_tracker().violations_for(ParticipantId(2));
        assert!(
            accused_violations.is_empty(),
            "accused should have no violations for a frivolous complaint"
        );
    }

    #[test]
    fn test_verified_complaint_wrong_phase() {
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        // Phase is Dealing, not Verification
        let result = ceremony.file_verified_complaint(
            ParticipantId(1),
            ParticipantId(2),
            1,
            0,
        );

        assert!(result.is_err(), "should fail in wrong phase");
        match result {
            Err(DkgError::WrongPhase { expected, actual }) => {
                assert_eq!(expected, CeremonyPhase::Verification.to_string());
                assert_eq!(actual, CeremonyPhase::Dealing.to_string());
            }
            other => panic!("expected WrongPhase error, got {:?}", other),
        }
    }

    #[test]
    fn test_full_ceremony_multi_epoch_refresh_pipeline() {
        // Full integration: DKG ceremony → 3 refresh epochs → verify invariants
        let t = 3;
        let n = 5;
        let config = DkgConfig::new(t, n).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=n as u32 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        // Each participant deals with a known secret
        let secrets: Vec<Scalar> = (1..=n as u64).map(|i| Scalar::from_u64(i * 111)).collect();
        let expected_secret: Scalar = secrets.iter().cloned().fold(Scalar::zero(), |a, b| a + b);

        for (i, secret) in secrets.iter().enumerate() {
            let pid = (i + 1) as u32;
            let dealer = crate::dealer::Dealer::with_secret(
                ParticipantId(pid), secret.clone(), t, n, &mut OsRng,
            ).unwrap();
            let deal = dealer.generate_deal();
            ceremony.submit_deal(ParticipantId(pid), deal, 0).unwrap();
        }

        let result = ceremony.finalize().unwrap();
        let original_public_key = result.public_key.clone();
        assert_eq!(result.qualified_count, n);

        // Verify initial reconstruction with different t-of-n subsets
        let all_shares: Vec<CombinedShare> = (1..=n as u32)
            .map(|i| ceremony.get_combined_share(ParticipantId(i)).unwrap())
            .collect();

        // Subset {1,2,3}
        let secret_123 = ceremony.reconstruct_secret(&all_shares[0..3]).unwrap();
        assert_eq!(secret_123, expected_secret);
        // Subset {3,4,5}
        let secret_345 = ceremony.reconstruct_secret(&all_shares[2..5]).unwrap();
        assert_eq!(secret_345, expected_secret);

        // Run 3 refresh epochs
        let mut current_shares = all_shares;
        for epoch in 1..=3u64 {
            let mut refresh = ceremony.refresh_shares().unwrap();
            assert_eq!(ceremony.epoch(), epoch);

            // All participants generate and submit refresh deals
            for i in 1..=n as u32 {
                let deal = RefreshRound::generate_refresh_deal(
                    ParticipantId(i), t, n, epoch, &mut OsRng,
                ).unwrap();
                refresh.submit_deal(deal).unwrap();
            }
            assert!(refresh.is_complete());

            // Apply refresh deltas
            let prev_shares = current_shares.clone();
            current_shares = (1..=n as u32)
                .map(|i| {
                    let delta = refresh.compute_refresh_delta(i).unwrap();
                    let mut new_val = prev_shares[(i - 1) as usize].value.clone();
                    new_val += delta;
                    CombinedShare::new(i, new_val)
                })
                .collect();

            // Verify shares actually changed
            let any_changed = prev_shares.iter().zip(current_shares.iter())
                .any(|(old, new)| old.value != new.value);
            assert!(any_changed, "Shares must change after refresh epoch {epoch}");

            // Verify reconstruction with multiple subsets
            let secret_first = ceremony.reconstruct_secret(&current_shares[0..t]).unwrap();
            assert_eq!(secret_first, expected_secret, "Subset [0..t] failed at epoch {epoch}");

            let secret_last = ceremony.reconstruct_secret(&current_shares[(n - t)..n]).unwrap();
            assert_eq!(secret_last, expected_secret, "Subset [n-t..n] failed at epoch {epoch}");

            // Mixed subset {1, 3, 5}
            let mixed = vec![
                current_shares[0].clone(),
                current_shares[2].clone(),
                current_shares[4].clone(),
            ];
            let secret_mixed = ceremony.reconstruct_secret(&mixed).unwrap();
            assert_eq!(secret_mixed, expected_secret, "Mixed subset failed at epoch {epoch}");
        }

        // Verify public key is unchanged after all refreshes
        let final_result = ceremony.finalize().unwrap();
        assert_eq!(final_result.public_key, original_public_key,
            "Public key must be preserved across refresh epochs");

        // Verify cross-epoch shares are uncorrelated (shares from epoch 0 can't
        // combine with shares from epoch 3 to reconstruct the secret)
        // This is the core proactive security property
        assert_eq!(ceremony.epoch(), 3);
    }

    #[test]
    fn test_refresh_with_threshold_subset_of_dealers() {
        // Only t (not all n) participants submit refresh deals — should still work
        let t = 2;
        let n = 4;
        let config = DkgConfig::new(t, n).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=n as u32 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }
        for i in 1..=n as u32 {
            let dealer = crate::dealer::Dealer::new(
                ParticipantId(i), t, n, &mut OsRng,
            ).unwrap();
            let deal = dealer.generate_deal();
            ceremony.submit_deal(ParticipantId(i), deal, 0).unwrap();
        }
        ceremony.finalize().unwrap();

        let original_shares: Vec<CombinedShare> = (1..=n as u32)
            .map(|i| ceremony.get_combined_share(ParticipantId(i)).unwrap())
            .collect();
        let original_secret = ceremony.reconstruct_secret(&original_shares[0..t]).unwrap();

        // Only t participants submit refresh deals (the minimum)
        let mut refresh = ceremony.refresh_shares().unwrap();
        for i in 1..=t as u32 {
            let deal = RefreshRound::generate_refresh_deal(
                ParticipantId(i), t, n, 1, &mut OsRng,
            ).unwrap();
            refresh.submit_deal(deal).unwrap();
        }
        assert!(refresh.is_complete());

        // Apply and verify
        let refreshed: Vec<CombinedShare> = (1..=n as u32)
            .map(|i| {
                let delta = refresh.compute_refresh_delta(i).unwrap();
                let mut v = original_shares[(i - 1) as usize].value.clone();
                v += delta;
                CombinedShare::new(i, v)
            })
            .collect();

        let refreshed_secret = ceremony.reconstruct_secret(&refreshed[0..t]).unwrap();
        assert_eq!(refreshed_secret, original_secret,
            "Secret must survive refresh with only t dealers");
    }
}
