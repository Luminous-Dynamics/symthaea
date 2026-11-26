//! Evidence-gated updates for a joint scientific-hypothesis / instrument-state belief.
//!
//! Build:
//!   rustc --edition=2024 --test scripts/research/sdcb_joint_belief_transition.rs -o /tmp/sdcb-joint-belief
//! Run:
//!   /tmp/sdcb-joint-belief
//!
//! This is a finite reference contract. It binds each observation to an exact
//! device registry, device identity, calibration context/epoch, device evidence
//! version, proposal, capture interval, and independently verified receipt.
//! The posterior is joint over scientific hypotheses and binary device states.
//!
//! It checks digest shape and a verifier test seam; it does not authenticate
//! signatures, verify hash preimages, prove custody, or establish physical
//! metrological traceability.

use std::collections::HashSet;

const STATES: usize = 2;
const PROB_TOL: f64 = 1e-9;
const EPS: f64 = 1e-12;
const MAX_HYPOTHESES: usize = 4;
const MAX_DEVICES: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentRole {
    Calibration,
    Science,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidIdentity,
    InvalidDigest,
    InvalidDistribution,
    InvalidLikelihoods,
    InvalidProposal,
    IdentityMismatch,
    HypothesisOrderMismatch,
    DeviceRegistryMismatch,
    DeviceMismatch,
    ProposalMismatch,
    UnverifiedEvidence,
    InvalidVerifier,
    StaleProposal,
    StaleDeviceEvidence,
    DuplicateExperiment,
    BudgetExceeded,
    CostMismatch,
    InvalidOutcome,
    ImpossibleObservation,
    InvalidCalibrationWindow,
    CalibrationExpired,
    CalibrationLeaksHypothesis,
    VersionOverflow,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn validate_probabilities(values: &[f64]) -> Result<(), Error> {
    if values.is_empty()
        || values.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
        || (values.iter().sum::<f64>() - 1.0).abs() > PROB_TOL
    {
        return Err(Error::InvalidDistribution);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeIdentity {
    pub run_id: String,
    pub task_id: String,
    pub task_digest: String,
    pub system_digest: String,
    pub hypothesis_set_digest: String,
    pub device_registry_digest: String,
}

impl EpisodeIdentity {
    fn validate(&self) -> Result<(), Error> {
        if self.run_id.trim().is_empty() || self.task_id.trim().is_empty() {
            return Err(Error::InvalidIdentity);
        }
        if [
            &self.task_digest,
            &self.system_digest,
            &self.hypothesis_set_digest,
            &self.device_registry_digest,
        ].iter().any(|d| !valid_digest(d.as_str()))
        {
            return Err(Error::InvalidDigest);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct JointExperimentProposal {
    pub identity: EpisodeIdentity,
    pub experiment_id: String,
    pub proposal_digest: String,
    pub prior_version: u64,
    pub hypothesis_ids: Vec<String>,
    pub device_ids: Vec<String>,
    pub device_id: String,
    /// Monotonic evidence version for the named device.
    pub device_evidence_version: u64,
    /// Version/epoch of the calibration record used to construct likelihoods.
    pub calibration_epoch: u64,
    /// Digest of the exact calibration record or explicit uncalibrated context.
    pub calibration_context_digest: String,
    pub calibration_valid_from_ms: u64,
    pub calibration_valid_until_ms: u64,
    pub role: ExperimentRole,
    pub cost_units: u64,
    /// Row (hypothesis * STATES + device_state), columns are outcomes.
    pub outcome_likelihoods: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JointEvidenceReceipt {
    pub identity: EpisodeIdentity,
    pub experiment_id: String,
    pub proposal_digest: String,
    pub observation_digest: String,
    pub verifier_id: String,
    pub observed_outcome: usize,
    pub cost_units: u64,
    pub device_id: String,
    pub device_evidence_version: u64,
    pub calibration_epoch: u64,
    pub calibration_context_digest: String,
    pub capture_start_ms: u64,
    pub capture_end_ms: u64,
    /// Test seam for an independently governed verifier decision.
    /// An untrusted agent must never be allowed to set this field.
    pub independently_verified: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JointBeliefTransition {
    pub experiment_id: String,
    pub proposal_digest: String,
    pub observation_digest: String,
    pub verifier_id: String,
    pub device_id: String,
    pub calibration_epoch: u64,
    pub calibration_context_digest: String,
    pub from_version: u64,
    pub to_version: u64,
    pub device_evidence_version_before: u64,
    pub device_evidence_version_after: u64,
    pub prior_joint: Vec<f64>,
    pub posterior_joint: Vec<f64>,
    pub scientific_marginal_before: Vec<f64>,
    pub scientific_marginal_after: Vec<f64>,
    pub device_marginal_before: Vec<f64>,
    pub device_marginal_after: Vec<f64>,
    pub predictive_probability: f64,
    pub observed_outcome: usize,
    pub cost_units: u64,
    pub budget_before: u64,
    pub budget_after: u64,
    pub capture_start_ms: u64,
    pub capture_end_ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JointBeliefLedger {
    identity: EpisodeIdentity,
    hypothesis_ids: Vec<String>,
    device_ids: Vec<String>,
    posterior_joint: Vec<f64>,
    version: u64,
    device_evidence_versions: Vec<u64>,
    budget_remaining: u64,
    used_experiments: HashSet<String>,
    accepted_transition_count: usize,
}

impl JointBeliefLedger {
    pub fn new(
        identity: EpisodeIdentity,
        hypothesis_ids: Vec<String>,
        device_ids: Vec<String>,
        prior_joint: Vec<f64>,
        budget_units: u64,
    ) -> Result<Self, Error> {
        identity.validate()?;
        let h = hypothesis_ids.len();
        let d = device_ids.len();
        if h < 2 || h > MAX_HYPOTHESES || d == 0 || d > MAX_DEVICES
            || hypothesis_ids.iter().any(|id| id.trim().is_empty())
            || device_ids.iter().any(|id| id.trim().is_empty())
            || (0..h).any(|i| hypothesis_ids[..i].contains(&hypothesis_ids[i]))
            || (0..d).any(|i| device_ids[..i].contains(&device_ids[i]))
            || prior_joint.len() != h * (1usize << d)
        {
            return Err(Error::InvalidIdentity);
        }
        validate_probabilities(&prior_joint)?;
        Ok(Self {
            identity,
            hypothesis_ids,
            device_evidence_versions: vec![0; d],
            device_ids,
            posterior_joint: prior_joint,
            version: 0,
            budget_remaining: budget_units,
            used_experiments: HashSet::new(),
            accepted_transition_count: 0,
        })
    }

    pub fn joint_posterior(&self) -> &[f64] {
        &self.posterior_joint
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn budget_remaining(&self) -> u64 {
        self.budget_remaining
    }

    pub fn accepted_transition_count(&self) -> usize {
        self.accepted_transition_count
    }

    pub fn device_evidence_version(&self, device_id: &str) -> Option<u64> {
        self.device_ids.iter().position(|id| id == device_id)
            .map(|index| self.device_evidence_versions[index])
    }

    pub fn scientific_marginal(&self) -> Vec<f64> {
        let states = 1usize << self.device_ids.len();
        let mut marginal = vec![0.0; self.hypothesis_ids.len()];
        for (index, p) in self.posterior_joint.iter().enumerate() {
            marginal[index / states] += *p;
        }
        marginal
    }

    pub fn device_marginal(&self, device_id: &str) -> Result<Vec<f64>, Error> {
        let device = self.device_ids.iter().position(|id| id == device_id)
            .ok_or(Error::DeviceMismatch)?;
        let states = 1usize << self.device_ids.len();
        let mut marginal = vec![0.0; STATES];
        for (index, p) in self.posterior_joint.iter().enumerate() {
            let code = index % states;
            marginal[(code >> device) & 1] += *p;
        }
        Ok(marginal)
    }

    fn posterior_for_device_outcome(
        &self,
        proposal: &JointExperimentProposal,
        device_index: usize,
        outcome: usize,
    ) -> Result<(Vec<f64>, f64), Error> {
        let outcome_count = proposal.outcome_likelihoods.first()
            .map(Vec::len).unwrap_or(0);
        if outcome_count < 2 || outcome >= outcome_count {
            return Err(Error::InvalidOutcome);
        }
        let expected_rows = self.hypothesis_ids.len() * STATES;
        if proposal.outcome_likelihoods.len() != expected_rows
            || proposal.outcome_likelihoods.iter().any(|row| {
                row.len() != outcome_count
                    || row.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
                    || (row.iter().sum::<f64>() - 1.0).abs() > PROB_TOL
            })
        {
            return Err(Error::InvalidLikelihoods);
        }

        if proposal.role == ExperimentRole::Calibration {
            for state in 0..STATES {
                let reference = &proposal.outcome_likelihoods[state];
                for hypothesis in 1..self.hypothesis_ids.len() {
                    if proposal.outcome_likelihoods[hypothesis * STATES + state]
                        .iter().zip(reference.iter())
                        .any(|(a, b)| (a - b).abs() > PROB_TOL)
                    {
                        return Err(Error::CalibrationLeaksHypothesis);
                    }
                }
            }
        }

        let state_count = 1usize << self.device_ids.len();
        let mut predictive = 0.0;
        let mut weighted = vec![0.0; self.posterior_joint.len()];
        for (joint_index, prior) in self.posterior_joint.iter().enumerate() {
            let hypothesis = joint_index / state_count;
            let code = joint_index % state_count;
            let device_state = (code >> device_index) & 1;
            let likelihood = proposal.outcome_likelihoods[hypothesis * STATES + device_state][outcome];
            weighted[joint_index] = *prior * likelihood;
            predictive += weighted[joint_index];
        }
        if !predictive.is_finite() || predictive <= 0.0 {
            return Err(Error::ImpossibleObservation);
        }
        for p in &mut weighted {
            *p /= predictive;
        }
        validate_probabilities(&weighted)?;
        Ok((weighted, predictive))
    }

    /// Validate every binding and derive the complete next state before mutation.
    /// Rejected receipts leave both posteriors, both version layers, budget, and
    /// accepted-transition count unchanged.
    pub fn apply_verified(
        &mut self,
        proposal: &JointExperimentProposal,
        receipt: &JointEvidenceReceipt,
    ) -> Result<JointBeliefTransition, Error> {
        self.identity.validate()?;
        proposal.identity.validate()?;
        receipt.identity.validate()?;

        if proposal.identity != self.identity || receipt.identity != self.identity {
            return Err(Error::IdentityMismatch);
        }
        if proposal.hypothesis_ids != self.hypothesis_ids {
            return Err(Error::HypothesisOrderMismatch);
        }
        if proposal.device_ids != self.device_ids {
            return Err(Error::DeviceRegistryMismatch);
        }
        if proposal.device_id != receipt.device_id {
            return Err(Error::DeviceMismatch);
        }
        if proposal.experiment_id.trim().is_empty()
            || !valid_digest(&proposal.proposal_digest)
            || !valid_digest(&receipt.observation_digest)
            || !valid_digest(&proposal.calibration_context_digest)
        {
            return Err(Error::InvalidDigest);
        }
        if proposal.experiment_id != receipt.experiment_id
            || proposal.proposal_digest != receipt.proposal_digest
        {
            return Err(Error::ProposalMismatch);
        }
        if proposal.calibration_epoch != receipt.calibration_epoch
            || proposal.calibration_context_digest != receipt.calibration_context_digest
            || proposal.device_evidence_version != receipt.device_evidence_version
        {
            return Err(Error::ProposalMismatch);
        }
        if !receipt.independently_verified {
            return Err(Error::UnverifiedEvidence);
        }
        if receipt.verifier_id.trim().is_empty() {
            return Err(Error::InvalidVerifier);
        }
        if proposal.calibration_valid_from_ms > proposal.calibration_valid_until_ms
            || receipt.capture_start_ms > receipt.capture_end_ms
        {
            return Err(Error::InvalidCalibrationWindow);
        }
        if receipt.capture_start_ms < proposal.calibration_valid_from_ms
            || receipt.capture_end_ms > proposal.calibration_valid_until_ms
        {
            return Err(Error::CalibrationExpired);
        }
        if self.used_experiments.contains(&proposal.experiment_id) {
            return Err(Error::DuplicateExperiment);
        }
        if proposal.prior_version != self.version {
            return Err(Error::StaleProposal);
        }
        let device_index = self.device_ids.iter().position(|id| id == &proposal.device_id)
            .ok_or(Error::DeviceMismatch)?;
        if proposal.device_evidence_version != self.device_evidence_versions[device_index] {
            return Err(Error::StaleDeviceEvidence);
        }
        if proposal.cost_units == 0 || proposal.cost_units != receipt.cost_units {
            return Err(Error::CostMismatch);
        }
        if proposal.cost_units > self.budget_remaining {
            return Err(Error::BudgetExceeded);
        }

        let (next_joint, predictive_probability) = self.posterior_for_device_outcome(
            proposal, device_index, receipt.observed_outcome,
        )?;
        let next_version = self.version.checked_add(1).ok_or(Error::VersionOverflow)?;
        let next_device_version = self.device_evidence_versions[device_index]
            .checked_add(1).ok_or(Error::VersionOverflow)?;
        let budget_after = self.budget_remaining - proposal.cost_units;
        let scientific_before = self.scientific_marginal();
        let device_before = self.device_marginal(&proposal.device_id)?;
        let prior_joint = self.posterior_joint.clone();

        let state_count = 1usize << self.device_ids.len();
        let mut scientific_after = vec![0.0; self.hypothesis_ids.len()];
        let mut device_after = vec![0.0; STATES];
        for (joint_index, p) in next_joint.iter().enumerate() {
            scientific_after[joint_index / state_count] += *p;
            let code = joint_index % state_count;
            device_after[(code >> device_index) & 1] += *p;
        }

        let transition = JointBeliefTransition {
            experiment_id: proposal.experiment_id.clone(),
            proposal_digest: proposal.proposal_digest.clone(),
            observation_digest: receipt.observation_digest.clone(),
            verifier_id: receipt.verifier_id.clone(),
            device_id: proposal.device_id.clone(),
            calibration_epoch: proposal.calibration_epoch,
            calibration_context_digest: proposal.calibration_context_digest.clone(),
            from_version: self.version,
            to_version: next_version,
            device_evidence_version_before: self.device_evidence_versions[device_index],
            device_evidence_version_after: next_device_version,
            prior_joint,
            posterior_joint: next_joint.clone(),
            scientific_marginal_before: scientific_before,
            scientific_marginal_after: scientific_after,
            device_marginal_before: device_before,
            device_marginal_after: device_after,
            predictive_probability,
            observed_outcome: receipt.observed_outcome,
            cost_units: proposal.cost_units,
            budget_before: self.budget_remaining,
            budget_after,
            capture_start_ms: receipt.capture_start_ms,
            capture_end_ms: receipt.capture_end_ms,
        };

        self.posterior_joint = next_joint;
        self.version = next_version;
        self.device_evidence_versions[device_index] = next_device_version;
        self.budget_remaining = budget_after;
        self.used_experiments.insert(proposal.experiment_id.clone());
        self.accepted_transition_count += 1;

        Ok(transition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat(ch).take(64).collect()
    }

    fn identity() -> EpisodeIdentity {
        EpisodeIdentity {
            run_id: String::from("run-001"),
            task_id: String::from("joint-state-fixture"),
            task_digest: digest('a'),
            system_digest: digest('b'),
            hypothesis_set_digest: digest('c'),
            device_registry_digest: digest('d'),
        }
    }

    fn make_ledger(budget: u64) -> JointBeliefLedger {
        let mut prior = vec![0.0; 8];
        for h in 0..2 {
            for code in 0..4 {
                let state_b = (code >> 1) & 1;
                let p_b = if state_b == 0 { 0.9 } else { 0.1 };
                prior[h * 4 + code] = 0.5 * 0.5 * p_b;
            }
        }
        JointBeliefLedger::new(
            identity(),
            vec![String::from("h0"), String::from("h1")],
            vec![String::from("device-a"), String::from("device-b")],
            prior,
            budget,
        ).unwrap()
    }

    fn likelihoods(role: ExperimentRole, accuracy: f64) -> Vec<Vec<f64>> {
        let mut rows = Vec::new();
        for h in 0..2 {
            for state in 0..STATES {
                let truth = match role {
                    ExperimentRole::Calibration => state,
                    ExperimentRole::Science => h ^ state,
                };
                let mut row = vec![0.0; 2];
                row[truth] = accuracy;
                row[1 - truth] = 1.0 - accuracy;
                rows.push(row);
            }
        }
        rows
    }

    fn proposal(ledger: &JointBeliefLedger, id: &str, device: &str, role: ExperimentRole) -> JointExperimentProposal {
        JointExperimentProposal {
            identity: identity(),
            experiment_id: id.to_string(),
            proposal_digest: digest('e'),
            prior_version: ledger.version(),
            hypothesis_ids: vec![String::from("h0"), String::from("h1")],
            device_ids: vec![String::from("device-a"), String::from("device-b")],
            device_id: device.to_string(),
            device_evidence_version: ledger.device_evidence_version(device).unwrap(),
            calibration_epoch: 7,
            calibration_context_digest: digest('f'),
            calibration_valid_from_ms: 1_000,
            calibration_valid_until_ms: 2_000,
            role,
            cost_units: 1,
            outcome_likelihoods: likelihoods(role, 0.95),
        }
    }

    fn receipt(p: &JointExperimentProposal, outcome: usize, verified: bool) -> JointEvidenceReceipt {
        JointEvidenceReceipt {
            identity: identity(),
            experiment_id: p.experiment_id.clone(),
            proposal_digest: p.proposal_digest.clone(),
            observation_digest: digest('9'),
            verifier_id: String::from("independent-verifier"),
            observed_outcome: outcome,
            cost_units: p.cost_units,
            device_id: p.device_id.clone(),
            device_evidence_version: p.device_evidence_version,
            calibration_epoch: p.calibration_epoch,
            calibration_context_digest: p.calibration_context_digest.clone(),
            capture_start_ms: 1_200,
            capture_end_ms: 1_300,
            independently_verified: verified,
        }
    }

    fn snapshot(ledger: &JointBeliefLedger) -> (Vec<f64>, u64, Vec<u64>, u64, usize) {
        (
            ledger.joint_posterior().to_vec(),
            ledger.version(),
            ledger.device_evidence_versions.clone(),
            ledger.budget_remaining(),
            ledger.accepted_transition_count(),
        )
    }

    #[test]
    fn calibration_updates_device_marginal_without_spurious_hypothesis_shift() {
        let mut ledger = make_ledger(4);
        let h_before = ledger.scientific_marginal();
        let c_before = ledger.device_marginal("device-a").unwrap();
        let p = proposal(&ledger, "cal-a-1", "device-a", ExperimentRole::Calibration);
        let transition = ledger.apply_verified(&p, &receipt(&p, 0, true)).unwrap();
        let h_after = ledger.scientific_marginal();
        let c_after = ledger.device_marginal("device-a").unwrap();

        assert_eq!(h_before, vec![0.5, 0.5]);
        assert!((h_after[0] - h_before[0]).abs() < EPS);
        assert!((h_after[1] - h_before[1]).abs() < EPS);
        assert!((c_before[0] - 0.5).abs() < EPS);
        assert!((c_after[0] - 0.95).abs() < EPS);
        assert!((c_after[1] - 0.05).abs() < EPS);
        assert_eq!(transition.device_evidence_version_before, 0);
        assert_eq!(transition.device_evidence_version_after, 1);
    }

    #[test]
    fn verified_science_observation_updates_joint_and_scientific_marginals() {
        let mut ledger = make_ledger(4);
        let calibration = proposal(&ledger, "cal-a-1", "device-a", ExperimentRole::Calibration);
        ledger.apply_verified(&calibration, &receipt(&calibration, 0, true)).unwrap();

        let science = proposal(&ledger, "science-a-1", "device-a", ExperimentRole::Science);
        let transition = ledger.apply_verified(&science, &receipt(&science, 0, true)).unwrap();
        let scientific = ledger.scientific_marginal();

        assert!((scientific[0] - 0.905).abs() < EPS);
        assert!((scientific[1] - 0.095).abs() < EPS);
        assert_eq!(transition.to_version, 2);
        assert_eq!(transition.device_evidence_version_after, 2);
        assert_eq!(ledger.accepted_transition_count(), 2);
    }

    #[test]
    fn wrong_device_receipt_rejects_without_any_mutation() {
        let mut ledger = make_ledger(4);
        let p = proposal(&ledger, "science-a-1", "device-a", ExperimentRole::Science);
        let mut r = receipt(&p, 0, true);
        r.device_id = String::from("device-b");
        let before = snapshot(&ledger);
        assert_eq!(ledger.apply_verified(&p, &r), Err(Error::DeviceMismatch));
        assert_eq!(snapshot(&ledger), before);
    }

    #[test]
    fn stale_device_evidence_version_rejects_without_any_mutation() {
        let mut ledger = make_ledger(4);
        let p1 = proposal(&ledger, "cal-a-1", "device-a", ExperimentRole::Calibration);
        ledger.apply_verified(&p1, &receipt(&p1, 0, true)).unwrap();

        let mut stale = proposal(&ledger, "science-a-stale", "device-a", ExperimentRole::Science);
        stale.device_evidence_version = 0;
        let r = receipt(&stale, 0, true);
        let before = snapshot(&ledger);
        assert_eq!(ledger.apply_verified(&stale, &r), Err(Error::StaleDeviceEvidence));
        assert_eq!(snapshot(&ledger), before);
    }

    #[test]
    fn expired_calibration_context_rejects_without_any_mutation() {
        let mut ledger = make_ledger(4);
        let mut p = proposal(&ledger, "science-a-expired", "device-a", ExperimentRole::Science);
        p.calibration_valid_until_ms = 1_250;
        let r = receipt(&p, 0, true);
        let before = snapshot(&ledger);
        assert_eq!(ledger.apply_verified(&p, &r), Err(Error::CalibrationExpired));
        assert_eq!(snapshot(&ledger), before);
    }

    #[test]
    fn unverified_receipt_rejects_without_any_mutation() {
        let mut ledger = make_ledger(4);
        let p = proposal(&ledger, "science-a-unverified", "device-a", ExperimentRole::Science);
        let r = receipt(&p, 0, false);
        let before = snapshot(&ledger);
        assert_eq!(ledger.apply_verified(&p, &r), Err(Error::UnverifiedEvidence));
        assert_eq!(snapshot(&ledger), before);
    }

    #[test]
    fn calibration_likelihood_cannot_leak_hypothesis_information() {
        let mut ledger = make_ledger(4);
        let mut p = proposal(&ledger, "cal-a-bad", "device-a", ExperimentRole::Calibration);
        p.outcome_likelihoods[2] = vec![0.7, 0.3];
        let r = receipt(&p, 0, true);
        let before = snapshot(&ledger);
        assert_eq!(ledger.apply_verified(&p, &r), Err(Error::CalibrationLeaksHypothesis));
        assert_eq!(snapshot(&ledger), before);
    }

    #[test]
    fn receipt_from_another_calibration_context_is_rejected() {
        let mut ledger = make_ledger(4);
        let p = proposal(&ledger, "science-a-wrong-cal", "device-a", ExperimentRole::Science);
        let mut r = receipt(&p, 0, true);
        r.calibration_context_digest = digest('8');
        let before = snapshot(&ledger);
        assert_eq!(ledger.apply_verified(&p, &r), Err(Error::ProposalMismatch));
        assert_eq!(snapshot(&ledger), before);
    }

    #[test]
    fn duplicate_experiment_is_consume_once() {
        let mut ledger = make_ledger(4);
        let p = proposal(&ledger, "science-a-1", "device-a", ExperimentRole::Science);
        let r = receipt(&p, 0, true);
        ledger.apply_verified(&p, &r).unwrap();

        let mut duplicate = proposal(&ledger, "science-a-1", "device-a", ExperimentRole::Science);
        duplicate.prior_version = ledger.version();
        duplicate.device_evidence_version = ledger.device_evidence_version("device-a").unwrap();
        let duplicate_receipt = receipt(&duplicate, 0, true);
        let before = snapshot(&ledger);
        assert_eq!(ledger.apply_verified(&duplicate, &duplicate_receipt), Err(Error::DuplicateExperiment));
        assert_eq!(snapshot(&ledger), before);
    }

    #[test]
    fn over_budget_receipt_rejects_without_any_mutation() {
        let mut ledger = make_ledger(0);
        let p = proposal(&ledger, "science-a-1", "device-a", ExperimentRole::Science);
        let r = receipt(&p, 0, true);
        let before = snapshot(&ledger);
        assert_eq!(ledger.apply_verified(&p, &r), Err(Error::BudgetExceeded));
        assert_eq!(snapshot(&ledger), before);
    }
}
