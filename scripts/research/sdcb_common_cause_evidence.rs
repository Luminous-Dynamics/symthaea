//! Evidence-gated joint posterior over scientific, shared common-cause,
//! and device-local binary states.
//!
//! Build:
//!   rustc --edition=2024 --test scripts/research/sdcb_common_cause_evidence.rs -o /tmp/sdcb-common-cause-evidence
//! Run:
//!   /tmp/sdcb-common-cause-evidence
//!
//! Digest fields are shape-checked only and independently_verified is a test
//! seam. This reference does not authenticate signatures, hash preimages,
//! source custody, calibration traceability, or physical metrology.

use std::collections::HashSet;

const B: usize = 2;
const MAX_OUTCOMES: usize = 4;
const MAX_HYPOTHESES: usize = 4;
const MAX_DEVICES: usize = 3;
const PROB_TOL: f64 = 1e-9;
const EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    SharedReferenceCalibration,
    ScientificMeasurement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidIdentity,
    InvalidDigest,
    InvalidDistribution,
    InvalidLikelihoods,
    InvalidRegistry,
    InvalidProposal,
    IdentityMismatch,
    HypothesisOrderMismatch,
    DeviceRegistryMismatch,
    CommonCauseMismatch,
    DeviceMismatch,
    ProposalMismatch,
    UnverifiedEvidence,
    InvalidVerifier,
    StaleProposal,
    StaleCommonCauseEvidence,
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

fn validate_distribution(values: &[f64]) -> Result<(), Error> {
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
    pub common_cause_registry_digest: String,
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
            &self.common_cause_registry_digest,
        ].iter().any(|d| !valid_digest(d))
        {
            return Err(Error::InvalidDigest);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    pub identity: EpisodeIdentity,
    pub experiment_id: String,
    pub proposal_digest: String,
    pub prior_version: u64,
    pub hypothesis_ids: Vec<String>,
    pub device_ids: Vec<String>,
    pub common_cause_id: String,
    pub common_cause_version: u64,
    pub device_id: String,
    pub device_evidence_version: u64,
    pub calibration_epoch: u64,
    pub calibration_context_digest: String,
    pub calibration_valid_from_ms: u64,
    pub calibration_valid_until_ms: u64,
    pub role: Role,
    pub cost_units: u64,
    /// Row order: h * 4 + common_state * 2 + local_device_state.
    pub outcome_likelihoods: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub identity: EpisodeIdentity,
    pub experiment_id: String,
    pub proposal_digest: String,
    pub observation_digest: String,
    pub verifier_id: String,
    pub observed_outcome: usize,
    pub cost_units: u64,
    pub common_cause_id: String,
    pub common_cause_version: u64,
    pub device_id: String,
    pub device_evidence_version: u64,
    pub calibration_epoch: u64,
    pub calibration_context_digest: String,
    pub capture_start_ms: u64,
    pub capture_end_ms: u64,
    /// Test seam for an independently governed verifier decision.
    pub independently_verified: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    pub experiment_id: String,
    pub proposal_digest: String,
    pub observation_digest: String,
    pub verifier_id: String,
    pub common_cause_id: String,
    pub common_cause_version_before: u64,
    pub common_cause_version_after: u64,
    pub device_id: String,
    pub device_version_before: u64,
    pub device_version_after: u64,
    pub calibration_epoch: u64,
    pub calibration_context_digest: String,
    pub from_version: u64,
    pub to_version: u64,
    pub prior_joint: Vec<f64>,
    pub posterior_joint: Vec<f64>,
    pub scientific_marginal_before: Vec<f64>,
    pub scientific_marginal_after: Vec<f64>,
    pub common_cause_marginal_before: Vec<f64>,
    pub common_cause_marginal_after: Vec<f64>,
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
pub struct Ledger {
    identity: EpisodeIdentity,
    hypothesis_ids: Vec<String>,
    device_ids: Vec<String>,
    common_cause_id: String,
    posterior_joint: Vec<f64>,
    version: u64,
    common_cause_version: u64,
    device_evidence_versions: Vec<u64>,
    budget_remaining: u64,
    used_experiments: HashSet<String>,
    accepted_transition_count: usize,
}

impl Ledger {
    pub fn new(
        identity: EpisodeIdentity,
        hypothesis_ids: Vec<String>,
        device_ids: Vec<String>,
        common_cause_id: String,
        prior_joint: Vec<f64>,
        budget_units: u64,
    ) -> Result<Self, Error> {
        identity.validate()?;
        let h = hypothesis_ids.len();
        let d = device_ids.len();
        if h < 2 || h > MAX_HYPOTHESES || d == 0 || d > MAX_DEVICES
            || common_cause_id.trim().is_empty()
            || hypothesis_ids.iter().any(|id| id.trim().is_empty())
            || device_ids.iter().any(|id| id.trim().is_empty())
            || (0..h).any(|i| hypothesis_ids[..i].contains(&hypothesis_ids[i]))
            || (0..d).any(|i| device_ids[..i].contains(&device_ids[i]))
        {
            return Err(Error::InvalidRegistry);
        }
        let state_count = 1usize << (d + 1);
        if prior_joint.len() != h * state_count {
            return Err(Error::InvalidDistribution);
        }
        validate_distribution(&prior_joint)?;
        Ok(Self {
            identity,
            hypothesis_ids,
            device_evidence_versions: vec![0; d],
            device_ids,
            common_cause_id,
            posterior_joint: prior_joint,
            version: 0,
            common_cause_version: 0,
            budget_remaining: budget_units,
            used_experiments: HashSet::new(),
            accepted_transition_count: 0,
        })
    }

    pub fn joint_posterior(&self) -> &[f64] { &self.posterior_joint }
    pub fn version(&self) -> u64 { self.version }
    pub fn common_cause_version(&self) -> u64 { self.common_cause_version }
    pub fn budget_remaining(&self) -> u64 { self.budget_remaining }
    pub fn accepted_transition_count(&self) -> usize { self.accepted_transition_count }

    pub fn device_evidence_version(&self, device_id: &str) -> Option<u64> {
        self.device_ids.iter().position(|id| id == device_id)
            .map(|i| self.device_evidence_versions[i])
    }

    fn state_count(&self) -> usize {
        1usize << (self.device_ids.len() + 1)
    }

    pub fn scientific_marginal(&self) -> Vec<f64> {
        let states = self.state_count();
        let mut marginal = vec![0.0; self.hypothesis_ids.len()];
        for (i, p) in self.posterior_joint.iter().enumerate() {
            marginal[i / states] += *p;
        }
        marginal
    }

    pub fn common_cause_marginal(&self) -> Vec<f64> {
        let states = self.state_count();
        let mut marginal = vec![0.0; B];
        for (i, p) in self.posterior_joint.iter().enumerate() {
            let code = i % states;
            marginal[code & 1] += *p;
        }
        marginal
    }

    pub fn device_marginal(&self, device_id: &str) -> Result<Vec<f64>, Error> {
        let device = self.device_ids.iter().position(|id| id == device_id)
            .ok_or(Error::DeviceMismatch)?;
        let states = self.state_count();
        let mut marginal = vec![0.0; B];
        for (i, p) in self.posterior_joint.iter().enumerate() {
            let code = i % states;
            marginal[(code >> (device + 1)) & 1] += *p;
        }
        Ok(marginal)
    }

    fn posterior_for_outcome(
        &self,
        proposal: &Proposal,
        device_index: usize,
        observed: usize,
    ) -> Result<(Vec<f64>, f64), Error> {
        let outcomes = proposal.outcome_likelihoods.first().map(Vec::len).unwrap_or(0);
        if outcomes < 2 || outcomes > MAX_OUTCOMES {
            return Err(Error::InvalidLikelihoods);
        }
        if observed >= outcomes {
            return Err(Error::InvalidOutcome);
        }
        if proposal.outcome_likelihoods.len() != self.hypothesis_ids.len() * 4
            || proposal.outcome_likelihoods.iter().any(|row| {
                row.len() != outcomes
                    || row.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
                    || (row.iter().sum::<f64>() - 1.0).abs() > PROB_TOL
            })
        {
            return Err(Error::InvalidLikelihoods);
        }

        if proposal.role == Role::SharedReferenceCalibration {
            for common in 0..B {
                for local in 0..B {
                    let row_index = common * B + local;
                    let reference = &proposal.outcome_likelihoods[row_index];
                    for hypothesis in 1..self.hypothesis_ids.len() {
                        if proposal.outcome_likelihoods[hypothesis * 4 + row_index]
                            .iter().zip(reference.iter())
                            .any(|(a, b)| (*a - *b).abs() > PROB_TOL)
                        {
                            return Err(Error::CalibrationLeaksHypothesis);
                        }
                    }
                }
            }
        }

        let states = self.state_count();
        let mut predictive = 0.0;
        let mut weighted = vec![0.0; self.posterior_joint.len()];
        for (index, prior) in self.posterior_joint.iter().enumerate() {
            let h = index / states;
            let code = index % states;
            let common = code & 1;
            let local = (code >> (device_index + 1)) & 1;
            let row = h * 4 + common * 2 + local;
            let likelihood = proposal.outcome_likelihoods[row][observed];
            weighted[index] = *prior * likelihood;
            predictive += weighted[index];
        }
        if !predictive.is_finite() || predictive <= 0.0 {
            return Err(Error::ImpossibleObservation);
        }
        for p in &mut weighted { *p /= predictive; }
        validate_distribution(&weighted)?;
        Ok((weighted, predictive))
    }

    /// Apply only after every source, device, common-cause, calibration and
    /// budget binding succeeds. A rejection leaves every state/version unchanged.
    pub fn apply_verified(
        &mut self,
        proposal: &Proposal,
        receipt: &Receipt,
    ) -> Result<Transition, Error> {
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
        if proposal.common_cause_id != self.common_cause_id
            || receipt.common_cause_id != self.common_cause_id
        {
            return Err(Error::CommonCauseMismatch);
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
            || proposal.common_cause_version != receipt.common_cause_version
            || proposal.device_evidence_version != receipt.device_evidence_version
            || proposal.calibration_epoch != receipt.calibration_epoch
            || proposal.calibration_context_digest != receipt.calibration_context_digest
        {
            return Err(Error::ProposalMismatch);
        }
        if !receipt.independently_verified { return Err(Error::UnverifiedEvidence); }
        if receipt.verifier_id.trim().is_empty() { return Err(Error::InvalidVerifier); }
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
        if proposal.prior_version != self.version { return Err(Error::StaleProposal); }
        if proposal.common_cause_version != self.common_cause_version {
            return Err(Error::StaleCommonCauseEvidence);
        }
        let device_index = self.device_ids.iter().position(|id| id == &proposal.device_id)
            .ok_or(Error::DeviceMismatch)?;
        if proposal.device_evidence_version != self.device_evidence_versions[device_index] {
            return Err(Error::StaleDeviceEvidence);
        }
        if proposal.cost_units == 0 || proposal.cost_units != receipt.cost_units {
            return Err(Error::CostMismatch);
        }
        if proposal.cost_units > self.budget_remaining { return Err(Error::BudgetExceeded); }

        let (next_joint, predictive_probability) =
            self.posterior_for_outcome(proposal, device_index, receipt.observed_outcome)?;
        let next_version = self.version.checked_add(1).ok_or(Error::VersionOverflow)?;
        let next_common_version = self.common_cause_version.checked_add(1).ok_or(Error::VersionOverflow)?;
        let next_device_version = self.device_evidence_versions[device_index]
            .checked_add(1).ok_or(Error::VersionOverflow)?;
        let budget_after = self.budget_remaining - proposal.cost_units;

        let scientific_before = self.scientific_marginal();
        let common_before = self.common_cause_marginal();
        let device_before = self.device_marginal(&proposal.device_id)?;
        let prior_joint = self.posterior_joint.clone();
        let mut scientific_after = vec![0.0; self.hypothesis_ids.len()];
        let mut common_after = vec![0.0; B];
        let mut device_after = vec![0.0; B];
        let states = self.state_count();
        for (index, p) in next_joint.iter().enumerate() {
            let code = index % states;
            scientific_after[index / states] += *p;
            common_after[code & 1] += *p;
            device_after[(code >> (device_index + 1)) & 1] += *p;
        }

        let transition = Transition {
            experiment_id: proposal.experiment_id.clone(),
            proposal_digest: proposal.proposal_digest.clone(),
            observation_digest: receipt.observation_digest.clone(),
            verifier_id: receipt.verifier_id.clone(),
            common_cause_id: self.common_cause_id.clone(),
            common_cause_version_before: self.common_cause_version,
            common_cause_version_after: next_common_version,
            device_id: proposal.device_id.clone(),
            device_version_before: self.device_evidence_versions[device_index],
            device_version_after: next_device_version,
            calibration_epoch: proposal.calibration_epoch,
            calibration_context_digest: proposal.calibration_context_digest.clone(),
            from_version: self.version,
            to_version: next_version,
            prior_joint,
            posterior_joint: next_joint.clone(),
            scientific_marginal_before: scientific_before,
            scientific_marginal_after: scientific_after,
            common_cause_marginal_before: common_before,
            common_cause_marginal_after: common_after,
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
        self.common_cause_version = next_common_version;
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
            run_id: String::from("run-common-001"),
            task_id: String::from("shared-reference-control"),
            task_digest: digest('a'),
            system_digest: digest('b'),
            hypothesis_set_digest: digest('c'),
            device_registry_digest: digest('d'),
            common_cause_registry_digest: digest('e'),
        }
    }

    fn ledger(budget: u64) -> Ledger {
        let mut prior = vec![0.0; 16];
        for h in 0..2 {
            for common in 0..2 {
                prior[h * 8 + common] = 0.25; // both device-local states known zero
            }
        }
        Ledger::new(
            identity(),
            vec![String::from("h0"), String::from("h1")],
            vec![String::from("science-a"), String::from("shared-reference")],
            String::from("lab-environment-main"),
            prior,
            budget,
        ).unwrap()
    }

    fn likelihoods(role: Role) -> Vec<Vec<f64>> {
        let mut rows = Vec::new();
        for h in 0..2 {
            for common in 0..2 {
                for local in 0..2 {
                    let truth = match role {
                        Role::SharedReferenceCalibration => common ^ local,
                        Role::ScientificMeasurement => h ^ common ^ local,
                    };
                    let mut row = vec![0.0; 2];
                    row[truth] = 0.95;
                    row[1 - truth] = 0.05;
                    rows.push(row);
                }
            }
        }
        rows
    }

    fn proposal(ledger: &Ledger, id: &str, device: &str, role: Role) -> Proposal {
        Proposal {
            identity: identity(),
            experiment_id: id.to_string(),
            proposal_digest: digest('f'),
            prior_version: ledger.version(),
            hypothesis_ids: vec![String::from("h0"), String::from("h1")],
            device_ids: vec![String::from("science-a"), String::from("shared-reference")],
            common_cause_id: String::from("lab-environment-main"),
            common_cause_version: ledger.common_cause_version(),
            device_id: device.to_string(),
            device_evidence_version: ledger.device_evidence_version(device).unwrap(),
            calibration_epoch: 9,
            calibration_context_digest: digest('1'),
            calibration_valid_from_ms: 10_000,
            calibration_valid_until_ms: 20_000,
            role,
            cost_units: 1,
            outcome_likelihoods: likelihoods(role),
        }
    }

    fn receipt(p: &Proposal, outcome: usize, verified: bool) -> Receipt {
        Receipt {
            identity: identity(),
            experiment_id: p.experiment_id.clone(),
            proposal_digest: p.proposal_digest.clone(),
            observation_digest: digest('2'),
            verifier_id: String::from("independent-verifier-fixture"),
            observed_outcome: outcome,
            cost_units: p.cost_units,
            common_cause_id: p.common_cause_id.clone(),
            common_cause_version: p.common_cause_version,
            device_id: p.device_id.clone(),
            device_evidence_version: p.device_evidence_version,
            calibration_epoch: p.calibration_epoch,
            calibration_context_digest: p.calibration_context_digest.clone(),
            capture_start_ms: 12_000,
            capture_end_ms: 13_000,
            independently_verified: verified,
        }
    }

    fn snapshot(ledger: &Ledger) -> (Vec<f64>, u64, u64, Vec<u64>, u64, usize) {
        (
            ledger.joint_posterior().to_vec(),
            ledger.version(),
            ledger.common_cause_version(),
            ledger.device_evidence_versions.clone(),
            ledger.budget_remaining(),
            ledger.accepted_transition_count(),
        )
    }

    #[test]
    fn shared_reference_updates_common_cause_not_scientific_marginal() {
        let mut l = ledger(5);
        let p = proposal(&l, "reference-001", "shared-reference", Role::SharedReferenceCalibration);
        let t = l.apply_verified(&p, &receipt(&p, 0, true)).unwrap();
        assert_eq!(t.scientific_marginal_before, vec![0.5, 0.5]);
        assert_eq!(t.scientific_marginal_after, vec![0.5, 0.5]);
        assert!((t.common_cause_marginal_before[0] - 0.5).abs() < EPS);
        assert!((t.common_cause_marginal_after[0] - 0.95).abs() < EPS);
        assert!((t.common_cause_marginal_after[1] - 0.05).abs() < EPS);
        assert_eq!(t.common_cause_version_before, 0);
        assert_eq!(t.common_cause_version_after, 1);
        assert_eq!(t.device_version_after, 1);
        assert_eq!(t.budget_after, 4);
    }

    #[test]
    fn science_observation_after_reference_updates_scientific_marginal() {
        let mut l = ledger(5);
        let r = proposal(&l, "reference-001", "shared-reference", Role::SharedReferenceCalibration);
        l.apply_verified(&r, &receipt(&r, 0, true)).unwrap();
        let p = proposal(&l, "science-001", "science-a", Role::ScientificMeasurement);
        let t = l.apply_verified(&p, &receipt(&p, 0, true)).unwrap();
        assert!((t.scientific_marginal_after[0] - 0.905).abs() < EPS);
        assert!((t.scientific_marginal_after[1] - 0.095).abs() < EPS);
        assert_eq!(t.common_cause_version_after, 2);
        assert_eq!(t.device_version_after, 1);
        assert_eq!(l.accepted_transition_count(), 2);
    }

    #[test]
    fn wrong_common_cause_receipt_rejects_without_mutation() {
        let mut l = ledger(5);
        let p = proposal(&l, "science-wrong-context", "science-a", Role::ScientificMeasurement);
        let mut r = receipt(&p, 0, true);
        r.common_cause_id = String::from("different-lab");
        let before = snapshot(&l);
        assert_eq!(l.apply_verified(&p, &r), Err(Error::CommonCauseMismatch));
        assert_eq!(snapshot(&l), before);
    }

    #[test]
    fn stale_common_cause_version_rejects_without_mutation() {
        let mut l = ledger(5);
        let r = proposal(&l, "reference-001", "shared-reference", Role::SharedReferenceCalibration);
        l.apply_verified(&r, &receipt(&r, 0, true)).unwrap();
        let mut p = proposal(&l, "science-stale-common", "science-a", Role::ScientificMeasurement);
        p.common_cause_version = 0;
        let rec = receipt(&p, 0, true);
        let before = snapshot(&l);
        assert_eq!(l.apply_verified(&p, &rec), Err(Error::StaleCommonCauseEvidence));
        assert_eq!(snapshot(&l), before);
    }

    #[test]
    fn stale_device_version_rejects_without_mutation() {
        let mut l = ledger(5);
        let p1 = proposal(&l, "science-a-1", "science-a", Role::ScientificMeasurement);
        l.apply_verified(&p1, &receipt(&p1, 0, true)).unwrap();
        let mut stale = proposal(&l, "science-a-2-stale", "science-a", Role::ScientificMeasurement);
        stale.device_evidence_version = 0;
        let rec = receipt(&stale, 0, true);
        let before = snapshot(&l);
        assert_eq!(l.apply_verified(&stale, &rec), Err(Error::StaleDeviceEvidence));
        assert_eq!(snapshot(&l), before);
    }

    #[test]
    fn expired_calibration_context_rejects_without_mutation() {
        let mut l = ledger(5);
        let mut p = proposal(&l, "science-expired", "science-a", Role::ScientificMeasurement);
        p.calibration_valid_until_ms = 12_500;
        let rec = receipt(&p, 0, true);
        let before = snapshot(&l);
        assert_eq!(l.apply_verified(&p, &rec), Err(Error::CalibrationExpired));
        assert_eq!(snapshot(&l), before);
    }

    #[test]
    fn calibration_cannot_leak_scientific_hypothesis() {
        let mut l = ledger(5);
        let mut p = proposal(&l, "bad-calibration", "shared-reference", Role::SharedReferenceCalibration);
        p.outcome_likelihoods[4] = vec![0.7, 0.3];
        let rec = receipt(&p, 0, true);
        let before = snapshot(&l);
        assert_eq!(l.apply_verified(&p, &rec), Err(Error::CalibrationLeaksHypothesis));
        assert_eq!(snapshot(&l), before);
    }

    #[test]
    fn unverified_receipt_rejects_without_mutation() {
        let mut l = ledger(5);
        let p = proposal(&l, "unverified", "science-a", Role::ScientificMeasurement);
        let rec = receipt(&p, 0, false);
        let before = snapshot(&l);
        assert_eq!(l.apply_verified(&p, &rec), Err(Error::UnverifiedEvidence));
        assert_eq!(snapshot(&l), before);
    }

    #[test]
    fn duplicate_experiment_is_consume_once() {
        let mut l = ledger(5);
        let p = proposal(&l, "duplicate", "science-a", Role::ScientificMeasurement);
        l.apply_verified(&p, &receipt(&p, 0, true)).unwrap();
        let mut duplicate = proposal(&l, "duplicate", "science-a", Role::ScientificMeasurement);
        duplicate.prior_version = l.version();
        duplicate.common_cause_version = l.common_cause_version();
        duplicate.device_evidence_version = l.device_evidence_version("science-a").unwrap();
        let rec = receipt(&duplicate, 0, true);
        let before = snapshot(&l);
        assert_eq!(l.apply_verified(&duplicate, &rec), Err(Error::DuplicateExperiment));
        assert_eq!(snapshot(&l), before);
    }

    #[test]
    fn over_budget_receipt_rejects_without_mutation() {
        let mut l = ledger(0);
        let p = proposal(&l, "over-budget", "science-a", Role::ScientificMeasurement);
        let rec = receipt(&p, 0, true);
        let before = snapshot(&l);
        assert_eq!(l.apply_verified(&p, &rec), Err(Error::BudgetExceeded));
        assert_eq!(snapshot(&l), before);
    }
}
