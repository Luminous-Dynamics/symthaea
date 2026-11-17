// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Conservative candidate retention under bounded and persistent outcome noise.
//!
//! This crate does not estimate probabilities. A hypothesis remains compatible
//! while its declared allowed-outcome model can explain every observation.
//! Persistent-noise predictions share one latent error state across all probes.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use symmetry_active_experiment_reference::ActiveExperimentProblemV1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoiseProbeV1 {
    pub probe_id: String,
    pub cost: u64,
    pub admissible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoiseObservationV1 {
    pub probe_id: String,
    /// Finite set of outcomes consistent with the measurement record.
    pub outcomes: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedNoiseProblemV1 {
    pub problem_id: String,
    pub hypothesis_ids: Vec<String>,
    pub probes: Vec<NoiseProbeV1>,
    /// [hypothesis][probe] -> allowed finite outcomes for that pair.
    pub allowed_outcomes_by_hypothesis: Vec<Vec<Vec<u32>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentNoiseProblemV1 {
    pub problem_id: String,
    pub hypothesis_ids: Vec<String>,
    pub probes: Vec<NoiseProbeV1>,
    /// A globally shared latent error realization, fixed for the campaign.
    pub error_state_ids: Vec<String>,
    /// [hypothesis][probe][shared error state] -> deterministic outcome.
    pub predicted_outcomes_by_hypothesis: Vec<Vec<Vec<u32>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoiseErrorV1 {
    EmptyProblemId,
    EmptyHypothesisSet,
    TooManyHypotheses,
    EmptyHypothesisId,
    DuplicateHypothesisId,
    EmptyProbeSet,
    TooManyProbes,
    EmptyProbeId,
    DuplicateProbeId,
    ZeroProbeCost {
        probe_id: String,
    },
    BoundedHypothesisCountMismatch,
    BoundedProbeCountMismatch {
        hypothesis_id: String,
    },
    EmptyAllowedOutcomeSet {
        hypothesis_id: String,
        probe_id: String,
    },
    DuplicateAllowedOutcome {
        hypothesis_id: String,
        probe_id: String,
    },
    EmptyErrorStateSet,
    TooManyErrorStates,
    EmptyErrorStateId,
    DuplicateErrorStateId,
    PersistentHypothesisCountMismatch,
    PersistentProbeCountMismatch {
        hypothesis_id: String,
    },
    PersistentErrorStateCountMismatch {
        hypothesis_id: String,
        probe_id: String,
    },
    UnknownProbe {
        probe_id: String,
    },
    InadmissibleProbe {
        probe_id: String,
    },
    EmptyObservedOutcomeSet {
        probe_id: String,
    },
    DuplicateObservedOutcome {
        probe_id: String,
    },
    ReceiptMismatch,
    InvalidDeterministicInput,
}

impl fmt::Display for NoiseErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for NoiseErrorV1 {}

fn append_bytes(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u64).to_le_bytes());
    target.extend_from_slice(value);
}

fn append_string(target: &mut Vec<u8>, value: &str) {
    append_bytes(target, value.as_bytes());
}

fn append_outcomes(target: &mut Vec<u8>, outcomes: &[u32]) {
    let mut sorted = outcomes.to_vec();
    sorted.sort_unstable();
    target.extend_from_slice(&(sorted.len() as u64).to_le_bytes());
    for outcome in sorted {
        target.extend_from_slice(&outcome.to_le_bytes());
    }
}

fn validate_identifiers(
    problem_id: &str,
    hypothesis_ids: &[String],
    probes: &[NoiseProbeV1],
) -> Result<(), NoiseErrorV1> {
    if problem_id.trim().is_empty() {
        return Err(NoiseErrorV1::EmptyProblemId);
    }
    if hypothesis_ids.is_empty() {
        return Err(NoiseErrorV1::EmptyHypothesisSet);
    }
    if hypothesis_ids.len() > 64 {
        return Err(NoiseErrorV1::TooManyHypotheses);
    }
    if probes.is_empty() {
        return Err(NoiseErrorV1::EmptyProbeSet);
    }
    if probes.len() > 64 {
        return Err(NoiseErrorV1::TooManyProbes);
    }

    let mut seen_hypotheses = BTreeSet::new();
    for hypothesis_id in hypothesis_ids {
        if hypothesis_id.trim().is_empty() {
            return Err(NoiseErrorV1::EmptyHypothesisId);
        }
        if !seen_hypotheses.insert(hypothesis_id) {
            return Err(NoiseErrorV1::DuplicateHypothesisId);
        }
    }

    let mut seen_probes = BTreeSet::new();
    for probe in probes {
        if probe.probe_id.trim().is_empty() {
            return Err(NoiseErrorV1::EmptyProbeId);
        }
        if !seen_probes.insert(&probe.probe_id) {
            return Err(NoiseErrorV1::DuplicateProbeId);
        }
        if probe.cost == 0 {
            return Err(NoiseErrorV1::ZeroProbeCost {
                probe_id: probe.probe_id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_observation_set(observation: &NoiseObservationV1) -> Result<(), NoiseErrorV1> {
    if observation.outcomes.is_empty() {
        return Err(NoiseErrorV1::EmptyObservedOutcomeSet {
            probe_id: observation.probe_id.clone(),
        });
    }
    let mut outcomes = BTreeSet::new();
    for outcome in &observation.outcomes {
        if !outcomes.insert(*outcome) {
            return Err(NoiseErrorV1::DuplicateObservedOutcome {
                probe_id: observation.probe_id.clone(),
            });
        }
    }
    Ok(())
}

impl BoundedNoiseProblemV1 {
    pub fn validate(&self) -> Result<(), NoiseErrorV1> {
        validate_identifiers(&self.problem_id, &self.hypothesis_ids, &self.probes)?;
        if self.allowed_outcomes_by_hypothesis.len() != self.hypothesis_ids.len() {
            return Err(NoiseErrorV1::BoundedHypothesisCountMismatch);
        }
        for (hypothesis_index, by_probe) in self.allowed_outcomes_by_hypothesis.iter().enumerate() {
            if by_probe.len() != self.probes.len() {
                return Err(NoiseErrorV1::BoundedProbeCountMismatch {
                    hypothesis_id: self.hypothesis_ids[hypothesis_index].clone(),
                });
            }
            for (probe_index, allowed) in by_probe.iter().enumerate() {
                if allowed.is_empty() {
                    return Err(NoiseErrorV1::EmptyAllowedOutcomeSet {
                        hypothesis_id: self.hypothesis_ids[hypothesis_index].clone(),
                        probe_id: self.probes[probe_index].probe_id.clone(),
                    });
                }
                let mut unique = BTreeSet::new();
                if allowed.iter().any(|outcome| !unique.insert(*outcome)) {
                    return Err(NoiseErrorV1::DuplicateAllowedOutcome {
                        hypothesis_id: self.hypothesis_ids[hypothesis_index].clone(),
                        probe_id: self.probes[probe_index].probe_id.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Canonical versioned bytes; this is not a cryptographic digest/signature.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symm_active_noise.bounded.v1\0".to_vec();
        append_string(&mut bytes, &self.problem_id);
        bytes.extend_from_slice(&(self.hypothesis_ids.len() as u64).to_le_bytes());
        for hypothesis_id in &self.hypothesis_ids {
            append_string(&mut bytes, hypothesis_id);
        }
        bytes.extend_from_slice(&(self.probes.len() as u64).to_le_bytes());
        for probe in &self.probes {
            append_string(&mut bytes, &probe.probe_id);
            bytes.extend_from_slice(&probe.cost.to_le_bytes());
            bytes.push(u8::from(probe.admissible));
        }
        for by_probe in &self.allowed_outcomes_by_hypothesis {
            for allowed in by_probe {
                append_outcomes(&mut bytes, allowed);
            }
        }
        bytes
    }

    /// Lift a deterministic experiment table into singleton allowed-outcome
    /// sets. This is the zero-noise bridge to the exact planner.
    pub fn from_deterministic(problem: &ActiveExperimentProblemV1) -> Result<Self, NoiseErrorV1> {
        if problem.hypotheses.is_empty()
            || problem.probes.is_empty()
            || problem
                .probes
                .iter()
                .any(|probe| probe.outcomes.len() != problem.hypotheses.len())
        {
            return Err(NoiseErrorV1::InvalidDeterministicInput);
        }
        let adapted = Self {
            problem_id: problem.problem_id.clone(),
            hypothesis_ids: problem.hypotheses.clone(),
            probes: problem
                .probes
                .iter()
                .map(|probe| NoiseProbeV1 {
                    probe_id: probe.probe_id.clone(),
                    cost: probe.cost,
                    admissible: probe.admissible,
                })
                .collect(),
            allowed_outcomes_by_hypothesis: (0..problem.hypotheses.len())
                .map(|hypothesis_index| {
                    problem
                        .probes
                        .iter()
                        .map(|probe| vec![probe.outcomes[hypothesis_index]])
                        .collect()
                })
                .collect(),
        };
        adapted.validate()?;
        Ok(adapted)
    }
}

impl PersistentNoiseProblemV1 {
    pub fn validate(&self) -> Result<(), NoiseErrorV1> {
        validate_identifiers(&self.problem_id, &self.hypothesis_ids, &self.probes)?;
        if self.error_state_ids.is_empty() {
            return Err(NoiseErrorV1::EmptyErrorStateSet);
        }
        if self.error_state_ids.len() > 64 {
            return Err(NoiseErrorV1::TooManyErrorStates);
        }
        let mut error_states = BTreeSet::new();
        for error_state_id in &self.error_state_ids {
            if error_state_id.trim().is_empty() {
                return Err(NoiseErrorV1::EmptyErrorStateId);
            }
            if !error_states.insert(error_state_id) {
                return Err(NoiseErrorV1::DuplicateErrorStateId);
            }
        }
        if self.predicted_outcomes_by_hypothesis.len() != self.hypothesis_ids.len() {
            return Err(NoiseErrorV1::PersistentHypothesisCountMismatch);
        }
        for (hypothesis_index, by_probe) in self.predicted_outcomes_by_hypothesis.iter().enumerate()
        {
            if by_probe.len() != self.probes.len() {
                return Err(NoiseErrorV1::PersistentProbeCountMismatch {
                    hypothesis_id: self.hypothesis_ids[hypothesis_index].clone(),
                });
            }
            for (probe_index, by_error_state) in by_probe.iter().enumerate() {
                if by_error_state.len() != self.error_state_ids.len() {
                    return Err(NoiseErrorV1::PersistentErrorStateCountMismatch {
                        hypothesis_id: self.hypothesis_ids[hypothesis_index].clone(),
                        probe_id: self.probes[probe_index].probe_id.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Canonical versioned bytes; the shared error-state axis is explicit.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symm_active_noise.persistent.v1\0".to_vec();
        append_string(&mut bytes, &self.problem_id);
        bytes.extend_from_slice(&(self.hypothesis_ids.len() as u64).to_le_bytes());
        for hypothesis_id in &self.hypothesis_ids {
            append_string(&mut bytes, hypothesis_id);
        }
        bytes.extend_from_slice(&(self.probes.len() as u64).to_le_bytes());
        for probe in &self.probes {
            append_string(&mut bytes, &probe.probe_id);
            bytes.extend_from_slice(&probe.cost.to_le_bytes());
            bytes.push(u8::from(probe.admissible));
        }
        bytes.extend_from_slice(&(self.error_state_ids.len() as u64).to_le_bytes());
        for error_state_id in &self.error_state_ids {
            append_string(&mut bytes, error_state_id);
        }
        for by_probe in &self.predicted_outcomes_by_hypothesis {
            for by_error_state in by_probe {
                bytes.extend_from_slice(&(by_error_state.len() as u64).to_le_bytes());
                for outcome in by_error_state {
                    bytes.extend_from_slice(&outcome.to_le_bytes());
                }
            }
        }
        bytes
    }
}

fn resolve_probe(probes: &[NoiseProbeV1], probe_id: &str) -> Result<usize, NoiseErrorV1> {
    let index = probes
        .iter()
        .position(|probe| probe.probe_id == probe_id)
        .ok_or_else(|| NoiseErrorV1::UnknownProbe {
            probe_id: probe_id.to_owned(),
        })?;
    if !probes[index].admissible {
        return Err(NoiseErrorV1::InadmissibleProbe {
            probe_id: probe_id.to_owned(),
        });
    }
    Ok(index)
}

fn observation_bytes(observations: &[NoiseObservationV1]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(observations.len() as u64).to_le_bytes());
    for observation in observations {
        append_string(&mut bytes, &observation.probe_id);
        append_outcomes(&mut bytes, &observation.outcomes);
    }
    bytes
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedSurvivorSnapshotV1 {
    pub after_observation_count: usize,
    pub active_hypothesis_ids: Vec<String>,
    /// False means the observed record is outside every declared model.
    pub model_consistent: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedNoiseReceiptV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub observation_contract_bytes: Vec<u8>,
    pub snapshots: Vec<BoundedSurvivorSnapshotV1>,
    pub final_active_hypothesis_ids: Vec<String>,
}

fn bounded_snapshot(
    problem: &BoundedNoiseProblemV1,
    observed_by_probe: &BTreeMap<usize, BTreeSet<u32>>,
    after_observation_count: usize,
) -> BoundedSurvivorSnapshotV1 {
    let mut active_hypothesis_ids = Vec::new();
    for (hypothesis_index, hypothesis_id) in problem.hypothesis_ids.iter().enumerate() {
        let compatible = observed_by_probe.iter().all(|(probe_index, observed)| {
            problem.allowed_outcomes_by_hypothesis[hypothesis_index][*probe_index]
                .iter()
                .any(|allowed| observed.contains(allowed))
        });
        if compatible {
            active_hypothesis_ids.push(hypothesis_id.clone());
        }
    }
    let model_consistent = !active_hypothesis_ids.is_empty();
    BoundedSurvivorSnapshotV1 {
        after_observation_count,
        active_hypothesis_ids,
        model_consistent,
    }
}

/// Replay bounded adversarial observations. Repeated readings of one probe
/// constrain the same per-probe outcome; they are not treated as independent
/// resamples. A hypothesis is retained while every observed probe has at
/// least one allowed outcome in the intersection of its observed sets.
pub fn replay_bounded_observations(
    problem: &BoundedNoiseProblemV1,
    observations: &[NoiseObservationV1],
) -> Result<BoundedNoiseReceiptV1, NoiseErrorV1> {
    problem.validate()?;
    let mut observed_by_probe: BTreeMap<usize, BTreeSet<u32>> = BTreeMap::new();
    let mut snapshots = Vec::with_capacity(observations.len() + 1);
    snapshots.push(bounded_snapshot(problem, &observed_by_probe, 0));

    for (observation_index, observation) in observations.iter().enumerate() {
        validate_observation_set(observation)?;
        let probe_index = resolve_probe(&problem.probes, &observation.probe_id)?;
        let incoming: BTreeSet<u32> = observation.outcomes.iter().copied().collect();
        if let Some(existing) = observed_by_probe.get_mut(&probe_index) {
            existing.retain(|value| incoming.contains(value));
        } else {
            observed_by_probe.insert(probe_index, incoming);
        }
        snapshots.push(bounded_snapshot(
            problem,
            &observed_by_probe,
            observation_index + 1,
        ));
    }

    let final_active_hypothesis_ids = snapshots
        .last()
        .map(|snapshot| snapshot.active_hypothesis_ids.clone())
        .unwrap_or_default();
    Ok(BoundedNoiseReceiptV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        observation_contract_bytes: observation_bytes(observations),
        snapshots,
        final_active_hypothesis_ids,
    })
}

pub fn verify_bounded_noise_receipt(
    problem: &BoundedNoiseProblemV1,
    observations: &[NoiseObservationV1],
    receipt: &BoundedNoiseReceiptV1,
) -> Result<(), NoiseErrorV1> {
    if receipt.schema_version != 1
        || receipt.problem_id != problem.problem_id
        || receipt.problem_contract_bytes != problem.canonical_bytes()
        || receipt.observation_contract_bytes != observation_bytes(observations)
    {
        return Err(NoiseErrorV1::ReceiptMismatch);
    }
    let replayed = replay_bounded_observations(problem, observations)?;
    if *receipt != replayed {
        return Err(NoiseErrorV1::ReceiptMismatch);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentHypothesisStateV1 {
    pub hypothesis_id: String,
    pub compatible_error_state_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentSurvivorSnapshotV1 {
    pub after_observation_count: usize,
    pub hypotheses: Vec<PersistentHypothesisStateV1>,
    pub active_hypothesis_ids: Vec<String>,
    pub model_consistent: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentNoiseReceiptV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub observation_contract_bytes: Vec<u8>,
    pub snapshots: Vec<PersistentSurvivorSnapshotV1>,
    pub final_active_hypothesis_ids: Vec<String>,
}

fn persistent_snapshot(
    problem: &PersistentNoiseProblemV1,
    compatible_states: &[Vec<bool>],
    after_observation_count: usize,
) -> PersistentSurvivorSnapshotV1 {
    let mut hypotheses = Vec::with_capacity(problem.hypothesis_ids.len());
    let mut active_hypothesis_ids = Vec::new();
    for (hypothesis_index, hypothesis_id) in problem.hypothesis_ids.iter().enumerate() {
        let compatible_error_state_ids: Vec<String> = compatible_states[hypothesis_index]
            .iter()
            .enumerate()
            .filter(|(_, compatible)| **compatible)
            .map(|(state_index, _)| problem.error_state_ids[state_index].clone())
            .collect();
        if !compatible_error_state_ids.is_empty() {
            active_hypothesis_ids.push(hypothesis_id.clone());
        }
        hypotheses.push(PersistentHypothesisStateV1 {
            hypothesis_id: hypothesis_id.clone(),
            compatible_error_state_ids,
        });
    }
    let model_consistent = !active_hypothesis_ids.is_empty();
    PersistentSurvivorSnapshotV1 {
        after_observation_count,
        hypotheses,
        active_hypothesis_ids,
        model_consistent,
    }
}

/// Replay observations under a globally shared, campaign-persistent error
/// state. Each hypothesis must explain the entire sequence using one and the
/// same error-state index. Repeating a probe therefore cannot create
/// independent evidence or silently switch the nuisance state.
pub fn replay_persistent_observations(
    problem: &PersistentNoiseProblemV1,
    observations: &[NoiseObservationV1],
) -> Result<PersistentNoiseReceiptV1, NoiseErrorV1> {
    problem.validate()?;
    let mut compatible_states =
        vec![vec![true; problem.error_state_ids.len()]; problem.hypothesis_ids.len()];
    let mut snapshots = Vec::with_capacity(observations.len() + 1);
    snapshots.push(persistent_snapshot(problem, &compatible_states, 0));

    for (observation_index, observation) in observations.iter().enumerate() {
        validate_observation_set(observation)?;
        let probe_index = resolve_probe(&problem.probes, &observation.probe_id)?;
        let observed: BTreeSet<u32> = observation.outcomes.iter().copied().collect();
        for (hypothesis_index, by_probe) in
            problem.predicted_outcomes_by_hypothesis.iter().enumerate()
        {
            for (state_index, compatible) in
                compatible_states[hypothesis_index].iter_mut().enumerate()
            {
                if *compatible && !observed.contains(&by_probe[probe_index][state_index]) {
                    *compatible = false;
                }
            }
        }
        snapshots.push(persistent_snapshot(
            problem,
            &compatible_states,
            observation_index + 1,
        ));
    }

    let final_active_hypothesis_ids = snapshots
        .last()
        .map(|snapshot| snapshot.active_hypothesis_ids.clone())
        .unwrap_or_default();
    Ok(PersistentNoiseReceiptV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        observation_contract_bytes: observation_bytes(observations),
        snapshots,
        final_active_hypothesis_ids,
    })
}

pub fn verify_persistent_noise_receipt(
    problem: &PersistentNoiseProblemV1,
    observations: &[NoiseObservationV1],
    receipt: &PersistentNoiseReceiptV1,
) -> Result<(), NoiseErrorV1> {
    if receipt.schema_version != 1
        || receipt.problem_id != problem.problem_id
        || receipt.problem_contract_bytes != problem.canonical_bytes()
        || receipt.observation_contract_bytes != observation_bytes(observations)
    {
        return Err(NoiseErrorV1::ReceiptMismatch);
    }
    let replayed = replay_persistent_observations(problem, observations)?;
    if *receipt != replayed {
        return Err(NoiseErrorV1::ReceiptMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symmetry_active_experiment_reference::{
        ActiveExperimentProblemV1, ExperimentProbeV1, replay_observations,
    };

    fn bounded_fixture() -> BoundedNoiseProblemV1 {
        BoundedNoiseProblemV1 {
            problem_id: "bounded-noise-four-hypothesis-v1".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned(), "H-C".to_owned()],
            probes: vec![
                NoiseProbeV1 {
                    probe_id: "p0".to_owned(),
                    cost: 1,
                    admissible: true,
                },
                NoiseProbeV1 {
                    probe_id: "p1".to_owned(),
                    cost: 2,
                    admissible: true,
                },
            ],
            allowed_outcomes_by_hypothesis: vec![
                vec![vec![0, 1], vec![0]],
                vec![vec![1, 2], vec![1]],
                vec![vec![2], vec![0, 1]],
            ],
        }
    }

    fn persistent_fixture() -> PersistentNoiseProblemV1 {
        PersistentNoiseProblemV1 {
            problem_id: "persistent-global-bias-two-hypothesis-v1".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned()],
            probes: vec![
                NoiseProbeV1 {
                    probe_id: "p-left".to_owned(),
                    cost: 1,
                    admissible: true,
                },
                NoiseProbeV1 {
                    probe_id: "p-right".to_owned(),
                    cost: 1,
                    admissible: true,
                },
            ],
            error_state_ids: vec!["bias-low".to_owned(), "bias-high".to_owned()],
            predicted_outcomes_by_hypothesis: vec![
                vec![vec![0, 1], vec![0, 1]],
                vec![vec![0, 1], vec![1, 0]],
            ],
        }
    }

    fn deterministic_fixture() -> ActiveExperimentProblemV1 {
        ActiveExperimentProblemV1 {
            problem_id: "zero-noise-bridge-v1".to_owned(),
            hypotheses: vec!["H-A".to_owned(), "H-B".to_owned(), "H-C".to_owned()],
            probes: vec![
                ExperimentProbeV1 {
                    probe_id: "p0".to_owned(),
                    cost: 1,
                    admissible: true,
                    outcomes: vec![0, 0, 1],
                },
                ExperimentProbeV1 {
                    probe_id: "p1".to_owned(),
                    cost: 1,
                    admissible: true,
                    outcomes: vec![0, 1, 1],
                },
            ],
        }
    }

    #[test]
    fn bounded_uncertainty_retains_every_hypothesis_whose_allowed_set_overlaps() {
        let problem = bounded_fixture();
        let observations = vec![NoiseObservationV1 {
            probe_id: "p0".to_owned(),
            outcomes: vec![1],
        }];
        let receipt = replay_bounded_observations(&problem, &observations).unwrap();
        assert_eq!(
            receipt.final_active_hypothesis_ids,
            vec!["H-A".to_owned(), "H-B".to_owned()]
        );
        assert!(receipt.snapshots[1].model_consistent);
    }

    #[test]
    fn repeated_bounded_probe_observations_intersect_one_per_probe_outcome() {
        let problem = bounded_fixture();
        let observations = vec![
            NoiseObservationV1 {
                probe_id: "p0".to_owned(),
                outcomes: vec![1],
            },
            NoiseObservationV1 {
                probe_id: "p0".to_owned(),
                outcomes: vec![0],
            },
        ];
        let receipt = replay_bounded_observations(&problem, &observations).unwrap();
        assert_eq!(
            receipt.snapshots[1].active_hypothesis_ids,
            vec!["H-A", "H-B"]
        );
        assert!(receipt.snapshots[2].active_hypothesis_ids.is_empty());
        assert!(!receipt.snapshots[2].model_consistent);
    }

    #[test]
    fn persistent_bias_requires_one_shared_error_state_across_probes() {
        let problem = persistent_fixture();
        let observations = vec![
            NoiseObservationV1 {
                probe_id: "p-left".to_owned(),
                outcomes: vec![0],
            },
            NoiseObservationV1 {
                probe_id: "p-right".to_owned(),
                outcomes: vec![1],
            },
        ];
        let receipt = replay_persistent_observations(&problem, &observations).unwrap();
        assert_eq!(
            receipt.snapshots[1].active_hypothesis_ids,
            vec!["H-A", "H-B"]
        );
        assert_eq!(receipt.final_active_hypothesis_ids, vec!["H-B"]);
        assert_eq!(
            receipt.snapshots[2].hypotheses[0].compatible_error_state_ids,
            Vec::<String>::new()
        );
        assert_eq!(
            receipt.snapshots[2].hypotheses[1].compatible_error_state_ids,
            vec!["bias-low".to_owned()]
        );
    }

    #[test]
    fn repeating_a_persistent_probe_does_not_create_independent_evidence() {
        let problem = persistent_fixture();
        let observations = vec![
            NoiseObservationV1 {
                probe_id: "p-left".to_owned(),
                outcomes: vec![0],
            },
            NoiseObservationV1 {
                probe_id: "p-left".to_owned(),
                outcomes: vec![0],
            },
        ];
        let receipt = replay_persistent_observations(&problem, &observations).unwrap();
        assert_eq!(
            receipt.snapshots[1].active_hypothesis_ids,
            receipt.snapshots[2].active_hypothesis_ids
        );
        assert_eq!(
            receipt.snapshots[1].hypotheses,
            receipt.snapshots[2].hypotheses
        );
    }

    #[test]
    fn zero_noise_adapter_matches_deterministic_active_planner() {
        let deterministic = deterministic_fixture();
        let bounded = BoundedNoiseProblemV1::from_deterministic(&deterministic).unwrap();
        let observation = NoiseObservationV1 {
            probe_id: "p0".to_owned(),
            outcomes: vec![1],
        };
        let noise_receipt =
            replay_bounded_observations(&bounded, std::slice::from_ref(&observation)).unwrap();
        let exact_survivors = replay_observations(
            &deterministic,
            &symmetry_active_experiment_reference::compile_active_experiment(
                &deterministic,
                Default::default(),
            )
            .unwrap(),
            &[1],
            Default::default(),
        )
        .unwrap();
        assert_eq!(noise_receipt.final_active_hypothesis_ids, exact_survivors);
    }

    #[test]
    fn malformed_deterministic_table_is_rejected_before_singleton_adaptation() {
        let mut deterministic = deterministic_fixture();
        deterministic.probes[0].outcomes.pop();
        assert_eq!(
            BoundedNoiseProblemV1::from_deterministic(&deterministic),
            Err(NoiseErrorV1::InvalidDeterministicInput)
        );
    }

    #[test]
    fn no_compatible_hypothesis_is_reported_as_model_mismatch_not_a_symmetry_proof() {
        let problem = bounded_fixture();
        let observations = vec![NoiseObservationV1 {
            probe_id: "p0".to_owned(),
            outcomes: vec![99],
        }];
        let receipt = replay_bounded_observations(&problem, &observations).unwrap();
        assert!(receipt.final_active_hypothesis_ids.is_empty());
        assert!(!receipt.snapshots[1].model_consistent);
    }

    #[test]
    fn unknown_and_inadmissible_probes_fail_closed() {
        let mut problem = bounded_fixture();
        problem.probes[1].admissible = false;
        assert_eq!(
            replay_bounded_observations(
                &problem,
                &[NoiseObservationV1 {
                    probe_id: "p1".to_owned(),
                    outcomes: vec![0]
                }]
            ),
            Err(NoiseErrorV1::InadmissibleProbe {
                probe_id: "p1".to_owned()
            })
        );
        assert_eq!(
            replay_bounded_observations(
                &problem,
                &[NoiseObservationV1 {
                    probe_id: "missing".to_owned(),
                    outcomes: vec![0]
                }]
            ),
            Err(NoiseErrorV1::UnknownProbe {
                probe_id: "missing".to_owned()
            })
        );
    }

    #[test]
    fn malformed_contracts_and_duplicate_outcomes_are_rejected() {
        let mut bounded = bounded_fixture();
        bounded.allowed_outcomes_by_hypothesis[0][0] = vec![0, 0];
        assert_eq!(
            bounded.validate(),
            Err(NoiseErrorV1::DuplicateAllowedOutcome {
                hypothesis_id: "H-A".to_owned(),
                probe_id: "p0".to_owned()
            })
        );
        let persistent = persistent_fixture();
        assert!(persistent.validate().is_ok());
        assert_eq!(
            replay_persistent_observations(
                &persistent,
                &[NoiseObservationV1 {
                    probe_id: "p-left".to_owned(),
                    outcomes: vec![]
                }]
            ),
            Err(NoiseErrorV1::EmptyObservedOutcomeSet {
                probe_id: "p-left".to_owned()
            })
        );
    }

    #[test]
    fn bounded_and_persistent_receipts_reject_forgery_and_mutated_models() {
        let bounded = bounded_fixture();
        let observations = vec![NoiseObservationV1 {
            probe_id: "p0".to_owned(),
            outcomes: vec![1],
        }];
        let receipt = replay_bounded_observations(&bounded, &observations).unwrap();
        verify_bounded_noise_receipt(&bounded, &observations, &receipt).unwrap();

        let mut forged = receipt.clone();
        forged.final_active_hypothesis_ids.clear();
        assert_eq!(
            verify_bounded_noise_receipt(&bounded, &observations, &forged),
            Err(NoiseErrorV1::ReceiptMismatch)
        );

        let mut changed = bounded.clone();
        changed.allowed_outcomes_by_hypothesis[0][0].push(7);
        assert_eq!(
            verify_bounded_noise_receipt(&changed, &observations, &receipt),
            Err(NoiseErrorV1::ReceiptMismatch)
        );

        let persistent = persistent_fixture();
        let persistent_observations = vec![NoiseObservationV1 {
            probe_id: "p-left".to_owned(),
            outcomes: vec![0],
        }];
        let persistent_receipt =
            replay_persistent_observations(&persistent, &persistent_observations).unwrap();
        verify_persistent_noise_receipt(&persistent, &persistent_observations, &persistent_receipt)
            .unwrap();
    }
}
