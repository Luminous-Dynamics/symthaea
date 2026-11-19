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
    InvalidHypothesisPair,
    EmptyQuantityUnitId,
    InvalidInterval,
    IntervalArithmeticOverflow,
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

/// Closed interval with exact signed fixed-point endpoints.
///
/// The scale is supplied by the problem contract (for example, micrometres,
/// microseconds, or millionths of a normalized quantity). No floating-point
/// tolerance is hidden inside this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedIntervalV1 {
    pub lower: i128,
    pub upper: i128,
}

impl FixedIntervalV1 {
    pub fn new(lower: i128, upper: i128) -> Result<Self, NoiseErrorV1> {
        if lower > upper {
            return Err(NoiseErrorV1::InvalidInterval);
        }
        Ok(Self { lower, upper })
    }

    pub fn point(value: i128) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    pub fn intersection(self, other: Self) -> Option<Self> {
        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);
        (lower <= upper).then_some(Self { lower, upper })
    }

    pub fn checked_subtract(self, value: i128) -> Result<Self, NoiseErrorV1> {
        let lower = self
            .lower
            .checked_sub(value)
            .ok_or(NoiseErrorV1::IntervalArithmeticOverflow)?;
        let upper = self
            .upper
            .checked_sub(value)
            .ok_or(NoiseErrorV1::IntervalArithmeticOverflow)?;
        Self::new(lower, upper)
    }
}

fn append_interval(target: &mut Vec<u8>, interval: FixedIntervalV1) {
    target.extend_from_slice(&interval.lower.to_le_bytes());
    target.extend_from_slice(&interval.upper.to_le_bytes());
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntervalObservationV1 {
    pub probe_id: String,
    pub observed: FixedIntervalV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedIntervalProblemV1 {
    pub problem_id: String,
    /// Stable unit-and-scale identifier (for example, "micrometre-integer").
    pub quantity_unit_id: String,
    pub hypothesis_ids: Vec<String>,
    pub probes: Vec<NoiseProbeV1>,
    /// [hypothesis][probe] -> closed interval of allowed exact outcomes.
    pub predicted_intervals_by_hypothesis: Vec<Vec<FixedIntervalV1>>,
}

impl BoundedIntervalProblemV1 {
    pub fn validate(&self) -> Result<(), NoiseErrorV1> {
        validate_identifiers(&self.problem_id, &self.hypothesis_ids, &self.probes)?;
        if self.quantity_unit_id.trim().is_empty() {
            return Err(NoiseErrorV1::EmptyQuantityUnitId);
        }
        if self.predicted_intervals_by_hypothesis.len() != self.hypothesis_ids.len() {
            return Err(NoiseErrorV1::BoundedHypothesisCountMismatch);
        }
        for (hypothesis_index, by_probe) in
            self.predicted_intervals_by_hypothesis.iter().enumerate()
        {
            if by_probe.len() != self.probes.len() {
                return Err(NoiseErrorV1::BoundedProbeCountMismatch {
                    hypothesis_id: self.hypothesis_ids[hypothesis_index].clone(),
                });
            }
            if by_probe
                .iter()
                .any(|interval| interval.lower > interval.upper)
            {
                return Err(NoiseErrorV1::InvalidInterval);
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symm_active_noise.bounded_intervals.v1\\0".to_vec();
        append_string(&mut bytes, &self.problem_id);
        append_string(&mut bytes, &self.quantity_unit_id);
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
        for by_probe in &self.predicted_intervals_by_hypothesis {
            for interval in by_probe {
                append_interval(&mut bytes, *interval);
            }
        }
        bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntervalSurvivorSnapshotV1 {
    pub after_observation_count: usize,
    pub active_hypothesis_ids: Vec<String>,
    pub model_consistent: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedIntervalReceiptV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub observation_contract_bytes: Vec<u8>,
    pub snapshots: Vec<IntervalSurvivorSnapshotV1>,
    pub final_active_hypothesis_ids: Vec<String>,
}

fn interval_observation_bytes(observations: &[IntervalObservationV1]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(observations.len() as u64).to_le_bytes());
    for observation in observations {
        append_string(&mut bytes, &observation.probe_id);
        append_interval(&mut bytes, observation.observed);
    }
    bytes
}

fn bounded_interval_snapshot(
    problem: &BoundedIntervalProblemV1,
    observed_by_probe: &BTreeMap<usize, FixedIntervalV1>,
    after_observation_count: usize,
) -> IntervalSurvivorSnapshotV1 {
    let mut active_hypothesis_ids = Vec::new();
    for (hypothesis_index, hypothesis_id) in problem.hypothesis_ids.iter().enumerate() {
        let compatible = observed_by_probe.iter().all(|(probe_index, observed)| {
            problem.predicted_intervals_by_hypothesis[hypothesis_index][*probe_index]
                .intersection(*observed)
                .is_some()
        });
        if compatible {
            active_hypothesis_ids.push(hypothesis_id.clone());
        }
    }
    let model_consistent = !active_hypothesis_ids.is_empty();
    IntervalSurvivorSnapshotV1 {
        after_observation_count,
        active_hypothesis_ids,
        model_consistent,
    }
}

/// Conservative bounded-interval replay. Every repeated observation for a
/// probe intersects the same observed interval; it is not treated as a new
/// independent draw. A hypothesis survives only when its predicted interval
/// overlaps the accumulated observation interval for every measured probe.
pub fn replay_bounded_intervals(
    problem: &BoundedIntervalProblemV1,
    observations: &[IntervalObservationV1],
) -> Result<BoundedIntervalReceiptV1, NoiseErrorV1> {
    problem.validate()?;
    let mut observed_by_probe: BTreeMap<usize, FixedIntervalV1> = BTreeMap::new();
    let mut model_consistent = true;
    let mut snapshots = Vec::with_capacity(observations.len() + 1);
    snapshots.push(bounded_interval_snapshot(problem, &observed_by_probe, 0));

    for (observation_index, observation) in observations.iter().enumerate() {
        if observation.observed.lower > observation.observed.upper {
            return Err(NoiseErrorV1::InvalidInterval);
        }
        let probe_index = resolve_probe(&problem.probes, &observation.probe_id)?;
        if model_consistent {
            if let Some(existing) = observed_by_probe.get(&probe_index).copied() {
                if let Some(intersection) = existing.intersection(observation.observed) {
                    observed_by_probe.insert(probe_index, intersection);
                } else {
                    // Once repeated readings have an empty intersection, a
                    // later reading cannot repair the campaign's evidence.
                    model_consistent = false;
                    observed_by_probe.remove(&probe_index);
                }
            } else {
                observed_by_probe.insert(probe_index, observation.observed);
            }
        }
        if model_consistent {
            snapshots.push(bounded_interval_snapshot(
                problem,
                &observed_by_probe,
                observation_index + 1,
            ));
        } else {
            snapshots.push(IntervalSurvivorSnapshotV1 {
                after_observation_count: observation_index + 1,
                active_hypothesis_ids: Vec::new(),
                model_consistent: false,
            });
        }
    }

    let final_active_hypothesis_ids = snapshots
        .last()
        .map(|snapshot| snapshot.active_hypothesis_ids.clone())
        .unwrap_or_default();
    Ok(BoundedIntervalReceiptV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        observation_contract_bytes: interval_observation_bytes(observations),
        snapshots,
        final_active_hypothesis_ids,
    })
}

pub fn verify_bounded_interval_receipt(
    problem: &BoundedIntervalProblemV1,
    observations: &[IntervalObservationV1],
    receipt: &BoundedIntervalReceiptV1,
) -> Result<(), NoiseErrorV1> {
    if receipt.schema_version != 1
        || receipt.problem_id != problem.problem_id
        || receipt.problem_contract_bytes != problem.canonical_bytes()
        || receipt.observation_contract_bytes != interval_observation_bytes(observations)
    {
        return Err(NoiseErrorV1::ReceiptMismatch);
    }
    let replayed = replay_bounded_intervals(problem, observations)?;
    if *receipt != replayed {
        return Err(NoiseErrorV1::ReceiptMismatch);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedAdditiveBiasProblemV1 {
    pub problem_id: String,
    /// Stable unit-and-scale identifier for both predictions and observations.
    pub quantity_unit_id: String,
    pub hypothesis_ids: Vec<String>,
    pub probes: Vec<NoiseProbeV1>,
    /// Nominal exact prediction [hypothesis][probe].
    pub nominal_predictions_by_hypothesis: Vec<Vec<i128>>,
    /// One bounded additive bias, shared across every probe in the campaign.
    pub global_bias_interval: FixedIntervalV1,
}

impl SharedAdditiveBiasProblemV1 {
    pub fn validate(&self) -> Result<(), NoiseErrorV1> {
        validate_identifiers(&self.problem_id, &self.hypothesis_ids, &self.probes)?;
        if self.quantity_unit_id.trim().is_empty() {
            return Err(NoiseErrorV1::EmptyQuantityUnitId);
        }
        if self.global_bias_interval.lower > self.global_bias_interval.upper {
            return Err(NoiseErrorV1::InvalidInterval);
        }
        if self.nominal_predictions_by_hypothesis.len() != self.hypothesis_ids.len() {
            return Err(NoiseErrorV1::PersistentHypothesisCountMismatch);
        }
        for (hypothesis_index, by_probe) in
            self.nominal_predictions_by_hypothesis.iter().enumerate()
        {
            if by_probe.len() != self.probes.len() {
                return Err(NoiseErrorV1::PersistentProbeCountMismatch {
                    hypothesis_id: self.hypothesis_ids[hypothesis_index].clone(),
                });
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symm_active_noise.shared_additive_bias.v1\\0".to_vec();
        append_string(&mut bytes, &self.problem_id);
        append_string(&mut bytes, &self.quantity_unit_id);
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
        append_interval(&mut bytes, self.global_bias_interval);
        for by_probe in &self.nominal_predictions_by_hypothesis {
            for prediction in by_probe {
                bytes.extend_from_slice(&prediction.to_le_bytes());
            }
        }
        bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedBiasHypothesisStateV1 {
    pub hypothesis_id: String,
    /// Set of global additive biases that explain all observations so far.
    pub feasible_global_bias_interval: Option<FixedIntervalV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedBiasSnapshotV1 {
    pub after_observation_count: usize,
    pub hypotheses: Vec<SharedBiasHypothesisStateV1>,
    pub active_hypothesis_ids: Vec<String>,
    pub model_consistent: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedAdditiveBiasReceiptV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub observation_contract_bytes: Vec<u8>,
    pub snapshots: Vec<SharedBiasSnapshotV1>,
    pub final_active_hypothesis_ids: Vec<String>,
}

fn shared_bias_snapshot(
    problem: &SharedAdditiveBiasProblemV1,
    feasible_bias_by_hypothesis: &[Option<FixedIntervalV1>],
    after_observation_count: usize,
) -> SharedBiasSnapshotV1 {
    let mut hypotheses = Vec::with_capacity(problem.hypothesis_ids.len());
    let mut active_hypothesis_ids = Vec::new();
    for (hypothesis_index, hypothesis_id) in problem.hypothesis_ids.iter().enumerate() {
        let feasible_global_bias_interval = feasible_bias_by_hypothesis[hypothesis_index];
        if feasible_global_bias_interval.is_some() {
            active_hypothesis_ids.push(hypothesis_id.clone());
        }
        hypotheses.push(SharedBiasHypothesisStateV1 {
            hypothesis_id: hypothesis_id.clone(),
            feasible_global_bias_interval,
        });
    }
    let model_consistent = !active_hypothesis_ids.is_empty();
    SharedBiasSnapshotV1 {
        after_observation_count,
        hypotheses,
        active_hypothesis_ids,
        model_consistent,
    }
}

/// Replay interval observations under one globally shared additive bias.
/// For nominal prediction μ and observed interval [l,u], the compatible bias
/// interval is [l-μ,u-μ]. Intersecting all probes forces one single bias value
/// to explain the whole campaign. Arithmetic is checked and exact.
pub fn replay_shared_additive_bias(
    problem: &SharedAdditiveBiasProblemV1,
    observations: &[IntervalObservationV1],
) -> Result<SharedAdditiveBiasReceiptV1, NoiseErrorV1> {
    problem.validate()?;
    let mut feasible_bias_by_hypothesis =
        vec![Some(problem.global_bias_interval); problem.hypothesis_ids.len()];
    let mut snapshots = Vec::with_capacity(observations.len() + 1);
    snapshots.push(shared_bias_snapshot(
        problem,
        &feasible_bias_by_hypothesis,
        0,
    ));

    for (observation_index, observation) in observations.iter().enumerate() {
        if observation.observed.lower > observation.observed.upper {
            return Err(NoiseErrorV1::InvalidInterval);
        }
        let probe_index = resolve_probe(&problem.probes, &observation.probe_id)?;
        for (hypothesis_index, feasible_bias) in feasible_bias_by_hypothesis.iter_mut().enumerate()
        {
            let Some(current) = *feasible_bias else {
                continue;
            };
            let nominal = problem.nominal_predictions_by_hypothesis[hypothesis_index][probe_index];
            let allowed_bias = observation.observed.checked_subtract(nominal)?;
            *feasible_bias = current.intersection(allowed_bias);
        }
        snapshots.push(shared_bias_snapshot(
            problem,
            &feasible_bias_by_hypothesis,
            observation_index + 1,
        ));
    }

    let final_active_hypothesis_ids = snapshots
        .last()
        .map(|snapshot| snapshot.active_hypothesis_ids.clone())
        .unwrap_or_default();
    Ok(SharedAdditiveBiasReceiptV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        observation_contract_bytes: interval_observation_bytes(observations),
        snapshots,
        final_active_hypothesis_ids,
    })
}

pub fn verify_shared_additive_bias_receipt(
    problem: &SharedAdditiveBiasProblemV1,
    observations: &[IntervalObservationV1],
    receipt: &SharedAdditiveBiasReceiptV1,
) -> Result<(), NoiseErrorV1> {
    if receipt.schema_version != 1
        || receipt.problem_id != problem.problem_id
        || receipt.problem_contract_bytes != problem.canonical_bytes()
        || receipt.observation_contract_bytes != interval_observation_bytes(observations)
    {
        return Err(NoiseErrorV1::ReceiptMismatch);
    }
    let replayed = replay_shared_additive_bias(problem, observations)?;
    if *receipt != replayed {
        return Err(NoiseErrorV1::ReceiptMismatch);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedPairIdentifiabilityV1 {
    pub hypothesis_a: String,
    pub hypothesis_b: String,
    /// A single disjoint probe is a witness that the pair can be separated.
    pub separating_probe_ids: Vec<String>,
    /// These probes have overlapping allowed outcomes for the pair.
    pub overlapping_probe_ids: Vec<String>,
    /// True only when every admissible probe has overlapping outcome sets.
    pub indistinguishable_under_available_probes: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentStateWitnessV1 {
    pub error_state_id: String,
    pub probe_id: String,
    pub outcome_a: u32,
    pub outcome_b: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentUnknownStateWitnessV1 {
    pub error_state_a_id: String,
    pub error_state_b_id: String,
    pub probe_id: String,
    pub outcome_a: u32,
    pub outcome_b: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentUnknownStateCollisionV1 {
    pub error_state_a_id: String,
    pub error_state_b_id: String,
    /// Identical complete signature over all admissible probes.
    pub common_signature: Vec<(String, u32)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentUnknownStateIdentifiabilityV1 {
    pub hypothesis_a: String,
    pub hypothesis_b: String,
    pub separating_state_pair_witnesses: Vec<PersistentUnknownStateWitnessV1>,
    pub unresolved_state_pair_collisions: Vec<PersistentUnknownStateCollisionV1>,
    pub indistinguishable_under_unknown_state: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentStateCollisionV1 {
    pub error_state_id: String,
    /// Complete identical signature over every admissible probe.
    pub common_signature: Vec<(String, u32)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentPairIdentifiabilityV1 {
    pub hypothesis_a: String,
    pub hypothesis_b: String,
    /// For each possible shared campaign error state, one cheapest probe
    /// that separates the pair, when such a probe exists.
    pub state_witnesses: Vec<PersistentStateWitnessV1>,
    /// Any one collision means the pair cannot be guaranteed distinguishable
    /// for every allowed persistent campaign state using this probe family.
    pub unresolved_error_states: Vec<PersistentStateCollisionV1>,
    pub indistinguishable_under_available_probes: bool,
}

fn validate_hypothesis_pair(
    hypothesis_count: usize,
    hypothesis_a: usize,
    hypothesis_b: usize,
) -> Result<(), NoiseErrorV1> {
    if hypothesis_a >= hypothesis_count
        || hypothesis_b >= hypothesis_count
        || hypothesis_a == hypothesis_b
    {
        return Err(NoiseErrorV1::InvalidHypothesisPair);
    }
    Ok(())
}

/// Determine whether two hypotheses can be robustly separated by the full
/// bounded-noise probe family. Because per-probe uncertainty is independent,
/// their complete outcome-signature sets overlap iff every admissible probe's
/// allowed sets overlap. A disjoint probe is a constructive witness.
pub fn bounded_pair_identifiability(
    problem: &BoundedNoiseProblemV1,
    hypothesis_a: usize,
    hypothesis_b: usize,
) -> Result<BoundedPairIdentifiabilityV1, NoiseErrorV1> {
    problem.validate()?;
    validate_hypothesis_pair(problem.hypothesis_ids.len(), hypothesis_a, hypothesis_b)?;

    let mut separating_probe_ids = Vec::new();
    let mut overlapping_probe_ids = Vec::new();
    let mut separating: Vec<&NoiseProbeV1> = Vec::new();

    for (probe_index, probe) in problem.probes.iter().enumerate() {
        if !probe.admissible {
            continue;
        }
        let allowed_a: BTreeSet<u32> = problem.allowed_outcomes_by_hypothesis[hypothesis_a]
            [probe_index]
            .iter()
            .copied()
            .collect();
        let allowed_b: BTreeSet<u32> = problem.allowed_outcomes_by_hypothesis[hypothesis_b]
            [probe_index]
            .iter()
            .copied()
            .collect();
        if allowed_a.is_disjoint(&allowed_b) {
            separating_probe_ids.push(probe.probe_id.clone());
            separating.push(probe);
        } else {
            overlapping_probe_ids.push(probe.probe_id.clone());
        }
    }
    separating.sort_by(|left, right| {
        (left.cost, left.probe_id.as_str()).cmp(&(right.cost, right.probe_id.as_str()))
    });
    overlapping_probe_ids.sort();

    Ok(BoundedPairIdentifiabilityV1 {
        hypothesis_a: problem.hypothesis_ids[hypothesis_a].clone(),
        hypothesis_b: problem.hypothesis_ids[hypothesis_b].clone(),
        separating_probe_ids,
        overlapping_probe_ids,
        indistinguishable_under_available_probes: separating.is_empty(),
    })
}

/// Determine whether two hypotheses can be separated under every possible
/// campaign-persistent shared error state. The same state index is compared
/// on both sides: the environment cannot change merely because the candidate
/// hypothesis changed. For each state, the cheapest admissible probe with
/// unequal predictions is retained as a replayable witness.
/// Robust pairwise identifiability when the persistent nuisance state is
/// unobserved. Unlike the same-state conditional check below, this compares
/// every possible nuisance state for hypothesis A with every possible state
/// for hypothesis B. If any cross-state pair has the same complete signature,
/// the two hypotheses are observationally confounded without state calibration.
pub fn persistent_pair_identifiability_under_unknown_state(
    problem: &PersistentNoiseProblemV1,
    hypothesis_a: usize,
    hypothesis_b: usize,
) -> Result<PersistentUnknownStateIdentifiabilityV1, NoiseErrorV1> {
    problem.validate()?;
    validate_hypothesis_pair(problem.hypothesis_ids.len(), hypothesis_a, hypothesis_b)?;

    let mut separating_state_pair_witnesses = Vec::new();
    let mut unresolved_state_pair_collisions = Vec::new();

    for (state_a_index, state_a_id) in problem.error_state_ids.iter().enumerate() {
        for (state_b_index, state_b_id) in problem.error_state_ids.iter().enumerate() {
            let mut separating: Vec<(&NoiseProbeV1, u32, u32)> = Vec::new();
            let mut common_signature = Vec::new();

            for (probe_index, probe) in problem.probes.iter().enumerate() {
                if !probe.admissible {
                    continue;
                }
                let outcome_a = problem.predicted_outcomes_by_hypothesis[hypothesis_a]
                    [probe_index][state_a_index];
                let outcome_b = problem.predicted_outcomes_by_hypothesis[hypothesis_b]
                    [probe_index][state_b_index];
                common_signature.push((probe.probe_id.clone(), outcome_a));
                if outcome_a != outcome_b {
                    separating.push((probe, outcome_a, outcome_b));
                }
            }

            separating.sort_by(|left, right| {
                (left.0.cost, left.0.probe_id.as_str())
                    .cmp(&(right.0.cost, right.0.probe_id.as_str()))
            });
            if let Some((probe, outcome_a, outcome_b)) = separating.first() {
                separating_state_pair_witnesses.push(PersistentUnknownStateWitnessV1 {
                    error_state_a_id: state_a_id.clone(),
                    error_state_b_id: state_b_id.clone(),
                    probe_id: probe.probe_id.clone(),
                    outcome_a: *outcome_a,
                    outcome_b: *outcome_b,
                });
            } else {
                unresolved_state_pair_collisions.push(PersistentUnknownStateCollisionV1 {
                    error_state_a_id: state_a_id.clone(),
                    error_state_b_id: state_b_id.clone(),
                    common_signature,
                });
            }
        }
    }

    Ok(PersistentUnknownStateIdentifiabilityV1 {
        hypothesis_a: problem.hypothesis_ids[hypothesis_a].clone(),
        hypothesis_b: problem.hypothesis_ids[hypothesis_b].clone(),
        separating_state_pair_witnesses,
        indistinguishable_under_unknown_state: !unresolved_state_pair_collisions.is_empty(),
        unresolved_state_pair_collisions,
    })
}

pub fn persistent_pair_identifiability(
    problem: &PersistentNoiseProblemV1,
    hypothesis_a: usize,
    hypothesis_b: usize,
) -> Result<PersistentPairIdentifiabilityV1, NoiseErrorV1> {
    problem.validate()?;
    validate_hypothesis_pair(problem.hypothesis_ids.len(), hypothesis_a, hypothesis_b)?;

    let mut state_witnesses = Vec::new();
    let mut unresolved_error_states = Vec::new();
    for (state_index, error_state_id) in problem.error_state_ids.iter().enumerate() {
        let mut distinguishing: Vec<(&NoiseProbeV1, u32, u32)> = Vec::new();
        let mut common_signature = Vec::new();
        for (probe_index, probe) in problem.probes.iter().enumerate() {
            if !probe.admissible {
                continue;
            }
            let outcome_a =
                problem.predicted_outcomes_by_hypothesis[hypothesis_a][probe_index][state_index];
            let outcome_b =
                problem.predicted_outcomes_by_hypothesis[hypothesis_b][probe_index][state_index];
            common_signature.push((probe.probe_id.clone(), outcome_a));
            if outcome_a != outcome_b {
                distinguishing.push((probe, outcome_a, outcome_b));
            }
        }
        distinguishing.sort_by(|left, right| {
            (left.0.cost, left.0.probe_id.as_str()).cmp(&(right.0.cost, right.0.probe_id.as_str()))
        });
        if let Some((probe, outcome_a, outcome_b)) = distinguishing.first() {
            state_witnesses.push(PersistentStateWitnessV1 {
                error_state_id: error_state_id.clone(),
                probe_id: probe.probe_id.clone(),
                outcome_a: *outcome_a,
                outcome_b: *outcome_b,
            });
        } else {
            unresolved_error_states.push(PersistentStateCollisionV1 {
                error_state_id: error_state_id.clone(),
                common_signature,
            });
        }
    }

    Ok(PersistentPairIdentifiabilityV1 {
        hypothesis_a: problem.hypothesis_ids[hypothesis_a].clone(),
        hypothesis_b: problem.hypothesis_ids[hypothesis_b].clone(),
        state_witnesses,
        indistinguishable_under_available_probes: !unresolved_error_states.is_empty(),
        unresolved_error_states,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedBiasCrossStateCollisionV1 {
    pub bias_a: i128,
    pub bias_b: i128,
    pub common_signature: Vec<(String, i128)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedBiasCrossStateIdentifiabilityV1 {
    pub hypothesis_a: String,
    pub hypothesis_b: String,
    /// One probe if every nominal offset is outside the bias-difference
    /// envelope; two probes if their nominal offsets differ.
    pub separating_probe_ids: Vec<String>,
    /// Present when different permissible campaign-bias values make the
    /// complete prediction signatures identical.
    pub collision: Option<SharedBiasCrossStateCollisionV1>,
    pub indistinguishable_under_unknown_bias: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericPairIdentifiabilityV1 {
    pub hypothesis_a: String,
    pub hypothesis_b: String,
    pub separating_probe_ids: Vec<String>,
    pub overlapping_probe_ids: Vec<String>,
    pub indistinguishable_under_available_probes: bool,
}

/// For bounded numeric predictions, one admissible probe with disjoint
/// predicted intervals is a robust pairwise separation witness.
pub fn bounded_interval_pair_identifiability(
    problem: &BoundedIntervalProblemV1,
    hypothesis_a: usize,
    hypothesis_b: usize,
) -> Result<NumericPairIdentifiabilityV1, NoiseErrorV1> {
    problem.validate()?;
    validate_hypothesis_pair(problem.hypothesis_ids.len(), hypothesis_a, hypothesis_b)?;

    let mut separating = Vec::new();
    let mut overlapping_probe_ids = Vec::new();
    for (probe_index, probe) in problem.probes.iter().enumerate() {
        if !probe.admissible {
            continue;
        }
        let interval_a = problem.predicted_intervals_by_hypothesis[hypothesis_a][probe_index];
        let interval_b = problem.predicted_intervals_by_hypothesis[hypothesis_b][probe_index];
        if interval_a.intersection(interval_b).is_none() {
            separating.push(probe);
        } else {
            overlapping_probe_ids.push(probe.probe_id.clone());
        }
    }
    separating.sort_by(|left, right| {
        (left.cost, left.probe_id.as_str()).cmp(&(right.cost, right.probe_id.as_str()))
    });
    overlapping_probe_ids.sort();
    Ok(NumericPairIdentifiabilityV1 {
        hypothesis_a: problem.hypothesis_ids[hypothesis_a].clone(),
        hypothesis_b: problem.hypothesis_ids[hypothesis_b].clone(),
        separating_probe_ids: separating
            .iter()
            .map(|probe| probe.probe_id.clone())
            .collect(),
        overlapping_probe_ids,
        indistinguishable_under_available_probes: separating.is_empty(),
    })
}

/// Robust pairwise identifiability when each candidate may explain the
/// campaign using any value in the same declared *unknown* bias interval.
/// The candidate-specific nuisance values may differ: the model only asserts
/// that each is campaign-persistent, not that its value has been calibrated.
pub fn shared_bias_pair_identifiability_under_unknown_bias(
    problem: &SharedAdditiveBiasProblemV1,
    hypothesis_a: usize,
    hypothesis_b: usize,
) -> Result<SharedBiasCrossStateIdentifiabilityV1, NoiseErrorV1> {
    problem.validate()?;
    validate_hypothesis_pair(problem.hypothesis_ids.len(), hypothesis_a, hypothesis_b)?;

    let mut differences: Vec<(&NoiseProbeV1, i128)> = Vec::new();
    for (probe_index, probe) in problem.probes.iter().enumerate() {
        if !probe.admissible {
            continue;
        }
        let difference = problem.nominal_predictions_by_hypothesis[hypothesis_a][probe_index]
            .checked_sub(problem.nominal_predictions_by_hypothesis[hypothesis_b][probe_index])
            .ok_or(NoiseErrorV1::IntervalArithmeticOverflow)?;
        differences.push((probe, difference));
    }
    differences.sort_by(|left, right| {
        (left.0.cost, left.0.probe_id.as_str())
            .cmp(&(right.0.cost, right.0.probe_id.as_str()))
    });

    let bias = problem.global_bias_interval;
    let minimum_bias_difference = bias
        .lower
        .checked_sub(bias.upper)
        .ok_or(NoiseErrorV1::IntervalArithmeticOverflow)?;
    let maximum_bias_difference = bias
        .upper
        .checked_sub(bias.lower)
        .ok_or(NoiseErrorV1::IntervalArithmeticOverflow)?;

    // A cross-hypothesis collision exists exactly when one constant
    // difference d = bias_b - bias_a can cancel every nominal difference.
    let constant_difference = differences
        .first()
        .map(|(_, difference)| *difference)
        .filter(|first| differences.iter().all(|(_, value)| value == first));

    let collision = if let Some(difference) = constant_difference {
        if (minimum_bias_difference..=maximum_bias_difference).contains(&difference) {
            let (bias_a, bias_b) = if difference >= 0 {
                (
                    bias.lower,
                    bias.lower
                        .checked_add(difference)
                        .ok_or(NoiseErrorV1::IntervalArithmeticOverflow)?,
                )
            } else {
                (
                    bias.lower
                        .checked_sub(difference)
                        .ok_or(NoiseErrorV1::IntervalArithmeticOverflow)?,
                    bias.lower,
                )
            };
            let mut common_signature = Vec::with_capacity(differences.len());
            for (probe_index, probe) in problem.probes.iter().enumerate() {
                if !probe.admissible {
                    continue;
                }
                let prediction_a = problem.nominal_predictions_by_hypothesis[hypothesis_a]
                    [probe_index]
                    .checked_add(bias_a)
                    .ok_or(NoiseErrorV1::IntervalArithmeticOverflow)?;
                let prediction_b = problem.nominal_predictions_by_hypothesis[hypothesis_b]
                    [probe_index]
                    .checked_add(bias_b)
                    .ok_or(NoiseErrorV1::IntervalArithmeticOverflow)?;
                if prediction_a != prediction_b {
                    return Err(NoiseErrorV1::InvalidDeterministicInput);
                }
                common_signature.push((probe.probe_id.clone(), prediction_a));
            }
            Some(SharedBiasCrossStateCollisionV1 {
                bias_a,
                bias_b,
                common_signature,
            })
        } else {
            None
        }
    } else if differences.is_empty() {
        Some(SharedBiasCrossStateCollisionV1 {
            bias_a: bias.lower,
            bias_b: bias.lower,
            common_signature: Vec::new(),
        })
    } else {
        None
    };

    let separating_probe_ids = if collision.is_some() {
        Vec::new()
    } else if let Some((probe, _)) = differences.first() {
        // If all probe offsets are the same but outside the possible
        // bias-difference envelope, one probe is sufficient.
        if differences
            .iter()
            .all(|(_, difference)| *difference == differences[0].1)
        {
            vec![probe.probe_id.clone()]
        } else {
            let mut witness_pair = None;
            for left_index in 0..differences.len() {
                for right_index in (left_index + 1)..differences.len() {
                    if differences[left_index].1 != differences[right_index].1 {
                        let left = differences[left_index].0;
                        let right = differences[right_index].0;
                        let candidate = (
                            left.cost
                                .checked_add(right.cost)
                                .ok_or(NoiseErrorV1::IntervalArithmeticOverflow)?,
                            left.probe_id.as_str(),
                            right.probe_id.as_str(),
                            left.probe_id.clone(),
                            right.probe_id.clone(),
                        );
                        if witness_pair.as_ref().is_none_or(|current| {
                            (&candidate.0, candidate.1, candidate.2)
                                < (&current.0, current.1, current.2)
                        }) {
                            witness_pair = Some(candidate);
                        }
                    }
                }
            }
            witness_pair
                .map(|candidate| vec![candidate.3, candidate.4])
                .unwrap_or_default()
        }
    } else {
        Vec::new()
    };

    Ok(SharedBiasCrossStateIdentifiabilityV1 {
        hypothesis_a: problem.hypothesis_ids[hypothesis_a].clone(),
        hypothesis_b: problem.hypothesis_ids[hypothesis_b].clone(),
        separating_probe_ids,
        indistinguishable_under_unknown_bias: collision.is_some(),
        collision,
    })
}

/// Conditional identifiability when the *same fixed bias value* is held
/// common across both competing hypotheses. For unobserved candidate-specific
/// nuisance values, use shared_bias_pair_identifiability_under_unknown_bias.
pub fn shared_bias_pair_identifiability(
    problem: &SharedAdditiveBiasProblemV1,
    hypothesis_a: usize,
    hypothesis_b: usize,
) -> Result<NumericPairIdentifiabilityV1, NoiseErrorV1> {
    problem.validate()?;
    validate_hypothesis_pair(problem.hypothesis_ids.len(), hypothesis_a, hypothesis_b)?;

    let mut separating = Vec::new();
    let mut overlapping_probe_ids = Vec::new();
    for (probe_index, probe) in problem.probes.iter().enumerate() {
        if !probe.admissible {
            continue;
        }
        let prediction_a = problem.nominal_predictions_by_hypothesis[hypothesis_a][probe_index];
        let prediction_b = problem.nominal_predictions_by_hypothesis[hypothesis_b][probe_index];
        if prediction_a != prediction_b {
            separating.push(probe);
        } else {
            overlapping_probe_ids.push(probe.probe_id.clone());
        }
    }
    separating.sort_by(|left, right| {
        (left.cost, left.probe_id.as_str()).cmp(&(right.cost, right.probe_id.as_str()))
    });
    overlapping_probe_ids.sort();
    Ok(NumericPairIdentifiabilityV1 {
        hypothesis_a: problem.hypothesis_ids[hypothesis_a].clone(),
        hypothesis_b: problem.hypothesis_ids[hypothesis_b].clone(),
        separating_probe_ids: separating
            .iter()
            .map(|probe| probe.probe_id.clone())
            .collect(),
        overlapping_probe_ids,
        indistinguishable_under_available_probes: separating.is_empty(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProbeDesignObligationV1 {
    pub hypothesis_a: String,
    pub hypothesis_b: String,
    /// Present when both hypotheses are compared under one calibrated,
    /// common campaign state.
    pub error_state_id: Option<String>,
    /// Present when nuisance states are unobserved and each candidate may
    /// use a different persistent state over the campaign.
    pub error_state_pair: Option<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeDesignReceiptV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub problem_contract_bytes: Vec<u8>,
    /// Greedy selection order, which is deterministic for a frozen contract.
    pub selected_probe_ids: Vec<String>,
    pub selected_cost: u64,
    pub required_obligations: Vec<ProbeDesignObligationV1>,
    pub unresolved_obligations: Vec<ProbeDesignObligationV1>,
    pub covered_obligations: Vec<ProbeDesignObligationV1>,
    /// Greedy weighted set-cover is a baseline, not an optimality certificate.
    pub optimality_certified: bool,
    pub complete_pairwise_separation: bool,
}

fn pair_obligation(
    hypothesis_ids: &[String],
    hypothesis_a: usize,
    hypothesis_b: usize,
    error_state_id: Option<String>,
) -> ProbeDesignObligationV1 {
    ProbeDesignObligationV1 {
        hypothesis_a: hypothesis_ids[hypothesis_a].clone(),
        hypothesis_b: hypothesis_ids[hypothesis_b].clone(),
        error_state_id,
        error_state_pair: None,
    }
}

fn greedy_probe_cover(
    problem_id: &str,
    problem_contract_bytes: Vec<u8>,
    probes: &[NoiseProbeV1],
    required_obligations: Vec<ProbeDesignObligationV1>,
    unresolved_obligations: Vec<ProbeDesignObligationV1>,
    coverage_by_probe: Vec<BTreeSet<usize>>,
) -> Result<ProbeDesignReceiptV1, NoiseErrorV1> {
    let mut remaining: BTreeSet<usize> = (0..required_obligations.len()).collect();
    let mut selected_probe_indices = Vec::new();
    let mut selected_cost = 0_u64;

    while !remaining.is_empty() {
        let mut best: Option<(usize, usize)> = None;
        for (probe_index, probe) in probes.iter().enumerate() {
            if !probe.admissible || selected_probe_indices.contains(&probe_index) {
                continue;
            }
            let newly_covered = coverage_by_probe[probe_index]
                .intersection(&remaining)
                .count();
            if newly_covered == 0 {
                continue;
            }
            let replace = match best {
                None => true,
                Some((best_index, best_count)) => {
                    let best_probe = &probes[best_index];
                    let left = (newly_covered as u128) * (best_probe.cost as u128);
                    let right = (best_count as u128) * (probe.cost as u128);
                    left > right
                        || (left == right
                            && (probe.cost, probe.probe_id.as_str())
                                < (best_probe.cost, best_probe.probe_id.as_str()))
                }
            };
            if replace {
                best = Some((probe_index, newly_covered));
            }
        }

        let Some((selected, _)) = best else {
            // Every required obligation was constructed only when at least one
            // admissible separating probe exists. Reaching this branch means
            // the design bookkeeping is internally inconsistent.
            return Err(NoiseErrorV1::InvalidDeterministicInput);
        };
        selected_cost = selected_cost
            .checked_add(probes[selected].cost)
            .ok_or(NoiseErrorV1::IntervalArithmeticOverflow)?;
        selected_probe_indices.push(selected);
        for obligation in &coverage_by_probe[selected] {
            remaining.remove(obligation);
        }
    }

    let covered_obligations = required_obligations.clone();
    let selected_probe_ids = selected_probe_indices
        .iter()
        .map(|index| probes[*index].probe_id.clone())
        .collect();
    Ok(ProbeDesignReceiptV1 {
        schema_version: 1,
        problem_id: problem_id.to_owned(),
        problem_contract_bytes,
        selected_probe_ids,
        selected_cost,
        required_obligations,
        unresolved_obligations: unresolved_obligations.clone(),
        covered_obligations,
        optimality_certified: false,
        complete_pairwise_separation: unresolved_obligations.is_empty(),
    })
}

/// Greedy minimum-cost non-adaptive diagnostic design for bounded categorical
/// outcomes. One probe covers a hypothesis pair when their allowed outcome
/// sets for that probe are disjoint.
pub fn design_bounded_noise_probe_set(
    problem: &BoundedNoiseProblemV1,
) -> Result<ProbeDesignReceiptV1, NoiseErrorV1> {
    problem.validate()?;
    let mut required = Vec::new();
    let mut unresolved = Vec::new();
    let mut coverage = vec![BTreeSet::new(); problem.probes.len()];

    for hypothesis_a in 0..problem.hypothesis_ids.len() {
        for hypothesis_b in (hypothesis_a + 1)..problem.hypothesis_ids.len() {
            let obligation =
                pair_obligation(&problem.hypothesis_ids, hypothesis_a, hypothesis_b, None);
            let mut separating_probes = Vec::new();
            for (probe_index, probe) in problem.probes.iter().enumerate() {
                if !probe.admissible {
                    continue;
                }
                let allowed_a: BTreeSet<u32> = problem.allowed_outcomes_by_hypothesis[hypothesis_a]
                    [probe_index]
                    .iter()
                    .copied()
                    .collect();
                let allowed_b: BTreeSet<u32> = problem.allowed_outcomes_by_hypothesis[hypothesis_b]
                    [probe_index]
                    .iter()
                    .copied()
                    .collect();
                if allowed_a.is_disjoint(&allowed_b) {
                    separating_probes.push(probe_index);
                }
            }
            if separating_probes.is_empty() {
                unresolved.push(obligation);
            } else {
                let obligation_index = required.len();
                required.push(obligation);
                for probe_index in separating_probes {
                    coverage[probe_index].insert(obligation_index);
                }
            }
        }
    }

    greedy_probe_cover(
        &problem.problem_id,
        problem.canonical_bytes(),
        &problem.probes,
        required,
        unresolved,
        coverage,
    )
}

/// Greedy minimum-cost non-adaptive diagnostic design under one persistent
/// categorical error state. Each (hypothesis pair, global error state) is a
/// separate obligation; a selected probe must separate the pair under that
/// same state. A nuisance state cannot switch between measurements.
pub fn design_persistent_noise_probe_set(
    problem: &PersistentNoiseProblemV1,
) -> Result<ProbeDesignReceiptV1, NoiseErrorV1> {
    problem.validate()?;
    let mut required = Vec::new();
    let mut unresolved = Vec::new();
    let mut coverage = vec![BTreeSet::new(); problem.probes.len()];

    for hypothesis_a in 0..problem.hypothesis_ids.len() {
        for hypothesis_b in (hypothesis_a + 1)..problem.hypothesis_ids.len() {
            for (state_index, state_id) in problem.error_state_ids.iter().enumerate() {
                let obligation = pair_obligation(
                    &problem.hypothesis_ids,
                    hypothesis_a,
                    hypothesis_b,
                    Some(state_id.clone()),
                );
                let separating_probes: Vec<usize> = problem
                    .probes
                    .iter()
                    .enumerate()
                    .filter(|(_, probe)| probe.admissible)
                    .filter_map(|(probe_index, _)| {
                        let outcome_a = problem.predicted_outcomes_by_hypothesis[hypothesis_a]
                            [probe_index][state_index];
                        let outcome_b = problem.predicted_outcomes_by_hypothesis[hypothesis_b]
                            [probe_index][state_index];
                        (outcome_a != outcome_b).then_some(probe_index)
                    })
                    .collect();
                if separating_probes.is_empty() {
                    unresolved.push(obligation);
                } else {
                    let obligation_index = required.len();
                    required.push(obligation);
                    for probe_index in separating_probes {
                        coverage[probe_index].insert(obligation_index);
                    }
                }
            }
        }
    }

    greedy_probe_cover(
        &problem.problem_id,
        problem.canonical_bytes(),
        &problem.probes,
        required,
        unresolved,
        coverage,
    )
}

/// Greedy non-adaptive probe design when the persistent categorical
/// nuisance state is unobserved. Each (hypothesis pair, state-A, state-B)
/// combination is a separate obligation. This is stronger than
/// `design_persistent_noise_probe_set`, which assumes the same indexed state
/// is known/common across both candidates.
pub fn design_persistent_noise_probe_set_under_unknown_state(
    problem: &PersistentNoiseProblemV1,
) -> Result<ProbeDesignReceiptV1, NoiseErrorV1> {
    problem.validate()?;
    let mut required = Vec::new();
    let mut unresolved = Vec::new();
    let mut coverage = vec![BTreeSet::new(); problem.probes.len()];

    for hypothesis_a in 0..problem.hypothesis_ids.len() {
        for hypothesis_b in (hypothesis_a + 1)..problem.hypothesis_ids.len() {
            for (state_a_index, state_a_id) in problem.error_state_ids.iter().enumerate() {
                for (state_b_index, state_b_id) in problem.error_state_ids.iter().enumerate() {
                    let obligation = ProbeDesignObligationV1 {
                        hypothesis_a: problem.hypothesis_ids[hypothesis_a].clone(),
                        hypothesis_b: problem.hypothesis_ids[hypothesis_b].clone(),
                        error_state_id: None,
                        error_state_pair: Some((state_a_id.clone(), state_b_id.clone())),
                    };
                    let separating_probes: Vec<usize> = problem
                        .probes
                        .iter()
                        .enumerate()
                        .filter(|(_, probe)| probe.admissible)
                        .filter_map(|(probe_index, _)| {
                            let outcome_a = problem.predicted_outcomes_by_hypothesis
                                [hypothesis_a][probe_index][state_a_index];
                            let outcome_b = problem.predicted_outcomes_by_hypothesis
                                [hypothesis_b][probe_index][state_b_index];
                            (outcome_a != outcome_b).then_some(probe_index)
                        })
                        .collect();
                    if separating_probes.is_empty() {
                        unresolved.push(obligation);
                    } else {
                        let obligation_index = required.len();
                        required.push(obligation);
                        for probe_index in separating_probes {
                            coverage[probe_index].insert(obligation_index);
                        }
                    }
                }
            }
        }
    }

    greedy_probe_cover(
        &problem.problem_id,
        problem.canonical_bytes(),
        &problem.probes,
        required,
        unresolved,
        coverage,
    )
}

/// Greedy minimum-cost non-adaptive diagnostic design for bounded numeric
/// intervals. Disjoint predicted intervals are robust pair-separation witnesses.
pub fn design_bounded_interval_probe_set(
    problem: &BoundedIntervalProblemV1,
) -> Result<ProbeDesignReceiptV1, NoiseErrorV1> {
    problem.validate()?;
    let mut required = Vec::new();
    let mut unresolved = Vec::new();
    let mut coverage = vec![BTreeSet::new(); problem.probes.len()];

    for hypothesis_a in 0..problem.hypothesis_ids.len() {
        for hypothesis_b in (hypothesis_a + 1)..problem.hypothesis_ids.len() {
            let obligation =
                pair_obligation(&problem.hypothesis_ids, hypothesis_a, hypothesis_b, None);
            let separating_probes: Vec<usize> = problem
                .probes
                .iter()
                .enumerate()
                .filter(|(_, probe)| probe.admissible)
                .filter_map(|(probe_index, _)| {
                    let interval_a =
                        problem.predicted_intervals_by_hypothesis[hypothesis_a][probe_index];
                    let interval_b =
                        problem.predicted_intervals_by_hypothesis[hypothesis_b][probe_index];
                    interval_a
                        .intersection(interval_b)
                        .is_none()
                        .then_some(probe_index)
                })
                .collect();
            if separating_probes.is_empty() {
                unresolved.push(obligation);
            } else {
                let obligation_index = required.len();
                required.push(obligation);
                for probe_index in separating_probes {
                    coverage[probe_index].insert(obligation_index);
                }
            }
        }
    }

    greedy_probe_cover(
        &problem.problem_id,
        problem.canonical_bytes(),
        &problem.probes,
        required,
        unresolved,
        coverage,
    )
}

/// Greedy minimum-cost non-adaptive diagnostic design under one campaign-wide
/// additive numeric bias. A common bias cancels between the two hypotheses,
/// so unequal nominal predictions are separating witnesses.
pub fn design_shared_bias_probe_set(
    problem: &SharedAdditiveBiasProblemV1,
) -> Result<ProbeDesignReceiptV1, NoiseErrorV1> {
    problem.validate()?;
    let mut required = Vec::new();
    let mut unresolved = Vec::new();
    let mut coverage = vec![BTreeSet::new(); problem.probes.len()];

    for hypothesis_a in 0..problem.hypothesis_ids.len() {
        for hypothesis_b in (hypothesis_a + 1)..problem.hypothesis_ids.len() {
            let obligation =
                pair_obligation(&problem.hypothesis_ids, hypothesis_a, hypothesis_b, None);
            let separating_probes: Vec<usize> = problem
                .probes
                .iter()
                .enumerate()
                .filter(|(_, probe)| probe.admissible)
                .filter_map(|(probe_index, _)| {
                    let prediction_a =
                        problem.nominal_predictions_by_hypothesis[hypothesis_a][probe_index];
                    let prediction_b =
                        problem.nominal_predictions_by_hypothesis[hypothesis_b][probe_index];
                    (prediction_a != prediction_b).then_some(probe_index)
                })
                .collect();
            if separating_probes.is_empty() {
                unresolved.push(obligation);
            } else {
                let obligation_index = required.len();
                required.push(obligation);
                for probe_index in separating_probes {
                    coverage[probe_index].insert(obligation_index);
                }
            }
        }
    }

    greedy_probe_cover(
        &problem.problem_id,
        problem.canonical_bytes(),
        &problem.probes,
        required,
        unresolved,
        coverage,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedBiasProbeWitnessV1 {
    pub hypothesis_a: String,
    pub hypothesis_b: String,
    pub witness_probe_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedBiasProbeDesignReceiptV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub selected_probe_ids: Vec<String>,
    pub selected_cost: u64,
    pub covered_pair_witnesses: Vec<SharedBiasProbeWitnessV1>,
    pub unresolved_pairs: Vec<ProbeDesignObligationV1>,
    /// This construction unions each pair's cheapest witness; it is not a
    /// globally optimal weighted set-cover solution.
    pub optimality_certified: bool,
    pub complete_pairwise_separation: bool,
}

/// Replay the deterministic greedy set-cover implementation and reject any
/// modified or forged receipt. This verifies reproducibility, not optimality.
/// Construct a non-adaptive probe set robust to an unknown campaign-wide
/// additive bias. Every candidate pair must be separated even when the two
/// hypotheses are allowed to use different values from the same bounded bias
/// interval. A witness can be one probe (constant offset outside the bias
/// difference range) or two probes (non-constant nominal offset).
///
/// This deterministic construction unions the cheapest witness set per pair.
/// It is a valid upper-bound design, not a globally optimal minimum-cost set.
pub fn design_shared_bias_probe_set_under_unknown_bias(
    problem: &SharedAdditiveBiasProblemV1,
) -> Result<SharedBiasProbeDesignReceiptV1, NoiseErrorV1> {
    problem.validate()?;
    let mut selected = BTreeSet::new();
    let mut covered_pair_witnesses = Vec::new();
    let mut unresolved_pairs = Vec::new();

    for hypothesis_a in 0..problem.hypothesis_ids.len() {
        for hypothesis_b in (hypothesis_a + 1)..problem.hypothesis_ids.len() {
            let report = shared_bias_pair_identifiability_under_unknown_bias(
                problem,
                hypothesis_a,
                hypothesis_b,
            )?;
            if report.indistinguishable_under_unknown_bias {
                unresolved_pairs.push(pair_obligation(
                    &problem.hypothesis_ids,
                    hypothesis_a,
                    hypothesis_b,
                    None,
                ));
                continue;
            }

            if report.separating_probe_ids.is_empty() {
                return Err(NoiseErrorV1::InvalidDeterministicInput);
            }
            for probe_id in &report.separating_probe_ids {
                if !problem.probes.iter().any(|probe| probe.probe_id == *probe_id && probe.admissible) {
                    return Err(NoiseErrorV1::InvalidDeterministicInput);
                }
                selected.insert(probe_id.clone());
            }
            covered_pair_witnesses.push(SharedBiasProbeWitnessV1 {
                hypothesis_a: problem.hypothesis_ids[hypothesis_a].clone(),
                hypothesis_b: problem.hypothesis_ids[hypothesis_b].clone(),
                witness_probe_ids: report.separating_probe_ids,
            });
        }
    }

    let selected_probe_ids: Vec<String> = selected.into_iter().collect();
    let selected_cost = selected_probe_ids.iter().try_fold(0_u64, |sum, probe_id| {
        let probe = problem
            .probes
            .iter()
            .find(|probe| probe.probe_id == *probe_id)
            .ok_or(NoiseErrorV1::InvalidDeterministicInput)?;
        sum.checked_add(probe.cost)
            .ok_or(NoiseErrorV1::IntervalArithmeticOverflow)
    })?;

    Ok(SharedBiasProbeDesignReceiptV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        selected_probe_ids,
        selected_cost,
        covered_pair_witnesses,
        complete_pairwise_separation: unresolved_pairs.is_empty(),
        unresolved_pairs,
        optimality_certified: false,
    })
}

pub fn verify_shared_bias_probe_design_under_unknown_bias(
    problem: &SharedAdditiveBiasProblemV1,
    receipt: &SharedBiasProbeDesignReceiptV1,
) -> Result<(), NoiseErrorV1> {
    let replayed = design_shared_bias_probe_set_under_unknown_bias(problem)?;
    if *receipt != replayed {
        return Err(NoiseErrorV1::ReceiptMismatch);
    }
    Ok(())
}

pub fn verify_bounded_noise_probe_design_receipt(
    problem: &BoundedNoiseProblemV1,
    receipt: &ProbeDesignReceiptV1,
) -> Result<(), NoiseErrorV1> {
    let replayed = design_bounded_noise_probe_set(problem)?;
    if *receipt != replayed {
        return Err(NoiseErrorV1::ReceiptMismatch);
    }
    Ok(())
}

pub fn verify_persistent_noise_probe_design_receipt(
    problem: &PersistentNoiseProblemV1,
    receipt: &ProbeDesignReceiptV1,
) -> Result<(), NoiseErrorV1> {
    let replayed = design_persistent_noise_probe_set(problem)?;
    if *receipt != replayed {
        return Err(NoiseErrorV1::ReceiptMismatch);
    }
    Ok(())
}

pub fn verify_persistent_noise_probe_design_under_unknown_state(
    problem: &PersistentNoiseProblemV1,
    receipt: &ProbeDesignReceiptV1,
) -> Result<(), NoiseErrorV1> {
    let replayed = design_persistent_noise_probe_set_under_unknown_state(problem)?;
    if *receipt != replayed {
        return Err(NoiseErrorV1::ReceiptMismatch);
    }
    Ok(())
}

pub fn verify_bounded_interval_probe_design_receipt(
    problem: &BoundedIntervalProblemV1,
    receipt: &ProbeDesignReceiptV1,
) -> Result<(), NoiseErrorV1> {
    let replayed = design_bounded_interval_probe_set(problem)?;
    if *receipt != replayed {
        return Err(NoiseErrorV1::ReceiptMismatch);
    }
    Ok(())
}

pub fn verify_shared_bias_probe_design_receipt(
    problem: &SharedAdditiveBiasProblemV1,
    receipt: &ProbeDesignReceiptV1,
) -> Result<(), NoiseErrorV1> {
    let replayed = design_shared_bias_probe_set(problem)?;
    if *receipt != replayed {
        return Err(NoiseErrorV1::ReceiptMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symmetry_active_experiment_reference::{
        ActiveExperimentProblemV1, ExperimentProbeV1, StaticProbeBudgetV1,
        compile_active_experiment, replay_observations, solve_exact_static_probe_design,
        verify_exact_static_probe_design,
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
    fn bounded_identifiability_reports_a_witness_or_unavoidable_pair_ambiguity() {
        let problem = bounded_fixture();
        let separable = bounded_pair_identifiability(&problem, 0, 2).unwrap();
        assert!(!separable.indistinguishable_under_available_probes);
        assert_eq!(separable.separating_probe_ids, vec!["p0".to_owned()]);

        let indistinguishable = bounded_pair_identifiability(&problem, 1, 2).unwrap();
        assert!(indistinguishable.indistinguishable_under_available_probes);
        assert!(indistinguishable.separating_probe_ids.is_empty());
        assert_eq!(
            indistinguishable.overlapping_probe_ids,
            vec!["p0".to_owned(), "p1".to_owned()]
        );
    }

    #[test]
    fn unknown_persistent_state_detects_cross_state_signature_collisions() {
        let problem = PersistentNoiseProblemV1 {
            problem_id: "cross-state-collision-v1".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned()],
            probes: vec![
                NoiseProbeV1 {
                    probe_id: "p0".to_owned(),
                    cost: 1,
                    admissible: true,
                },
                NoiseProbeV1 {
                    probe_id: "p1".to_owned(),
                    cost: 1,
                    admissible: true,
                },
            ],
            error_state_ids: vec!["low".to_owned(), "high".to_owned()],
            predicted_outcomes_by_hypothesis: vec![
                vec![vec![0, 1], vec![0, 1]],
                vec![vec![1, 0], vec![1, 0]],
            ],
        };

        // Conditional on the same state index, every comparison is separated.
        let same_state = persistent_pair_identifiability(&problem, 0, 1).unwrap();
        assert!(!same_state.indistinguishable_under_available_probes);

        // But H-A/low and H-B/high both produce [0, 0], and H-A/high and
        // H-B/low both produce [1, 1]. Without state calibration the hypotheses
        // are not robustly identifiable from this probe family.
        let unknown_state =
            persistent_pair_identifiability_under_unknown_state(&problem, 0, 1).unwrap();
        assert!(unknown_state.indistinguishable_under_unknown_state);
        assert_eq!(unknown_state.unresolved_state_pair_collisions.len(), 2);
        assert_eq!(
            unknown_state.unresolved_state_pair_collisions[0].common_signature,
            vec![("p0".to_owned(), 0), ("p1".to_owned(), 0)]
        );
        assert_eq!(
            unknown_state.unresolved_state_pair_collisions[1].common_signature,
            vec![("p0".to_owned(), 1), ("p1".to_owned(), 1)]
        );
    }

    #[test]
    fn exhaustive_two_state_binary_tables_match_unknown_state_collision_oracle() {
        for encoded in 0_u32..256 {
            let mut predictions = vec![vec![vec![0_u32; 2]; 2]; 2];
            for hypothesis in 0..2 {
                for probe in 0..2 {
                    for state in 0..2 {
                        let bit = hypothesis * 4 + probe * 2 + state;
                        predictions[hypothesis][probe][state] = (encoded >> bit) & 1;
                    }
                }
            }
            let problem = PersistentNoiseProblemV1 {
                problem_id: format!("exhaustive-cross-state-{encoded}"),
                hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned()],
                probes: vec![
                    NoiseProbeV1 {
                        probe_id: "p0".to_owned(),
                        cost: 1,
                        admissible: true,
                    },
                    NoiseProbeV1 {
                        probe_id: "p1".to_owned(),
                        cost: 1,
                        admissible: true,
                    },
                ],
                error_state_ids: vec!["s0".to_owned(), "s1".to_owned()],
                predicted_outcomes_by_hypothesis: predictions,
            };

            let report =
                persistent_pair_identifiability_under_unknown_state(&problem, 0, 1).unwrap();
            let mut brute_force_collision = false;
            for state_a in 0..2 {
                for state_b in 0..2 {
                    let identical = (0..2).all(|probe| {
                        problem.predicted_outcomes_by_hypothesis[0][probe][state_a]
                            == problem.predicted_outcomes_by_hypothesis[1][probe][state_b]
                    });
                    brute_force_collision |= identical;
                }
            }
            assert_eq!(
                report.indistinguishable_under_unknown_state,
                brute_force_collision,
                "cross-state mismatch for encoded prediction table {encoded}"
            );
        }
    }

    #[test]
    fn persistent_identifiability_uses_the_same_global_error_state_on_both_sides() {
        let problem = persistent_fixture();
        let report = persistent_pair_identifiability(&problem, 0, 1).unwrap();
        assert!(!report.indistinguishable_under_available_probes);
        assert_eq!(report.state_witnesses.len(), 2);
        assert!(report.unresolved_error_states.is_empty());

        let mut restricted = problem.clone();
        restricted.probes[1].admissible = false;
        let report = persistent_pair_identifiability(&restricted, 0, 1).unwrap();
        assert!(report.indistinguishable_under_available_probes);
        assert_eq!(
            report
                .unresolved_error_states
                .iter()
                .map(|state| state.error_state_id.as_str())
                .collect::<Vec<_>>(),
            vec!["bias-low", "bias-high"]
        );
    }

    #[test]
    fn pair_identifiability_rejects_same_or_out_of_range_hypothesis() {
        let bounded = bounded_fixture();
        assert_eq!(
            bounded_pair_identifiability(&bounded, 1, 1),
            Err(NoiseErrorV1::InvalidHypothesisPair)
        );
        let persistent = persistent_fixture();
        assert_eq!(
            persistent_pair_identifiability(&persistent, 0, 2),
            Err(NoiseErrorV1::InvalidHypothesisPair)
        );
    }

    #[test]
    fn bounded_categorical_probe_set_covers_every_resolvable_pair_and_reports_the_frontier() {
        let problem = bounded_fixture();
        let receipt = design_bounded_noise_probe_set(&problem).unwrap();
        assert_eq!(
            receipt.selected_probe_ids,
            vec!["p0".to_owned(), "p1".to_owned()]
        );
        assert_eq!(receipt.selected_cost, 3);
        assert_eq!(receipt.required_obligations.len(), 2);
        assert_eq!(receipt.covered_obligations, receipt.required_obligations);
        assert_eq!(receipt.unresolved_obligations.len(), 1);
        assert_eq!(
            (
                receipt.unresolved_obligations[0].hypothesis_a.as_str(),
                receipt.unresolved_obligations[0].hypothesis_b.as_str()
            ),
            ("H-B", "H-C")
        );
        assert!(!receipt.complete_pairwise_separation);
        assert!(!receipt.optimality_certified);
    }

    #[test]
    fn persistent_categorical_probe_set_covers_every_shared_error_state() {
        let problem = persistent_fixture();
        let receipt = design_persistent_noise_probe_set(&problem).unwrap();
        assert_eq!(receipt.selected_probe_ids, vec!["p-right".to_owned()]);
        assert_eq!(receipt.required_obligations.len(), 2);
        assert!(receipt.unresolved_obligations.is_empty());
        assert!(receipt.complete_pairwise_separation);
        assert!(
            receipt
                .required_obligations
                .iter()
                .all(|obligation| obligation.error_state_id.is_some())
        );
    }

    #[test]
    fn unknown_state_persistent_design_receipt_rejects_forgery() {
        let problem = PersistentNoiseProblemV1 {
            problem_id: "unknown-state-receipt-v1".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned()],
            probes: vec![
                NoiseProbeV1 {
                    probe_id: "p0".to_owned(),
                    cost: 1,
                    admissible: true,
                },
            ],
            error_state_ids: vec!["low".to_owned(), "high".to_owned()],
            predicted_outcomes_by_hypothesis: vec![
                vec![vec![0, 1]],
                vec![vec![1, 0]],
            ],
        };
        let receipt = design_persistent_noise_probe_set_under_unknown_state(&problem).unwrap();
        verify_persistent_noise_probe_design_under_unknown_state(&problem, &receipt).unwrap();
        let mut forged = receipt.clone();
        forged.complete_pairwise_separation = !forged.complete_pairwise_separation;
        assert_eq!(
            verify_persistent_noise_probe_design_under_unknown_state(&problem, &forged),
            Err(NoiseErrorV1::ReceiptMismatch)
        );
    }

    #[test]
    fn unknown_state_persistent_probe_design_covers_cross_state_obligations() {
        let problem = PersistentNoiseProblemV1 {
            problem_id: "unknown-state-design-v1".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned()],
            probes: vec![
                NoiseProbeV1 {
                    probe_id: "p0".to_owned(),
                    cost: 1,
                    admissible: true,
                },
                NoiseProbeV1 {
                    probe_id: "p1".to_owned(),
                    cost: 1,
                    admissible: true,
                },
            ],
            error_state_ids: vec!["low".to_owned(), "high".to_owned()],
            predicted_outcomes_by_hypothesis: vec![
                vec![vec![0, 1], vec![0, 1]],
                vec![vec![1, 0], vec![1, 0]],
            ],
        };
        let conditional = design_persistent_noise_probe_set(&problem).unwrap();
        assert!(conditional.complete_pairwise_separation);

        let robust = design_persistent_noise_probe_set_under_unknown_state(&problem).unwrap();
        assert!(!robust.complete_pairwise_separation);
        assert_eq!(robust.unresolved_obligations.len(), 2);
        assert_eq!(
            robust.unresolved_obligations[0].error_state_pair.as_ref(),
            Some(&("low".to_owned(), "high".to_owned()))
        );
        assert_eq!(
            robust.unresolved_obligations[1].error_state_pair.as_ref(),
            Some(&("high".to_owned(), "low".to_owned()))
        );
        assert!(!robust.optimality_certified);
    }

    #[test]
    fn unknown_bias_probe_design_carries_pair_witnesses_and_unresolved_frontier() {
        let problem = SharedAdditiveBiasProblemV1 {
            problem_id: "unknown-bias-probe-design-v1".to_owned(),
            quantity_unit_id: "millimetre-integer".to_owned(),
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
            nominal_predictions_by_hypothesis: vec![
                vec![0, 10],
                vec![3, 13],
                vec![3, 14],
            ],
            global_bias_interval: FixedIntervalV1::new(-5, 5).unwrap(),
        };
        let receipt = design_shared_bias_probe_set_under_unknown_bias(&problem).unwrap();
        assert_eq!(receipt.unresolved_pairs.len(), 1);
        assert_eq!(
            (receipt.unresolved_pairs[0].hypothesis_a.as_str(),
             receipt.unresolved_pairs[0].hypothesis_b.as_str()),
            ("H-A", "H-B")
        );
        assert_eq!(receipt.covered_pair_witnesses.len(), 2);
        let h_a_h_c = receipt.covered_pair_witnesses.iter()
            .find(|witness| witness.hypothesis_a == "H-A" && witness.hypothesis_b == "H-C")
            .unwrap();
        assert_eq!(h_a_h_c.witness_probe_ids.len(), 2);
        assert_eq!(receipt.selected_probe_ids, vec!["p0", "p1"]);
        assert!(!receipt.complete_pairwise_separation);
        assert!(!receipt.optimality_certified);
        verify_shared_bias_probe_design_under_unknown_bias(&problem, &receipt).unwrap();

        let mut forged = receipt.clone();
        forged.selected_cost += 1;
        assert_eq!(
            verify_shared_bias_probe_design_under_unknown_bias(&problem, &forged),
            Err(NoiseErrorV1::ReceiptMismatch)
        );
    }

    #[test]
    fn unknown_bias_probe_design_replays_pair_witnesses_and_ambiguity_frontier() {
        let problem = SharedAdditiveBiasProblemV1 {
            problem_id: "unknown-bias-probe-design-v1".to_owned(),
            quantity_unit_id: "millimetre-integer".to_owned(),
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
            nominal_predictions_by_hypothesis: vec![
                vec![0, 10],
                vec![3, 13],
                vec![3, 14],
            ],
            global_bias_interval: FixedIntervalV1::new(-5, 5).unwrap(),
        };
        let receipt = design_shared_bias_probe_set_under_unknown_bias(&problem).unwrap();
        assert_eq!(receipt.unresolved_pairs.len(), 1);
        assert_eq!(
            (
                receipt.unresolved_pairs[0].hypothesis_a.as_str(),
                receipt.unresolved_pairs[0].hypothesis_b.as_str()
            ),
            ("H-A", "H-B")
        );
        assert_eq!(receipt.covered_pair_witnesses.len(), 2);
        let h_a_h_c = receipt
            .covered_pair_witnesses
            .iter()
            .find(|witness| witness.hypothesis_a == "H-A" && witness.hypothesis_b == "H-C")
            .unwrap();
        assert_eq!(h_a_h_c.witness_probe_ids.len(), 2);
        assert_eq!(receipt.selected_probe_ids, vec!["p0", "p1"]);
        assert!(!receipt.complete_pairwise_separation);
        assert!(!receipt.optimality_certified);
        verify_shared_bias_probe_design_under_unknown_bias(&problem, &receipt).unwrap();

        let mut forged = receipt.clone();
        forged.selected_cost += 1;
        assert_eq!(
            verify_shared_bias_probe_design_under_unknown_bias(&problem, &forged),
            Err(NoiseErrorV1::ReceiptMismatch)
        );
    }

    #[test]
    fn bounded_numeric_probe_set_reports_minimum_cost_greedy_witness_basis() {
        let problem = BoundedIntervalProblemV1 {
            problem_id: "bounded-interval-probe-design-v1".to_owned(),
            quantity_unit_id: "millimetre-integer".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned(), "H-C".to_owned()],
            probes: vec![
                NoiseProbeV1 {
                    probe_id: "p-overlap".to_owned(),
                    cost: 1,
                    admissible: true,
                },
                NoiseProbeV1 {
                    probe_id: "p-separate-ab".to_owned(),
                    cost: 2,
                    admissible: true,
                },
                NoiseProbeV1 {
                    probe_id: "p-separate-ac".to_owned(),
                    cost: 1,
                    admissible: true,
                },
            ],
            predicted_intervals_by_hypothesis: vec![
                vec![
                    FixedIntervalV1::new(0, 10).unwrap(),
                    FixedIntervalV1::new(0, 2).unwrap(),
                    FixedIntervalV1::new(0, 10).unwrap(),
                ],
                vec![
                    FixedIntervalV1::new(5, 15).unwrap(),
                    FixedIntervalV1::new(20, 30).unwrap(),
                    FixedIntervalV1::new(5, 15).unwrap(),
                ],
                vec![
                    FixedIntervalV1::new(6, 9).unwrap(),
                    FixedIntervalV1::new(6, 9).unwrap(),
                    FixedIntervalV1::new(20, 30).unwrap(),
                ],
            ],
        };
        let receipt = design_bounded_interval_probe_set(&problem).unwrap();
        assert!(
            receipt
                .selected_probe_ids
                .contains(&"p-separate-ab".to_owned())
        );
        assert!(
            receipt
                .selected_probe_ids
                .contains(&"p-separate-ac".to_owned())
        );
        assert_eq!(receipt.required_obligations.len(), 3);
        assert_eq!(receipt.covered_obligations, receipt.required_obligations);
        assert!(receipt.complete_pairwise_separation);
        assert!(!receipt.optimality_certified);
    }

    #[test]
    fn shared_bias_probe_set_uses_a_common_shift_for_every_pair() {
        let problem = SharedAdditiveBiasProblemV1 {
            problem_id: "shared-bias-probe-design-v1".to_owned(),
            quantity_unit_id: "micrometre-integer".to_owned(),
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
                NoiseProbeV1 {
                    probe_id: "p2".to_owned(),
                    cost: 1,
                    admissible: true,
                },
            ],
            nominal_predictions_by_hypothesis: vec![vec![0, 0, 0], vec![0, 10, 0], vec![0, 0, 20]],
            global_bias_interval: FixedIntervalV1::new(-100, 100).unwrap(),
        };
        let receipt = design_shared_bias_probe_set(&problem).unwrap();
        assert_eq!(receipt.required_obligations.len(), 3);
        assert_eq!(receipt.covered_obligations, receipt.required_obligations);
        assert!(receipt.complete_pairwise_separation);
        assert_eq!(receipt.selected_cost, 3);
    }

    #[test]
    fn static_probe_basis_cost_is_compared_with_adaptive_minimax_cost_on_same_d4_problem() {
        let deterministic = ActiveExperimentProblemV1 {
            problem_id: "d4-adaptive-versus-static-probe-basis-v1".to_owned(),
            hypotheses: vec![
                "D4-H-A".to_owned(),
                "D4-H-B".to_owned(),
                "D4-H-C".to_owned(),
                "D4-H-D".to_owned(),
            ],
            probes: vec![
                ExperimentProbeV1 {
                    probe_id: "p-balanced-first".to_owned(),
                    cost: 1,
                    admissible: true,
                    outcomes: vec![0, 0, 1, 1],
                },
                ExperimentProbeV1 {
                    probe_id: "p-separate-ab".to_owned(),
                    cost: 1,
                    admissible: true,
                    outcomes: vec![0, 1, 0, 0],
                },
                ExperimentProbeV1 {
                    probe_id: "p-separate-cd".to_owned(),
                    cost: 1,
                    admissible: true,
                    outcomes: vec![0, 0, 1, 0],
                },
            ],
        };
        let bounded = BoundedNoiseProblemV1::from_deterministic(&deterministic).unwrap();
        let static_design = design_bounded_noise_probe_set(&bounded).unwrap();
        let exact_static = solve_exact_static_probe_design(
            &deterministic,
            Default::default(),
            StaticProbeBudgetV1::default(),
        )
        .unwrap();
        let adaptive = compile_active_experiment(&deterministic, Default::default()).unwrap();

        assert_eq!(static_design.selected_probe_ids.len(), 3);
        assert_eq!(static_design.selected_cost, 3);
        assert!(static_design.complete_pairwise_separation);
        assert!(!static_design.optimality_certified);
        assert_eq!(exact_static.selected_cost, 3);
        assert!(exact_static.optimality_certified);
        assert!(exact_static.intrinsically_unresolved_pairs.is_empty());
        assert_eq!(
            static_design.covered_obligations.len(),
            exact_static.distinguished_pairs.len()
        );
        assert_eq!(
            static_design.required_obligations.len(),
            exact_static.distinguished_pairs.len()
        );
        verify_exact_static_probe_design(
            &deterministic,
            &exact_static,
            Default::default(),
            StaticProbeBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(adaptive.root_worst_case_cost, 2);
        assert!(exact_static.selected_cost > adaptive.root_worst_case_cost);

        verify_bounded_noise_probe_design_receipt(&bounded, &static_design).unwrap();
        let mut forged = static_design.clone();
        forged.selected_cost = 2;
        assert_eq!(
            verify_bounded_noise_probe_design_receipt(&bounded, &forged),
            Err(NoiseErrorV1::ReceiptMismatch)
        );
    }

    #[test]
    fn every_noise_aware_static_design_receipt_replays_or_rejects_mutation() {
        let bounded = bounded_fixture();
        let bounded_receipt = design_bounded_noise_probe_set(&bounded).unwrap();
        verify_bounded_noise_probe_design_receipt(&bounded, &bounded_receipt).unwrap();

        let persistent = persistent_fixture();
        let persistent_receipt = design_persistent_noise_probe_set(&persistent).unwrap();
        verify_persistent_noise_probe_design_receipt(&persistent, &persistent_receipt).unwrap();

        let numeric = BoundedIntervalProblemV1 {
            problem_id: "numeric-probe-receipt-v1".to_owned(),
            quantity_unit_id: "millimetre-integer".to_owned(),
            hypothesis_ids: vec!["A".to_owned(), "B".to_owned()],
            probes: vec![NoiseProbeV1 {
                probe_id: "p0".to_owned(),
                cost: 1,
                admissible: true,
            }],
            predicted_intervals_by_hypothesis: vec![
                vec![FixedIntervalV1::new(0, 1).unwrap()],
                vec![FixedIntervalV1::new(2, 3).unwrap()],
            ],
        };
        let numeric_receipt = design_bounded_interval_probe_set(&numeric).unwrap();
        verify_bounded_interval_probe_design_receipt(&numeric, &numeric_receipt).unwrap();

        let shared_bias = SharedAdditiveBiasProblemV1 {
            problem_id: "shared-bias-probe-receipt-v1".to_owned(),
            quantity_unit_id: "millimetre-integer".to_owned(),
            hypothesis_ids: vec!["A".to_owned(), "B".to_owned()],
            probes: vec![NoiseProbeV1 {
                probe_id: "p0".to_owned(),
                cost: 1,
                admissible: true,
            }],
            nominal_predictions_by_hypothesis: vec![vec![0], vec![1]],
            global_bias_interval: FixedIntervalV1::new(-100, 100).unwrap(),
        };
        let bias_receipt = design_shared_bias_probe_set(&shared_bias).unwrap();
        verify_shared_bias_probe_design_receipt(&shared_bias, &bias_receipt).unwrap();
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
    fn bounded_numeric_identifiability_reports_disjoint_interval_witnesses() {
        let problem = BoundedIntervalProblemV1 {
            problem_id: "numeric-identifiability-v1".to_owned(),
            quantity_unit_id: "millimetre-integer".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned(), "H-C".to_owned()],
            probes: vec![
                NoiseProbeV1 {
                    probe_id: "p0".to_owned(),
                    cost: 2,
                    admissible: true,
                },
                NoiseProbeV1 {
                    probe_id: "p1".to_owned(),
                    cost: 1,
                    admissible: true,
                },
            ],
            predicted_intervals_by_hypothesis: vec![
                vec![
                    FixedIntervalV1::new(0, 10).unwrap(),
                    FixedIntervalV1::new(0, 10).unwrap(),
                ],
                vec![
                    FixedIntervalV1::new(5, 15).unwrap(),
                    FixedIntervalV1::new(20, 30).unwrap(),
                ],
                vec![
                    FixedIntervalV1::new(6, 9).unwrap(),
                    FixedIntervalV1::new(8, 9).unwrap(),
                ],
            ],
        };
        let report = bounded_interval_pair_identifiability(&problem, 0, 1).unwrap();
        assert!(!report.indistinguishable_under_available_probes);
        assert_eq!(report.separating_probe_ids, vec!["p1".to_owned()]);
        assert_eq!(report.overlapping_probe_ids, vec!["p0".to_owned()]);

        let ambiguous = bounded_interval_pair_identifiability(&problem, 0, 2).unwrap();
        assert!(ambiguous.indistinguishable_under_available_probes);
        assert!(ambiguous.separating_probe_ids.is_empty());
    }

    #[test]
    fn unknown_campaign_bias_detects_cross_bias_signature_collisions() {
        let problem = SharedAdditiveBiasProblemV1 {
            problem_id: "cross-bias-collision-v1".to_owned(),
            quantity_unit_id: "millimetre-integer".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned(), "H-C".to_owned()],
            probes: vec![
                NoiseProbeV1 {
                    probe_id: "p0".to_owned(),
                    cost: 1,
                    admissible: true,
                },
                NoiseProbeV1 {
                    probe_id: "p1".to_owned(),
                    cost: 1,
                    admissible: true,
                },
            ],
            nominal_predictions_by_hypothesis: vec![
                vec![0, 10],
                vec![3, 13],
                vec![3, 14],
            ],
            global_bias_interval: FixedIntervalV1::new(-5, 5).unwrap(),
        };

        let same_bias = shared_bias_pair_identifiability(&problem, 0, 1).unwrap();
        assert!(!same_bias.indistinguishable_under_available_probes);
        let unknown_bias =
            shared_bias_pair_identifiability_under_unknown_bias(&problem, 0, 1).unwrap();
        assert!(unknown_bias.indistinguishable_under_unknown_bias);
        let collision = unknown_bias.collision.unwrap();
        assert_eq!((collision.bias_a, collision.bias_b), (-2, -5));
        assert_eq!(
            collision.common_signature,
            vec![("p0".to_owned(), -2), ("p1".to_owned(), 8)]
        );

        // A non-constant nominal offset cannot be canceled by one campaign-wide
        // bias pair, even though either individual probe can be matched.
        let nonconstant =
            shared_bias_pair_identifiability_under_unknown_bias(&problem, 0, 2).unwrap();
        assert!(!nonconstant.indistinguishable_under_unknown_bias);
        assert_eq!(nonconstant.separating_probe_ids.len(), 2);
        assert_ne!(
            nonconstant.separating_probe_ids[0],
            nonconstant.separating_probe_ids[1]
        );
    }

    #[test]
    fn exhaustive_small_numeric_tables_match_unknown_bias_collision_oracle() {
        for encoded in 0_u32..81 {
            let mut value = encoded;
            let mut nominal = vec![vec![0_i128; 2]; 2];
            for hypothesis in 0..2 {
                for probe in 0..2 {
                    nominal[hypothesis][probe] = i128::from(value % 3);
                    value /= 3;
                }
            }
            let problem = SharedAdditiveBiasProblemV1 {
                problem_id: format!("exhaustive-unknown-bias-{encoded}"),
                quantity_unit_id: "integer-unit".to_owned(),
                hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned()],
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
                nominal_predictions_by_hypothesis: nominal,
                global_bias_interval: FixedIntervalV1::new(-1, 1).unwrap(),
            };

            let report =
                shared_bias_pair_identifiability_under_unknown_bias(&problem, 0, 1).unwrap();
            let mut brute_force_collision = false;
            for bias_a in -1_i128..=1 {
                for bias_b in -1_i128..=1 {
                    let identical = (0..2).all(|probe| {
                        problem.nominal_predictions_by_hypothesis[0][probe] + bias_a
                            == problem.nominal_predictions_by_hypothesis[1][probe] + bias_b
                    });
                    brute_force_collision |= identical;
                }
            }
            assert_eq!(
                report.indistinguishable_under_unknown_bias,
                brute_force_collision,
                "cross-bias mismatch for encoded prediction table {encoded}"
            );
            if brute_force_collision {
                assert!(report.separating_probe_ids.is_empty());
                assert!(report.collision.is_some());
            } else {
                assert!(!report.separating_probe_ids.is_empty());
                assert!(report.collision.is_none());
            }
        }
    }

    #[test]
    fn shared_bias_identifiability_uses_a_single_common_shift() {
        let problem = SharedAdditiveBiasProblemV1 {
            problem_id: "shared-bias-identifiability-v1".to_owned(),
            quantity_unit_id: "millimetre-integer".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned()],
            probes: vec![
                NoiseProbeV1 {
                    probe_id: "p0".to_owned(),
                    cost: 2,
                    admissible: true,
                },
                NoiseProbeV1 {
                    probe_id: "p1".to_owned(),
                    cost: 1,
                    admissible: true,
                },
            ],
            nominal_predictions_by_hypothesis: vec![vec![0, 0], vec![0, 10]],
            global_bias_interval: FixedIntervalV1::new(-100, 100).unwrap(),
        };
        let report = shared_bias_pair_identifiability(&problem, 0, 1).unwrap();
        assert!(!report.indistinguishable_under_available_probes);
        assert_eq!(report.separating_probe_ids, vec!["p1".to_owned()]);
        assert_eq!(report.overlapping_probe_ids, vec!["p0".to_owned()]);

        let mut restricted = problem.clone();
        restricted.probes[1].admissible = false;
        let report = shared_bias_pair_identifiability(&restricted, 0, 1).unwrap();
        assert!(report.indistinguishable_under_available_probes);
        assert_eq!(report.overlapping_probe_ids, vec!["p0".to_owned()]);
    }

    #[test]
    fn bounded_numeric_intervals_retain_overlapping_hypotheses_only() {
        let problem = BoundedIntervalProblemV1 {
            problem_id: "numeric-bounded-interval-v1".to_owned(),
            quantity_unit_id: "millimetre-integer".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned()],
            probes: vec![
                NoiseProbeV1 {
                    probe_id: "p0".to_owned(),
                    cost: 1,
                    admissible: true,
                },
                NoiseProbeV1 {
                    probe_id: "p1".to_owned(),
                    cost: 1,
                    admissible: true,
                },
            ],
            predicted_intervals_by_hypothesis: vec![
                vec![
                    FixedIntervalV1::new(0, 10).unwrap(),
                    FixedIntervalV1::new(20, 30).unwrap(),
                ],
                vec![
                    FixedIntervalV1::new(5, 15).unwrap(),
                    FixedIntervalV1::new(5, 15).unwrap(),
                ],
            ],
        };
        let observations = vec![
            IntervalObservationV1 {
                probe_id: "p0".to_owned(),
                observed: FixedIntervalV1::new(8, 9).unwrap(),
            },
            IntervalObservationV1 {
                probe_id: "p1".to_owned(),
                observed: FixedIntervalV1::new(24, 25).unwrap(),
            },
        ];
        let receipt = replay_bounded_intervals(&problem, &observations).unwrap();
        assert_eq!(receipt.final_active_hypothesis_ids, vec!["H-A"]);
        verify_bounded_interval_receipt(&problem, &observations, &receipt).unwrap();
    }

    #[test]
    fn repeated_numeric_observations_intersect_the_same_probe_interval() {
        let problem = BoundedIntervalProblemV1 {
            problem_id: "numeric-repeat-intersection-v1".to_owned(),
            quantity_unit_id: "millimetre-integer".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned()],
            probes: vec![NoiseProbeV1 {
                probe_id: "p0".to_owned(),
                cost: 1,
                admissible: true,
            }],
            predicted_intervals_by_hypothesis: vec![vec![FixedIntervalV1::new(0, 10).unwrap()]],
        };
        let observations = vec![
            IntervalObservationV1 {
                probe_id: "p0".to_owned(),
                observed: FixedIntervalV1::new(2, 4).unwrap(),
            },
            IntervalObservationV1 {
                probe_id: "p0".to_owned(),
                observed: FixedIntervalV1::new(6, 7).unwrap(),
            },
        ];
        let receipt = replay_bounded_intervals(&problem, &observations).unwrap();
        assert_eq!(receipt.final_active_hypothesis_ids, Vec::<String>::new());
        assert_eq!(receipt.snapshots.len(), observations.len() + 1);
        assert!(!receipt.snapshots[2].model_consistent);
    }

    #[test]
    fn one_shared_additive_bias_must_explain_every_probe() {
        let problem = SharedAdditiveBiasProblemV1 {
            problem_id: "shared-numeric-bias-v1".to_owned(),
            quantity_unit_id: "millimetre-integer".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned(), "H-B".to_owned()],
            probes: vec![
                NoiseProbeV1 {
                    probe_id: "p0".to_owned(),
                    cost: 1,
                    admissible: true,
                },
                NoiseProbeV1 {
                    probe_id: "p1".to_owned(),
                    cost: 1,
                    admissible: true,
                },
            ],
            nominal_predictions_by_hypothesis: vec![vec![0, 0], vec![0, 10]],
            global_bias_interval: FixedIntervalV1::new(-5, 5).unwrap(),
        };
        let observations = vec![
            IntervalObservationV1 {
                probe_id: "p0".to_owned(),
                observed: FixedIntervalV1::point(0),
            },
            IntervalObservationV1 {
                probe_id: "p1".to_owned(),
                observed: FixedIntervalV1::point(10),
            },
        ];
        let receipt = replay_shared_additive_bias(&problem, &observations).unwrap();
        assert_eq!(
            receipt.snapshots[1].active_hypothesis_ids,
            vec!["H-A", "H-B"]
        );
        assert_eq!(receipt.final_active_hypothesis_ids, vec!["H-B"]);
        assert_eq!(
            receipt.snapshots[2].hypotheses[0].feasible_global_bias_interval,
            None
        );
        assert_eq!(
            receipt.snapshots[2].hypotheses[1].feasible_global_bias_interval,
            Some(FixedIntervalV1::point(0))
        );
        verify_shared_additive_bias_receipt(&problem, &observations, &receipt).unwrap();
    }

    #[test]
    fn shared_bias_repetition_does_not_allow_a_fresh_nuisance_value() {
        let problem = SharedAdditiveBiasProblemV1 {
            problem_id: "shared-bias-repeat-v1".to_owned(),
            quantity_unit_id: "millimetre-integer".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned()],
            probes: vec![NoiseProbeV1 {
                probe_id: "p0".to_owned(),
                cost: 1,
                admissible: true,
            }],
            nominal_predictions_by_hypothesis: vec![vec![0]],
            global_bias_interval: FixedIntervalV1::new(-10, 10).unwrap(),
        };
        let observations = vec![
            IntervalObservationV1 {
                probe_id: "p0".to_owned(),
                observed: FixedIntervalV1::point(1),
            },
            IntervalObservationV1 {
                probe_id: "p0".to_owned(),
                observed: FixedIntervalV1::point(2),
            },
        ];
        let receipt = replay_shared_additive_bias(&problem, &observations).unwrap();
        assert_eq!(receipt.snapshots[1].active_hypothesis_ids, vec!["H-A"]);
        assert!(receipt.snapshots[2].active_hypothesis_ids.is_empty());
    }

    #[test]
    fn numeric_interval_unit_scale_is_required_and_receipt_bound() {
        let mut problem = SharedAdditiveBiasProblemV1 {
            problem_id: "numeric-unit-contract-v1".to_owned(),
            quantity_unit_id: String::new(),
            hypothesis_ids: vec!["H-A".to_owned()],
            probes: vec![NoiseProbeV1 {
                probe_id: "p0".to_owned(),
                cost: 1,
                admissible: true,
            }],
            nominal_predictions_by_hypothesis: vec![vec![0]],
            global_bias_interval: FixedIntervalV1::new(0, 1).unwrap(),
        };
        assert_eq!(problem.validate(), Err(NoiseErrorV1::EmptyQuantityUnitId));
        problem.quantity_unit_id = "millimetre-integer".to_owned();
        problem.validate().unwrap();
        let original = problem.canonical_bytes();
        problem.quantity_unit_id = "micrometre-integer".to_owned();
        assert_ne!(original, problem.canonical_bytes());
    }

    #[test]
    fn interval_endpoints_reject_reversal_and_checked_bias_arithmetic_overflow() {
        assert_eq!(
            FixedIntervalV1::new(5, 4),
            Err(NoiseErrorV1::InvalidInterval)
        );
        let problem = SharedAdditiveBiasProblemV1 {
            problem_id: "interval-overflow-v1".to_owned(),
            quantity_unit_id: "millimetre-integer".to_owned(),
            hypothesis_ids: vec!["H-A".to_owned()],
            probes: vec![NoiseProbeV1 {
                probe_id: "p0".to_owned(),
                cost: 1,
                admissible: true,
            }],
            nominal_predictions_by_hypothesis: vec![vec![1]],
            global_bias_interval: FixedIntervalV1::new(i128::MIN, i128::MAX).unwrap(),
        };
        let observations = vec![IntervalObservationV1 {
            probe_id: "p0".to_owned(),
            observed: FixedIntervalV1::new(i128::MIN, i128::MIN).unwrap(),
        }];
        assert_eq!(
            replay_shared_additive_bias(&problem, &observations),
            Err(NoiseErrorV1::IntervalArithmeticOverflow)
        );
    }

    #[test]
    fn identical_observation_models_remain_ambiguous() {
        let problem = BoundedIntervalProblemV1 {
            problem_id: "observationally-equivalent-symmetries-v1".to_owned(),
            quantity_unit_id: "millimetre-integer".to_owned(),
            hypothesis_ids: vec!["H-rotation".to_owned(), "H-reflection".to_owned()],
            probes: vec![NoiseProbeV1 {
                probe_id: "non-identifying-probe".to_owned(),
                cost: 1,
                admissible: true,
            }],
            predicted_intervals_by_hypothesis: vec![
                vec![FixedIntervalV1::new(3, 7).unwrap()],
                vec![FixedIntervalV1::new(3, 7).unwrap()],
            ],
        };
        let observations = vec![IntervalObservationV1 {
            probe_id: "non-identifying-probe".to_owned(),
            observed: FixedIntervalV1::new(4, 6).unwrap(),
        }];
        let receipt = replay_bounded_intervals(&problem, &observations).unwrap();
        assert_eq!(
            receipt.final_active_hypothesis_ids,
            vec!["H-rotation".to_owned(), "H-reflection".to_owned()]
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
