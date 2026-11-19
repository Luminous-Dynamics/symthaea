// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Explicit, replayable contracts for hypothesis-conditioned sensor dropout.
//!
//! This crate separates hard support claims from probabilistic dropout models.
//! It does not infer either model from data and does not establish physical
//! truth. Evidence identifiers are bound into receipts but are not verified
//! cryptographic signatures or proof objects.

use std::collections::BTreeSet;
use std::fmt;

pub const PROBABILITY_SCALE_PPM_V1: u32 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardSupportEvidenceKindV1 {
    EngineeringInterlock,
    ProvenPhysicalBound,
    FormalProof,
    ExternalCertificate,
}

impl HardSupportEvidenceKindV1 {
    fn tag(self) -> u8 {
        match self {
            Self::EngineeringInterlock => 1,
            Self::ProvenPhysicalBound => 2,
            Self::FormalProof => 3,
            Self::ExternalCertificate => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardSupportJustificationV1 {
    /// Stable reference to the evidence artifact; the referenced artifact is
    /// not downloaded or cryptographically verified by this crate.
    pub evidence_id: String,
    pub evidence_kind: HardSupportEvidenceKindV1,
    /// The operating envelope within which the impossibility claim is valid.
    pub validity_domain_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingnessProbeV1 {
    pub probe_id: String,
    pub cost: u64,
    pub admissible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HypothesisConditionedMissingnessProblemV1 {
    pub problem_id: String,
    pub validity_domain_id: String,
    pub hypothesis_ids: Vec<String>,
    pub probes: Vec<MissingnessProbeV1>,
    /// [hypothesis][probe] -> is abstention possible under this hypothesis?
    pub abstention_possible_by_hypothesis: Vec<Vec<bool>>,
    /// For every false support entry, a caller-supplied hard-impossibility
    /// justification scoped to the same validity domain. A missing reference
    /// is a validation error, not permission to eliminate.
    pub impossibility_justifications_by_hypothesis:
        Vec<Vec<Option<HardSupportJustificationV1>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissingnessErrorV1 {
    EmptyProblemId,
    EmptyValidityDomainId,
    EmptyHypothesisSet,
    TooManyHypotheses,
    EmptyHypothesisId,
    DuplicateHypothesisId,
    EmptyProbeSet,
    TooManyProbes,
    EmptyProbeId,
    DuplicateProbeId,
    EmptyProbeSet,
    TooManyProbes,
    ZeroProbeCost { probe_id: String },
    HypothesisCountMismatch,
    ProbeCountMismatch { hypothesis_id: String },
    MissingImpossibilityJustification {
        hypothesis_id: String,
        probe_id: String,
    },
    UnexpectedImpossibilityJustification {
        hypothesis_id: String,
        probe_id: String,
    },
    EmptyEvidenceId { hypothesis_id: String, probe_id: String },
    EmptyEvidenceDomain { hypothesis_id: String, probe_id: String },
    EvidenceDomainMismatch {
        hypothesis_id: String,
        probe_id: String,
    },
    UnknownProbe { probe_id: String },
    InadmissibleProbe { probe_id: String },
    DuplicateProbeObservation { probe_id: String },
    EmptyCalibrationEvidenceId,
    EmptyIndependenceContractId,
    InvalidProbabilityInterval,
    ProbabilityArithmeticOverflow,
    ReceiptMismatch,
}

impl fmt::Display for MissingnessErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MissingnessErrorV1 {}

fn append_bytes(output: &mut Vec<u8>, value: &[u8]) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value);
}

fn append_string(output: &mut Vec<u8>, value: &str) {
    append_bytes(output, value.as_bytes());
}

impl HypothesisConditionedMissingnessProblemV1 {
    pub fn validate(&self) -> Result<(), MissingnessErrorV1> {
        if self.problem_id.trim().is_empty() {
            return Err(MissingnessErrorV1::EmptyProblemId);
        }
        if self.validity_domain_id.trim().is_empty() {
            return Err(MissingnessErrorV1::EmptyValidityDomainId);
        }
        if self.hypothesis_ids.is_empty() {
            return Err(MissingnessErrorV1::EmptyHypothesisSet);
        }
        if self.hypothesis_ids.len() > 64 {
            return Err(MissingnessErrorV1::TooManyHypotheses);
        }
        let mut hypothesis_ids = BTreeSet::new();
        for id in &self.hypothesis_ids {
            if id.trim().is_empty() {
                return Err(MissingnessErrorV1::EmptyHypothesisId);
            }
            if !hypothesis_ids.insert(id) {
                return Err(MissingnessErrorV1::DuplicateHypothesisId);
            }
        }
        if self.probes.is_empty() {
            return Err(MissingnessErrorV1::EmptyProbeSet);
        }
        if self.probes.len() > 64 {
            return Err(MissingnessErrorV1::TooManyProbes);
        }
        let mut probe_ids = BTreeSet::new();
        for probe in &self.probes {
            if probe.probe_id.trim().is_empty() {
                return Err(MissingnessErrorV1::EmptyProbeId);
            }
            if !probe_ids.insert(&probe.probe_id) {
                return Err(MissingnessErrorV1::DuplicateProbeId);
            }
            if probe.cost == 0 {
                return Err(MissingnessErrorV1::ZeroProbeCost {
                    probe_id: probe.probe_id.clone(),
                });
            }
        }
        if self.abstention_possible_by_hypothesis.len() != self.hypothesis_ids.len()
            || self.impossibility_justifications_by_hypothesis.len()
                != self.hypothesis_ids.len()
        {
            return Err(MissingnessErrorV1::HypothesisCountMismatch);
        }
        for (hypothesis_index, hypothesis_id) in self.hypothesis_ids.iter().enumerate() {
            let possibilities = &self.abstention_possible_by_hypothesis[hypothesis_index];
            let justifications =
                &self.impossibility_justifications_by_hypothesis[hypothesis_index];
            if possibilities.len() != self.probes.len()
                || justifications.len() != self.probes.len()
            {
                return Err(MissingnessErrorV1::ProbeCountMismatch {
                    hypothesis_id: hypothesis_id.clone(),
                });
            }
            for (probe_index, possible) in possibilities.iter().enumerate() {
                let justification = &justifications[probe_index];
                let probe_id = &self.probes[probe_index].probe_id;
                match (*possible, justification) {
                    (true, Some(_)) => {
                        return Err(
                            MissingnessErrorV1::UnexpectedImpossibilityJustification {
                                hypothesis_id: hypothesis_id.clone(),
                                probe_id: probe_id.clone(),
                            },
                        );
                    }
                    (true, None) => {}
                    (false, None) => {
                        return Err(MissingnessErrorV1::MissingImpossibilityJustification {
                            hypothesis_id: hypothesis_id.clone(),
                            probe_id: probe_id.clone(),
                        });
                    }
                    (false, Some(justification)) => {
                        if justification.evidence_id.trim().is_empty() {
                            return Err(MissingnessErrorV1::EmptyEvidenceId {
                                hypothesis_id: hypothesis_id.clone(),
                                probe_id: probe_id.clone(),
                            });
                        }
                        if justification.validity_domain_id.trim().is_empty() {
                            return Err(MissingnessErrorV1::EmptyEvidenceDomain {
                                hypothesis_id: hypothesis_id.clone(),
                                probe_id: probe_id.clone(),
                            });
                        }
                        if justification.validity_domain_id != self.validity_domain_id {
                            return Err(MissingnessErrorV1::EvidenceDomainMismatch {
                                hypothesis_id: hypothesis_id.clone(),
                                probe_id: probe_id.clone(),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Canonical, versioned bytes. The evidence IDs are bound, not verified.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symmetry_conditioned_missingness.hard_support.v1\0".to_vec();
        append_string(&mut bytes, &self.problem_id);
        append_string(&mut bytes, &self.validity_domain_id);
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
        for (hypothesis_index, possibilities) in
            self.abstention_possible_by_hypothesis.iter().enumerate()
        {
            for (probe_index, possible) in possibilities.iter().enumerate() {
                bytes.push(u8::from(*possible));
                if let Some(justification) =
                    &self.impossibility_justifications_by_hypothesis[hypothesis_index][probe_index]
                {
                    append_string(&mut bytes, &justification.evidence_id);
                    bytes.push(justification.evidence_kind.tag());
                    append_string(&mut bytes, &justification.validity_domain_id);
                } else {
                    bytes.extend_from_slice(&0_u64.to_le_bytes());
                    bytes.push(0);
                    bytes.extend_from_slice(&0_u64.to_le_bytes());
                }
            }
        }
        bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingnessObservationV1 {
    pub probe_id: String,
    pub abstained: bool,
}

fn observation_bytes(observations: &[MissingnessObservationV1]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(observations.len() as u64).to_le_bytes());
    for observation in observations {
        append_string(&mut bytes, &observation.probe_id);
        bytes.push(u8::from(observation.abstained));
    }
    bytes
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardMissingnessSnapshotV1 {
    pub after_observation_count: usize,
    pub active_hypothesis_ids: Vec<String>,
    pub model_consistent: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardMissingnessReceiptV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub observation_contract_bytes: Vec<u8>,
    pub snapshots: Vec<HardMissingnessSnapshotV1>,
    pub final_active_hypothesis_ids: Vec<String>,
}

fn resolve_probe(
    probes: &[MissingnessProbeV1],
    probe_id: &str,
) -> Result<usize, MissingnessErrorV1> {
    let index = probes
        .iter()
        .position(|probe| probe.probe_id == probe_id)
        .ok_or_else(|| MissingnessErrorV1::UnknownProbe {
            probe_id: probe_id.to_owned(),
        })?;
    if !probes[index].admissible {
        return Err(MissingnessErrorV1::InadmissibleProbe {
            probe_id: probe_id.to_owned(),
        });
    }
    Ok(index)
}

/// Replay hard-support observations. A candidate is removed only when the
/// frozen contract supplies an explicit impossibility justification for the
/// observed abstention in the active validity domain. An empty set means the
/// declared missingness model is inconsistent with the observation history.
pub fn replay_hard_missingness(
    problem: &HypothesisConditionedMissingnessProblemV1,
    observations: &[MissingnessObservationV1],
) -> Result<HardMissingnessReceiptV1, MissingnessErrorV1> {
    problem.validate()?;
    let mut active = vec![true; problem.hypothesis_ids.len()];
    let mut snapshots = Vec::with_capacity(observations.len() + 1);
    snapshots.push(HardMissingnessSnapshotV1 {
        after_observation_count: 0,
        active_hypothesis_ids: problem.hypothesis_ids.clone(),
        model_consistent: true,
    });
    let mut seen = BTreeSet::new();

    for (observation_index, observation) in observations.iter().enumerate() {
        let probe_index = resolve_probe(&problem.probes, &observation.probe_id)?;
        if !seen.insert(probe_index) {
            return Err(MissingnessErrorV1::DuplicateProbeObservation {
                probe_id: observation.probe_id.clone(),
            });
        }
        if observation.abstained {
            for (hypothesis_index, is_active) in active.iter_mut().enumerate() {
                if *is_active
                    && !problem.abstention_possible_by_hypothesis[hypothesis_index][probe_index]
                {
                    *is_active = false;
                }
            }
        }
        let active_hypothesis_ids = problem
            .hypothesis_ids
            .iter()
            .zip(active.iter())
            .filter(|(_, is_active)| **is_active)
            .map(|(hypothesis_id, _)| hypothesis_id.clone())
            .collect::<Vec<_>>();
        snapshots.push(HardMissingnessSnapshotV1 {
            after_observation_count: observation_index + 1,
            model_consistent: !active_hypothesis_ids.is_empty(),
            active_hypothesis_ids,
        });
    }

    let final_active_hypothesis_ids = snapshots
        .last()
        .map(|snapshot| snapshot.active_hypothesis_ids.clone())
        .unwrap_or_default();
    Ok(HardMissingnessReceiptV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        observation_contract_bytes: observation_bytes(observations),
        snapshots,
        final_active_hypothesis_ids,
    })
}

pub fn verify_hard_missingness_receipt(
    problem: &HypothesisConditionedMissingnessProblemV1,
    observations: &[MissingnessObservationV1],
    receipt: &HardMissingnessReceiptV1,
) -> Result<(), MissingnessErrorV1> {
    if receipt.schema_version != 1
        || receipt.problem_id != problem.problem_id
        || receipt.problem_contract_bytes != problem.canonical_bytes()
        || receipt.observation_contract_bytes != observation_bytes(observations)
    {
        return Err(MissingnessErrorV1::ReceiptMismatch);
    }
    let replayed = replay_hard_missingness(problem, observations)?;
    if *receipt != replayed {
        return Err(MissingnessErrorV1::ReceiptMismatch);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DropoutProbabilityIntervalPpmV1 {
    pub lower_ppm: u32,
    pub upper_ppm: u32,
}

impl DropoutProbabilityIntervalPpmV1 {
    pub fn new(lower_ppm: u32, upper_ppm: u32) -> Result<Self, MissingnessErrorV1> {
        if lower_ppm > upper_ppm || upper_ppm > PROBABILITY_SCALE_PPM_V1 {
            return Err(MissingnessErrorV1::InvalidProbabilityInterval);
        }
        Ok(Self {
            lower_ppm,
            upper_ppm,
        })
    }

    fn complement(self) -> Self {
        Self {
            lower_ppm: PROBABILITY_SCALE_PPM_V1 - self.upper_ppm,
            upper_ppm: PROBABILITY_SCALE_PPM_V1 - self.lower_ppm,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbabilisticMissingnessProblemV1 {
    pub problem_id: String,
    pub validity_domain_id: String,
    pub calibration_evidence_id: String,
    pub independence_contract_id: String,
    pub hypothesis_ids: Vec<String>,
    pub probes: Vec<MissingnessProbeV1>,
    /// [hypothesis][probe] -> calibrated interval for dropout probability.
    pub dropout_probabilities_by_hypothesis: Vec<Vec<DropoutProbabilityIntervalPpmV1>>,
}

impl ProbabilisticMissingnessProblemV1 {
    pub fn validate(&self) -> Result<(), MissingnessErrorV1> {
        if self.problem_id.trim().is_empty() {
            return Err(MissingnessErrorV1::EmptyProblemId);
        }
        if self.validity_domain_id.trim().is_empty() {
            return Err(MissingnessErrorV1::EmptyValidityDomainId);
        }
        if self.calibration_evidence_id.trim().is_empty() {
            return Err(MissingnessErrorV1::EmptyCalibrationEvidenceId);
        }
        if self.independence_contract_id.trim().is_empty() {
            return Err(MissingnessErrorV1::EmptyIndependenceContractId);
        }
        if self.hypothesis_ids.is_empty() || self.hypothesis_ids.len() > 64 {
            return Err(MissingnessErrorV1::HypothesisCountMismatch);
        }
        let mut hypothesis_ids = BTreeSet::new();
        for id in &self.hypothesis_ids {
            if id.trim().is_empty() {
                return Err(MissingnessErrorV1::EmptyHypothesisId);
            }
            if !hypothesis_ids.insert(id) {
                return Err(MissingnessErrorV1::DuplicateHypothesisId);
            }
        }
        if self.probes.is_empty() {
            return Err(MissingnessErrorV1::EmptyProbeSet);
        }
        if self.probes.len() > 64 {
            return Err(MissingnessErrorV1::TooManyProbes);
        }
        let mut probe_ids = BTreeSet::new();
        for probe in &self.probes {
            if probe.probe_id.trim().is_empty() {
                return Err(MissingnessErrorV1::EmptyProbeId);
            }
            if !probe_ids.insert(&probe.probe_id) {
                return Err(MissingnessErrorV1::DuplicateProbeId);
            }
            if probe.cost == 0 {
                return Err(MissingnessErrorV1::ZeroProbeCost {
                    probe_id: probe.probe_id.clone(),
                });
            }
        }
        if self.dropout_probabilities_by_hypothesis.len() != self.hypothesis_ids.len() {
            return Err(MissingnessErrorV1::HypothesisCountMismatch);
        }
        for (hypothesis_index, probabilities) in
            self.dropout_probabilities_by_hypothesis.iter().enumerate()
        {
            if probabilities.len() != self.probes.len() {
                return Err(MissingnessErrorV1::ProbeCountMismatch {
                    hypothesis_id: self.hypothesis_ids[hypothesis_index].clone(),
                });
            }
            if probabilities.iter().any(|interval| {
                interval.lower_ppm > interval.upper_ppm
                    || interval.upper_ppm > PROBABILITY_SCALE_PPM_V1
            }) {
                return Err(MissingnessErrorV1::InvalidProbabilityInterval);
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes =
            b"luminous.symmetry_conditioned_missingness.probability_intervals.v1\0".to_vec();
        append_string(&mut bytes, &self.problem_id);
        append_string(&mut bytes, &self.validity_domain_id);
        append_string(&mut bytes, &self.calibration_evidence_id);
        append_string(&mut bytes, &self.independence_contract_id);
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
        for probabilities in &self.dropout_probabilities_by_hypothesis {
            for interval in probabilities {
                bytes.extend_from_slice(&interval.lower_ppm.to_le_bytes());
                bytes.extend_from_slice(&interval.upper_ppm.to_le_bytes());
            }
        }
        bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropoutLikelihoodIntervalV1 {
    pub hypothesis_id: String,
    pub lower_ppm: u32,
    pub upper_ppm: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbabilisticMissingnessReceiptV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub observation_contract_bytes: Vec<u8>,
    pub conditional_independence_assumed: bool,
    pub dropout_likelihoods: Vec<DropoutLikelihoodIntervalV1>,
}

fn multiply_probability_intervals(
    left: DropoutProbabilityIntervalPpmV1,
    right: DropoutProbabilityIntervalPpmV1,
) -> Result<DropoutProbabilityIntervalPpmV1, MissingnessErrorV1> {
    let scale = u128::from(PROBABILITY_SCALE_PPM_V1);
    let lower_product = u128::from(left.lower_ppm)
        .checked_mul(u128::from(right.lower_ppm))
        .ok_or(MissingnessErrorV1::ProbabilityArithmeticOverflow)?;
    let upper_product = u128::from(left.upper_ppm)
        .checked_mul(u128::from(right.upper_ppm))
        .ok_or(MissingnessErrorV1::ProbabilityArithmeticOverflow)?;
    let lower = lower_product / scale;
    let upper = upper_product
        .checked_add(scale - 1)
        .ok_or(MissingnessErrorV1::ProbabilityArithmeticOverflow)?
        / scale;
    if upper > scale {
        return Err(MissingnessErrorV1::ProbabilityArithmeticOverflow);
    }
    Ok(DropoutProbabilityIntervalPpmV1 {
        lower_ppm: lower as u32,
        upper_ppm: upper as u32,
    })
}

/// Return likelihood intervals for the observed dropout/non-dropout pattern.
/// This assumes conditional independence across the distinct probes, as
/// explicitly named by independence_contract_id. It never eliminates a
/// hypothesis or converts a small estimated probability into hard support.
pub fn replay_probabilistic_missingness(
    problem: &ProbabilisticMissingnessProblemV1,
    observations: &[MissingnessObservationV1],
) -> Result<ProbabilisticMissingnessReceiptV1, MissingnessErrorV1> {
    problem.validate()?;
    let mut seen = BTreeSet::new();
    let mut likelihoods = Vec::with_capacity(problem.hypothesis_ids.len());
    for (hypothesis_index, hypothesis_id) in problem.hypothesis_ids.iter().enumerate() {
        let mut likelihood = DropoutProbabilityIntervalPpmV1 {
            lower_ppm: PROBABILITY_SCALE_PPM_V1,
            upper_ppm: PROBABILITY_SCALE_PPM_V1,
        };
        for observation in observations {
            let probe_index = resolve_probe(&problem.probes, &observation.probe_id)?;
            if !seen.insert(probe_index) {
                return Err(MissingnessErrorV1::DuplicateProbeObservation {
                    probe_id: observation.probe_id.clone(),
                });
            }
            let dropout = problem.dropout_probabilities_by_hypothesis[hypothesis_index]
                [probe_index];
            let event_probability = if observation.abstained {
                dropout
            } else {
                dropout.complement()
            };
            likelihood = multiply_probability_intervals(likelihood, event_probability)?;
        }
        // Reset the distinct-probe set for the next candidate.
        seen.clear();
        likelihoods.push(DropoutLikelihoodIntervalV1 {
            hypothesis_id: hypothesis_id.clone(),
            lower_ppm: likelihood.lower_ppm,
            upper_ppm: likelihood.upper_ppm,
        });
    }

    let mut obs_bytes = Vec::new();
    obs_bytes.extend_from_slice(&(observations.len() as u64).to_le_bytes());
    for observation in observations {
        append_string(&mut obs_bytes, &observation.probe_id);
        obs_bytes.push(u8::from(observation.abstained));
    }

    Ok(ProbabilisticMissingnessReceiptV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        observation_contract_bytes: obs_bytes,
        conditional_independence_assumed: true,
        dropout_likelihoods: likelihoods,
    })
}

pub fn verify_probabilistic_missingness_receipt(
    problem: &ProbabilisticMissingnessProblemV1,
    observations: &[MissingnessObservationV1],
    receipt: &ProbabilisticMissingnessReceiptV1,
) -> Result<(), MissingnessErrorV1> {
    if receipt.schema_version != 1
        || receipt.problem_id != problem.problem_id
        || receipt.problem_contract_bytes != problem.canonical_bytes()
    {
        return Err(MissingnessErrorV1::ReceiptMismatch);
    }
    let replayed = replay_probabilistic_missingness(problem, observations)?;
    if *receipt != replayed {
        return Err(MissingnessErrorV1::ReceiptMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hard_support_fixture() -> HypothesisConditionedMissingnessProblemV1 {
        HypothesisConditionedMissingnessProblemV1 {
            problem_id: "sensor-dropout-hard-support-v1".to_owned(),
            validity_domain_id: "lab-envelope-load-20-80-temp-10-40-v1".to_owned(),
            hypothesis_ids: vec!["H-normal".to_owned(), "H-overload".to_owned()],
            probes: vec![
                MissingnessProbeV1 {
                    probe_id: "strain-gauge".to_owned(),
                    cost: 4,
                    admissible: true,
                },
                MissingnessProbeV1 {
                    probe_id: "thermal-camera".to_owned(),
                    cost: 2,
                    admissible: true,
                },
            ],
            abstention_possible_by_hypothesis: vec![vec![false, true], vec![true, true]],
            impossibility_justifications_by_hypothesis: vec![
                vec![
                    Some(HardSupportJustificationV1 {
                        evidence_id: "interlock-test-report-2026-11-12".to_owned(),
                        evidence_kind: HardSupportEvidenceKindV1::EngineeringInterlock,
                        validity_domain_id: "lab-envelope-load-20-80-temp-10-40-v1".to_owned(),
                    }),
                    None,
                ],
                vec![None, None],
            ],
        }
    }

    fn probabilistic_fixture() -> ProbabilisticMissingnessProblemV1 {
        ProbabilisticMissingnessProblemV1 {
            problem_id: "sensor-dropout-probability-v1".to_owned(),
            validity_domain_id: "calibration-site-A-load-20-80-v1".to_owned(),
            calibration_evidence_id: "dropout-calibration-split-2026-11-14".to_owned(),
            independence_contract_id: "distinct-probe-dropout-independent-given-hypothesis-v1"
                .to_owned(),
            hypothesis_ids: vec!["H-normal".to_owned(), "H-overload".to_owned()],
            probes: vec![
                MissingnessProbeV1 {
                    probe_id: "strain-gauge".to_owned(),
                    cost: 4,
                    admissible: true,
                },
                MissingnessProbeV1 {
                    probe_id: "thermal-camera".to_owned(),
                    cost: 2,
                    admissible: true,
                },
            ],
            dropout_probabilities_by_hypothesis: vec![
                vec![
                    DropoutProbabilityIntervalPpmV1::new(1, 1_000).unwrap(),
                    DropoutProbabilityIntervalPpmV1::new(100, 2_000).unwrap(),
                ],
                vec![
                    DropoutProbabilityIntervalPpmV1::new(100_000, 200_000).unwrap(),
                    DropoutProbabilityIntervalPpmV1::new(300_000, 500_000).unwrap(),
                ],
            ],
        }
    }

    #[test]
    fn hard_support_dropout_eliminates_only_the_candidate_with_a_bound_reference() {
        let problem = hard_support_fixture();
        problem.validate().unwrap();
        let observations = vec![MissingnessObservationV1 {
            probe_id: "strain-gauge".to_owned(),
            abstained: true,
        }];
        let receipt = replay_hard_missingness(&problem, &observations).unwrap();
        assert_eq!(receipt.final_active_hypothesis_ids, vec!["H-overload"]);
        verify_hard_missingness_receipt(&problem, &observations, &receipt).unwrap();
    }

    #[test]
    fn ordinary_measurement_does_not_create_missingness_evidence() {
        let problem = hard_support_fixture();
        let observations = vec![MissingnessObservationV1 {
            probe_id: "strain-gauge".to_owned(),
            abstained: false,
        }];
        let receipt = replay_hard_missingness(&problem, &observations).unwrap();
        assert_eq!(
            receipt.final_active_hypothesis_ids,
            problem.hypothesis_ids
        );
    }

    #[test]
    fn impossible_dropout_without_justification_is_rejected() {
        let mut problem = hard_support_fixture();
        problem.impossibility_justifications_by_hypothesis[0][0] = None;
        assert_eq!(
            problem.validate(),
            Err(MissingnessErrorV1::MissingImpossibilityJustification {
                hypothesis_id: "H-normal".to_owned(),
                probe_id: "strain-gauge".to_owned(),
            })
        );
    }

    #[test]
    fn evidence_outside_the_declared_validity_domain_is_rejected() {
        let mut problem = hard_support_fixture();
        problem.impossibility_justifications_by_hypothesis[0][0]
            .as_mut()
            .unwrap()
            .validity_domain_id = "different-site-v1".to_owned();
        assert_eq!(
            problem.validate(),
            Err(MissingnessErrorV1::EvidenceDomainMismatch {
                hypothesis_id: "H-normal".to_owned(),
                probe_id: "strain-gauge".to_owned(),
            })
        );
    }

    #[test]
    fn all_hypotheses_excluding_dropout_yields_model_mismatch_not_physical_proof() {
        let mut problem = hard_support_fixture();
        problem.abstention_possible_by_hypothesis[1][0] = false;
        problem.impossibility_justifications_by_hypothesis[1][0] =
            Some(HardSupportJustificationV1 {
                evidence_id: "overload-hardening-study-v1".to_owned(),
                evidence_kind: HardSupportEvidenceKindV1::ProvenPhysicalBound,
                validity_domain_id: problem.validity_domain_id.clone(),
            });
        let observations = vec![MissingnessObservationV1 {
            probe_id: "strain-gauge".to_owned(),
            abstained: true,
        }];
        let receipt = replay_hard_missingness(&problem, &observations).unwrap();
        assert!(receipt.final_active_hypothesis_ids.is_empty());
        assert!(!receipt.snapshots[1].model_consistent);
    }

    #[test]
    fn repeated_probe_observation_is_rejected_without_an_independence_contract() {
        let problem = hard_support_fixture();
        let observations = vec![
            MissingnessObservationV1 {
                probe_id: "strain-gauge".to_owned(),
                abstained: true,
            },
            MissingnessObservationV1 {
                probe_id: "strain-gauge".to_owned(),
                abstained: true,
            },
        ];
        assert_eq!(
            replay_hard_missingness(&problem, &observations),
            Err(MissingnessErrorV1::DuplicateProbeObservation {
                probe_id: "strain-gauge".to_owned()
            })
        );
    }

    #[test]
    fn hard_support_receipt_binds_evidence_references_and_rejects_forgery() {
        let problem = hard_support_fixture();
        let observations = vec![MissingnessObservationV1 {
            probe_id: "strain-gauge".to_owned(),
            abstained: true,
        }];
        let receipt = replay_hard_missingness(&problem, &observations).unwrap();
        verify_hard_missingness_receipt(&problem, &observations, &receipt).unwrap();
        let mut changed = problem.clone();
        changed.impossibility_justifications_by_hypothesis[0][0]
            .as_mut()
            .unwrap()
            .evidence_id = "mutated-reference".to_owned();
        assert_eq!(
            verify_hard_missingness_receipt(&changed, &observations, &receipt),
            Err(MissingnessErrorV1::ReceiptMismatch)
        );
        let mut forged = receipt;
        forged.final_active_hypothesis_ids.clear();
        assert_eq!(
            verify_hard_missingness_receipt(&problem, &observations, &forged),
            Err(MissingnessErrorV1::ReceiptMismatch)
        );
    }

    #[test]
    fn low_but_nonzero_dropout_probability_never_deletes_a_hypothesis() {
        let problem = probabilistic_fixture();
        let observations = vec![MissingnessObservationV1 {
            probe_id: "strain-gauge".to_owned(),
            abstained: true,
        }];
        let receipt = replay_probabilistic_missingness(&problem, &observations).unwrap();
        assert_eq!(receipt.dropout_likelihoods.len(), 2);
        assert_eq!(
            receipt.dropout_likelihoods[0].lower_ppm,
            0,
            "the calibrated dropout likelihood is small but strictly nonzero"
        );
        assert!(receipt.dropout_likelihoods[0].upper_ppm > 0);
        assert!(receipt.dropout_likelihoods[1].lower_ppm > 0);
        assert!(receipt.conditional_independence_assumed);
    }

    #[test]
    fn non_abstention_uses_the_complement_dropout_interval() {
        let problem = probabilistic_fixture();
        let observations = vec![MissingnessObservationV1 {
            probe_id: "strain-gauge".to_owned(),
            abstained: false,
        }];
        let receipt = replay_probabilistic_missingness(&problem, &observations).unwrap();
        assert_eq!(receipt.dropout_likelihoods[0].lower_ppm, 999_000);
        assert_eq!(receipt.dropout_likelihoods[0].upper_ppm, 1_000_000);
        assert_eq!(receipt.dropout_likelihoods[1].lower_ppm, 800_000);
        assert_eq!(receipt.dropout_likelihoods[1].upper_ppm, 900_000);
    }

    #[test]
    fn independent_dropout_likelihoods_multiply_with_outward_rounding() {
        let problem = probabilistic_fixture();
        let observations = vec![
            MissingnessObservationV1 {
                probe_id: "strain-gauge".to_owned(),
                abstained: true,
            },
            MissingnessObservationV1 {
                probe_id: "thermal-camera".to_owned(),
                abstained: false,
            },
        ];
        let receipt = replay_probabilistic_missingness(&problem, &observations).unwrap();
        let normal = &receipt.dropout_likelihoods[0];
        let overload = &receipt.dropout_likelihoods[1];
        assert!(normal.lower_ppm <= 0);
        assert!(normal.upper_ppm > 0);
        assert!(overload.lower_ppm > normal.lower_ppm);
        assert!(overload.upper_ppm > normal.upper_ppm);
    }

    #[test]
    fn probabilistic_model_requires_calibration_and_independence_contracts() {
        let mut problem = probabilistic_fixture();
        problem.calibration_evidence_id.clear();
        assert_eq!(
            problem.validate(),
            Err(MissingnessErrorV1::EmptyCalibrationEvidenceId)
        );
        problem.calibration_evidence_id = "calibration-ref".to_owned();
        problem.independence_contract_id.clear();
        assert_eq!(
            problem.validate(),
            Err(MissingnessErrorV1::EmptyIndependenceContractId)
        );
    }

    #[test]
    fn probabilistic_receipt_rejects_forgery_and_duplicate_attempts() {
        let problem = probabilistic_fixture();
        let observations = vec![MissingnessObservationV1 {
            probe_id: "strain-gauge".to_owned(),
            abstained: true,
        }];
        let receipt = replay_probabilistic_missingness(&problem, &observations).unwrap();
        verify_probabilistic_missingness_receipt(&problem, &observations, &receipt).unwrap();
        let mut forged = receipt;
        forged.dropout_likelihoods[0].upper_ppm = 0;
        assert_eq!(
            verify_probabilistic_missingness_receipt(&problem, &observations, &forged),
            Err(MissingnessErrorV1::ReceiptMismatch)
        );
        let repeated = vec![observations[0].clone(), observations[0].clone()];
        assert_eq!(
            replay_probabilistic_missingness(&problem, &repeated),
            Err(MissingnessErrorV1::DuplicateProbeObservation {
                probe_id: "strain-gauge".to_owned()
            })
        );
    }

    #[test]
    fn probability_intervals_reject_reversal_and_values_above_one() {
        assert_eq!(
            DropoutProbabilityIntervalPpmV1::new(5, 4),
            Err(MissingnessErrorV1::InvalidProbabilityInterval)
        );
        assert_eq!(
            DropoutProbabilityIntervalPpmV1::new(0, 1_000_001),
            Err(MissingnessErrorV1::InvalidProbabilityInterval)
        );
    }
}
