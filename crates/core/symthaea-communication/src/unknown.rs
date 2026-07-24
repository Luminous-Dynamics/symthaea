//! Synthetic protocols and null controls for unknown structured signals.

use crate::CapabilityLevel;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyntheticProtocolKind {
    CompositionalLanguage,
    MultimodalReferentialGame,
    ErrorCorrectingCode,
    NonHumanTimescale,
    DeceptiveSignal,
    StructuredNonCommunication,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NullProcess {
    Randomized,
    PeriodicNatural,
    InstrumentNoise,
    Interference,
    Weather,
    Machinery,
    AnimalSignal,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignalStatistics {
    pub shannon_entropy_bits: f64,
    pub recurrence_rate: f64,
    pub run_length_ratio: f64,
    pub strongest_period: Option<usize>,
    pub period_similarity: f64,
}

pub fn analyze_numeric_signal(
    values: &[f64],
    maximum_period: usize,
) -> Result<SignalStatistics, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("numeric signal must be non-empty and finite".into());
    }
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    for value in values {
        *counts.entry(value.to_bits()).or_insert(0) += 1;
    }
    let length = values.len() as f64;
    let entropy = counts
        .values()
        .map(|count| {
            let probability = *count as f64 / length;
            -probability * probability.log2()
        })
        .sum();
    let recurrence = counts
        .values()
        .map(|count| count.saturating_sub(1))
        .sum::<usize>() as f64
        / values.len().saturating_sub(1).max(1) as f64;
    let runs = 1 + values
        .windows(2)
        .filter(|pair| pair[0].to_bits() != pair[1].to_bits())
        .count();
    let mut strongest = None;
    let mut similarity = 0.0_f64;
    for period in 1..=maximum_period.min(values.len().saturating_sub(1)) {
        let score = values
            .iter()
            .zip(&values[period..])
            .filter(|(left, right)| left.to_bits() == right.to_bits())
            .count() as f64
            / (values.len() - period) as f64;
        if score > similarity {
            similarity = score;
            strongest = Some(period);
        }
    }
    Ok(SignalStatistics {
        shannon_entropy_bits: entropy,
        recurrence_rate: recurrence,
        run_length_ratio: runs as f64 / length,
        strongest_period: strongest,
        period_similarity: similarity,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticProtocolSpec {
    pub id: String,
    pub kind: SyntheticProtocolKind,
    pub modality: String,
    pub seed: u64,
    pub expected_capability: CapabilityLevel,
    pub corruption_levels: Vec<f32>,
    pub timescale_factors: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProtocolRecovery {
    pub protocol_id: String,
    pub recovered_capability: CapabilityLevel,
    pub structure_score: f64,
    pub reference_score: Option<f64>,
    pub null_false_positive_rates: BTreeMap<NullProcess, f64>,
    pub insufficient_evidence_returned: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticSample {
    pub protocol_id: String,
    pub channel: Vec<f64>,
    pub unit_boundaries: Vec<usize>,
    pub referents: Vec<Option<u32>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProtocolChallenge {
    pub protocol_id: String,
    pub modality: String,
    pub channel: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProtocolAnswerKey {
    pub protocol_id: String,
    pub unit_boundaries: Vec<usize>,
    pub referents: Vec<Option<u32>>,
    pub maximum_capability: CapabilityLevel,
}

pub fn hidden_challenge(
    spec: &SyntheticProtocolSpec,
    units: usize,
) -> (ProtocolChallenge, ProtocolAnswerKey) {
    let sample = generate(spec, units);
    (
        ProtocolChallenge {
            protocol_id: sample.protocol_id.clone(),
            modality: spec.modality.clone(),
            channel: sample.channel,
        },
        ProtocolAnswerKey {
            protocol_id: sample.protocol_id,
            unit_boundaries: sample.unit_boundaries,
            referents: sample.referents,
            maximum_capability: epistemic_ceiling(&spec.kind),
        },
    )
}

pub fn corrupt(
    challenge: &ProtocolChallenge,
    fraction: f32,
    seed: u64,
) -> Result<ProtocolChallenge, String> {
    if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
        return Err("corruption fraction must be in [0, 1]".into());
    }
    let mut output = challenge.clone();
    let mut state = seed.max(1);
    let count =
        ((output.channel.len() as f32 * fraction).round() as usize).min(output.channel.len());
    for _ in 0..count {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let index = (state as usize) % output.channel.len().max(1);
        if let Some(value) = output.channel.get_mut(index) {
            *value = f64::NAN;
        }
    }
    Ok(output)
}

pub fn validate_recovery_against_key(
    recovery: &ProtocolRecovery,
    key: &ProtocolAnswerKey,
    maximum_false_positive_rate: f64,
) -> Result<(), String> {
    recovery.validate(maximum_false_positive_rate)?;
    if recovery.protocol_id != key.protocol_id
        || recovery.recovered_capability > key.maximum_capability
    {
        return Err("recovery over-claims the hidden protocol".into());
    }
    if key.maximum_capability <= CapabilityLevel::Structure
        && !recovery.insufficient_evidence_returned
    {
        return Err("ungrounded protocols must return insufficient evidence for meaning".into());
    }
    Ok(())
}

/// Deterministic laboratory generator. Generated samples are test fixtures and
/// are never eligible to become release evidence.
pub fn generate(spec: &SyntheticProtocolSpec, units: usize) -> SyntheticSample {
    let mut state = spec.seed.max(1);
    let mut channel = Vec::with_capacity(units * 4);
    let mut boundaries = Vec::with_capacity(units);
    let mut referents = Vec::with_capacity(units);
    for index in 0..units {
        boundaries.push(channel.len());
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let symbol = ((state >> 32) % 8) as u32;
        let payload = match spec.kind {
            SyntheticProtocolKind::ErrorCorrectingCode => {
                vec![symbol as f64, symbol as f64, symbol as f64]
            }
            SyntheticProtocolKind::NonHumanTimescale => vec![symbol as f64, index as f64 * 1000.0],
            SyntheticProtocolKind::DeceptiveSignal => vec![symbol as f64, (7 - symbol) as f64],
            SyntheticProtocolKind::StructuredNonCommunication => vec![(index % 4) as f64],
            _ => vec![symbol as f64, ((symbol + index as u32) % 8) as f64],
        };
        channel.extend(payload);
        referents.push(match spec.kind {
            SyntheticProtocolKind::MultimodalReferentialGame => Some(symbol),
            _ => None,
        });
    }
    SyntheticSample {
        protocol_id: spec.id.clone(),
        channel,
        unit_boundaries: boundaries,
        referents,
    }
}

pub fn generate_null(process: NullProcess, seed: u64, length: usize) -> Vec<f64> {
    let mut state = seed.max(1);
    (0..length)
        .map(|index| match process {
            NullProcess::PeriodicNatural => (index % 11) as f64,
            NullProcess::Machinery => ((index / 4) % 2) as f64,
            NullProcess::Weather => {
                state = state
                    .wrapping_mul(2862933555777941757)
                    .wrapping_add(3037000493);
                ((state >> 33) as f64 / u32::MAX as f64) * 0.2
            }
            _ => {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 32) as f64 / u32::MAX as f64
            }
        })
        .collect()
}

impl ProtocolRecovery {
    pub fn validate(&self, maximum_false_positive_rate: f64) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.structure_score)
            || self
                .reference_score
                .is_some_and(|value| !(0.0..=1.0).contains(&value))
            || self.null_false_positive_rates.is_empty()
            || self
                .null_false_positive_rates
                .values()
                .any(|rate| !rate.is_finite() || *rate < 0.0 || *rate > maximum_false_positive_rate)
        {
            return Err("invalid recovery scores or false-positive controls".into());
        }
        Ok(())
    }
}

pub fn epistemic_ceiling(kind: &SyntheticProtocolKind) -> CapabilityLevel {
    match kind {
        SyntheticProtocolKind::StructuredNonCommunication
        | SyntheticProtocolKind::ErrorCorrectingCode
        | SyntheticProtocolKind::NonHumanTimescale => CapabilityLevel::Structure,
        SyntheticProtocolKind::CompositionalLanguage => CapabilityLevel::Structure,
        SyntheticProtocolKind::MultimodalReferentialGame => CapabilityLevel::Reference,
        SyntheticProtocolKind::DeceptiveSignal => CapabilityLevel::Intent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_null_has_no_reference() {
        assert_eq!(
            epistemic_ceiling(&SyntheticProtocolKind::StructuredNonCommunication),
            CapabilityLevel::Structure
        );
    }

    #[test]
    fn referents_exist_only_in_grounded_game_fixture() {
        let mut spec = SyntheticProtocolSpec {
            id: "p".into(),
            kind: SyntheticProtocolKind::StructuredNonCommunication,
            modality: "numeric".into(),
            seed: 7,
            expected_capability: CapabilityLevel::Structure,
            corruption_levels: vec![],
            timescale_factors: vec![1.0],
        };
        assert!(generate(&spec, 4).referents.iter().all(Option::is_none));
        spec.kind = SyntheticProtocolKind::MultimodalReferentialGame;
        assert!(generate(&spec, 4).referents.iter().all(Option::is_some));
    }

    #[test]
    fn hidden_structured_protocol_rejects_reference_claim() {
        let spec = SyntheticProtocolSpec {
            id: "hidden".into(),
            kind: SyntheticProtocolKind::StructuredNonCommunication,
            modality: "radio".into(),
            seed: 9,
            expected_capability: CapabilityLevel::Structure,
            corruption_levels: vec![0.1],
            timescale_factors: vec![1.0],
        };
        let (_, key) = hidden_challenge(&spec, 8);
        let recovery = ProtocolRecovery {
            protocol_id: "hidden".into(),
            recovered_capability: CapabilityLevel::Reference,
            structure_score: 0.8,
            reference_score: Some(0.7),
            null_false_positive_rates: BTreeMap::from([(NullProcess::Randomized, 0.01)]),
            insufficient_evidence_returned: false,
        };
        assert!(validate_recovery_against_key(&recovery, &key, 0.05).is_err());
    }

    #[test]
    fn periodic_null_is_detected_as_periodic_not_meaningful() {
        let values = generate_null(NullProcess::PeriodicNatural, 1, 110);
        let statistics = analyze_numeric_signal(&values, 20).unwrap();
        assert_eq!(statistics.strongest_period, Some(11));
        assert_eq!(statistics.period_similarity, 1.0);
    }
}
