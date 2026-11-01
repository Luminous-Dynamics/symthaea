//! Counter-based bounded synthetic noise with explicit cross-objective correlation.
//!
//! This module supplies reproducible common-random-number observations for matched MOO
//! policy comparisons. It is a synthetic benchmark generator, not a claim about real-world
//! sample independence. One shared Rademacher factor plus per-objective idiosyncratic
//! Rademacher factors gives a known covariance matrix and bounded observations.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, error::Error, fmt};

const SCHEMA: &str = "symthaea.architecture-engineering.correlated-bounded-noise";
const VERSION: &str = "0.1.0";
const MAX_OBJECTIVES: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorrelatedNoiseError { pub path: String, pub message: String }
impl CorrelatedNoiseError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}
impl fmt::Display for CorrelatedNoiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}
impl Error for CorrelatedNoiseError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorrelatedNoiseObjective {
    pub objective_id: String,
    /// Positive marginal standard deviation scale.
    pub scale: f64,
    /// Loading on the common Rademacher factor; must lie in [-1, 1].
    pub shared_loading: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorrelatedBoundedNoiseModel {
    pub schema: String,
    pub version: String,
    pub model_id: String,
    /// Objective ordering is semantic and must match the EventTape objective contract.
    pub objectives: Vec<CorrelatedNoiseObjective>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorrelatedNoiseObservation {
    pub model_sha256: String,
    pub campaign_seed: u64,
    pub candidate_id: String,
    pub replicate_id: String,
    pub noise_values: Vec<f64>,
    pub observation_values: Vec<f64>,
    pub observation_sha256: String,
}

impl CorrelatedBoundedNoiseModel {
    pub fn new(
        model_id: impl Into<String>,
        objectives: Vec<CorrelatedNoiseObjective>,
    ) -> Result<Self, CorrelatedNoiseError> {
        let model = Self { schema: SCHEMA.into(), version: VERSION.into(),
            model_id: model_id.into(), objectives };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), CorrelatedNoiseError> {
        if self.schema != SCHEMA || self.version != VERSION {
            return Err(CorrelatedNoiseError::new("model.schema", "unsupported schema/version"));
        }
        if self.model_id.trim().is_empty() {
            return Err(CorrelatedNoiseError::new("model.model_id", "model ID must be non-empty"));
        }
        if !(2..=MAX_OBJECTIVES).contains(&self.objectives.len()) {
            return Err(CorrelatedNoiseError::new("model.objectives", "objective count must be between 2 and 16"));
        }
        let mut ids = BTreeSet::new();
        for (index, objective) in self.objectives.iter().enumerate() {
            if objective.objective_id.trim().is_empty() || !ids.insert(objective.objective_id.as_str()) {
                return Err(CorrelatedNoiseError::new(
                    format!("model.objectives[{index}].objective_id"),
                    "objective IDs must be non-empty and unique",
                ));
            }
            if !objective.scale.is_finite() || objective.scale <= 0.0 {
                return Err(CorrelatedNoiseError::new(
                    format!("model.objectives[{index}].scale"),
                    "noise scale must be finite and positive",
                ));
            }
            if !objective.shared_loading.is_finite() || !(-1.0..=1.0).contains(&objective.shared_loading) {
                return Err(CorrelatedNoiseError::new(
                    format!("model.objectives[{index}].shared_loading"),
                    "shared loading must be finite and lie in [-1, 1]",
                ));
            }
            let bound = objective.scale * (objective.shared_loading.abs()
                + (1.0 - objective.shared_loading * objective.shared_loading).sqrt());
            if !bound.is_finite() || bound <= 0.0 {
                return Err(CorrelatedNoiseError::new(
                    format!("model.objectives[{index}].bound"),
                    "derived noise bound is non-finite or non-positive",
                ));
            }
        }
        Ok(())
    }

    pub fn model_sha256(&self) -> Result<String, CorrelatedNoiseError> {
        self.validate()?;
        digest(self)
    }

    /// Maximum absolute noise for each objective under the bounded factor construction.
    pub fn noise_bounds(&self) -> Result<Vec<f64>, CorrelatedNoiseError> {
        self.validate()?;
        Ok(self.objectives.iter().map(|objective| {
            objective.scale * (objective.shared_loading.abs()
                + (1.0 - objective.shared_loading * objective.shared_loading).sqrt())
        }).collect())
    }

    /// Known correlation matrix: Corr(e_i,e_j) = loading_i * loading_j.
    pub fn theoretical_correlation_matrix(&self) -> Result<Vec<Vec<f64>>, CorrelatedNoiseError> {
        self.validate()?;
        Ok(self.objectives.iter().enumerate().map(|(row, left)| {
            self.objectives.iter().enumerate().map(|(column, right)| {
                if row == column { 1.0 } else { left.shared_loading * right.shared_loading }
            }).collect()
        }).collect())
    }

    /// Deterministic, policy-independent potential observation for matched campaigns.
    ///
    /// The counter key intentionally excludes policy identity. Two methods evaluating the
    /// same candidate/replicate under the same campaign seed receive identical noise, enabling
    /// paired comparisons with common random numbers. Changing a candidate or replicate changes
    /// the counter stream. This is a synthetic pseudorandom construction, not proof that physical
    /// measurements in a real campaign are statistically independent.
    pub fn observe(
        &self,
        campaign_seed: u64,
        candidate_id: &str,
        replicate_id: &str,
        objective_means: &[f64],
    ) -> Result<CorrelatedNoiseObservation, CorrelatedNoiseError> {
        self.validate()?;
        if candidate_id.trim().is_empty() || replicate_id.trim().is_empty() {
            return Err(CorrelatedNoiseError::new("observation.identity", "candidate and replicate IDs must be non-empty"));
        }
        if objective_means.len() != self.objectives.len()
            || objective_means.iter().any(|value| !value.is_finite())
        {
            return Err(CorrelatedNoiseError::new(
                "observation.objective_means",
                "means must be finite and match the frozen objective dimension",
            ));
        }
        let model_sha256 = self.model_sha256()?;
        let counter = counter_key(campaign_seed, candidate_id, replicate_id);
        let shared = sign(splitmix64(counter ^ 0xa076_1d64_78bd_642f));
        let mut noise_values = Vec::with_capacity(self.objectives.len());
        let mut observation_values = Vec::with_capacity(self.objectives.len());
        for (index, objective) in self.objectives.iter().enumerate() {
            let idiosyncratic = sign(splitmix64(
                counter ^ 0xe703_7ed1_a0b4_28db ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ));
            let residual_loading = (1.0 - objective.shared_loading.powi(2)).sqrt();
            let noise = objective.scale * (
                objective.shared_loading * shared + residual_loading * idiosyncratic
            );
            let observed = objective_means[index] + noise;
            if !noise.is_finite() || !observed.is_finite() {
                return Err(CorrelatedNoiseError::new(
                    format!("observation.objective_values[{index}]"),
                    "noise or observed objective overflowed",
                ));
            }
            noise_values.push(noise);
            observation_values.push(observed);
        }
        let payload = ObservationDigestPayload {
            model_sha256: &model_sha256, campaign_seed, candidate_id, replicate_id,
            noise_values: &noise_values, observation_values: &observation_values,
        };
        let observation_sha256 = digest(&payload)?;
        Ok(CorrelatedNoiseObservation {
            model_sha256, campaign_seed, candidate_id: candidate_id.into(),
            replicate_id: replicate_id.into(), noise_values, observation_values,
            observation_sha256,
        })
    }
}

#[derive(Serialize)]
struct ObservationDigestPayload<'a> {
    model_sha256: &'a str,
    campaign_seed: u64,
    candidate_id: &'a str,
    replicate_id: &'a str,
    noise_values: &'a [f64],
    observation_values: &'a [f64],
}

fn counter_key(seed: u64, candidate_id: &str, replicate_id: &str) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-correlated-bounded-noise-counter-v0.1");
    hasher.update(seed.to_le_bytes());
    hasher.update((candidate_id.len() as u64).to_le_bytes());
    hasher.update(candidate_id.as_bytes());
    hasher.update((replicate_id.len() as u64).to_le_bytes());
    hasher.update(replicate_id.as_bytes());
    let digest = hasher.finalize();
    let mut prefix = [0_u8; 8];
    prefix.copy_from_slice(&digest[0..8]);
    u64::from_le_bytes(prefix)
}

fn sign(value: u64) -> f64 {
    if value & 1 == 0 { -1.0 } else { 1.0 }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn digest<T: Serialize>(value: &T) -> Result<String, CorrelatedNoiseError> {
    let bytes = serde_json::to_vec(value).map_err(|e| {
        CorrelatedNoiseError::new("serialization", e.to_string())
    })?;
    Ok(Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CorrelatedBoundedNoiseModel {
        CorrelatedBoundedNoiseModel::new("shared-latent-noise-v1", vec![
            CorrelatedNoiseObjective { objective_id: "quality".into(), scale: 0.2, shared_loading: 0.8 },
            CorrelatedNoiseObjective { objective_id: "cost".into(), scale: 0.4, shared_loading: -0.5 },
            CorrelatedNoiseObjective { objective_id: "energy".into(), scale: 0.1, shared_loading: 0.0 },
        ]).unwrap()
    }

    #[test]
    fn theoretical_covariance_is_known_and_psd_by_construction() {
        let model = model();
        let matrix = model.theoretical_correlation_matrix().unwrap();
        assert_eq!(matrix.len(), 3);
        assert!((matrix[0][0] - 1.0).abs() < 1e-12);
        assert!((matrix[0][1] + 0.4).abs() < 1e-12);
        assert!((matrix[0][2]).abs() < 1e-12);
        assert!((matrix[1][1] - 1.0).abs() < 1e-12);
        assert!((matrix[1][2]).abs() < 1e-12);
        assert!((matrix[2][2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn common_random_numbers_are_policy_independent_and_replayable() {
        let model = model();
        let means = [0.5, 10.0, 2.0];
        let first = model.observe(42, "candidate-sha", "replicate-1", &means).unwrap();
        let second = model.observe(42, "candidate-sha", "replicate-1", &means).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.observation_values.len(), 3);
        assert_eq!(first.model_sha256, model.model_sha256().unwrap());
    }

    #[test]
    fn generated_noise_respects_declared_bounds_and_keeps_objective_order() {
        let model = model();
        let bounds = model.noise_bounds().unwrap();
        for seed in 0..1024_u64 {
            let observed = model.observe(seed, "candidate-001", &format!("rep-{seed}"), &[1.0, 2.0, 3.0]).unwrap();
            assert_eq!(observed.noise_values.len(), 3);
            assert!(observed.noise_values.iter().zip(&bounds).all(|(value, bound)| value.abs() <= *bound + 1e-12));
            assert_eq!(observed.observation_values[0], 1.0 + observed.noise_values[0]);
        }
    }

    #[test]
    fn invalid_noise_contracts_fail_closed() {
        assert!(CorrelatedBoundedNoiseModel::new("bad", vec![]).is_err());
        assert!(CorrelatedBoundedNoiseModel::new("bad", vec![
            CorrelatedNoiseObjective { objective_id: "x".into(), scale: 0.1, shared_loading: 0.0 },
            CorrelatedNoiseObjective { objective_id: "x".into(), scale: 0.2, shared_loading: 0.0 },
        ]).is_err());
        assert!(CorrelatedBoundedNoiseModel::new("bad", vec![
            CorrelatedNoiseObjective { objective_id: "x".into(), scale: 0.1, shared_loading: 1.1 },
            CorrelatedNoiseObjective { objective_id: "y".into(), scale: 0.2, shared_loading: 0.0 },
        ]).is_err());
    }

    #[test]
    fn observation_identity_and_dimension_are_required() {
        let model = model();
        assert!(model.observe(7, "", "rep-1", &[1.0, 2.0, 3.0]).is_err());
        assert!(model.observe(7, "candidate", "", &[1.0, 2.0, 3.0]).is_err());
        assert!(model.observe(7, "candidate", "rep-1", &[1.0, 2.0]).is_err());
        assert!(model.observe(7, "candidate", "rep-1", &[1.0, f64::NAN, 3.0]).is_err());
    }
}
