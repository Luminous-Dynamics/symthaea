// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Candidate-only HDC/HDC-LTC retrieval over MACE event identifiers.
//! The index returns IDs and diagnostics, never evidence or action authority.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::mem::size_of;
use std::time::Instant;

use broca_mace_policy::{Action, Experience, Objective, Regime, Scope};
use sha2::{Digest, Sha256};
use symthaea_hdc_ltc::{Activation, ContinuousHV, HdcLtcUnifiedNeuron, NeuronConfig};

const MIN_DIMENSION: usize = 64;
const MAX_DIMENSION: usize = 262_144;

#[derive(Clone, Copy, Debug)]
pub enum RepresentationProfile {
    StaticContinuousHv,
    HdcLtcZeroInput {
        elapsed_seconds: f32,
        tau_base_seconds: f32,
        tau_min_seconds: f32,
        tau_max_seconds: f32,
        neuron_seed: u64,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct IndexConfig {
    pub dimension: usize,
    pub codebook_seed: u64,
    pub profile: RepresentationProfile,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dimension: 16_384,
            codebook_seed: 0x4D41_4345_4844_4301,
            profile: RepresentationProfile::StaticContinuousHv,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct BucketKey {
    regime: Regime,
    objective: Objective,
    action: Action,
}

impl From<&Experience> for BucketKey {
    fn from(item: &Experience) -> Self {
        Self {
            regime: item.scope.regime,
            objective: item.scope.objective,
            action: item.action,
        }
    }
}

#[derive(Clone, Debug)]
struct CandidateCode {
    event_id: u32,
    view_digest: [u8; 32],
    code: ContinuousHV,
}

#[derive(Clone, Debug)]
struct Bucket {
    role_key: ContinuousHV,
    memory_state: ContinuousHV,
    candidates: Vec<CandidateCode>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BuildError {
    DimensionOutOfRange { actual: usize },
    EmptyMemory,
    DuplicateEventId { event_id: u32 },
    InvalidTemporalProfile,
    PayloadSizeOverflow,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionOutOfRange { actual } => {
                write!(f, "HDC dimension {actual} is outside {MIN_DIMENSION}..={MAX_DIMENSION}")
            }
            Self::EmptyMemory => write!(f, "cannot build an HDC index from empty memory"),
            Self::DuplicateEventId { event_id } => {
                write!(f, "duplicate event ID {event_id} makes resolution ambiguous")
            }
            Self::InvalidTemporalProfile => {
                write!(f, "invalid HDC-LTC elapsed time or tau configuration")
            }
            Self::PayloadSizeOverflow => write!(f, "HDC index payload size overflow"),
        }
    }
}

impl std::error::Error for BuildError {}

#[derive(Clone, Debug)]
pub struct CandidateHit {
    pub event_id: u32,
    /// SHA-256 of the candidate-visible Experience view at index-build time.
    /// A consistency check only; not a source signature or trust attestation.
    pub view_digest: [u8; 32],
    pub similarity: f32,
    pub rank: usize,
}

#[derive(Clone, Debug)]
pub struct RetrievalResult {
    pub profile_digest: String,
    /// Candidate IDs only. Resolve each ID against the canonical evidence registry.
    pub candidates: Vec<CandidateHit>,
    pub abstained: bool,
    pub best_similarity: Option<f32>,
    pub best_margin: Option<f32>,
    pub latency_ns: u64,
    /// Vector/ID payload lower bound; excludes allocator and tree overhead.
    pub index_payload_bytes: u64,
    pub candidate_count: usize,
}

#[derive(Clone, Debug)]
pub struct CandidateIndex {
    config: IndexConfig,
    profile_digest: String,
    buckets: BTreeMap<BucketKey, Bucket>,
    candidate_count: usize,
    index_payload_bytes: u64,
}

impl CandidateIndex {
    /// Build a lossy candidate index. Event IDs must be globally unique.
    /// Source, causal, trust and lifecycle fields are not copied into the index.
    pub fn build(memory: &[Experience], config: IndexConfig) -> Result<Self, BuildError> {
        validate_config(config)?;
        if memory.is_empty() {
            return Err(BuildError::EmptyMemory);
        }

        let mut seen = BTreeSet::new();
        let mut grouped: BTreeMap<BucketKey, Vec<&Experience>> = BTreeMap::new();
        for item in memory {
            if !seen.insert(item.event_id) {
                return Err(BuildError::DuplicateEventId {
                    event_id: item.event_id,
                });
            }
            grouped.entry(BucketKey::from(item)).or_default().push(item);
        }

        let mut buckets = BTreeMap::new();
        let mut payload_bytes = 0_u64;
        for (key, items) in grouped {
            let role_key = bipolar(ContinuousHV::new_random(
                config.dimension,
                role_seed(config.codebook_seed, key),
            ));
            let candidates: Vec<CandidateCode> = items
                .iter()
                .map(|item| CandidateCode {
                    event_id: item.event_id,
                    view_digest: experience_view_digest(item),
                    code: event_code(config.dimension, config.codebook_seed, item.event_id),
                })
                .collect();
            let associations: Vec<ContinuousHV> = candidates
                .iter()
                .map(|candidate| role_key.bind(&candidate.code))
                .collect();
            let association_refs: Vec<&ContinuousHV> = associations.iter().collect();
            let mut memory_state = ContinuousHV::bundle(&association_refs);

            if let RepresentationProfile::HdcLtcZeroInput {
                elapsed_seconds,
                tau_base_seconds,
                tau_min_seconds,
                tau_max_seconds,
                neuron_seed,
            } = config.profile
            {
                let mut neuron_config = NeuronConfig::default();
                neuron_config.dim = config.dimension;
                neuron_config.activation = Activation::Tanh;
                neuron_config.tau_base = tau_base_seconds;
                neuron_config.tau_min = tau_min_seconds;
                neuron_config.tau_max = tau_max_seconds;
                let mut neuron = HdcLtcUnifiedNeuron::new(neuron_config, neuron_seed);
                neuron.set_state(memory_state);
                if elapsed_seconds > 0.0 {
                    neuron.evolve_closed_form_fused(
                        elapsed_seconds,
                        &ContinuousHV::new(config.dimension),
                    );
                }
                memory_state = neuron.state().clone();
            }

            let vector_payload = vector_bytes(config.dimension)?;
            let role_and_state_bytes = vector_payload
                .checked_mul(2)
                .ok_or(BuildError::PayloadSizeOverflow)?;
            let per_candidate_bytes = vector_payload
                .checked_add(size_of::<u32>() as u64)
                .ok_or(BuildError::PayloadSizeOverflow)?;
            let candidate_count = u64::try_from(candidates.len())
                .map_err(|_| BuildError::PayloadSizeOverflow)?;
            let candidate_bytes = per_candidate_bytes
                .checked_mul(candidate_count)
                .ok_or(BuildError::PayloadSizeOverflow)?;
            let bucket_bytes = role_and_state_bytes
                .checked_add(candidate_bytes)
                .ok_or(BuildError::PayloadSizeOverflow)?;
            payload_bytes = payload_bytes
                .checked_add(bucket_bytes)
                .ok_or(BuildError::PayloadSizeOverflow)?;

            buckets.insert(
                key,
                Bucket {
                    role_key,
                    memory_state,
                    candidates,
                },
            );
        }

        Ok(Self {
            config,
            profile_digest: profile_digest(config),
            buckets,
            candidate_count: memory.len(),
            index_payload_bytes: payload_bytes,
        })
    }

    /// Retrieve a ranked list for one exact scope/action key.
    ///
    /// This is candidate generation only. The caller must resolve IDs and
    /// pass canonical records through unchanged MACE policy gates.
    pub fn retrieve(&self, scope: Scope, action: Action, top_k: usize) -> RetrievalResult {
        let started = Instant::now();
        let key = BucketKey {
            regime: scope.regime,
            objective: scope.objective,
            action,
        };
        let Some(bucket) = self.buckets.get(&key) else {
            return self.result(Vec::new(), started);
        };
        if top_k == 0 {
            return self.result(Vec::new(), started);
        }

        // Bipolar role keys are self-inverse under element-wise multiplication.
        let decoded = bucket.role_key.bind(&bucket.memory_state);
        let mut ranked: Vec<(u32, [u8; 32], f32)> = bucket
            .candidates
            .iter()
            .map(|candidate| {
                (
                    candidate.event_id,
                    candidate.view_digest,
                    decoded.similarity(&candidate.code),
                )
            })
            .filter(|(_, _, similarity)| similarity.is_finite())
            .collect();
        ranked.sort_by(|left, right| {
            right
                .2
                .total_cmp(&left.2)
                .then_with(|| left.0.cmp(&right.0))
        });

        let candidates = ranked
            .into_iter()
            .take(top_k)
            .enumerate()
            .map(|(rank, (event_id, view_digest, similarity))| CandidateHit {
                event_id,
                view_digest,
                similarity,
                rank: rank + 1,
            })
            .collect();
        self.result(candidates, started)
    }

    fn result(&self, candidates: Vec<CandidateHit>, started: Instant) -> RetrievalResult {
        let best_similarity = candidates.first().map(|candidate| candidate.similarity);
        let best_margin = candidates
            .first()
            .zip(candidates.get(1))
            .map(|(best, second)| best.similarity - second.similarity);
        let elapsed = started.elapsed().as_nanos();
        RetrievalResult {
            profile_digest: self.profile_digest.clone(),
            abstained: candidates.is_empty(),
            candidates,
            best_similarity,
            best_margin,
            latency_ns: u64::try_from(elapsed).unwrap_or(u64::MAX),
            index_payload_bytes: self.index_payload_bytes,
            candidate_count: self.candidate_count,
        }
    }

    pub fn profile_digest(&self) -> &str {
        &self.profile_digest
    }

    pub fn index_payload_bytes(&self) -> u64 {
        self.index_payload_bytes
    }

    pub fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    pub fn dimension(&self) -> usize {
        self.config.dimension
    }
}

/// Exact-scan candidate ceiling for the same scope/action question.
/// It also returns IDs only and does not qualify or authorize experiences.
pub fn exact_scan_candidate_ids(
    memory: &[Experience],
    scope: Scope,
    action: Action,
    top_k: usize,
) -> Vec<u32> {
    if top_k == 0 {
        return Vec::new();
    }
    memory
        .iter()
        .filter(|item| item.scope == scope && item.action == action)
        .take(top_k)
        .map(|item| item.event_id)
        .collect()
}

/// Fingerprint the candidate-visible Experience view so a resolver can
/// reject an index hit if the canonical record changed after index construction.
/// This does not bind raw source bytes and is not a provenance/trust proof.
pub fn experience_view_digest(item: &Experience) -> [u8; 32] {
    use broca_mace_policy::IdentificationMethodV1;

    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:experience-view:v1\\0");
    hasher.update(item.event_id.to_be_bytes());
    hasher.update(item.source_family.to_be_bytes());
    hasher.update(item.source_root_id.to_be_bytes());
    hasher.update(item.source_root_digest.as_bytes());
    hasher.update([
        match item.scope.regime {
            Regime::Baseline => 1,
            Regime::Shifted => 2,
        },
        match item.scope.objective {
            Objective::Quality => 1,
            Objective::Cost => 2,
            Objective::Speed => 3,
        },
        match item.action {
            Action::A => 1,
            Action::B => 2,
        },
    ]);
    hasher.update(item.observed_utility.to_bits().to_be_bytes());
    hasher.update(item.measurement_error_bound.to_bits().to_be_bytes());

    let identification = item.causal_identification;
    hasher.update([
        match identification.method {
            IdentificationMethodV1::RandomizedControlledTrial => 1,
            IdentificationMethodV1::QuasiExperimental => 2,
            IdentificationMethodV1::Observational => 3,
            IdentificationMethodV1::Unknown => 4,
        },
        u8::from(identification.interference_controlled),
    ]);
    hasher.update(identification.design_digest.as_bytes());
    hasher.update(identification.estimand_digest.as_bytes());
    hasher.update(identification.semantic_estimand.canonical_digest().as_bytes());
    hasher.update(identification.propensity_min_bps.to_be_bytes());
    hasher.update(identification.propensity_max_bps.to_be_bytes());
    hasher.update(identification.max_standardized_mean_difference_micros.to_be_bytes());
    hasher.update([
        u8::from(item.intervention),
        u8::from(item.prediction_pre_registered),
        u8::from(item.outcome_reconciled),
        u8::from(item.source_trusted),
        u8::from(item.source_retracted),
        u8::from(item.confounders_controlled),
        u8::from(item.delayed_horizon_complete),
    ]);
    let digest = hasher.finalize();
    let mut output = [0_u8; 32];
    output.copy_from_slice(&digest);
    output
}

fn validate_config(config: IndexConfig) -> Result<(), BuildError> {
    if !(MIN_DIMENSION..=MAX_DIMENSION).contains(&config.dimension) {
        return Err(BuildError::DimensionOutOfRange {
            actual: config.dimension,
        });
    }
    if let RepresentationProfile::HdcLtcZeroInput {
        elapsed_seconds,
        tau_base_seconds,
        tau_min_seconds,
        tau_max_seconds,
        ..
    } = config.profile
    {
        if !elapsed_seconds.is_finite()
            || elapsed_seconds < 0.0
            || !tau_base_seconds.is_finite()
            || !tau_min_seconds.is_finite()
            || !tau_max_seconds.is_finite()
            || tau_min_seconds <= 0.0
            || tau_min_seconds > tau_base_seconds
            || tau_base_seconds > tau_max_seconds
        {
            return Err(BuildError::InvalidTemporalProfile);
        }
    }
    Ok(())
}

fn vector_bytes(dimension: usize) -> Result<u64, BuildError> {
    let count = u64::try_from(dimension).map_err(|_| BuildError::PayloadSizeOverflow)?;
    count
        .checked_mul(size_of::<f32>() as u64)
        .ok_or(BuildError::PayloadSizeOverflow)
}

fn bipolar(mut vector: ContinuousHV) -> ContinuousHV {
    for value in &mut vector.values {
        *value = if *value < 0.0 { -1.0 } else { 1.0 };
    }
    vector
}

fn role_seed(seed: u64, key: BucketKey) -> u64 {
    let regime = match key.regime {
        Regime::Baseline => 1_u64,
        Regime::Shifted => 2_u64,
    };
    let objective = match key.objective {
        Objective::Quality => 11_u64,
        Objective::Cost => 13_u64,
        Objective::Speed => 17_u64,
    };
    let action = match key.action {
        Action::A => 23_u64,
        Action::B => 29_u64,
    };
    mix64(seed ^ regime.rotate_left(11) ^ objective.rotate_left(23) ^ action.rotate_left(37))
}

fn event_code(dimension: usize, seed: u64, event_id: u32) -> ContinuousHV {
    bipolar(ContinuousHV::new_random(
        dimension,
        mix64(seed ^ u64::from(event_id).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
    ))
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn profile_digest(config: IndexConfig) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:hdc-candidate-index:v1\0");
    hasher.update((config.dimension as u64).to_be_bytes());
    hasher.update(config.codebook_seed.to_be_bytes());
    match config.profile {
        RepresentationProfile::StaticContinuousHv => {
            hasher.update([0_u8]);
        }
        RepresentationProfile::HdcLtcZeroInput {
            elapsed_seconds,
            tau_base_seconds,
            tau_min_seconds,
            tau_max_seconds,
            neuron_seed,
        } => {
            hasher.update([1_u8]);
            hasher.update(elapsed_seconds.to_bits().to_be_bytes());
            hasher.update(tau_base_seconds.to_bits().to_be_bytes());
            hasher.update(tau_min_seconds.to_bits().to_be_bytes());
            hasher.update(tau_max_seconds.to_bits().to_be_bytes());
            hasher.update(neuron_seed.to_be_bytes());
        }
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use broca_mace_policy::{CausalIdentificationV1, SourceRootDigest};

    fn fixture(event_id: u32, action: Action, scope: Scope) -> Experience {
        Experience {
            event_id,
            source_family: u16::try_from(event_id % 65_534 + 1).unwrap(),
            source_root_id: event_id.saturating_add(1),
            source_root_digest: SourceRootDigest::UNKNOWN,
            scope,
            action,
            observed_utility: 0.5,
            measurement_error_bound: 0.02,
            causal_identification: CausalIdentificationV1::UNKNOWN,
            intervention: false,
            prediction_pre_registered: false,
            outcome_reconciled: false,
            source_trusted: false,
            source_retracted: false,
            confounders_controlled: false,
            delayed_horizon_complete: false,
        }
    }

    fn scope(regime: Regime, objective: Objective) -> Scope {
        Scope { regime, objective }
    }

    fn memory() -> Vec<Experience> {
        vec![
            fixture(1, Action::A, scope(Regime::Baseline, Objective::Quality)),
            fixture(2, Action::A, scope(Regime::Baseline, Objective::Quality)),
            fixture(3, Action::A, scope(Regime::Baseline, Objective::Quality)),
            fixture(4, Action::B, scope(Regime::Baseline, Objective::Quality)),
            fixture(5, Action::B, scope(Regime::Baseline, Objective::Quality)),
            fixture(6, Action::B, scope(Regime::Baseline, Objective::Quality)),
            fixture(7, Action::A, scope(Regime::Shifted, Objective::Quality)),
        ]
    }

    #[test]
    fn static_hdc_recovers_exact_single_event_binding() {
        let memory = vec![fixture(
            41,
            Action::A,
            scope(Regime::Baseline, Objective::Quality),
        )];
        let index = CandidateIndex::build(
            &memory,
            IndexConfig {
                dimension: 1024,
                ..IndexConfig::default()
            },
        )
        .unwrap();
        let result = index.retrieve(
            scope(Regime::Baseline, Objective::Quality),
            Action::A,
            4,
        );
        assert_eq!(result.candidates.len(), 1);
        assert_eq!(result.candidates[0].event_id, 41);
        assert_eq!(
            result.candidates[0].view_digest,
            experience_view_digest(&memory[0])
        );
        assert!((result.candidates[0].similarity - 1.0).abs() < 1e-5);
        assert!(!result.abstained);
    }

    #[test]
    fn exact_scope_and_action_buckets_do_not_cross_contaminate() {
        let memory = memory();
        let index = CandidateIndex::build(
            &memory,
            IndexConfig {
                dimension: 2048,
                ..IndexConfig::default()
            },
        )
        .unwrap();
        let a = index.retrieve(
            scope(Regime::Baseline, Objective::Quality),
            Action::A,
            16,
        );
        let b = index.retrieve(
            scope(Regime::Baseline, Objective::Quality),
            Action::B,
            16,
        );
        let shifted = index.retrieve(
            scope(Regime::Shifted, Objective::Quality),
            Action::B,
            16,
        );
        assert_eq!(
            a.candidates.iter().map(|hit| hit.event_id).collect::<BTreeSet<_>>(),
            BTreeSet::from([1, 2, 3])
        );
        assert_eq!(
            b.candidates.iter().map(|hit| hit.event_id).collect::<BTreeSet<_>>(),
            BTreeSet::from([4, 5, 6])
        );
        assert!(shifted.candidates.is_empty());
        assert!(shifted.abstained);
    }

    #[test]
    fn duplicate_ids_and_invalid_temporal_inputs_fail_closed() {
        let duplicate = vec![
            fixture(9, Action::A, scope(Regime::Baseline, Objective::Quality)),
            fixture(9, Action::B, scope(Regime::Baseline, Objective::Quality)),
        ];
        assert_eq!(
            CandidateIndex::build(&duplicate, IndexConfig::default()).unwrap_err(),
            BuildError::DuplicateEventId { event_id: 9 }
        );
        assert!(matches!(
            CandidateIndex::build(
                &memory(),
                IndexConfig {
                    dimension: 32,
                    ..IndexConfig::default()
                }
            ),
            Err(BuildError::DimensionOutOfRange { .. })
        ));
        assert_eq!(
            CandidateIndex::build(
                &memory(),
                IndexConfig {
                    dimension: 1024,
                    codebook_seed: 1,
                    profile: RepresentationProfile::HdcLtcZeroInput {
                        elapsed_seconds: f32::NAN,
                        tau_base_seconds: 0.1,
                        tau_min_seconds: 0.01,
                        tau_max_seconds: 10.0,
                        neuron_seed: 7,
                    },
                }
            )
            .unwrap_err(),
            BuildError::InvalidTemporalProfile
        );
    }

    #[test]
    fn hdc_ltc_zero_input_returns_only_original_candidate_ids() {
        let memory = memory();
        let index = CandidateIndex::build(
            &memory,
            IndexConfig {
                dimension: 2048,
                codebook_seed: 42,
                profile: RepresentationProfile::HdcLtcZeroInput {
                    elapsed_seconds: 0.25,
                    tau_base_seconds: 10.0,
                    tau_min_seconds: 0.01,
                    tau_max_seconds: 20.0,
                    neuron_seed: 99,
                },
            },
        )
        .unwrap();
        let result = index.retrieve(
            scope(Regime::Baseline, Objective::Quality),
            Action::A,
            8,
        );
        let ids: BTreeSet<u32> = result.candidates.iter().map(|hit| hit.event_id).collect();
        assert!(ids.is_subset(&BTreeSet::from([1, 2, 3])));
        assert_eq!(result.candidates.len(), 3);
        assert!(result.candidates.iter().all(|hit| hit.similarity.is_finite()));
        assert!(!result.abstained);
        let static_profile = IndexConfig {
            dimension: 2048,
            codebook_seed: 42,
            profile: RepresentationProfile::StaticContinuousHv,
        };
        let static = CandidateIndex::build(&memory, static_profile).unwrap();
        assert_ne!(result.profile_digest, static.profile_digest);
    }

    #[test]
    fn index_returns_ids_not_experience_records_or_authority() {
        let index = CandidateIndex::build(&memory(), IndexConfig::default()).unwrap();
        let result = index.retrieve(
            scope(Regime::Baseline, Objective::Quality),
            Action::A,
            8,
        );
        assert!(result.candidates.iter().all(|hit| [1, 2, 3].contains(&hit.event_id)));
    }

    #[test]
    fn payload_footprint_and_profile_digest_are_stable() {
        let memory = memory();
        let config = IndexConfig {
            dimension: 1024,
            codebook_seed: 77,
            profile: RepresentationProfile::StaticContinuousHv,
        };
        let first = CandidateIndex::build(&memory, config).unwrap();
        let second = CandidateIndex::build(&memory, config).unwrap();
        assert_eq!(first.profile_digest(), second.profile_digest());
        assert_eq!(first.index_payload_bytes(), second.index_payload_bytes());
        assert!(first.index_payload_bytes() > 0);
        assert_eq!(first.candidate_count(), memory.len());
    }
}
