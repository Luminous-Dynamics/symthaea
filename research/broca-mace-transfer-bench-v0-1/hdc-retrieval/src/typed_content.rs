// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed-content, query-conditioned HDC retrieval for MACE.
//!
//! This module is deliberately separate from the event-ID associative index.
//! It represents candidate-visible Experience fields with deterministic
//! role/value binding, and represents a structured decision query in the same
//! field space. It returns candidate IDs and view digests only. It never
//! establishes provenance, causal qualification, freshness or authority.

use std::collections::BTreeSet;
use std::fmt;
use std::mem::size_of;
use std::time::Instant;

use broca_mace_policy::{
    Action, EvidenceDigest256, Experience, IdentificationMethodV1, Objective, Regime, Scope,
};
use sha2::{Digest, Sha256};
use symthaea_hdc_ltc::ContinuousHV;

use super::{experience_view_digest, CandidateHit, RetrievalResult};

const MIN_DIMENSION: usize = 256;
const MAX_DIMENSION: usize = 262_144;
const PROFILE_DOMAIN: &[u8] = b"luminous:mace:typed-content-hdc:v1\0";
const NUMERIC_KERNEL: &[u8] = b"rff-rbf-gaussian-v1";

/// Versioned encoder configuration. Bandwidths are part of the profile digest
/// because they determine the meaning of numeric similarity.
#[derive(Clone, Copy, Debug)]
pub struct TypedContentConfigV1 {
    pub dimension: usize,
    pub seed: u64,
    /// RBF length scale for normalized observed utility.
    pub utility_bandwidth: f64,
    /// RBF length scale for measurement-error bounds.
    pub measurement_error_bandwidth: f64,
    /// RBF length scale for propensity bounds, normalized to [0, 1].
    pub propensity_bandwidth: f64,
    /// RBF length scale for standardized mean difference, normalized to [0, 1].
    pub balance_bandwidth: f64,
}

impl Default for TypedContentConfigV1 {
    fn default() -> Self {
        Self {
            dimension: 4_096,
            seed: 0x4D41_4345_5459_5045,
            utility_bandwidth: 0.10,
            measurement_error_bandwidth: 0.025,
            propensity_bandwidth: 0.05,
            balance_bandwidth: 0.05,
        }
    }
}

/// Structured query over the same typed feature vocabulary as the candidate
/// Experience encoder. Optional fields add query constraints/cues without
/// changing the canonical evidence qualification rules.
#[derive(Clone, Copy, Debug)]
pub struct SemanticExperienceQueryV1 {
    pub scope: Scope,
    pub action: Action,
    pub target_utility: Option<f64>,
    pub target_measurement_error_bound: Option<f64>,
    pub target_propensity_min_bps: Option<u16>,
    pub target_propensity_max_bps: Option<u16>,
    pub target_max_standardized_mean_difference_micros: Option<u32>,
    pub identification_method: Option<IdentificationMethodV1>,
    /// Exact semantic-estimand identity cue. A digest match is not proof that
    /// the upstream semantic descriptor was correctly authored or attested.
    pub estimand_digest: Option<EvidenceDigest256>,
    pub require_intervention: Option<bool>,
    pub require_pre_registered: Option<bool>,
    pub require_outcome_reconciled: Option<bool>,
    pub require_source_trusted: Option<bool>,
    pub require_source_not_retracted: Option<bool>,
    pub require_confounders_controlled: Option<bool>,
    pub require_delayed_horizon_complete: Option<bool>,
}

impl SemanticExperienceQueryV1 {
    pub fn exact_scope_action(scope: Scope, action: Action) -> Self {
        Self {
            scope,
            action,
            target_utility: None,
            target_measurement_error_bound: None,
            target_propensity_min_bps: None,
            target_propensity_max_bps: None,
            target_max_standardized_mean_difference_micros: None,
            identification_method: None,
            estimand_digest: None,
            require_intervention: None,
            require_pre_registered: None,
            require_outcome_reconciled: None,
            require_source_trusted: None,
            require_source_not_retracted: None,
            require_confounders_controlled: None,
            require_delayed_horizon_complete: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TypedContentError {
    DimensionOutOfRange { actual: usize },
    InvalidBandwidth,
    EmptyMemory,
    DuplicateEventId { event_id: u32 },
    NonFiniteExperience { event_id: u32, field: &'static str },
    InvalidMeasurementError { event_id: u32 },
    InvalidPropensityBounds { event_id: u32 },
    NonFiniteQuery { field: &'static str },
    InvalidQueryPropensityBounds,
}

impl fmt::Display for TypedContentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionOutOfRange { actual } => write!(
                f,
                "typed-content HDC dimension {actual} is outside {MIN_DIMENSION}..={MAX_DIMENSION}"
            ),
            Self::InvalidBandwidth => write!(f, "typed-content HDC bandwidth must be finite and positive"),
            Self::EmptyMemory => write!(f, "cannot build typed-content HDC from empty memory"),
            Self::DuplicateEventId { event_id } => {
                write!(f, "duplicate event ID {event_id} makes canonical resolution ambiguous")
            }
            Self::NonFiniteExperience { event_id, field } => {
                write!(f, "event {event_id} has non-finite {field}")
            }
            Self::InvalidMeasurementError { event_id } => {
                write!(f, "event {event_id} has a negative measurement-error bound")
            }
            Self::InvalidPropensityBounds { event_id } => {
                write!(f, "event {event_id} has invalid propensity bounds")
            }
            Self::NonFiniteQuery { field } => write!(f, "query has non-finite {field}"),
            Self::InvalidQueryPropensityBounds => {
                write!(f, "query has invalid propensity target bounds")
            }
        }
    }
}

impl std::error::Error for TypedContentError {}

#[derive(Clone, Debug)]
struct EncodedExperience {
    event_id: u32,
    view_digest: [u8; 32],
    vector: ContinuousHV,
}

#[derive(Clone, Debug)]
pub struct TypedContentIndexV1 {
    config: TypedContentConfigV1,
    profile_digest: String,
    candidates: Vec<EncodedExperience>,
    index_payload_bytes: u64,
}

impl TypedContentIndexV1 {
    /// Build a typed content index. Event IDs and provenance identities are
    /// retained only as exact sidecar references; they are not encoded into
    /// approximate vector content.
    pub fn build(
        memory: &[Experience],
        config: TypedContentConfigV1,
    ) -> Result<Self, TypedContentError> {
        validate_config(config)?;
        if memory.is_empty() {
            return Err(TypedContentError::EmptyMemory);
        }

        let mut seen = BTreeSet::new();
        let mut candidates = Vec::with_capacity(memory.len());
        let mut payload_bytes = 0_u64;
        let per_candidate_bytes = vector_bytes(config.dimension)?
            .checked_add(size_of::<u32>() as u64)
            .and_then(|value| value.checked_add(32))
            .ok_or(TypedContentError::DimensionOutOfRange {
                actual: config.dimension,
            })?;

        for item in memory {
            if !seen.insert(item.event_id) {
                return Err(TypedContentError::DuplicateEventId {
                    event_id: item.event_id,
                });
            }
            validate_experience(item)?;
            candidates.push(EncodedExperience {
                event_id: item.event_id,
                view_digest: experience_view_digest(item),
                vector: encode_experience(item, config),
            });
            payload_bytes = payload_bytes
                .checked_add(per_candidate_bytes)
                .ok_or(TypedContentError::DimensionOutOfRange {
                    actual: config.dimension,
                })?;
        }

        Ok(Self {
            config,
            profile_digest: profile_digest(config),
            candidates,
            index_payload_bytes: payload_bytes,
        })
    }

    /// Return candidate IDs only. Callers must resolve them to canonical
    /// Experience records and re-run every MACE admission gate after retrieval.
    pub fn retrieve(
        &self,
        query: SemanticExperienceQueryV1,
        top_k: usize,
    ) -> Result<RetrievalResult, TypedContentError> {
        validate_query(query)?;
        let started = Instant::now();
        if top_k == 0 {
            return Ok(self.result(Vec::new(), started));
        }

        let query_vector = encode_query(query, self.config);
        let mut ranked: Vec<(u32, [u8; 32], f32)> = self
            .candidates
            .iter()
            .map(|candidate| {
                (
                    candidate.event_id,
                    candidate.view_digest,
                    candidate.vector.similarity(&query_vector),
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

        Ok(self.result(candidates, started))
    }

    fn result(&self, candidates: Vec<CandidateHit>, started: Instant) -> RetrievalResult {
        let best_similarity = candidates.first().map(|candidate| candidate.similarity);
        let best_margin = candidates
            .first()
            .zip(candidates.get(1))
            .map(|(best, second)| best.similarity - second.similarity);
        RetrievalResult {
            profile_digest: self.profile_digest.clone(),
            abstained: candidates.is_empty(),
            candidates,
            best_similarity,
            best_margin,
            latency_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            index_payload_bytes: self.index_payload_bytes,
            candidate_count: self.candidates.len(),
        }
    }

    pub fn profile_digest(&self) -> &str {
        &self.profile_digest
    }

    /// Vector + exact event-ID + candidate-view digest payload lower bound.
    /// Allocator/runtime/container overhead is excluded.
    pub fn index_payload_bytes(&self) -> u64 {
        self.index_payload_bytes
    }

    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    pub fn dimension(&self) -> usize {
        self.config.dimension
    }
}

fn validate_config(config: TypedContentConfigV1) -> Result<(), TypedContentError> {
    if !(MIN_DIMENSION..=MAX_DIMENSION).contains(&config.dimension) {
        return Err(TypedContentError::DimensionOutOfRange {
            actual: config.dimension,
        });
    }
    for bandwidth in [
        config.utility_bandwidth,
        config.measurement_error_bandwidth,
        config.propensity_bandwidth,
        config.balance_bandwidth,
    ] {
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err(TypedContentError::InvalidBandwidth);
        }
    }
    Ok(())
}

fn validate_experience(item: &Experience) -> Result<(), TypedContentError> {
    for (field, value) in [
        ("observed_utility", item.observed_utility),
        ("measurement_error_bound", item.measurement_error_bound),
    ] {
        if !value.is_finite() {
            return Err(TypedContentError::NonFiniteExperience {
                event_id: item.event_id,
                field,
            });
        }
    }
    if item.measurement_error_bound < 0.0 {
        return Err(TypedContentError::InvalidMeasurementError {
            event_id: item.event_id,
        });
    }
    if item.causal_identification.propensity_min_bps
        > item.causal_identification.propensity_max_bps
        || item.causal_identification.propensity_max_bps > 10_000
    {
        return Err(TypedContentError::InvalidPropensityBounds {
            event_id: item.event_id,
        });
    }
    Ok(())
}

fn validate_query(query: SemanticExperienceQueryV1) -> Result<(), TypedContentError> {
    for (field, value) in [
        ("target_utility", query.target_utility),
        (
            "target_measurement_error_bound",
            query.target_measurement_error_bound,
        ),
    ] {
        if value.is_some_and(|value| !value.is_finite()) {
            return Err(TypedContentError::NonFiniteQuery { field });
        }
    }
    if query
        .target_measurement_error_bound
        .is_some_and(|value| value < 0.0)
    {
        return Err(TypedContentError::NonFiniteQuery {
            field: "negative target_measurement_error_bound",
        });
    }
    if query.target_propensity_min_bps.zip(query.target_propensity_max_bps).is_some_and(
        |(minimum, maximum)| minimum > maximum || maximum > 10_000,
    ) {
        return Err(TypedContentError::InvalidQueryPropensityBounds);
    }
    Ok(())
}

fn encode_experience(
    item: &Experience,
    config: TypedContentConfigV1,
) -> ContinuousHV {
    let mut features = Vec::with_capacity(22);

    push_category(
        &mut features,
        config,
        b"scope.regime",
        &enum_bytes(match item.scope.regime {
            Regime::Baseline => 1,
            Regime::Shifted => 2,
        }),
    );
    push_category(
        &mut features,
        config,
        b"scope.objective",
        &enum_bytes(match item.scope.objective {
            Objective::Quality => 1,
            Objective::Cost => 2,
            Objective::Speed => 3,
        }),
    );
    push_category(
        &mut features,
        config,
        b"action",
        &enum_bytes(action_code(item.action)),
    );
    push_numeric(
        &mut features,
        config,
        b"observed_utility",
        item.observed_utility,
        config.utility_bandwidth,
    );
    push_numeric(
        &mut features,
        config,
        b"measurement_error_bound",
        item.measurement_error_bound,
        config.measurement_error_bandwidth,
    );
    push_category(
        &mut features,
        config,
        b"causal.identification_method",
        &enum_bytes(method_code(item.causal_identification.method)),
    );
    push_category(
        &mut features,
        config,
        b"causal.estimand_digest",
        item.causal_identification.estimand_digest.as_bytes(),
    );
    let semantic = item.causal_identification.semantic_estimand;
    let semantic_digest = if semantic.is_complete() {
        semantic.canonical_digest()
    } else {
        EvidenceDigest256::UNKNOWN
    };
    push_category(
        &mut features,
        config,
        b"causal.semantic_estimand_digest",
        semantic_digest.as_bytes(),
    );
    push_numeric(
        &mut features,
        config,
        b"causal.propensity_min_bps",
        f64::from(item.causal_identification.propensity_min_bps) / 10_000.0,
        config.propensity_bandwidth,
    );
    push_numeric(
        &mut features,
        config,
        b"causal.propensity_max_bps",
        f64::from(item.causal_identification.propensity_max_bps) / 10_000.0,
        config.propensity_bandwidth,
    );
    push_numeric(
        &mut features,
        config,
        b"causal.max_standardized_mean_difference",
        f64::from(item.causal_identification.max_standardized_mean_difference_micros)
            / 1_000_000.0,
        config.balance_bandwidth,
    );
    push_category(
        &mut features,
        config,
        b"causal.interference_controlled",
        &bool_bytes(item.causal_identification.interference_controlled),
    );
    push_category(
        &mut features,
        config,
        b"intervention",
        &bool_bytes(item.intervention),
    );
    push_category(
        &mut features,
        config,
        b"prediction_pre_registered",
        &bool_bytes(item.prediction_pre_registered),
    );
    push_category(
        &mut features,
        config,
        b"outcome_reconciled",
        &bool_bytes(item.outcome_reconciled),
    );
    push_category(
        &mut features,
        config,
        b"source_trusted",
        &bool_bytes(item.source_trusted),
    );
    push_category(
        &mut features,
        config,
        b"source_retracted",
        &bool_bytes(item.source_retracted),
    );
    push_category(
        &mut features,
        config,
        b"confounders_controlled",
        &bool_bytes(item.confounders_controlled),
    );
    push_category(
        &mut features,
        config,
        b"delayed_horizon_complete",
        &bool_bytes(item.delayed_horizon_complete),
    );
    push_category(
        &mut features,
        config,
        b"semantic_estimand_complete",
        &bool_bytes(semantic.is_complete()),
    );

    bundle_features(&features, config.dimension)
}

fn encode_query(
    query: SemanticExperienceQueryV1,
    config: TypedContentConfigV1,
) -> ContinuousHV {
    let mut features = Vec::with_capacity(18);
    push_category(
        &mut features,
        config,
        b"scope.regime",
        &enum_bytes(match query.scope.regime {
            Regime::Baseline => 1,
            Regime::Shifted => 2,
        }),
    );
    push_category(
        &mut features,
        config,
        b"scope.objective",
        &enum_bytes(match query.scope.objective {
            Objective::Quality => 1,
            Objective::Cost => 2,
            Objective::Speed => 3,
        }),
    );
    push_category(
        &mut features,
        config,
        b"action",
        &enum_bytes(action_code(query.action)),
    );

    if let Some(value) = query.target_utility {
        push_numeric(
            &mut features,
            config,
            b"observed_utility",
            value,
            config.utility_bandwidth,
        );
    }
    if let Some(value) = query.target_measurement_error_bound {
        push_numeric(
            &mut features,
            config,
            b"measurement_error_bound",
            value,
            config.measurement_error_bandwidth,
        );
    }
    if let Some(value) = query.target_propensity_min_bps {
        push_numeric(
            &mut features,
            config,
            b"causal.propensity_min_bps",
            f64::from(value) / 10_000.0,
            config.propensity_bandwidth,
        );
    }
    if let Some(value) = query.target_propensity_max_bps {
        push_numeric(
            &mut features,
            config,
            b"causal.propensity_max_bps",
            f64::from(value) / 10_000.0,
            config.propensity_bandwidth,
        );
    }
    if let Some(value) = query.target_max_standardized_mean_difference_micros {
        push_numeric(
            &mut features,
            config,
            b"causal.max_standardized_mean_difference",
            f64::from(value) / 1_000_000.0,
            config.balance_bandwidth,
        );
    }
    if let Some(method) = query.identification_method {
        push_category(
            &mut features,
            config,
            b"causal.identification_method",
            &enum_bytes(method_code(method)),
        );
    }
    if let Some(digest) = query.estimand_digest {
        push_category(
            &mut features,
            config,
            b"causal.estimand_digest",
            digest.as_bytes(),
        );
    }

    for (tag, value) in [
        ("intervention", query.require_intervention),
        ("prediction_pre_registered", query.require_pre_registered),
        ("outcome_reconciled", query.require_outcome_reconciled),
        ("source_trusted", query.require_source_trusted),
        ("source_retracted", query.require_source_not_retracted.map(|value| !value)),
        ("confounders_controlled", query.require_confounders_controlled),
        (
            "delayed_horizon_complete",
            query.require_delayed_horizon_complete,
        ),
    ] {
        if let Some(value) = value {
            push_category(
                &mut features,
                config,
                tag.as_bytes(),
                &bool_bytes(value),
            );
        }
    }

    bundle_features(&features, config.dimension)
}

fn bundle_features(features: &[ContinuousHV], dimension: usize) -> ContinuousHV {
    let references: Vec<&ContinuousHV> = features.iter().collect();
    if references.is_empty() {
        return ContinuousHV::new(dimension);
    }
    ContinuousHV::bundle(&references).normalize()
}

fn push_category(
    features: &mut Vec<ContinuousHV>,
    config: TypedContentConfigV1,
    tag: &[u8],
    value: &[u8],
) {
    let role = ContinuousHV::new_random(
        config.dimension,
        derive_seed(config.seed, b"role", tag, &[]),
    );
    let filler = ContinuousHV::new_random(
        config.dimension,
        derive_seed(config.seed, b"category", tag, value),
    );
    features.push(role.bind(&filler));
}

fn push_numeric(
    features: &mut Vec<ContinuousHV>,
    config: TypedContentConfigV1,
    tag: &[u8],
    value: f64,
    bandwidth: f64,
) {
    let role = ContinuousHV::new_random(
        config.dimension,
        derive_seed(config.seed, b"role", tag, &[]),
    );
    let filler = random_fourier_rbf(
        config.dimension,
        derive_seed(config.seed, NUMERIC_KERNEL, tag, &[]),
        value,
        bandwidth,
    );
    features.push(role.bind(&filler));
}

/// Deterministic random Fourier features for a one-dimensional Gaussian RBF
/// kernel. In expectation, nearby scalar values have higher cosine similarity;
/// bandwidth and seed are frozen in the profile digest.
fn random_fourier_rbf(
    dimension: usize,
    seed: u64,
    value: f64,
    bandwidth: f64,
) -> ContinuousHV {
    let mut state = if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    };
    let mut values = Vec::with_capacity(dimension);
    let scale = (2.0 / dimension as f64).sqrt();

    while values.len() < dimension {
        let u1 = unit_open(&mut state);
        let u2 = unit_open(&mut state);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        let normal_0 = radius * angle.cos();
        let normal_1 = radius * angle.sin();

        for normal in [normal_0, normal_1] {
            if values.len() == dimension {
                break;
            }
            let phase = std::f64::consts::TAU * unit_open(&mut state);
            let frequency = normal / bandwidth;
            values.push((scale * (frequency * value + phase).cos()) as f32);
        }
    }

    ContinuousHV::from_values(values)
}

fn unit_open(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let mantissa = *state >> 11;
    (mantissa as f64 + 0.5) / 9_007_199_254_740_992.0
}

fn derive_seed(seed: u64, domain: &[u8], tag: &[u8], value: &[u8]) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(PROFILE_DOMAIN);
    hasher.update(seed.to_be_bytes());
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((tag.len() as u64).to_be_bytes());
    hasher.update(tag);
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
    let digest = hasher.finalize();
    u64::from_be_bytes(digest[..8].try_into().unwrap_or([0; 8]))
}

fn enum_bytes(value: u8) -> [u8; 1] {
    [value]
}

fn bool_bytes(value: bool) -> [u8; 1] {
    [u8::from(value)]
}

fn action_code(action: Action) -> u8 {
    match action {
        Action::A => 1,
        Action::B => 2,
    }
}

fn method_code(method: IdentificationMethodV1) -> u8 {
    match method {
        IdentificationMethodV1::RandomizedControlledTrial => 1,
        IdentificationMethodV1::QuasiExperimental => 2,
        IdentificationMethodV1::Observational => 3,
        IdentificationMethodV1::Unknown => 4,
    }
}

fn profile_digest(config: TypedContentConfigV1) -> String {
    let mut hasher = Sha256::new();
    hasher.update(PROFILE_DOMAIN);
    hasher.update((config.dimension as u64).to_be_bytes());
    hasher.update(config.seed.to_be_bytes());
    hasher.update(NUMERIC_KERNEL);
    hasher.update(config.utility_bandwidth.to_bits().to_be_bytes());
    hasher.update(config.measurement_error_bandwidth.to_bits().to_be_bytes());
    hasher.update(config.propensity_bandwidth.to_bits().to_be_bytes());
    hasher.update(config.balance_bandwidth.to_bits().to_be_bytes());
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn vector_bytes(dimension: usize) -> Result<u64, TypedContentError> {
    u64::try_from(dimension)
        .map_err(|_| TypedContentError::DimensionOutOfRange { actual: dimension })?
        .checked_mul(size_of::<f32>() as u64)
        .ok_or(TypedContentError::DimensionOutOfRange { actual: dimension })
}

#[cfg(test)]
mod tests {
    use super::*;
    use broca_mace_policy::{CausalIdentificationV1, Scope, SourceRootDigest};

    fn scope(regime: Regime, objective: Objective) -> Scope {
        Scope { regime, objective }
    }

    fn fixture(
        event_id: u32,
        action: Action,
        scope: Scope,
        utility: f64,
        measurement_error: f64,
    ) -> Experience {
        Experience {
            event_id,
            source_family: (event_id % u16::MAX as u32) as u16,
            source_root_id: event_id.saturating_add(1),
            source_root_digest: SourceRootDigest::UNKNOWN,
            scope,
            action,
            observed_utility: utility,
            measurement_error_bound: measurement_error,
            causal_identification: CausalIdentificationV1::UNKNOWN,
            intervention: true,
            prediction_pre_registered: true,
            outcome_reconciled: true,
            source_trusted: true,
            source_retracted: false,
            confounders_controlled: true,
            delayed_horizon_complete: true,
        }
    }

    fn query(scope: Scope, action: Action, target_utility: Option<f64>) -> SemanticExperienceQueryV1 {
        let mut query = SemanticExperienceQueryV1::exact_scope_action(scope, action);
        query.target_utility = target_utility;
        query
    }

    fn ids(result: &RetrievalResult) -> Vec<u32> {
        result.candidates.iter().map(|candidate| candidate.event_id).collect()
    }

    #[test]
    fn content_ranking_is_invariant_to_event_id_permutation() {
        let target_scope = scope(Regime::Baseline, Objective::Quality);
        let first = vec![
            fixture(10, Action::A, target_scope, 0.84, 0.01),
            fixture(20, Action::A, target_scope, 0.22, 0.01),
        ];
        let permuted = vec![
            fixture(900, Action::A, target_scope, 0.84, 0.01),
            fixture(300, Action::A, target_scope, 0.22, 0.01),
        ];
        let config = TypedContentConfigV1 { dimension: 2_048, ..TypedContentConfigV1::default() };
        let first_index = TypedContentIndexV1::build(&first, config).unwrap();
        let second_index = TypedContentIndexV1::build(&permuted, config).unwrap();
        let first_result = first_index.retrieve(query(target_scope, Action::A, Some(0.80)), 2).unwrap();
        let second_result = second_index.retrieve(query(target_scope, Action::A, Some(0.80)), 2).unwrap();

        assert_eq!(first_result.candidates[0].event_id, 10);
        assert_eq!(second_result.candidates[0].event_id, 900);
        assert!((first_result.candidates[0].similarity - second_result.candidates[0].similarity).abs() < 1e-6);
    }

    #[test]
    fn content_mutation_changes_ranking_without_changing_event_ids() {
        let target_scope = scope(Regime::Baseline, Objective::Quality);
        let before = vec![
            fixture(10, Action::A, target_scope, 0.20, 0.01),
            fixture(20, Action::A, target_scope, 0.90, 0.01),
        ];
        let after = vec![
            fixture(10, Action::A, target_scope, 0.92, 0.01),
            fixture(20, Action::A, target_scope, 0.18, 0.01),
        ];
        let config = TypedContentConfigV1 { dimension: 4_096, ..TypedContentConfigV1::default() };
        let before_index = TypedContentIndexV1::build(&before, config).unwrap();
        let after_index = TypedContentIndexV1::build(&after, config).unwrap();
        let decision_query = query(target_scope, Action::A, Some(0.90));

        assert_eq!(before_index.retrieve(decision_query, 2).unwrap().candidates[0].event_id, 20);
        assert_eq!(after_index.retrieve(decision_query, 2).unwrap().candidates[0].event_id, 10);
    }

    #[test]
    fn query_scope_and_action_changes_the_retrieval_set() {
        let baseline_quality = scope(Regime::Baseline, Objective::Quality);
        let shifted_cost = scope(Regime::Shifted, Objective::Cost);
        let memory = vec![
            fixture(1, Action::A, baseline_quality, 0.7, 0.01),
            fixture(2, Action::B, shifted_cost, 0.7, 0.01),
        ];
        let index = TypedContentIndexV1::build(
            &memory,
            TypedContentConfigV1 { dimension: 2_048, ..TypedContentConfigV1::default() },
        ).unwrap();

        assert_eq!(
            index.retrieve(query(baseline_quality, Action::A, None), 1).unwrap().candidates[0].event_id,
            1
        );
        assert_eq!(
            index.retrieve(query(shifted_cost, Action::B, None), 1).unwrap().candidates[0].event_id,
            2
        );
    }

    #[test]
    fn numeric_rff_similarity_prefers_nearby_values() {
        let target_scope = scope(Regime::Baseline, Objective::Quality);
        let memory = vec![
            fixture(1, Action::A, target_scope, 0.50, 0.01),
            fixture(2, Action::A, target_scope, 0.55, 0.01),
            fixture(3, Action::A, target_scope, 0.95, 0.01),
        ];
        let index = TypedContentIndexV1::build(
            &memory,
            TypedContentConfigV1 { dimension: 8_192, ..TypedContentConfigV1::default() },
        ).unwrap();
        let result = index.retrieve(query(target_scope, Action::A, Some(0.52)), 3).unwrap();
        let score_by_id: std::collections::BTreeMap<u32, f32> = result
            .candidates
            .iter()
            .map(|candidate| (candidate.event_id, candidate.similarity))
            .collect();

        assert!(score_by_id[&1] > score_by_id[&3]);
        assert!(score_by_id[&2] > score_by_id[&3]);
    }

    #[test]
    fn profile_digest_binds_dimension_seed_and_numeric_bandwidths() {
        let base = TypedContentConfigV1::default();
        let base_digest = profile_digest(base);
        assert_ne!(
            base_digest,
            profile_digest(TypedContentConfigV1 { seed: base.seed + 1, ..base })
        );
        assert_ne!(
            base_digest,
            profile_digest(TypedContentConfigV1 { dimension: base.dimension * 2, ..base })
        );
        assert_ne!(
            base_digest,
            profile_digest(TypedContentConfigV1 { utility_bandwidth: 0.2, ..base })
        );
    }

    #[test]
    fn duplicate_ids_and_non_finite_inputs_fail_closed() {
        let target_scope = scope(Regime::Baseline, Objective::Quality);
        let duplicate = vec![
            fixture(7, Action::A, target_scope, 0.5, 0.01),
            fixture(7, Action::B, target_scope, 0.5, 0.01),
        ];
        assert_eq!(
            TypedContentIndexV1::build(&duplicate, TypedContentConfigV1::default()).unwrap_err(),
            TypedContentError::DuplicateEventId { event_id: 7 }
        );

        let mut non_finite = fixture(8, Action::A, target_scope, 0.5, 0.01);
        non_finite.observed_utility = f64::NAN;
        assert!(matches!(
            TypedContentIndexV1::build(&[non_finite], TypedContentConfigV1::default()),
            Err(TypedContentError::NonFiniteExperience { field: "observed_utility", .. })
        ));
    }

    #[test]
    fn query_can_target_causal_estimand_and_evidence_quality_without_authorizing() {
        let target_scope = scope(Regime::Baseline, Objective::Quality);
        let mut item = fixture(11, Action::A, target_scope, 0.8, 0.01);
        item.causal_identification.estimand_digest = EvidenceDigest256::from_bytes([7; 32]);
        let index = TypedContentIndexV1::build(&[item], TypedContentConfigV1::default()).unwrap();
        let mut decision_query = query(target_scope, Action::A, None);
        decision_query.estimand_digest = Some(EvidenceDigest256::from_bytes([7; 32]));
        decision_query.identification_method = Some(IdentificationMethodV1::Unknown);
        decision_query.require_source_trusted = Some(true);
        decision_query.require_outcome_reconciled = Some(true);

        let result = index.retrieve(decision_query, 1).unwrap();
        assert_eq!(result.candidates[0].event_id, 11);
        // The index returns a candidate only. MACE qualification is a separate call.
        assert_eq!(result.candidates[0].view_digest, experience_view_digest(&item));
    }
}
