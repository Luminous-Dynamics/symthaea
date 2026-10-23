// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persona continuity observatory with context-sensitive admissible ranges.
//!
//! This evaluates an explicit designed persona contract. It does not establish
//! consciousness/personhood, infer a human personality, or create authority.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PERSONA_CONTINUITY_SCHEMA_V1: &str =
    "symthaea.communication.persona-continuity-observatory.v1";
const MAX_FACETS_V1: usize = 256;
const MAX_CONTEXT_BANDS_PER_FACET_V1: usize = 256;
const MAX_OBSERVATIONS_V1: usize = 8192;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PersonaFacetCriticalityV1 {
    Critical,
    NonCritical,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersonaBandV1 {
    pub min: f32,
    pub max: f32,
}

impl PersonaBandV1 {
    pub fn new(min: f32, max: f32) -> Result<Self, PersonaObservatoryErrorV1> {
        if !min.is_finite()
            || !max.is_finite()
            || !(0.0..=1.0).contains(&min)
            || !(0.0..=1.0).contains(&max)
            || min > max
        {
            return Err(PersonaObservatoryErrorV1::InvalidBand);
        }
        Ok(Self { min, max })
    }

    pub fn contains(&self, value: f32) -> bool {
        value.is_finite() && value >= self.min && value <= self.max
    }

    fn is_subset_of(&self, parent: &Self) -> bool {
        self.min >= parent.min && self.max <= parent.max
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersonaFacetPolicyV1 {
    pub facet_key: String,
    pub global_band: PersonaBandV1,
    pub criticality: PersonaFacetCriticalityV1,
    pub required: bool,
    pub allow_context_broadening: bool,
    pub context_bands: BTreeMap<String, PersonaBandV1>,
}

impl PersonaFacetPolicyV1 {
    pub fn new(
        facet_key: impl Into<String>,
        global_band: PersonaBandV1,
        criticality: PersonaFacetCriticalityV1,
        required: bool,
        allow_context_broadening: bool,
        context_bands: Vec<(String, PersonaBandV1)>,
    ) -> Result<Self, PersonaObservatoryErrorV1> {
        if context_bands.len() > MAX_CONTEXT_BANDS_PER_FACET_V1 {
            return Err(PersonaObservatoryErrorV1::TooManyContextBands);
        }
        let facet_key = canonical_key(facet_key.into())?;
        let mut canonical_contexts = BTreeMap::new();
        for (context_id, band) in context_bands {
            let context_id = canonical_key(context_id)?;
            if !allow_context_broadening && !band.is_subset_of(&global_band) {
                return Err(PersonaObservatoryErrorV1::ContextBandBroadensGlobalWithoutPermission);
            }
            if canonical_contexts.insert(context_id, band).is_some() {
                return Err(PersonaObservatoryErrorV1::DuplicateContextBand);
            }
        }
        Ok(Self {
            facet_key,
            global_band,
            criticality,
            required,
            allow_context_broadening,
            context_bands: canonical_contexts,
        })
    }

    fn applicable_band(&self, context_id: &str) -> (PersonaBandV1, bool) {
        self.context_bands
            .get(context_id)
            .copied()
            .map(|band| (band, true))
            .unwrap_or((self.global_band, false))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersonaProfileV1 {
    pub profile_id: String,
    pub profile_version: String,
    pub evaluator_disagreement_tolerance: f32,
    pub facets: BTreeMap<String, PersonaFacetPolicyV1>,
    pub commitment: String,
}

impl PersonaProfileV1 {
    pub fn new(
        profile_id: impl Into<String>,
        profile_version: impl Into<String>,
        evaluator_disagreement_tolerance: f32,
        facets: Vec<PersonaFacetPolicyV1>,
    ) -> Result<Self, PersonaObservatoryErrorV1> {
        if facets.is_empty() || facets.len() > MAX_FACETS_V1 {
            return Err(PersonaObservatoryErrorV1::InvalidFacetCount);
        }
        if !evaluator_disagreement_tolerance.is_finite()
            || !(0.0..=1.0).contains(&evaluator_disagreement_tolerance)
        {
            return Err(PersonaObservatoryErrorV1::InvalidDisagreementTolerance);
        }
        let profile_id = canonical_id(profile_id.into())?;
        let profile_version = canonical_key(profile_version.into())?;
        let mut facet_map = BTreeMap::new();
        for facet in facets {
            if facet_map.insert(facet.facet_key.clone(), facet).is_some() {
                return Err(PersonaObservatoryErrorV1::DuplicateFacet);
            }
        }
        let mut profile = Self {
            profile_id,
            profile_version,
            evaluator_disagreement_tolerance,
            facets: facet_map,
            commitment: String::new(),
        };
        profile.commitment = profile_commitment(&profile);
        Ok(profile)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersonaObservationV1 {
    pub observation_id: String,
    pub episode_id: String,
    pub turn_index: u64,
    pub context_id: String,
    pub facet_key: String,
    pub value: f32,
    pub evaluator_id: String,
    pub evidence_ref: String,
}

impl PersonaObservationV1 {
    pub fn new(
        observation_id: impl Into<String>,
        episode_id: impl Into<String>,
        turn_index: u64,
        context_id: impl Into<String>,
        facet_key: impl Into<String>,
        value: f32,
        evaluator_id: impl Into<String>,
        evidence_ref: impl Into<String>,
    ) -> Result<Self, PersonaObservatoryErrorV1> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(PersonaObservatoryErrorV1::InvalidObservationValue);
        }
        Ok(Self {
            observation_id: canonical_id(observation_id.into())?,
            episode_id: canonical_id(episode_id.into())?,
            turn_index,
            context_id: canonical_key(context_id.into())?,
            facet_key: canonical_key(facet_key.into())?,
            value,
            evaluator_id: canonical_id(evaluator_id.into())?,
            evidence_ref: canonical_ref(evidence_ref.into())?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PersonaEvidenceStatusV1 {
    NotEstablished,
    Partial,
    Established,
    Uncertain,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersonaContinuityReceiptV1 {
    pub schema: String,
    pub profile_id: String,
    pub profile_version: String,
    pub profile_commitment: String,
    pub observation_set_commitment: String,
    pub receipt_commitment: String,
    pub evidence_status: PersonaEvidenceStatusV1,
    pub critical_violation_count: u64,
    pub noncritical_violation_count: u64,
    pub global_band_violation_count: u64,
    pub context_band_violation_count: u64,
    pub material_disagreement_group_count: u64,
    pub within_band_observation_count: u64,
    pub observed_facet_count: u64,
    pub profile_facet_count: u64,
    pub observed_context_count: u64,
    pub total_observation_count: u64,
    pub unobserved_required_facets: Vec<String>,
}

pub fn evaluate_persona_continuity(
    profile: &PersonaProfileV1,
    observations: &[PersonaObservationV1],
) -> Result<PersonaContinuityReceiptV1, PersonaObservatoryErrorV1> {
    validate_profile(profile)?;
    if observations.len() > MAX_OBSERVATIONS_V1 {
        return Err(PersonaObservatoryErrorV1::TooManyObservations);
    }

    let mut canonical = observations.to_vec();
    canonical.sort_by(|a, b| {
        (
            &a.episode_id,
            a.turn_index,
            &a.context_id,
            &a.facet_key,
            &a.evaluator_id,
            &a.observation_id,
        )
            .cmp(&(
                &b.episode_id,
                b.turn_index,
                &b.context_id,
                &b.facet_key,
                &b.evaluator_id,
                &b.observation_id,
            ))
    });

    let mut observation_ids = BTreeSet::new();
    let mut evaluator_slots = BTreeSet::new();
    let mut observed_facets = BTreeSet::new();
    let mut observed_contexts = BTreeSet::new();
    let mut groups: BTreeMap<(String, u64, String, String), (f32, f32)> = BTreeMap::new();

    let mut critical_violation_count = 0_u64;
    let mut noncritical_violation_count = 0_u64;
    let mut global_band_violation_count = 0_u64;
    let mut context_band_violation_count = 0_u64;
    let mut within_band_observation_count = 0_u64;

    for observation in &canonical {
        if !observation_ids.insert(observation.observation_id.clone()) {
            return Err(PersonaObservatoryErrorV1::DuplicateObservationId);
        }
        let facet = profile
            .facets
            .get(&observation.facet_key)
            .ok_or(PersonaObservatoryErrorV1::UnknownFacet)?;
        let evaluator_slot = (
            observation.episode_id.clone(),
            observation.turn_index,
            observation.context_id.clone(),
            observation.facet_key.clone(),
            observation.evaluator_id.clone(),
        );
        if !evaluator_slots.insert(evaluator_slot) {
            return Err(PersonaObservatoryErrorV1::DuplicateEvaluatorObservation);
        }

        observed_facets.insert(observation.facet_key.clone());
        observed_contexts.insert(observation.context_id.clone());
        let (band, is_context_band) = facet.applicable_band(&observation.context_id);
        if band.contains(observation.value) {
            within_band_observation_count += 1;
        } else {
            match facet.criticality {
                PersonaFacetCriticalityV1::Critical => critical_violation_count += 1,
                PersonaFacetCriticalityV1::NonCritical => noncritical_violation_count += 1,
            }
            if is_context_band {
                context_band_violation_count += 1;
            } else {
                global_band_violation_count += 1;
            }
        }

        let group_key = (
            observation.episode_id.clone(),
            observation.turn_index,
            observation.context_id.clone(),
            observation.facet_key.clone(),
        );
        groups
            .entry(group_key)
            .and_modify(|(min, max)| {
                *min = min.min(observation.value);
                *max = max.max(observation.value);
            })
            .or_insert((observation.value, observation.value));
    }

    let material_disagreement_group_count = groups
        .values()
        .filter(|(min, max)| *max - *min > profile.evaluator_disagreement_tolerance)
        .count() as u64;

    let unobserved_required_facets: Vec<String> = profile
        .facets
        .values()
        .filter(|facet| facet.required && !observed_facets.contains(&facet.facet_key))
        .map(|facet| facet.facet_key.clone())
        .collect();

    let evidence_status = if canonical.is_empty() {
        PersonaEvidenceStatusV1::NotEstablished
    } else if material_disagreement_group_count > 0 {
        PersonaEvidenceStatusV1::Uncertain
    } else if !unobserved_required_facets.is_empty() {
        PersonaEvidenceStatusV1::Partial
    } else {
        PersonaEvidenceStatusV1::Established
    };

    let observation_set_commitment = observation_set_commitment(profile, &canonical);
    let mut receipt = PersonaContinuityReceiptV1 {
        schema: PERSONA_CONTINUITY_SCHEMA_V1.into(),
        profile_id: profile.profile_id.clone(),
        profile_version: profile.profile_version.clone(),
        profile_commitment: profile.commitment.clone(),
        observation_set_commitment,
        receipt_commitment: String::new(),
        evidence_status,
        critical_violation_count,
        noncritical_violation_count,
        global_band_violation_count,
        context_band_violation_count,
        material_disagreement_group_count,
        within_band_observation_count,
        observed_facet_count: observed_facets.len() as u64,
        profile_facet_count: profile.facets.len() as u64,
        observed_context_count: observed_contexts.len() as u64,
        total_observation_count: canonical.len() as u64,
        unobserved_required_facets,
    };
    receipt.receipt_commitment = receipt_commitment(&receipt);
    Ok(receipt)
}

fn validate_profile(profile: &PersonaProfileV1) -> Result<(), PersonaObservatoryErrorV1> {
    if profile.commitment != profile_commitment(profile) {
        return Err(PersonaObservatoryErrorV1::ProfileCommitmentMismatch);
    }
    Ok(())
}

fn profile_commitment(profile: &PersonaProfileV1) -> String {
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, "symthaea-persona-profile-v1");
    put_str(&mut hasher, &profile.profile_id);
    put_str(&mut hasher, &profile.profile_version);
    put_f32(&mut hasher, profile.evaluator_disagreement_tolerance);
    put_u64(&mut hasher, profile.facets.len() as u64);
    for facet in profile.facets.values() {
        put_str(&mut hasher, &facet.facet_key);
        put_band(&mut hasher, facet.global_band);
        put_u8(&mut hasher, criticality_tag(facet.criticality));
        put_bool(&mut hasher, facet.required);
        put_bool(&mut hasher, facet.allow_context_broadening);
        put_u64(&mut hasher, facet.context_bands.len() as u64);
        for (context_id, band) in &facet.context_bands {
            put_str(&mut hasher, context_id);
            put_band(&mut hasher, *band);
        }
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn observation_set_commitment(
    profile: &PersonaProfileV1,
    observations: &[PersonaObservationV1],
) -> String {
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, "symthaea-persona-observation-set-v1");
    put_str(&mut hasher, &profile.commitment);
    put_u64(&mut hasher, observations.len() as u64);
    for observation in observations {
        put_str(&mut hasher, &observation.observation_id);
        put_str(&mut hasher, &observation.episode_id);
        put_u64(&mut hasher, observation.turn_index);
        put_str(&mut hasher, &observation.context_id);
        put_str(&mut hasher, &observation.facet_key);
        put_f32(&mut hasher, observation.value);
        put_str(&mut hasher, &observation.evaluator_id);
        put_str(&mut hasher, &observation.evidence_ref);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn receipt_commitment(receipt: &PersonaContinuityReceiptV1) -> String {
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, "symthaea-persona-continuity-receipt-v1");
    put_str(&mut hasher, &receipt.profile_commitment);
    put_str(&mut hasher, &receipt.observation_set_commitment);
    put_u8(&mut hasher, evidence_status_tag(receipt.evidence_status));
    put_u64(&mut hasher, receipt.critical_violation_count);
    put_u64(&mut hasher, receipt.noncritical_violation_count);
    put_u64(&mut hasher, receipt.global_band_violation_count);
    put_u64(&mut hasher, receipt.context_band_violation_count);
    put_u64(&mut hasher, receipt.material_disagreement_group_count);
    put_u64(&mut hasher, receipt.within_band_observation_count);
    put_u64(&mut hasher, receipt.observed_facet_count);
    put_u64(&mut hasher, receipt.profile_facet_count);
    put_u64(&mut hasher, receipt.observed_context_count);
    put_u64(&mut hasher, receipt.total_observation_count);
    put_u64(&mut hasher, receipt.unobserved_required_facets.len() as u64);
    for facet in &receipt.unobserved_required_facets {
        put_str(&mut hasher, facet);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn canonical_id(value: String) -> Result<String, PersonaObservatoryErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 256 {
        return Err(PersonaObservatoryErrorV1::InvalidIdentity);
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':' | '/'))
    {
        return Err(PersonaObservatoryErrorV1::InvalidIdentity);
    }
    Ok(value)
}

fn canonical_key(value: String) -> Result<String, PersonaObservatoryErrorV1> {
    let value = value.trim().to_ascii_lowercase();
    if value.is_empty() || value.len() > 256 {
        return Err(PersonaObservatoryErrorV1::InvalidKey);
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':' | '/'))
    {
        return Err(PersonaObservatoryErrorV1::InvalidKey);
    }
    Ok(value)
}

fn canonical_ref(value: String) -> Result<String, PersonaObservatoryErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 1024 {
        return Err(PersonaObservatoryErrorV1::InvalidReference);
    }
    Ok(value)
}

fn put_band(hasher: &mut blake3::Hasher, band: PersonaBandV1) {
    put_f32(hasher, band.min);
    put_f32(hasher, band.max);
}

fn put_f32(hasher: &mut blake3::Hasher, value: f32) {
    hasher.update(&value.to_bits().to_le_bytes());
}

fn put_str(hasher: &mut blake3::Hasher, value: &str) {
    put_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

fn put_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn put_u8(hasher: &mut blake3::Hasher, value: u8) {
    hasher.update(&[value]);
}

fn put_bool(hasher: &mut blake3::Hasher, value: bool) {
    put_u8(hasher, u8::from(value));
}

const fn criticality_tag(value: PersonaFacetCriticalityV1) -> u8 {
    match value {
        PersonaFacetCriticalityV1::Critical => 1,
        PersonaFacetCriticalityV1::NonCritical => 2,
    }
}

const fn evidence_status_tag(value: PersonaEvidenceStatusV1) -> u8 {
    match value {
        PersonaEvidenceStatusV1::NotEstablished => 1,
        PersonaEvidenceStatusV1::Partial => 2,
        PersonaEvidenceStatusV1::Established => 3,
        PersonaEvidenceStatusV1::Uncertain => 4,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PersonaObservatoryErrorV1 {
    InvalidBand,
    InvalidFacetCount,
    InvalidDisagreementTolerance,
    InvalidObservationValue,
    InvalidIdentity,
    InvalidKey,
    InvalidReference,
    TooManyContextBands,
    TooManyObservations,
    DuplicateContextBand,
    DuplicateFacet,
    DuplicateObservationId,
    DuplicateEvaluatorObservation,
    ContextBandBroadensGlobalWithoutPermission,
    ProfileCommitmentMismatch,
    UnknownFacet,
}
