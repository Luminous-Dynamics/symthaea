// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content-blind observatory for explicit conversational corrections.
//!
//! This module measures correction adaptation, recurrence, and scope spillover.
//! It does not decide whether arbitrary natural-language content is correct, mutate
//! memory/preferences itself, or create any physical/user authority.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CORRECTION_OBSERVATORY_SCHEMA_V1: &str =
    "symthaea.communication.correction-observatory.v1";
const MAX_OBSERVATIONS_V1: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrectionDurabilityV1 {
    TurnOnly,
    Session,
    DurableExplicit,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrectionPayloadV1 {
    Replace { value_ref: String },
    Retract,
    NarrowScope { scope_ref: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExplicitCorrectionV1 {
    pub correction_id: String,
    pub subject_ref: String,
    pub semantic_key: String,
    pub context_id: String,
    pub durability: CorrectionDurabilityV1,
    pub payload: CorrectionPayloadV1,
    pub corrected_at_turn: u64,
    pub corrected_at_ns: Option<u64>,
    pub commitment: String,
}

impl ExplicitCorrectionV1 {
    pub fn new(
        correction_id: impl Into<String>,
        subject_ref: impl Into<String>,
        semantic_key: impl Into<String>,
        context_id: impl Into<String>,
        durability: CorrectionDurabilityV1,
        payload: CorrectionPayloadV1,
        corrected_at_turn: u64,
        corrected_at_ns: Option<u64>,
    ) -> Result<Self, CorrectionObservatoryErrorV1> {
        let correction_id = canonical_id(correction_id.into())?;
        let subject_ref = canonical_ref(subject_ref.into())?;
        let semantic_key = canonical_key(semantic_key.into())?;
        let context_id = canonical_key(context_id.into())?;
        let payload = canonical_payload(payload)?;

        let mut value = Self {
            correction_id,
            subject_ref,
            semantic_key,
            context_id,
            durability,
            payload,
            corrected_at_turn,
            corrected_at_ns,
            commitment: String::new(),
        };
        value.commitment = correction_commitment(&value);
        Ok(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CorrectionObservationKindV1 {
    ApplicableCompliant,
    ApplicableRecurrence,
    ApplicableAmbiguous,
    OutOfScopeUnchanged,
    OutOfScopeChangedConsistentWithCorrection,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrectionObservationV1 {
    pub observation_id: String,
    pub turn_index: u64,
    pub observed_at_ns: Option<u64>,
    pub semantic_key: String,
    pub context_id: String,
    pub kind: CorrectionObservationKindV1,
}

impl CorrectionObservationV1 {
    pub fn new(
        observation_id: impl Into<String>,
        turn_index: u64,
        observed_at_ns: Option<u64>,
        semantic_key: impl Into<String>,
        context_id: impl Into<String>,
        kind: CorrectionObservationKindV1,
    ) -> Result<Self, CorrectionObservatoryErrorV1> {
        Ok(Self {
            observation_id: canonical_id(observation_id.into())?,
            turn_index,
            observed_at_ns,
            semantic_key: canonical_key(semantic_key.into())?,
            context_id: canonical_key(context_id.into())?,
            kind,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrectionAdaptationStatusV1 {
    NotEstablished,
    NoCompliantObservation,
    Adapted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrectionRecurrenceStatusV1 {
    NotEvaluated,
    NoRecurrenceObserved,
    RecurrenceObserved,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrectionSpilloverStatusV1 {
    NotEvaluated,
    NoSpilloverObserved,
    SpilloverObserved,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrectionObservationReceiptV1 {
    pub schema: String,
    pub correction_id: String,
    pub correction_commitment: String,
    pub observation_set_commitment: String,
    pub receipt_commitment: String,
    pub adaptation_status: CorrectionAdaptationStatusV1,
    pub recurrence_status: CorrectionRecurrenceStatusV1,
    pub spillover_status: CorrectionSpilloverStatusV1,
    pub first_compliant_turn_delta: Option<u64>,
    pub first_compliant_latency_ns: Option<u64>,
    pub pre_adaptation_recurrence_count: u64,
    pub post_adaptation_recurrence_count: u64,
    pub applicable_ambiguous_count: u64,
    pub applicable_observation_count: u64,
    pub out_of_scope_observation_count: u64,
    pub spillover_count: u64,
    pub timestamped_observation_count: u64,
    pub total_observation_count: u64,
}

pub fn evaluate_correction(
    correction: &ExplicitCorrectionV1,
    observations: &[CorrectionObservationV1],
) -> Result<CorrectionObservationReceiptV1, CorrectionObservatoryErrorV1> {
    validate_correction(correction)?;
    if observations.len() > MAX_OBSERVATIONS_V1 {
        return Err(CorrectionObservatoryErrorV1::TooManyObservations);
    }

    let mut canonical = observations.to_vec();
    canonical.sort_by(|a, b| {
        (a.turn_index, &a.observation_id).cmp(&(b.turn_index, &b.observation_id))
    });

    let mut ids = BTreeSet::new();
    for observation in &canonical {
        validate_observation(correction, observation)?;
        if !ids.insert(observation.observation_id.clone()) {
            return Err(CorrectionObservatoryErrorV1::DuplicateObservationId);
        }
    }

    let mut first_compliant_turn_delta = None;
    let mut first_compliant_latency_ns = None;
    let mut adaptation_seen = false;
    let mut pre_adaptation_recurrence_count = 0_u64;
    let mut post_adaptation_recurrence_count = 0_u64;
    let mut applicable_ambiguous_count = 0_u64;
    let mut applicable_observation_count = 0_u64;
    let mut out_of_scope_observation_count = 0_u64;
    let mut spillover_count = 0_u64;
    let mut timestamped_observation_count = 0_u64;

    for observation in &canonical {
        if observation.observed_at_ns.is_some() {
            timestamped_observation_count += 1;
        }
        match observation.kind {
            CorrectionObservationKindV1::ApplicableCompliant => {
                applicable_observation_count += 1;
                if !adaptation_seen {
                    adaptation_seen = true;
                    first_compliant_turn_delta = Some(
                        observation
                            .turn_index
                            .checked_sub(correction.corrected_at_turn)
                            .expect("validated observation is after correction turn"),
                    );
                    first_compliant_latency_ns = match (
                        correction.corrected_at_ns,
                        observation.observed_at_ns,
                    ) {
                        (Some(corrected), Some(observed)) => Some(
                            observed
                                .checked_sub(corrected)
                                .expect("validated observation timestamp is not stale"),
                        ),
                        _ => None,
                    };
                }
            }
            CorrectionObservationKindV1::ApplicableRecurrence => {
                applicable_observation_count += 1;
                if adaptation_seen {
                    post_adaptation_recurrence_count += 1;
                } else {
                    pre_adaptation_recurrence_count += 1;
                }
            }
            CorrectionObservationKindV1::ApplicableAmbiguous => {
                applicable_observation_count += 1;
                applicable_ambiguous_count += 1;
            }
            CorrectionObservationKindV1::OutOfScopeUnchanged => {
                out_of_scope_observation_count += 1;
            }
            CorrectionObservationKindV1::OutOfScopeChangedConsistentWithCorrection => {
                out_of_scope_observation_count += 1;
                spillover_count += 1;
            }
        }
    }

    let adaptation_status = if applicable_observation_count == 0 {
        CorrectionAdaptationStatusV1::NotEstablished
    } else if adaptation_seen {
        CorrectionAdaptationStatusV1::Adapted
    } else {
        CorrectionAdaptationStatusV1::NoCompliantObservation
    };

    let recurrence_status = if applicable_observation_count == 0 {
        CorrectionRecurrenceStatusV1::NotEvaluated
    } else if pre_adaptation_recurrence_count + post_adaptation_recurrence_count == 0 {
        CorrectionRecurrenceStatusV1::NoRecurrenceObserved
    } else {
        CorrectionRecurrenceStatusV1::RecurrenceObserved
    };

    let spillover_status = if out_of_scope_observation_count == 0 {
        CorrectionSpilloverStatusV1::NotEvaluated
    } else if spillover_count == 0 {
        CorrectionSpilloverStatusV1::NoSpilloverObserved
    } else {
        CorrectionSpilloverStatusV1::SpilloverObserved
    };

    let observation_set_commitment = observation_set_commitment(correction, &canonical);
    let mut receipt = CorrectionObservationReceiptV1 {
        schema: CORRECTION_OBSERVATORY_SCHEMA_V1.into(),
        correction_id: correction.correction_id.clone(),
        correction_commitment: correction.commitment.clone(),
        observation_set_commitment,
        receipt_commitment: String::new(),
        adaptation_status,
        recurrence_status,
        spillover_status,
        first_compliant_turn_delta,
        first_compliant_latency_ns,
        pre_adaptation_recurrence_count,
        post_adaptation_recurrence_count,
        applicable_ambiguous_count,
        applicable_observation_count,
        out_of_scope_observation_count,
        spillover_count,
        timestamped_observation_count,
        total_observation_count: canonical.len() as u64,
    };
    receipt.receipt_commitment = receipt_commitment(&receipt);
    Ok(receipt)
}

fn validate_correction(
    correction: &ExplicitCorrectionV1,
) -> Result<(), CorrectionObservatoryErrorV1> {
    if correction.commitment != correction_commitment(correction) {
        return Err(CorrectionObservatoryErrorV1::CorrectionCommitmentMismatch);
    }
    Ok(())
}

fn validate_observation(
    correction: &ExplicitCorrectionV1,
    observation: &CorrectionObservationV1,
) -> Result<(), CorrectionObservatoryErrorV1> {
    if observation.turn_index <= correction.corrected_at_turn {
        return Err(CorrectionObservatoryErrorV1::ObservationNotAfterCorrection);
    }
    if let (Some(corrected), Some(observed)) =
        (correction.corrected_at_ns, observation.observed_at_ns)
    {
        if observed < corrected {
            return Err(CorrectionObservatoryErrorV1::ObservationTimestampBeforeCorrection);
        }
    }

    let in_exact_scope = observation.semantic_key == correction.semantic_key
        && observation.context_id == correction.context_id;
    match observation.kind {
        CorrectionObservationKindV1::ApplicableCompliant
        | CorrectionObservationKindV1::ApplicableRecurrence
        | CorrectionObservationKindV1::ApplicableAmbiguous
            if !in_exact_scope =>
        {
            Err(CorrectionObservatoryErrorV1::ApplicableObservationOutOfScope)
        }
        CorrectionObservationKindV1::OutOfScopeUnchanged
        | CorrectionObservationKindV1::OutOfScopeChangedConsistentWithCorrection
            if in_exact_scope =>
        {
            Err(CorrectionObservatoryErrorV1::OutOfScopeObservationInExactScope)
        }
        _ => Ok(()),
    }
}

fn canonical_payload(
    payload: CorrectionPayloadV1,
) -> Result<CorrectionPayloadV1, CorrectionObservatoryErrorV1> {
    match payload {
        CorrectionPayloadV1::Replace { value_ref } => Ok(CorrectionPayloadV1::Replace {
            value_ref: canonical_ref(value_ref)?,
        }),
        CorrectionPayloadV1::Retract => Ok(CorrectionPayloadV1::Retract),
        CorrectionPayloadV1::NarrowScope { scope_ref } => Ok(CorrectionPayloadV1::NarrowScope {
            scope_ref: canonical_ref(scope_ref)?,
        }),
    }
}

fn correction_commitment(correction: &ExplicitCorrectionV1) -> String {
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, "symthaea-correction-directive-v1");
    put_str(&mut hasher, &correction.correction_id);
    put_str(&mut hasher, &correction.subject_ref);
    put_str(&mut hasher, &correction.semantic_key);
    put_str(&mut hasher, &correction.context_id);
    put_u8(&mut hasher, durability_tag(correction.durability));
    match &correction.payload {
        CorrectionPayloadV1::Replace { value_ref } => {
            put_u8(&mut hasher, 1);
            put_str(&mut hasher, value_ref);
        }
        CorrectionPayloadV1::Retract => put_u8(&mut hasher, 2),
        CorrectionPayloadV1::NarrowScope { scope_ref } => {
            put_u8(&mut hasher, 3);
            put_str(&mut hasher, scope_ref);
        }
    }
    put_u64(&mut hasher, correction.corrected_at_turn);
    put_optional_u64(&mut hasher, correction.corrected_at_ns);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn observation_set_commitment(
    correction: &ExplicitCorrectionV1,
    observations: &[CorrectionObservationV1],
) -> String {
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, "symthaea-correction-observation-set-v1");
    put_str(&mut hasher, &correction.commitment);
    put_u64(&mut hasher, observations.len() as u64);
    for observation in observations {
        put_str(&mut hasher, &observation.observation_id);
        put_u64(&mut hasher, observation.turn_index);
        put_optional_u64(&mut hasher, observation.observed_at_ns);
        put_str(&mut hasher, &observation.semantic_key);
        put_str(&mut hasher, &observation.context_id);
        put_u8(&mut hasher, observation_kind_tag(observation.kind));
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn receipt_commitment(receipt: &CorrectionObservationReceiptV1) -> String {
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, "symthaea-correction-observation-receipt-v1");
    put_str(&mut hasher, &receipt.correction_commitment);
    put_str(&mut hasher, &receipt.observation_set_commitment);
    put_u8(&mut hasher, adaptation_status_tag(receipt.adaptation_status));
    put_u8(&mut hasher, recurrence_status_tag(receipt.recurrence_status));
    put_u8(&mut hasher, spillover_status_tag(receipt.spillover_status));
    put_optional_u64(&mut hasher, receipt.first_compliant_turn_delta);
    put_optional_u64(&mut hasher, receipt.first_compliant_latency_ns);
    put_u64(&mut hasher, receipt.pre_adaptation_recurrence_count);
    put_u64(&mut hasher, receipt.post_adaptation_recurrence_count);
    put_u64(&mut hasher, receipt.applicable_ambiguous_count);
    put_u64(&mut hasher, receipt.applicable_observation_count);
    put_u64(&mut hasher, receipt.out_of_scope_observation_count);
    put_u64(&mut hasher, receipt.spillover_count);
    put_u64(&mut hasher, receipt.timestamped_observation_count);
    put_u64(&mut hasher, receipt.total_observation_count);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn canonical_id(value: String) -> Result<String, CorrectionObservatoryErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 256 {
        return Err(CorrectionObservatoryErrorV1::InvalidIdentity);
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':' | '/'))
    {
        return Err(CorrectionObservatoryErrorV1::InvalidIdentity);
    }
    Ok(value)
}

fn canonical_key(value: String) -> Result<String, CorrectionObservatoryErrorV1> {
    let value = value.trim().to_ascii_lowercase();
    if value.is_empty() || value.len() > 256 {
        return Err(CorrectionObservatoryErrorV1::InvalidSemanticKey);
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':' | '/'))
    {
        return Err(CorrectionObservatoryErrorV1::InvalidSemanticKey);
    }
    Ok(value)
}

fn canonical_ref(value: String) -> Result<String, CorrectionObservatoryErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 1024 {
        return Err(CorrectionObservatoryErrorV1::InvalidReference);
    }
    Ok(value)
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

fn put_optional_u64(hasher: &mut blake3::Hasher, value: Option<u64>) {
    match value {
        Some(value) => {
            put_u8(hasher, 1);
            put_u64(hasher, value);
        }
        None => put_u8(hasher, 0),
    }
}

const fn durability_tag(value: CorrectionDurabilityV1) -> u8 {
    match value {
        CorrectionDurabilityV1::TurnOnly => 1,
        CorrectionDurabilityV1::Session => 2,
        CorrectionDurabilityV1::DurableExplicit => 3,
    }
}

const fn observation_kind_tag(value: CorrectionObservationKindV1) -> u8 {
    match value {
        CorrectionObservationKindV1::ApplicableCompliant => 1,
        CorrectionObservationKindV1::ApplicableRecurrence => 2,
        CorrectionObservationKindV1::ApplicableAmbiguous => 3,
        CorrectionObservationKindV1::OutOfScopeUnchanged => 4,
        CorrectionObservationKindV1::OutOfScopeChangedConsistentWithCorrection => 5,
    }
}

const fn adaptation_status_tag(value: CorrectionAdaptationStatusV1) -> u8 {
    match value {
        CorrectionAdaptationStatusV1::NotEstablished => 1,
        CorrectionAdaptationStatusV1::NoCompliantObservation => 2,
        CorrectionAdaptationStatusV1::Adapted => 3,
    }
}

const fn recurrence_status_tag(value: CorrectionRecurrenceStatusV1) -> u8 {
    match value {
        CorrectionRecurrenceStatusV1::NotEvaluated => 1,
        CorrectionRecurrenceStatusV1::NoRecurrenceObserved => 2,
        CorrectionRecurrenceStatusV1::RecurrenceObserved => 3,
    }
}

const fn spillover_status_tag(value: CorrectionSpilloverStatusV1) -> u8 {
    match value {
        CorrectionSpilloverStatusV1::NotEvaluated => 1,
        CorrectionSpilloverStatusV1::NoSpilloverObserved => 2,
        CorrectionSpilloverStatusV1::SpilloverObserved => 3,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrectionObservatoryErrorV1 {
    InvalidIdentity,
    InvalidSemanticKey,
    InvalidReference,
    TooManyObservations,
    DuplicateObservationId,
    CorrectionCommitmentMismatch,
    ObservationNotAfterCorrection,
    ObservationTimestampBeforeCorrection,
    ApplicableObservationOutOfScope,
    OutOfScopeObservationInExactScope,
}
