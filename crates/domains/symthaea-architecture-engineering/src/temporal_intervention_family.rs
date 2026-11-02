//! Temporal intervention adapter for the canonical anytime-confidence family ledger.
//!
//! Each frozen treatment/control contrast expands to three bounded sign streams:
//! wrong-belief, abstention, and missing-output exposure. A beneficial paired sign
//! maps to 1.0, a harmful sign to 0.0, and a tie to 0.5. Under independently
//! randomized fair-coin pair orientation and Fisher's sharp no-effect null, each
//! stream observation has conditional mean 0.5.
//!
//! This adapter reuses the generic family manifest, alpha allocation, look ledger,
//! cumulative-prefix receipts, hash chain, and replay verifier. It does not create
//! a second generic ledger.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::anytime_confidence_ledger::{
    AnytimeConfidenceFamilyManifest, AnytimeConfidenceLedger, AnytimeLedgerError,
    AnytimeStreamDeclaration, append_anytime_confidence_look,
    create_anytime_confidence_family_manifest, create_anytime_confidence_ledger,
    validate_anytime_confidence_family_manifest, verify_anytime_confidence_ledger,
};
use crate::anytime_qmc_confidence::AnytimeBoundedMeanReceipt;

const FAMILY_SCHEMA: &str = "symthaea.temporal-intervention-family";
const FAMILY_VERSION: &str = "0.1.0";
const EVIDENCE_SCHEMA: &str = "symthaea.temporal-intervention-evidence-ledger";
const EVIDENCE_VERSION: &str = "0.1.0";
const CONTRAST_DOMAIN: &[u8] = b"symthaea.temporal-intervention-contrast.v0.1\0";
const PAIR_BATCH_DOMAIN: &[u8] = b"symthaea.temporal-intervention-pair-batch.v0.1\0";
const EVIDENCE_LEDGER_DOMAIN: &[u8] = b"symthaea.temporal-intervention-evidence-ledger.v0.1\0";
const MAX_CONTRASTS: usize = 30_000;
const MAX_BATCH_PAIRS: usize = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalInterventionAxis {
    Abstention,
    MissingOutput,
    WrongBelief,
}

impl TemporalInterventionAxis {
    pub const ALL: [Self; 3] = [
        Self::Abstention,
        Self::MissingOutput,
        Self::WrongBelief,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WrongBelief => "wrong_belief_exposure",
            Self::Abstention => "abstention_exposure",
            Self::MissingOutput => "missing_output_exposure",
        }
    }
}

/// Immutable identity of one preregistered treatment/control contrast.
/// Digests are lowercase hexadecimal SHA-256 values supplied by upstream authorities.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalInterventionContrast {
    pub contrast_id: String,
    pub task_sha256: String,
    pub model_sha256: String,
    pub evaluator_sha256: String,
    pub treatment_policy_sha256: String,
    pub control_policy_sha256: String,
    pub assignment_manifest_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalInterventionFamily {
    pub schema: String,
    pub version: String,
    pub contrasts: Vec<TemporalInterventionContrast>,
    /// Canonical generic manifest. Its family size is derived from its stream count.
    pub confidence_family: AnytimeConfidenceFamilyManifest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalInterventionAssignmentScheme {
    IndependentFairCoin,
    FixedCountBlocked,
    ClusterRandomized,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalInterventionPairOutcome {
    pub pair_id: u64,
    pub context_sha256: [u8; 32],
    pub source_family_sha256: [u8; 32],
    pub assignment_receipt_sha256: [u8; 32],
    pub assignment_scheme: TemporalInterventionAssignmentScheme,
    pub treatment_probability_ppm: u32,
    pub treatment_was_first: bool,
    pub episode_duration_us: u64,
    pub treatment_exposure_us: u64,
    pub control_exposure_us: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalInterventionBatchBinding {
    pub contrast_id: String,
    pub contrast_sha256: String,
    pub axis: TemporalInterventionAxis,
    pub stream_id: String,
    pub look_index: u64,
    pub pair_outcomes: Vec<TemporalInterventionPairOutcome>,
    pub pair_batch_sha256: String,
    pub confidence_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalInterventionEvidenceLedger {
    pub schema: String,
    pub version: String,
    pub family: TemporalInterventionFamily,
    /// Canonical append-only generic ledger; it owns alpha, looks, values and replay.
    pub confidence_ledger: AnytimeConfidenceLedger,
    /// Bindings connect each bounded sign observation to evaluator pair receipts.
    pub pair_bindings: Vec<TemporalInterventionBatchBinding>,
    pub evidence_ledger_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalInterventionLedgerError {
    pub field: String,
    pub message: String,
}

impl TemporalInterventionLedgerError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }

    fn from_ledger(error: AnytimeLedgerError) -> Self {
        Self::new(error.field, error.message)
    }
}

impl fmt::Display for TemporalInterventionLedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for TemporalInterventionLedgerError {}

/// Build a canonical family manifest with exactly three streams per contrast.
///
/// The contrast list must already be strictly sorted by contrast_id. Each stream has
/// support [0,1] and threshold 0.5. The generic ledger derives family size and
/// Bonferroni alpha from the complete stream set.
pub fn create_temporal_intervention_family(
    family_id: impl Into<String>,
    family_alpha: f64,
    contrasts: Vec<TemporalInterventionContrast>,
) -> Result<TemporalInterventionFamily, TemporalInterventionLedgerError> {
    validate_contrasts(&contrasts)?;
    let mut streams = Vec::with_capacity(contrasts.len() * 3);
    for contrast in &contrasts {
        let digest = contrast_sha256(contrast)?;
        for axis in TemporalInterventionAxis::ALL {
            streams.push(AnytimeStreamDeclaration {
                stream_id: temporal_stream_id(&contrast.contrast_id, &digest, axis),
                range_min: 0.0,
                range_max: 1.0,
                threshold: 0.5,
            });
        }
    }
    streams.sort_by(|left, right| left.stream_id.cmp(&right.stream_id));

    let confidence_family = create_anytime_confidence_family_manifest(
        family_id, family_alpha, streams,
    ).map_err(TemporalInterventionLedgerError::from_ledger)?;

    let family = TemporalInterventionFamily {
        schema: FAMILY_SCHEMA.to_owned(),
        version: FAMILY_VERSION.to_owned(),
        contrasts,
        confidence_family,
    };
    verify_temporal_intervention_family(&family)?;
    Ok(family)
}

/// Verify that the contrast identities reproduce every generic confidence stream.
pub fn verify_temporal_intervention_family(
    family: &TemporalInterventionFamily,
) -> Result<(), TemporalInterventionLedgerError> {
    if family.schema != FAMILY_SCHEMA || family.version != FAMILY_VERSION {
        return Err(TemporalInterventionLedgerError::new(
            "schema_version", "unsupported temporal intervention family schema or version",
        ));
    }
    validate_contrasts(&family.contrasts)?;
    validate_anytime_confidence_family_manifest(&family.confidence_family)
        .map_err(TemporalInterventionLedgerError::from_ledger)?;

    let mut expected = Vec::with_capacity(family.contrasts.len() * 3);
    for contrast in &family.contrasts {
        let digest = contrast_sha256(contrast)?;
        for axis in TemporalInterventionAxis::ALL {
            expected.push(AnytimeStreamDeclaration {
                stream_id: temporal_stream_id(&contrast.contrast_id, &digest, axis),
                range_min: 0.0,
                range_max: 1.0,
                threshold: 0.5,
            });
        }
    }
    expected.sort_by(|left, right| left.stream_id.cmp(&right.stream_id));
    let actual = &family.confidence_family;
    if actual.streams.len() != expected.len()
        || actual.streams.iter().zip(&expected).any(|(left, right)| {
            left.stream_id != right.stream_id
                || left.range_min.to_bits() != right.range_min.to_bits()
                || left.range_max.to_bits() != right.range_max.to_bits()
                || left.threshold.to_bits() != right.threshold.to_bits()
        })
    {
        return Err(TemporalInterventionLedgerError::new(
            "confidence_family.streams",
            "streams do not match the complete sorted contrast-by-axis declaration",
        ));
    }
    Ok(())
}

/// Create a temporal-intervention wrapper over the canonical generic look ledger.
pub fn create_temporal_intervention_evidence_ledger(
    family: TemporalInterventionFamily,
) -> Result<TemporalInterventionEvidenceLedger, TemporalInterventionLedgerError> {
    verify_temporal_intervention_family(&family)?;
    let confidence_ledger = create_anytime_confidence_ledger(family.confidence_family.clone())
        .map_err(TemporalInterventionLedgerError::from_ledger)?;
    let mut ledger = TemporalInterventionEvidenceLedger {
        schema: EVIDENCE_SCHEMA.to_owned(),
        version: EVIDENCE_VERSION.to_owned(),
        family,
        confidence_ledger,
        pair_bindings: Vec::new(),
        evidence_ledger_sha256: String::new(),
    };
    ledger.evidence_ledger_sha256 = compute_evidence_ledger_sha256(&ledger)?;
    Ok(ledger)
}

/// Append one batch to the preregistered contrast/axis stream.
///
/// The generic ledger computes its bounded-mean receipt and hash chain. This wrapper
/// additionally binds ordered pair IDs, contexts, source families, assignment-receipt
/// digests and realized paired signs to that exact generic look.
pub fn append_temporal_intervention_batch(
    ledger: &mut TemporalInterventionEvidenceLedger,
    contrast_id: &str,
    axis: TemporalInterventionAxis,
    pair_outcomes: &[TemporalInterventionPairOutcome],
) -> Result<AnytimeBoundedMeanReceipt, TemporalInterventionLedgerError> {
    verify_temporal_intervention_evidence_ledger(ledger)?;
    // Mutate a clone and commit only after the complete wrapper verifies.
    let mut next_ledger = ledger.clone();
    if pair_outcomes.is_empty() {
        return Err(TemporalInterventionLedgerError::new(
            "pair_outcomes", "an appended look must contain at least one pair",
        ));
    }
    if pair_outcomes.len() > MAX_BATCH_PAIRS {
        return Err(TemporalInterventionLedgerError::new(
            "pair_outcomes", format!("pair batch exceeds {MAX_BATCH_PAIRS}"),
        ));
    }

    let contrast = next_ledger.family.contrasts
        .binary_search_by(|item| item.contrast_id.as_str().cmp(contrast_id))
        .ok()
        .and_then(|index| ledger.family.contrasts.get(index))
        .ok_or_else(|| TemporalInterventionLedgerError::new(
            "contrast_id", format!("contrast {contrast_id:?} was not preregistered"),
        ))?;
    let digest = contrast_sha256(contrast)?;
    let stream_id = temporal_stream_id(contrast_id, &digest, axis);

    let prior_ids = existing_pair_ids_for_stream(&next_ledger, &stream_id);
    let prior_contexts = existing_contexts_for_stream(&next_ledger, &stream_id);
    let prior_receipts = existing_assignment_receipts_for_stream(&next_ledger, &stream_id);
    let (observations, pair_batch_sha256) = encode_pair_outcomes(
        &stream_id, pair_outcomes, &prior_ids, &prior_contexts, &prior_receipts,
    )?;

    let receipt = append_anytime_confidence_look(
        &mut next_ledger.confidence_ledger, &stream_id, &observations,
    ).map_err(TemporalInterventionLedgerError::from_ledger)?;

    next_ledger.pair_bindings.push(TemporalInterventionBatchBinding {
        contrast_id: contrast_id.to_owned(),
        contrast_sha256: digest,
        axis,
        stream_id,
        look_index: receipt.look_index,
        pair_outcomes: pair_outcomes.to_vec(),
        pair_batch_sha256,
        confidence_receipt_sha256: receipt.receipt_sha256.clone(),
    });
    next_ledger.evidence_ledger_sha256 = compute_evidence_ledger_sha256(&next_ledger)?;
    verify_temporal_intervention_evidence_ledger(&next_ledger)?;
    *ledger = next_ledger;
    Ok(receipt)
}

/// Replay-verify the full outer pair binding and underlying generic confidence ledger.
pub fn verify_temporal_intervention_evidence_ledger(
    ledger: &TemporalInterventionEvidenceLedger,
) -> Result<(), TemporalInterventionLedgerError> {
    if ledger.schema != EVIDENCE_SCHEMA || ledger.version != EVIDENCE_VERSION {
        return Err(TemporalInterventionLedgerError::new(
            "schema_version", "unsupported temporal evidence ledger schema or version",
        ));
    }
    verify_temporal_intervention_family(&ledger.family)?;
    verify_anytime_confidence_ledger(&ledger.confidence_ledger)
        .map_err(TemporalInterventionLedgerError::from_ledger)?;
    if ledger.confidence_ledger.manifest != ledger.family.confidence_family {
        return Err(TemporalInterventionLedgerError::new(
            "confidence_ledger.manifest", "generic ledger family differs from temporal family",
        ));
    }

    let expected_count: usize = ledger.confidence_ledger.streams
        .iter().map(|stream| stream.looks.len()).sum();
    if ledger.pair_bindings.len() != expected_count {
        return Err(TemporalInterventionLedgerError::new(
            "pair_bindings", "binding count differs from generic ledger look count",
        ));
    }

    let mut seen_looks = BTreeSet::new();
    let mut seen_pairs = BTreeMap::<String, BTreeSet<u64>>::new();
    let mut seen_contexts = BTreeMap::<String, BTreeSet<[u8; 32]>>::new();
    let mut seen_receipts = BTreeMap::<String, BTreeSet<[u8; 32]>>::new();

    for binding in &ledger.pair_bindings {
        let contrast = ledger.family.contrasts
            .binary_search_by(|item| item.contrast_id.as_str().cmp(binding.contrast_id.as_str()))
            .ok()
            .and_then(|index| ledger.family.contrasts.get(index))
            .ok_or_else(|| TemporalInterventionLedgerError::new(
                "pair_bindings.contrast_id", "binding references undeclared contrast",
            ))?;
        let digest = contrast_sha256(contrast)?;
        if binding.contrast_sha256 != digest
            || binding.stream_id != temporal_stream_id(&binding.contrast_id, &digest, binding.axis)
        {
            return Err(TemporalInterventionLedgerError::new(
                "pair_bindings.contrast_sha256", "binding identity/axis differs from frozen family",
            ));
        }
        if !seen_looks.insert((binding.stream_id.clone(), binding.look_index)) {
            return Err(TemporalInterventionLedgerError::new(
                "pair_bindings.look_index", "duplicate stream/look binding",
            ));
        }

        let ids = seen_pairs.entry(binding.stream_id.clone()).or_default();
        let contexts = seen_contexts.entry(binding.stream_id.clone()).or_default();
        let receipts = seen_receipts.entry(binding.stream_id.clone()).or_default();
        let (observations, batch_digest) = encode_pair_outcomes(
            &binding.stream_id, &binding.pair_outcomes, ids, contexts, receipts,
        )?;
        if batch_digest != binding.pair_batch_sha256 {
            return Err(TemporalInterventionLedgerError::new(
                "pair_bindings.pair_batch_sha256", "pair batch digest mismatch",
            ));
        }

        let stream = ledger.confidence_ledger.streams.iter()
            .find(|stream| stream.stream_id == binding.stream_id)
            .ok_or_else(|| TemporalInterventionLedgerError::new(
                "pair_bindings.stream_id", "binding references unknown generic stream",
            ))?;
        let index = usize::try_from(binding.look_index.saturating_sub(1)).unwrap_or(usize::MAX);
        let look = stream.looks.get(index)
            .filter(|look| look.look_index == binding.look_index)
            .ok_or_else(|| TemporalInterventionLedgerError::new(
                "pair_bindings.look_index", "binding has no corresponding generic look",
            ))?;
        if look.replicate_batch != observations
            || look.receipt.receipt_sha256 != binding.confidence_receipt_sha256
        {
            return Err(TemporalInterventionLedgerError::new(
                "pair_bindings", "pair signs do not reproduce stored generic observations/receipt",
            ));
        }
    }

    for stream in &ledger.confidence_ledger.streams {
        for look in &stream.looks {
            if !seen_looks.contains(&(stream.stream_id.clone(), look.look_index)) {
                return Err(TemporalInterventionLedgerError::new(
                    "pair_bindings", "generic ledger look has no evaluator-pair binding",
                ));
            }
        }
    }

    let expected_digest = compute_evidence_ledger_sha256(ledger)?;
    if expected_digest != ledger.evidence_ledger_sha256 {
        return Err(TemporalInterventionLedgerError::new(
            "evidence_ledger_sha256", "outer ledger digest mismatch",
        ));
    }
    Ok(())
}

fn validate_contrasts(
    contrasts: &[TemporalInterventionContrast],
) -> Result<(), TemporalInterventionLedgerError> {
    if contrasts.is_empty() || contrasts.len() > MAX_CONTRASTS {
        return Err(TemporalInterventionLedgerError::new(
            "contrasts", format!("contrast count must be in 1..={MAX_CONTRASTS}"),
        ));
    }
    if contrasts.windows(2).any(|pair| pair[0].contrast_id >= pair[1].contrast_id) {
        return Err(TemporalInterventionLedgerError::new(
            "contrasts", "contrast declarations must be strictly sorted by unique contrast_id",
        ));
    }
    for (index, contrast) in contrasts.iter().enumerate() {
        if !valid_stable_id(&contrast.contrast_id) {
            return Err(TemporalInterventionLedgerError::new(
                format!("contrasts[{index}].contrast_id"),
                "ID must be 1..=128 ASCII alphanumeric, dash, underscore or dot characters",
            ));
        }
        for (field, digest) in [
            ("task_sha256", &contrast.task_sha256),
            ("model_sha256", &contrast.model_sha256),
            ("evaluator_sha256", &contrast.evaluator_sha256),
            ("treatment_policy_sha256", &contrast.treatment_policy_sha256),
            ("control_policy_sha256", &contrast.control_policy_sha256),
            ("assignment_manifest_sha256", &contrast.assignment_manifest_sha256),
        ] {
            if !is_sha256_hex(digest) {
                return Err(TemporalInterventionLedgerError::new(
                    format!("contrasts[{index}].{field}"),
                    "must be lowercase hexadecimal SHA-256",
                ));
            }
        }
        if contrast.treatment_policy_sha256 == contrast.control_policy_sha256 {
            return Err(TemporalInterventionLedgerError::new(
                format!("contrasts[{index}].policy_digests"),
                "treatment and control policy digests must differ",
            ));
        }
    }
    Ok(())
}

fn contrast_sha256(
    contrast: &TemporalInterventionContrast,
) -> Result<String, TemporalInterventionLedgerError> {
    validate_contrasts(std::slice::from_ref(contrast))?;
    let mut hasher = Sha256::new();
    hasher.update(CONTRAST_DOMAIN);
    for value in [
        contrast.contrast_id.as_str(),
        contrast.task_sha256.as_str(),
        contrast.model_sha256.as_str(),
        contrast.evaluator_sha256.as_str(),
        contrast.treatment_policy_sha256.as_str(),
        contrast.control_policy_sha256.as_str(),
        contrast.assignment_manifest_sha256.as_str(),
    ] {
        hash_string(&mut hasher, value);
    }
    Ok(hex_digest(&hasher.finalize()))
}

fn temporal_stream_id(contrast_id: &str, contrast_digest: &str, axis: TemporalInterventionAxis) -> String {
    format!("temporal-intervention-v1/{contrast_id}/{contrast_digest}/{}", axis.as_str())
}

fn encode_pair_outcomes(
    stream_id: &str,
    pairs: &[TemporalInterventionPairOutcome],
    prior_pair_ids: &BTreeSet<u64>,
    prior_contexts: &BTreeSet<[u8; 32]>,
    prior_receipts: &BTreeSet<[u8; 32]>,
) -> Result<(Vec<f64>, String), TemporalInterventionLedgerError> {
    if pairs.is_empty() || pairs.len() > MAX_BATCH_PAIRS {
        return Err(TemporalInterventionLedgerError::new(
            "pair_outcomes", "pair batch must be non-empty and within the configured cap",
        ));
    }
    let mut pair_ids = prior_pair_ids.clone();
    let mut contexts = prior_contexts.clone();
    let mut receipts = prior_receipts.clone();
    let mut observations = Vec::with_capacity(pairs.len());

    let mut hasher = Sha256::new();
    hasher.update(PAIR_BATCH_DOMAIN);
    hash_string(&mut hasher, stream_id);
    hasher.update((pairs.len() as u64).to_le_bytes());

    for (index, pair) in pairs.iter().enumerate() {
        if pair.pair_id == 0 || pair.context_sha256 == [0; 32]
            || pair.source_family_sha256 == [0; 32]
            || pair.assignment_receipt_sha256 == [0; 32]
        {
            return Err(TemporalInterventionLedgerError::new(
                format!("pair_outcomes[{index}]"), "pair ID and all digests must be nonzero",
            ));
        }
        if !pair_ids.insert(pair.pair_id) {
            return Err(TemporalInterventionLedgerError::new(
                format!("pair_outcomes[{index}].pair_id"), "duplicate pair ID in this stream",
            ));
        }
        if !contexts.insert(pair.context_sha256) {
            return Err(TemporalInterventionLedgerError::new(
                format!("pair_outcomes[{index}].context_sha256"), "duplicate context in this stream",
            ));
        }
        if !receipts.insert(pair.assignment_receipt_sha256) {
            return Err(TemporalInterventionLedgerError::new(
                format!("pair_outcomes[{index}].assignment_receipt_sha256"),
                "duplicate assignment receipt in this stream",
            ));
        }

        if pair.assignment_scheme != TemporalInterventionAssignmentScheme::IndependentFairCoin {
            return Err(TemporalInterventionLedgerError::new(
                format!("pair_outcomes[{index}].assignment_scheme"),
                "this adapter supports independent fair-coin pair assignment only",
            ));
        }
        if pair.treatment_probability_ppm != 500_000 {
            return Err(TemporalInterventionLedgerError::new(
                format!("pair_outcomes[{index}].treatment_probability_ppm"),
                "treatment probability must be exactly 500000 PPM",
            ));
        }
        if pair.episode_duration_us == 0
            || pair.treatment_exposure_us > pair.episode_duration_us
            || pair.control_exposure_us > pair.episode_duration_us
        {
            return Err(TemporalInterventionLedgerError::new(
                format!("pair_outcomes[{index}].exposure"),
                "episode duration must be positive and each exposure must not exceed it",
            ));
        }
        let benefit_us = i128::from(pair.control_exposure_us)
            - i128::from(pair.treatment_exposure_us);
        observations.push(if benefit_us > 0 { 1.0 } else if benefit_us < 0 { 0.0 } else { 0.5 });
        hasher.update(pair.pair_id.to_le_bytes());
        hasher.update(pair.context_sha256);
        hasher.update(pair.source_family_sha256);
        hasher.update(pair.assignment_receipt_sha256);
        hasher.update([match pair.assignment_scheme {
            TemporalInterventionAssignmentScheme::IndependentFairCoin => 0,
            TemporalInterventionAssignmentScheme::FixedCountBlocked => 1,
            TemporalInterventionAssignmentScheme::ClusterRandomized => 2,
        }]);
        hasher.update(pair.treatment_probability_ppm.to_le_bytes());
        hasher.update([u8::from(pair.treatment_was_first)]);
        hasher.update(pair.episode_duration_us.to_le_bytes());
        hasher.update(pair.treatment_exposure_us.to_le_bytes());
        hasher.update(pair.control_exposure_us.to_le_bytes());
    }
    Ok((observations, hex_digest(&hasher.finalize())))
}

fn existing_pair_ids_for_stream(ledger: &TemporalInterventionEvidenceLedger, stream_id: &str) -> BTreeSet<u64> {
    ledger.pair_bindings.iter().filter(|b| b.stream_id == stream_id)
        .flat_map(|b| b.pair_outcomes.iter().map(|p| p.pair_id)).collect()
}

fn existing_contexts_for_stream(ledger: &TemporalInterventionEvidenceLedger, stream_id: &str) -> BTreeSet<[u8; 32]> {
    ledger.pair_bindings.iter().filter(|b| b.stream_id == stream_id)
        .flat_map(|b| b.pair_outcomes.iter().map(|p| p.context_sha256)).collect()
}

fn existing_assignment_receipts_for_stream(ledger: &TemporalInterventionEvidenceLedger, stream_id: &str) -> BTreeSet<[u8; 32]> {
    ledger.pair_bindings.iter().filter(|b| b.stream_id == stream_id)
        .flat_map(|b| b.pair_outcomes.iter().map(|p| p.assignment_receipt_sha256)).collect()
}

fn compute_evidence_ledger_sha256(
    ledger: &TemporalInterventionEvidenceLedger,
) -> Result<String, TemporalInterventionLedgerError> {
    let mut copy = ledger.clone();
    copy.evidence_ledger_sha256.clear();
    let bytes = serde_json::to_vec(&copy)
        .map_err(|error| TemporalInterventionLedgerError::new("evidence_ledger", error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(EVIDENCE_LEDGER_DOMAIN);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    Ok(hex_digest(&hasher.finalize()))
}

fn hash_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        && value.bytes().any(|b| b != b'0')
}

fn valid_stable_id(value: &str) -> bool {
    !value.is_empty() && value.len() <= 128
        && value.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anytime_qmc_confidence::AnytimeDecision;

    fn digest(value: u8) -> String {
        format!("{value:02x}").repeat(32)
    }

    fn contrast(id: &str, task: u8, model: u8, treatment: u8, control: u8) -> TemporalInterventionContrast {
        TemporalInterventionContrast {
            contrast_id: id.to_owned(),
            task_sha256: digest(task),
            model_sha256: digest(model),
            evaluator_sha256: digest(3),
            treatment_policy_sha256: digest(treatment),
            control_policy_sha256: digest(control),
            assignment_manifest_sha256: digest(6),
        }
    }

    fn pair(id: u64, benefit_us: i128) -> TemporalInterventionPairOutcome {
        let (treatment_exposure_us, control_exposure_us) = if benefit_us > 0 {
            (0, 10)
        } else if benefit_us < 0 {
            (10, 0)
        } else {
            (10, 10)
        };
        TemporalInterventionPairOutcome {
            pair_id: id,
            context_sha256: [20 + id as u8; 32],
            source_family_sha256: [9; 32],
            assignment_receipt_sha256: [80 + id as u8; 32],
            assignment_scheme: TemporalInterventionAssignmentScheme::IndependentFairCoin,
            treatment_probability_ppm: 500_000,
            treatment_was_first: id % 2 == 0,
            episode_duration_us: 100,
            treatment_exposure_us,
            control_exposure_us,
        }
    }

    fn family() -> TemporalInterventionFamily {
        create_temporal_intervention_family(
            "temporal-main-2026-11",
            0.05,
            vec![contrast("pareto-vs-greedy", 1, 2, 4, 5), contrast("value-vs-lru", 1, 2, 7, 8)],
        ).unwrap()
    }

    #[test]
    fn each_contrast_expands_to_three_bounded_axis_streams() {
        let family = family();
        assert_eq!(family.confidence_family.streams.len(), 6);
        assert!(family.confidence_family.streams.windows(2).all(|w| w[0].stream_id < w[1].stream_id));
        assert!(family.confidence_family.streams.iter().all(|s| s.range_min == 0.0 && s.range_max == 1.0 && s.threshold == 0.5));
        verify_temporal_intervention_family(&family).unwrap();
    }

    #[test]
    fn contrast_declarations_must_be_sorted() {
        assert!(create_temporal_intervention_family(
            "family", 0.05,
            vec![contrast("z-last", 1, 2, 4, 5), contrast("a-first", 1, 2, 7, 8)],
        ).is_err());
    }

    #[test]
    fn invalid_digest_and_identical_policies_are_rejected() {
        let mut invalid = contrast("bad-digest", 1, 2, 4, 5);
        invalid.task_sha256 = "not-a-digest".to_owned();
        assert!(create_temporal_intervention_family("family", 0.05, vec![invalid]).is_err());
        assert!(create_temporal_intervention_family(
            "family", 0.05, vec![contrast("same-policy", 1, 2, 4, 4)],
        ).is_err());
    }

    #[test]
    fn contrast_metadata_mutation_invalidates_frozen_manifest() {
        let original = family();
        let mut mutated = original.clone();
        mutated.contrasts[0].model_sha256 = digest(99);
        assert!(verify_temporal_intervention_family(&mutated).is_err());
        let mut declarations = original.contrasts.clone();
        declarations[0].model_sha256 = digest(99);
        let rebuilt = create_temporal_intervention_family("temporal-main-2026-11", 0.05, declarations).unwrap();
        assert_ne!(
            original.confidence_family.manifest_sha256,
            rebuilt.confidence_family.manifest_sha256
        );
    }

    #[test]
    fn sign_encoding_is_exact_for_benefit_harm_and_tie() {
        let family = family();
        let contrast = &family.contrasts[0];
        let digest = contrast_sha256(contrast).unwrap();
        let stream = temporal_stream_id(&contrast.contrast_id, &digest, TemporalInterventionAxis::WrongBelief);
        let (values, _) = encode_pair_outcomes(
            &stream, &[pair(1, 10), pair(2, -2), pair(3, 0)],
            &BTreeSet::new(), &BTreeSet::new(), &BTreeSet::new(),
        ).unwrap();
        assert_eq!(values, vec![1.0, 0.0, 0.5]);
    }

    #[test]
    fn append_derives_family_size_and_replay_verifies_bindings() {
        let family = family();
        let contrast_id = family.contrasts[0].contrast_id.clone();
        let mut ledger = create_temporal_intervention_evidence_ledger(family).unwrap();
        let receipt = append_temporal_intervention_batch(
            &mut ledger, &contrast_id, TemporalInterventionAxis::WrongBelief,
            &[pair(1, 10), pair(2, 20), pair(3, 30)],
        ).unwrap();
        assert_eq!(receipt.family_size, 6);
        assert_eq!(receipt.stream_alpha, 0.05 / 6.0);
        assert_eq!(receipt.sample_count, 3);
        verify_temporal_intervention_evidence_ledger(&ledger).unwrap();
    }

    #[test]
    fn confidence_sequence_certifies_a_strong_beneficial_sign_stream() {
        let family = family();
        let contrast_id = family.contrasts[0].contrast_id.clone();
        let mut ledger = create_temporal_intervention_evidence_ledger(family).unwrap();
        let pairs: Vec<_> = (1..=128).map(|id| pair(id, 100)).collect();
        let receipt = append_temporal_intervention_batch(
            &mut ledger, &contrast_id, TemporalInterventionAxis::WrongBelief, &pairs,
        ).unwrap();
        assert_eq!(receipt.decision, AnytimeDecision::AtOrAboveThreshold);
        verify_temporal_intervention_evidence_ledger(&ledger).unwrap();
    }

    #[test]
    fn multiple_looks_derive_monotone_indices_and_prefixes() {
        let family = family();
        let contrast_id = family.contrasts[0].contrast_id.clone();
        let mut ledger = create_temporal_intervention_evidence_ledger(family).unwrap();
        let first = append_temporal_intervention_batch(
            &mut ledger, &contrast_id, TemporalInterventionAxis::WrongBelief, &[pair(1, 10), pair(2, 10)],
        ).unwrap();
        let second = append_temporal_intervention_batch(
            &mut ledger, &contrast_id, TemporalInterventionAxis::WrongBelief, &[pair(3, 10), pair(4, 10)],
        ).unwrap();
        assert_eq!(first.look_index, 1);
        assert_eq!(second.look_index, 2);
        assert_eq!(second.sample_count, 4);
        verify_temporal_intervention_evidence_ledger(&ledger).unwrap();
    }

    #[test]
    fn duplicate_pair_id_context_and_receipt_fail_without_ledger_mutation() {
        let family = family();
        let contrast_id = family.contrasts[0].contrast_id.clone();
        let mut ledger = create_temporal_intervention_evidence_ledger(family).unwrap();
        append_temporal_intervention_batch(
            &mut ledger, &contrast_id, TemporalInterventionAxis::WrongBelief, &[pair(1, 10)],
        ).unwrap();
        let digest_before = ledger.evidence_ledger_sha256.clone();

        let mut duplicate_context = pair(2, 10);
        duplicate_context.context_sha256 = pair(1, 10).context_sha256;
        assert!(append_temporal_intervention_batch(
            &mut ledger, &contrast_id, TemporalInterventionAxis::WrongBelief, &[duplicate_context],
        ).is_err());

        let mut duplicate_receipt = pair(2, 10);
        duplicate_receipt.assignment_receipt_sha256 = pair(1, 10).assignment_receipt_sha256;
        assert!(append_temporal_intervention_batch(
            &mut ledger, &contrast_id, TemporalInterventionAxis::WrongBelief, &[duplicate_receipt],
        ).is_err());

        assert_eq!(ledger.evidence_ledger_sha256, digest_before);
        verify_temporal_intervention_evidence_ledger(&ledger).unwrap();
    }

    #[test]
    fn unsupported_assignment_and_exposure_outside_episode_are_rejected() {
        let family = family();
        let contrast_id = family.contrasts[0].contrast_id.clone();
        let mut ledger = create_temporal_intervention_evidence_ledger(family).unwrap();

        let mut blocked = pair(1, 10);
        blocked.assignment_scheme = TemporalInterventionAssignmentScheme::FixedCountBlocked;
        assert!(append_temporal_intervention_batch(
            &mut ledger, &contrast_id, TemporalInterventionAxis::WrongBelief, &[blocked],
        ).is_err());

        let mut biased = pair(1, 10);
        biased.treatment_probability_ppm = 250_000;
        assert!(append_temporal_intervention_batch(
            &mut ledger, &contrast_id, TemporalInterventionAxis::WrongBelief, &[biased],
        ).is_err());

        let mut overrun = pair(1, 10);
        overrun.treatment_exposure_us = 101;
        assert!(append_temporal_intervention_batch(
            &mut ledger, &contrast_id, TemporalInterventionAxis::WrongBelief, &[overrun],
        ).is_err());

        assert!(ledger.pair_bindings.is_empty());
        verify_temporal_intervention_evidence_ledger(&ledger).unwrap();
    }

    #[test]
    fn unknown_contrast_and_empty_batch_fail_closed() {
        let family = family();
        let mut ledger = create_temporal_intervention_evidence_ledger(family).unwrap();
        assert!(append_temporal_intervention_batch(
            &mut ledger, "undeclared", TemporalInterventionAxis::WrongBelief, &[pair(1, 1)],
        ).is_err());
        let known = ledger.family.contrasts[0].contrast_id.clone();
        assert!(append_temporal_intervention_batch(
            &mut ledger, &known, TemporalInterventionAxis::WrongBelief, &[],
        ).is_err());
    }

    #[test]
    fn pair_sign_tampering_and_generic_ledger_mutation_are_detected() {
        let family = family();
        let contrast_id = family.contrasts[0].contrast_id.clone();
        let mut ledger = create_temporal_intervention_evidence_ledger(family).unwrap();
        append_temporal_intervention_batch(
            &mut ledger, &contrast_id, TemporalInterventionAxis::WrongBelief, &[pair(1, 10), pair(2, -10)],
        ).unwrap();

        let mut pair_tamper = ledger.clone();
        pair_tamper.pair_bindings[0].pair_outcomes[0].control_exposure_us = 0;
        assert!(verify_temporal_intervention_evidence_ledger(&pair_tamper).is_err());

        let mut ledger_tamper = ledger;
        let stream_id = ledger_tamper.pair_bindings[0].stream_id.clone();
        let stream = ledger_tamper.confidence_ledger.streams.iter_mut()
            .find(|stream| stream.stream_id == stream_id).unwrap();
        stream.looks[0].replicate_batch[0] = 0.0;
        assert!(verify_temporal_intervention_evidence_ledger(&ledger_tamper).is_err());
    }

    #[test]
    fn missing_pair_binding_or_outer_digest_tampering_is_detected() {
        let family = family();
        let contrast_id = family.contrasts[0].contrast_id.clone();
        let mut ledger = create_temporal_intervention_evidence_ledger(family).unwrap();
        append_temporal_intervention_batch(
            &mut ledger, &contrast_id, TemporalInterventionAxis::Abstention, &[pair(1, 10)],
        ).unwrap();

        let mut missing = ledger.clone();
        missing.pair_bindings.clear();
        assert!(verify_temporal_intervention_evidence_ledger(&missing).is_err());

        let mut digest_tamper = ledger;
        digest_tamper.evidence_ledger_sha256 = digest(77);
        assert!(verify_temporal_intervention_evidence_ledger(&digest_tamper).is_err());
    }

    #[test]
    fn json_round_trip_preserves_full_replay_verification() {
        let family = family();
        let contrast_id = family.contrasts[0].contrast_id.clone();
        let mut ledger = create_temporal_intervention_evidence_ledger(family).unwrap();
        append_temporal_intervention_batch(
            &mut ledger, &contrast_id, TemporalInterventionAxis::MissingOutput, &[pair(1, -10), pair(2, 10)],
        ).unwrap();
        let bytes = serde_json::to_vec(&ledger).unwrap();
        let restored: TemporalInterventionEvidenceLedger = serde_json::from_slice(&bytes).unwrap();
        verify_temporal_intervention_evidence_ledger(&restored).unwrap();
    }

    #[test]
    fn exact_stream_count_is_three_times_the_contrast_count() {
        let family = family();
        assert_eq!(
            family.confidence_family.streams.len(),
            family.contrasts.len() * TemporalInterventionAxis::ALL.len()
        );
        assert_eq!(family.confidence_family.streams.len(), 6);
    }
}
