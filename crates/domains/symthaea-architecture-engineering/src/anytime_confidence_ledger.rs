//! Append-only preregistered family and look ledger for anytime confidence receipts.
//!
//! A confidence receipt is a statement about one look. This module adds the sequential
//! protocol around those receipts: a fixed family manifest, a complete set of unique
//! decision streams, immutable support/threshold declarations, monotonically derived
//! look indices, exact cumulative sample prefixes, parent-receipt links, and SHA-256
//! hash chaining for each appended replicate batch.
//!
//! This ledger proves internal consistency and append continuity relative to the
//! supplied manifest and recorded batches. It does not prove that the manifest was
//! externally timestamped before outcomes, that the ledger was never rewritten by an
//! actor able to recompute all hashes, that batches are statistically independent, or
//! that a pseudorandom generator has ideal randomness. Those are separate custody,
//! preregistration, and experimental-design obligations.
//!
//! Replicate batches contain one independent bounded observation per unit. For
//! randomized QMC, one unit is a complete independently randomized QMC replicate;
//! integration points inside a replicate are not independent observations.

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::anytime_qmc_confidence::{
    AnytimeBoundedMeanConfig, AnytimeBoundedMeanReceipt,
    evaluate_anytime_bounded_mean,
};

const MANIFEST_SCHEMA: &str = "symthaea.nas.anytime-confidence-family";
const MANIFEST_VERSION: &str = "0.1.0";
const LEDGER_SCHEMA: &str = "symthaea.nas.anytime-confidence-ledger";
const LEDGER_VERSION: &str = "0.1.0";
const MANIFEST_DOMAIN: &[u8] = b"symthaea.nas.anytime-family-manifest.v0.1";
const BATCH_DOMAIN: &[u8] = b"symthaea.nas.anytime-ledger-batch.v0.1";
const PREFIX_DOMAIN: &[u8] = b"symthaea.nas.anytime-ledger-prefix.v0.1";
const LOOK_DOMAIN: &[u8] = b"symthaea.nas.anytime-ledger-look.v0.1";
const LEDGER_DOMAIN: &[u8] = b"symthaea.nas.anytime-ledger.v0.1";
const MAX_STREAMS: usize = 100_000;
const MAX_LOOKS_PER_STREAM: usize = 1_000_000;
const MAX_REPLICATES_PER_STREAM: usize = 1_000_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeStreamDeclaration {
    pub stream_id: String,
    pub range_min: f64,
    pub range_max: f64,
    pub threshold: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeConfidenceFamilyManifest {
    pub schema: String,
    pub version: String,
    pub family_id: String,
    pub family_alpha: f64,
    /// The complete, strictly sorted set of streams declared before outcomes are observed.
    pub streams: Vec<AnytimeStreamDeclaration>,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeConfidenceLookRecord {
    /// Derived from the current stream length; callers cannot supply an arbitrary index.
    pub look_index: u64,
    /// New independent replicate observations added at this look.
    pub replicate_batch: Vec<f64>,
    pub sample_count_through_look: usize,
    pub batch_sha256: String,
    /// Hash chain over every ordered batch through this look.
    pub prefix_chain_sha256: String,
    pub previous_receipt_sha256: Option<String>,
    pub receipt: AnytimeBoundedMeanReceipt,
    pub record_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeConfidenceStreamLedger {
    pub stream_id: String,
    pub looks: Vec<AnytimeConfidenceLookRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeConfidenceLedger {
    pub schema: String,
    pub version: String,
    pub manifest: AnytimeConfidenceFamilyManifest,
    /// Streams use the exact canonical order declared by the manifest.
    pub streams: Vec<AnytimeConfidenceStreamLedger>,
    pub ledger_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnytimeLedgerError {
    pub field: String,
    pub message: String,
}

impl AnytimeLedgerError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for AnytimeLedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for AnytimeLedgerError {}

/// Construct and hash a frozen family manifest.
///
/// The stream list must already be strictly sorted by stream_id. Silently sorting here
/// would hide a caller's non-canonical preregistration artifact.
pub fn create_anytime_confidence_family_manifest(
    family_id: impl Into<String>,
    family_alpha: f64,
    streams: Vec<AnytimeStreamDeclaration>,
) -> Result<AnytimeConfidenceFamilyManifest, AnytimeLedgerError> {
    let mut manifest = AnytimeConfidenceFamilyManifest {
        schema: MANIFEST_SCHEMA.to_owned(),
        version: MANIFEST_VERSION.to_owned(),
        family_id: family_id.into(),
        family_alpha,
        streams,
        manifest_sha256: String::new(),
    };
    validate_manifest_fields(&manifest)?;
    manifest.manifest_sha256 = compute_manifest_sha256(&manifest)?;
    Ok(manifest)
}

/// Create an empty ledger from a previously frozen family manifest.
pub fn create_anytime_confidence_ledger(
    manifest: AnytimeConfidenceFamilyManifest,
) -> Result<AnytimeConfidenceLedger, AnytimeLedgerError> {
    validate_anytime_confidence_family_manifest(&manifest)?;
    let streams = manifest
        .streams
        .iter()
        .map(|declaration| AnytimeConfidenceStreamLedger {
            stream_id: declaration.stream_id.clone(),
            looks: Vec::new(),
        })
        .collect();
    let mut ledger = AnytimeConfidenceLedger {
        schema: LEDGER_SCHEMA.to_owned(),
        version: LEDGER_VERSION.to_owned(),
        manifest,
        streams,
        ledger_sha256: String::new(),
    };
    ledger.ledger_sha256 = compute_ledger_sha256(&ledger)?;
    Ok(ledger)
}

/// Append one batch to a declared stream and derive the next sequential receipt.
///
/// This operation verifies the existing ledger before mutation. The next look index,
/// previous-receipt link, cumulative sample prefix, and family size are derived from
/// existing ledger state and the immutable manifest.
pub fn append_anytime_confidence_look(
    ledger: &mut AnytimeConfidenceLedger,
    stream_id: &str,
    replicate_batch: &[f64],
) -> Result<AnytimeBoundedMeanReceipt, AnytimeLedgerError> {
    verify_anytime_confidence_ledger(ledger)?;
    if replicate_batch.is_empty() {
        return Err(AnytimeLedgerError::new(
            "replicate_batch",
            "a look must append at least one new independent replicate",
        ));
    }

    let stream_index = ledger
        .manifest
        .streams
        .binary_search_by(|declaration| declaration.stream_id.as_str().cmp(stream_id))
        .map_err(|_| {
            AnytimeLedgerError::new(
                "stream_id",
                format!("stream {stream_id:?} was not preregistered in this family"),
            )
        })?;

    let declaration = &ledger.manifest.streams[stream_index];
    let stream_ledger = &ledger.streams[stream_index];
    if stream_ledger.stream_id != stream_id {
        return Err(AnytimeLedgerError::new(
            "streams",
            "ledger stream order differs from the frozen manifest",
        ));
    }
    let next_look = (stream_ledger.looks.len() as u64)
        .checked_add(1)
        .ok_or_else(|| AnytimeLedgerError::new("look_index", "look index overflow"))?;
    if next_look > MAX_LOOKS_PER_STREAM as u64 {
        return Err(AnytimeLedgerError::new(
            "look_index",
            format!("look count exceeds {MAX_LOOKS_PER_STREAM}"),
        ));
    }

    let mut cumulative_values = Vec::new();
    for look in &stream_ledger.looks {
        cumulative_values.extend_from_slice(&look.replicate_batch);
    }
    let prior_count = cumulative_values.len();
    let total_count = prior_count
        .checked_add(replicate_batch.len())
        .ok_or_else(|| AnytimeLedgerError::new("sample_count", "sample count overflow"))?;
    if total_count > MAX_REPLICATES_PER_STREAM {
        return Err(AnytimeLedgerError::new(
            "sample_count",
            format!("cumulative replicate count exceeds {MAX_REPLICATES_PER_STREAM}"),
        ));
    }
    cumulative_values.extend_from_slice(replicate_batch);

    let config = AnytimeBoundedMeanConfig {
        family_id: ledger.manifest.family_id.clone(),
        stream_id: stream_id.to_owned(),
        family_alpha: ledger.manifest.family_alpha,
        family_size: ledger.manifest.streams.len(),
        look_index: next_look,
        range_min: declaration.range_min,
        range_max: declaration.range_max,
        threshold: declaration.threshold,
    };
    let receipt = evaluate_anytime_bounded_mean(&cumulative_values, &config)
        .map_err(|error| AnytimeLedgerError::new("receipt", error.to_string()))?;

    let previous_prefix = stream_ledger
        .looks
        .last()
        .map(|look| look.prefix_chain_sha256.as_str());
    let previous_receipt_sha256 = stream_ledger
        .looks
        .last()
        .map(|look| look.receipt.receipt_sha256.clone());
    let batch_sha256 = compute_batch_sha256(replicate_batch);
    let prefix_chain_sha256 = compute_prefix_chain_sha256(
        previous_prefix,
        &ledger.manifest.manifest_sha256,
        stream_id,
        next_look,
        prior_count,
        replicate_batch,
    );
    let mut record = AnytimeConfidenceLookRecord {
        look_index: next_look,
        replicate_batch: replicate_batch.to_vec(),
        sample_count_through_look: total_count,
        batch_sha256,
        prefix_chain_sha256,
        previous_receipt_sha256,
        receipt: receipt.clone(),
        record_sha256: String::new(),
    };
    record.record_sha256 = compute_look_record_sha256(&record)?;

    ledger.streams[stream_index].looks.push(record);
    ledger.ledger_sha256 = compute_ledger_sha256(ledger)?;
    Ok(receipt)
}

/// Validate a frozen manifest's schema, identity, stream order, and digest.
pub fn validate_anytime_confidence_family_manifest(
    manifest: &AnytimeConfidenceFamilyManifest,
) -> Result<(), AnytimeLedgerError> {
    validate_manifest_fields(manifest)?;
    if !is_sha256(&manifest.manifest_sha256) {
        return Err(AnytimeLedgerError::new(
            "manifest_sha256",
            "must be a lowercase hexadecimal SHA-256 digest",
        ));
    }
    let expected = compute_manifest_sha256(manifest)?;
    if expected != manifest.manifest_sha256 {
        return Err(AnytimeLedgerError::new(
            "manifest_sha256",
            "does not match canonical manifest contents",
        ));
    }
    Ok(())
}

fn validate_manifest_fields(
    manifest: &AnytimeConfidenceFamilyManifest,
) -> Result<(), AnytimeLedgerError> {
    if manifest.schema != MANIFEST_SCHEMA || manifest.version != MANIFEST_VERSION {
        return Err(AnytimeLedgerError::new(
            "schema_version",
            "unsupported family manifest schema or version",
        ));
    }
    if manifest.family_id.trim().is_empty() {
        return Err(AnytimeLedgerError::new(
            "family_id",
            "family ID must be non-empty",
        ));
    }
    if !manifest.family_alpha.is_finite()
        || manifest.family_alpha <= 0.0
        || manifest.family_alpha >= 1.0
    {
        return Err(AnytimeLedgerError::new(
            "family_alpha",
            "family alpha must be finite and strictly between zero and one",
        ));
    }
    if manifest.streams.is_empty() || manifest.streams.len() > MAX_STREAMS {
        return Err(AnytimeLedgerError::new(
            "streams",
            format!("stream count must be in 1..={MAX_STREAMS}"),
        ));
    }
    for (index, declaration) in manifest.streams.iter().enumerate() {
        if declaration.stream_id.trim().is_empty() {
            return Err(AnytimeLedgerError::new(
                format!("streams[{index}].stream_id"),
                "stream ID must be non-empty",
            ));
        }
        if !declaration.range_min.is_finite()
            || !declaration.range_max.is_finite()
            || declaration.range_max <= declaration.range_min
            || !(declaration.range_max - declaration.range_min).is_finite()
        {
            return Err(AnytimeLedgerError::new(
                format!("streams[{index}].range"),
                "support bounds must be finite, ordered, and have finite width",
            ));
        }
        if !declaration.threshold.is_finite()
            || declaration.threshold < declaration.range_min
            || declaration.threshold > declaration.range_max
        {
            return Err(AnytimeLedgerError::new(
                format!("streams[{index}].threshold"),
                "threshold must be finite and inside the declared support",
            ));
        }
    }
    if manifest
        .streams
        .windows(2)
        .any(|pair| pair[0].stream_id >= pair[1].stream_id)
    {
        return Err(AnytimeLedgerError::new(
            "streams",
            "stream declarations must be strictly sorted by stream_id with no duplicates",
        ));
    }
    Ok(())
}

/// Verify the entire family ledger by replaying every observation batch and receipt.
pub fn verify_anytime_confidence_ledger(
    ledger: &AnytimeConfidenceLedger,
) -> Result<(), AnytimeLedgerError> {
    if ledger.schema != LEDGER_SCHEMA || ledger.version != LEDGER_VERSION {
        return Err(AnytimeLedgerError::new(
            "schema_version",
            "unsupported anytime ledger schema or version",
        ));
    }
    validate_anytime_confidence_family_manifest(&ledger.manifest)?;
    if ledger.streams.len() != ledger.manifest.streams.len() {
        return Err(AnytimeLedgerError::new(
            "streams",
            "ledger must contain exactly one entry for every preregistered stream",
        ));
    }

    for (stream_index, (declaration, stream_ledger)) in ledger
        .manifest
        .streams
        .iter()
        .zip(&ledger.streams)
        .enumerate()
    {
        if stream_ledger.stream_id != declaration.stream_id {
            return Err(AnytimeLedgerError::new(
                format!("streams[{stream_index}].stream_id"),
                "ledger stream order or identity differs from the manifest",
            ));
        }
        if stream_ledger.looks.len() > MAX_LOOKS_PER_STREAM {
            return Err(AnytimeLedgerError::new(
                format!("streams[{stream_index}].looks"),
                "look count exceeds configured maximum",
            ));
        }

        let mut cumulative_values = Vec::new();
        let mut previous_prefix: Option<String> = None;
        let mut previous_receipt: Option<String> = None;
        for (look_offset, record) in stream_ledger.looks.iter().enumerate() {
            let expected_look = (look_offset as u64) + 1;
            if record.look_index != expected_look {
                return Err(AnytimeLedgerError::new(
                    format!("streams[{stream_index}].looks[{look_offset}].look_index"),
                    "look indices must be contiguous, one-based, and append-only",
                ));
            }
            if record.replicate_batch.is_empty() {
                return Err(AnytimeLedgerError::new(
                    format!("streams[{stream_index}].looks[{look_offset}].replicate_batch"),
                    "every look must append at least one new replicate",
                ));
            }
            let next_count = cumulative_values
                .len()
                .checked_add(record.replicate_batch.len())
                .ok_or_else(|| {
                    AnytimeLedgerError::new(
                        format!("streams[{stream_index}].looks[{look_offset}].sample_count"),
                        "sample count overflow",
                    )
                })?;
            if next_count > MAX_REPLICATES_PER_STREAM {
                return Err(AnytimeLedgerError::new(
                    format!("streams[{stream_index}].looks[{look_offset}].sample_count"),
                    "cumulative replicate count exceeds configured maximum",
                ));
            }
            if record.sample_count_through_look != next_count {
                return Err(AnytimeLedgerError::new(
                    format!("streams[{stream_index}].looks[{look_offset}].sample_count"),
                    "sample count does not equal the exact cumulative prefix length",
                ));
            }
            if record.batch_sha256 != compute_batch_sha256(&record.replicate_batch) {
                return Err(AnytimeLedgerError::new(
                    format!("streams[{stream_index}].looks[{look_offset}].batch_sha256"),
                    "batch digest does not match the ordered replicate batch",
                ));
            }
            if record.previous_receipt_sha256 != previous_receipt {
                return Err(AnytimeLedgerError::new(
                    format!("streams[{stream_index}].looks[{look_offset}].previous_receipt_sha256"),
                    "previous receipt link is missing or does not match the preceding look",
                ));
            }

            let expected_prefix = compute_prefix_chain_sha256(
                previous_prefix.as_deref(),
                &ledger.manifest.manifest_sha256,
                &declaration.stream_id,
                expected_look,
                cumulative_values.len(),
                &record.replicate_batch,
            );
            if record.prefix_chain_sha256 != expected_prefix {
                return Err(AnytimeLedgerError::new(
                    format!("streams[{stream_index}].looks[{look_offset}].prefix_chain_sha256"),
                    "prefix chain does not extend the preceding exact batch sequence",
                ));
            }

            cumulative_values.extend_from_slice(&record.replicate_batch);
            let config = AnytimeBoundedMeanConfig {
                family_id: ledger.manifest.family_id.clone(),
                stream_id: declaration.stream_id.clone(),
                family_alpha: ledger.manifest.family_alpha,
                family_size: ledger.manifest.streams.len(),
                look_index: expected_look,
                range_min: declaration.range_min,
                range_max: declaration.range_max,
                threshold: declaration.threshold,
            };
            let expected_receipt = evaluate_anytime_bounded_mean(&cumulative_values, &config)
                .map_err(|error| {
                    AnytimeLedgerError::new(
                        format!("streams[{stream_index}].looks[{look_offset}].receipt"),
                        error.to_string(),
                    )
                })?;
            if record.receipt != expected_receipt {
                return Err(AnytimeLedgerError::new(
                    format!("streams[{stream_index}].looks[{look_offset}].receipt"),
                    "receipt does not recompute from the exact cumulative prefix and manifest",
                ));
            }
            if record.record_sha256 != compute_look_record_sha256(record)? {
                return Err(AnytimeLedgerError::new(
                    format!("streams[{stream_index}].looks[{look_offset}].record_sha256"),
                    "look record digest does not match its contents",
                ));
            }

            previous_prefix = Some(record.prefix_chain_sha256.clone());
            previous_receipt = Some(record.receipt.receipt_sha256.clone());
        }
    }

    if !is_sha256(&ledger.ledger_sha256)
        || ledger.ledger_sha256 != compute_ledger_sha256(ledger)?
    {
        return Err(AnytimeLedgerError::new(
            "ledger_sha256",
            "ledger digest does not match the canonical complete ledger",
        ));
    }
    Ok(())
}

fn compute_manifest_sha256(
    manifest: &AnytimeConfidenceFamilyManifest,
) -> Result<String, AnytimeLedgerError> {
    let mut canonical = manifest.clone();
    canonical.manifest_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| AnytimeLedgerError::new("manifest", error.to_string()))?;
    Ok(digest_bytes(MANIFEST_DOMAIN, &bytes))
}

fn compute_batch_sha256(values: &[f64]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((BATCH_DOMAIN.len() as u64).to_be_bytes());
    hasher.update(BATCH_DOMAIN);
    hasher.update((values.len() as u64).to_be_bytes());
    for value in values {
        hasher.update(value.to_bits().to_be_bytes());
    }
    encode_sha256(hasher.finalize().as_slice())
}

fn compute_prefix_chain_sha256(
    previous_prefix: Option<&str>,
    manifest_sha256: &str,
    stream_id: &str,
    look_index: u64,
    prior_count: usize,
    batch: &[f64],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update((PREFIX_DOMAIN.len() as u64).to_be_bytes());
    hasher.update(PREFIX_DOMAIN);
    hash_text(&mut hasher, manifest_sha256);
    hash_text(&mut hasher, stream_id);
    match previous_prefix {
        Some(previous) => {
            hasher.update([1u8]);
            hash_text(&mut hasher, previous);
        }
        None => hasher.update([0u8]),
    }
    hasher.update(look_index.to_be_bytes());
    hasher.update((prior_count as u64).to_be_bytes());
    hasher.update((batch.len() as u64).to_be_bytes());
    for value in batch {
        hasher.update(value.to_bits().to_be_bytes());
    }
    encode_sha256(hasher.finalize().as_slice())
}

fn compute_look_record_sha256(
    record: &AnytimeConfidenceLookRecord,
) -> Result<String, AnytimeLedgerError> {
    let mut canonical = record.clone();
    canonical.record_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| AnytimeLedgerError::new("look_record", error.to_string()))?;
    Ok(digest_bytes(LOOK_DOMAIN, &bytes))
}

fn compute_ledger_sha256(
    ledger: &AnytimeConfidenceLedger,
) -> Result<String, AnytimeLedgerError> {
    let mut canonical = ledger.clone();
    canonical.ledger_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| AnytimeLedgerError::new("ledger", error.to_string()))?;
    Ok(digest_bytes(LEDGER_DOMAIN, &bytes))
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    encode_sha256(hasher.finalize().as_slice())
}

fn hash_text(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn encode_sha256(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> AnytimeConfidenceFamilyManifest {
        create_anytime_confidence_family_manifest(
            "nas-family-2026-10-v1",
            0.05,
            vec![
                AnytimeStreamDeclaration {
                    stream_id: "energy".into(),
                    range_min: 0.0,
                    range_max: 1.0,
                    threshold: 0.4,
                },
                AnytimeStreamDeclaration {
                    stream_id: "latency".into(),
                    range_min: 0.0,
                    range_max: 1.0,
                    threshold: 0.4,
                },
            ],
        )
        .unwrap()
    }

    fn ledger() -> AnytimeConfidenceLedger {
        create_anytime_confidence_ledger(manifest()).unwrap()
    }

    #[test]
    fn creates_a_canonical_manifest_and_empty_ledger() {
        let ledger = ledger();
        assert_eq!(ledger.streams.len(), 2);
        assert_eq!(ledger.streams[0].stream_id, "energy");
        assert_eq!(ledger.streams[1].stream_id, "latency");
        verify_anytime_confidence_ledger(&ledger).unwrap();
    }

    #[test]
    fn appends_three_monotone_looks_with_exact_prefixes() {
        let mut ledger = ledger();
        let first = append_anytime_confidence_look(
            &mut ledger,
            "energy",
            &[0.6, 0.7, 0.5, 0.8],
        )
        .unwrap();
        let second =
            append_anytime_confidence_look(&mut ledger, "energy", &[0.65, 0.75]).unwrap();
        let third =
            append_anytime_confidence_look(&mut ledger, "energy", &[0.7, 0.8]).unwrap();

        assert_eq!(first.look_index, 1);
        assert_eq!(second.look_index, 2);
        assert_eq!(third.look_index, 3);
        assert_eq!(third.sample_count, 8);
        assert_eq!(
            ledger.streams[0].looks[1].previous_receipt_sha256.as_deref(),
            Some(first.receipt_sha256.as_str())
        );
        verify_anytime_confidence_ledger(&ledger).unwrap();
    }

    #[test]
    fn streams_have_independent_sequences_but_one_fixed_family_alpha() {
        let mut ledger = ledger();
        let energy = append_anytime_confidence_look(
            &mut ledger,
            "energy",
            &[0.8, 0.8, 0.8],
        )
        .unwrap();
        let latency = append_anytime_confidence_look(
            &mut ledger,
            "latency",
            &[0.3, 0.3, 0.3],
        )
        .unwrap();
        assert_eq!(energy.look_index, 1);
        assert_eq!(latency.look_index, 1);
        assert_eq!(energy.stream_alpha, 0.025);
        assert_eq!(latency.stream_alpha, 0.025);
        verify_anytime_confidence_ledger(&ledger).unwrap();
    }

    #[test]
    fn rejects_unknown_streams() {
        let mut ledger = ledger();
        assert!(append_anytime_confidence_look(&mut ledger, "memory", &[0.5]).is_err());
    }

    #[test]
    fn rejects_empty_batches() {
        let mut ledger = ledger();
        assert!(append_anytime_confidence_look(&mut ledger, "energy", &[]).is_err());
    }

    #[test]
    fn rejects_non_finite_and_out_of_support_replicates() {
        let mut ledger = ledger();
        assert!(append_anytime_confidence_look(&mut ledger, "energy", &[f64::NAN]).is_err());
        assert!(append_anytime_confidence_look(&mut ledger, "energy", &[1.1]).is_err());
    }

    #[test]
    fn manifest_requires_nonempty_sorted_unique_streams() {
        let duplicate = vec![
            AnytimeStreamDeclaration {
                stream_id: "energy".into(),
                range_min: 0.0,
                range_max: 1.0,
                threshold: 0.5,
            },
            AnytimeStreamDeclaration {
                stream_id: "energy".into(),
                range_min: 0.0,
                range_max: 1.0,
                threshold: 0.5,
            },
        ];
        assert!(create_anytime_confidence_family_manifest("family", 0.05, duplicate).is_err());

        let unsorted = vec![
            AnytimeStreamDeclaration {
                stream_id: "z".into(),
                range_min: 0.0,
                range_max: 1.0,
                threshold: 0.5,
            },
            AnytimeStreamDeclaration {
                stream_id: "a".into(),
                range_min: 0.0,
                range_max: 1.0,
                threshold: 0.5,
            },
        ];
        assert!(create_anytime_confidence_family_manifest("family", 0.05, unsorted).is_err());
    }

    #[test]
    fn manifest_digest_rejects_family_alpha_threshold_and_support_drift() {
        let mut changed_alpha = manifest();
        changed_alpha.family_alpha = 0.1;
        assert!(validate_anytime_confidence_family_manifest(&changed_alpha).is_err());

        let mut changed_threshold = manifest();
        changed_threshold.streams[0].threshold = 0.5;
        assert!(validate_anytime_confidence_family_manifest(&changed_threshold).is_err());

        let mut changed_range = manifest();
        changed_range.streams[0].range_max = 2.0;
        assert!(validate_anytime_confidence_family_manifest(&changed_range).is_err());
    }

    #[test]
    fn rejects_manifest_reorder_without_rehashing() {
        let mut changed = manifest();
        changed.streams.swap(0, 1);
        assert!(validate_anytime_confidence_family_manifest(&changed).is_err());
    }

    #[test]
    fn ledger_rejects_changed_family_manifest() {
        let mut ledger = ledger();
        ledger.manifest.family_alpha = 0.1;
        assert!(verify_anytime_confidence_ledger(&ledger).is_err());
    }

    #[test]
    fn rejects_reordered_or_missing_ledger_streams() {
        let mut populated = ledger();
        populated.streams.swap(0, 1);
        assert!(verify_anytime_confidence_ledger(&populated).is_err());

        let mut missing = ledger();
        missing.streams.pop();
        assert!(verify_anytime_confidence_ledger(&missing).is_err());
    }

    #[test]
    fn detects_batch_tampering_even_when_outer_hashes_are_recomputed() {
        let mut ledger = ledger();
        append_anytime_confidence_look(&mut ledger, "energy", &[0.7, 0.8]).unwrap();
        let record = &mut ledger.streams[0].looks[0];
        record.replicate_batch[0] = 0.1;
        record.batch_sha256 = compute_batch_sha256(&record.replicate_batch);
        record.prefix_chain_sha256 =
            compute_prefix_chain_sha256(
                None,
                &ledger.manifest.manifest_sha256,
                "energy",
                1,
                0,
                &record.replicate_batch,
            );
        record.record_sha256 = compute_look_record_sha256(record).unwrap();
        ledger.ledger_sha256 = compute_ledger_sha256(&ledger).unwrap();
        assert!(verify_anytime_confidence_ledger(&ledger)
            .unwrap_err()
            .message
            .contains("receipt does not recompute"));
    }

    #[test]
    fn rejects_duplicate_or_rolled_back_look_indices() {
        let mut ledger = ledger();
        append_anytime_confidence_look(&mut ledger, "energy", &[0.6, 0.7]).unwrap();
        append_anytime_confidence_look(&mut ledger, "energy", &[0.7, 0.8]).unwrap();
        ledger.streams[0].looks[1].look_index = 1;
        assert!(verify_anytime_confidence_ledger(&ledger).is_err());
    }

    #[test]
    fn rejects_broken_parent_receipt_link() {
        let mut ledger = ledger();
        append_anytime_confidence_look(&mut ledger, "energy", &[0.6, 0.7]).unwrap();
        append_anytime_confidence_look(&mut ledger, "energy", &[0.7, 0.8]).unwrap();
        ledger.streams[0].looks[1].previous_receipt_sha256 = None;
        assert!(verify_anytime_confidence_ledger(&ledger).is_err());
    }

    #[test]
    fn rejects_corrupt_prefix_chain() {
        let mut ledger = ledger();
        append_anytime_confidence_look(&mut ledger, "energy", &[0.6, 0.7]).unwrap();
        ledger.streams[0].looks[0].prefix_chain_sha256 = "0".repeat(64);
        assert!(verify_anytime_confidence_ledger(&ledger).is_err());
    }

    #[test]
    fn verifies_json_round_trip_exactly() {
        let mut ledger = ledger();
        append_anytime_confidence_look(&mut ledger, "energy", &[0.6, 0.7, 0.8]).unwrap();
        append_anytime_confidence_look(&mut ledger, "energy", &[0.7, 0.8]).unwrap();
        let encoded = serde_json::to_string(&ledger).unwrap();
        let decoded: AnytimeConfidenceLedger = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, ledger);
        verify_anytime_confidence_ledger(&decoded).unwrap();
    }

    #[test]
    fn rejects_tampered_receipt_or_final_ledger_digest() {
        let mut ledger = ledger();
        append_anytime_confidence_look(&mut ledger, "energy", &[0.7, 0.8]).unwrap();
        ledger.streams[0].looks[0].receipt.upper_bound = 1.0;
        assert!(verify_anytime_confidence_ledger(&ledger).is_err());

        let mut second = ledger();
        append_anytime_confidence_look(&mut second, "latency", &[0.2, 0.3]).unwrap();
        second.ledger_sha256 = "0".repeat(64);
        assert!(verify_anytime_confidence_ledger(&second).is_err());
    }

    #[test]
    fn cumulative_family_error_allocation_never_exceeds_declared_alpha() {
        let ledger = ledger();
        let family_alpha = ledger.manifest.family_alpha;
        let family_size = ledger.manifest.streams.len() as f64;
        let stream_alpha = family_alpha / family_size;
        assert!(stream_alpha * family_size <= family_alpha + 1e-15);
        for look in 1..=10_000u64 {
            let spent = stream_alpha * (look as f64 / (look + 1) as f64);
            assert!(spent <= stream_alpha + 1e-15);
        }
    }
}
