// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Anytime-valid paired evaluation for sequential WeightProgram candidates.
//!
//! Every candidate is frozen before its confirmatory evaluation stream begins.
//! Each observation compares baseline and candidate losses on the same episode,
//! with both losses normalized to [0, 1]. Improvement is baseline loss minus
//! candidate loss, so positive values favor the candidate.
//!
//! A union-bound Hoeffding confidence sequence spends one campaign-wide alpha
//! budget across candidate indices and every possible interim look. This
//! protects against optional stopping and repeated candidate evaluation, so
//! long as each candidate's confirmatory episodes are fresh, its model/protocol
//! digests remain fixed, and a candidate is not edited mid-stream. New edits
//! require a new candidate index and a fresh confirmatory stream.
//!
//! This is a statistical qualification primitive, not a verifier of task
//! correctness, safety or model identity. A signed independent verifier must
//! still bind its report digest to the WeightProgram acceptance attestation.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

const DIGEST_HEX_LEN: usize = 64;

/// Immutable identities bound to one candidate's confirmatory evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightVerificationBinding {
    /// One-based candidate ordinal in a preregistered repair campaign.
    pub candidate_index: u64,
    /// Digest of the baseline model evaluated on each paired episode.
    pub baseline_model_digest: String,
    /// Digest of the frozen candidate model evaluated on each paired episode.
    pub candidate_model_digest: String,
    /// Digest of the frozen task, split, loss, and evaluation protocol.
    pub evaluation_protocol_digest: String,
    /// Digest of the fresh, candidate-specific confirmatory data shard.
    pub confirmatory_shard_digest: String,
    /// Digest of the canonical sorted episode-ID manifest reserved by this campaign.
    pub confirmatory_episode_manifest_digest: String,
    /// Digest of the open-world/model-adequacy precondition report.
    pub precondition_gate_digest: String,
}

/// Alpha-spending and practical-effect requirements for a repair candidate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SequentialWeightVerificationConfig {
    /// Total family-wise error budget across all candidate indices and looks.
    pub campaign_alpha: f64,
    /// Minimum practically meaningful average loss reduction in [0, 1].
    pub minimum_improvement: f64,
    /// Minimum paired episodes before a candidate can be qualified.
    pub minimum_observations: u64,
    /// Maximum paired episodes allowed for this candidate.
    pub maximum_observations: u64,
}

/// One paired observation from a frozen confirmatory episode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedLossObservation {
    /// Stable episode identifier; duplicates are rejected.
    pub episode_id: String,
    /// Baseline loss, normalized to [0, 1].
    pub baseline_loss: f64,
    /// Candidate loss on the same episode, normalized to [0, 1].
    pub candidate_loss: f64,
}

/// Canonical identity manifest for a confirmatory episode shard.
///
/// Episode IDs are sorted, unique, non-empty, and bound with a domain-separated
/// BLAKE3 digest. The manifest identifies which episodes may be consumed by a
/// candidate; the separate shard digest must still bind the actual immutable
/// episode payloads and their source provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryEpisodeManifest {
    pub format_version: u32,
    pub episode_ids: Vec<String>,
    pub manifest_digest: String,
}

impl ConfirmatoryEpisodeManifest {
    /// Construct a canonical manifest from episode identifiers.
    pub fn new(mut episode_ids: Vec<String>) -> Result<Self, SequentialWeightVerificationError> {
        if episode_ids.is_empty() {
            return Err(SequentialWeightVerificationError::EmptyConfirmatoryEpisodeManifest);
        }
        if episode_ids
            .iter()
            .any(|episode_id| episode_id.trim().is_empty())
        {
            return Err(SequentialWeightVerificationError::EmptyEpisodeId);
        }
        episode_ids.sort();
        if episode_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(
                SequentialWeightVerificationError::InvalidConfirmatoryEpisodeManifest(
                    "duplicate episode identifiers",
                ),
            );
        }
        let mut manifest = Self {
            format_version: 1,
            episode_ids,
            manifest_digest: String::new(),
        };
        manifest.manifest_digest = confirmatory_episode_manifest_digest(&manifest)?;
        Ok(manifest)
    }

    pub fn episode_count(&self) -> usize {
        self.episode_ids.len()
    }

    fn contains(&self, episode_id: &str) -> bool {
        self.episode_ids
            .binary_search_by(|candidate| candidate.as_str().cmp(episode_id))
            .is_ok()
    }
}

/// Digest a canonical manifest. Sorting and uniqueness are validated by callers.
fn confirmatory_episode_manifest_digest(
    manifest: &ConfirmatoryEpisodeManifest,
) -> Result<String, SequentialWeightVerificationError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-confirmatory-episode-manifest-v1\0");
    hasher.update(&manifest.format_version.to_le_bytes());
    hasher.update(&(manifest.episode_ids.len() as u64).to_le_bytes());
    for episode_id in &manifest.episode_ids {
        let bytes = episode_id.as_bytes();
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// Stable campaign-registry digest for one episode identity.
fn confirmatory_episode_id_digest(episode_id: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-confirmatory-episode-id-v1\0");
    hasher.update(&(episode_id.len() as u64).to_le_bytes());
    hasher.update(episode_id.as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn validate_confirmatory_episode_manifest(
    manifest: &ConfirmatoryEpisodeManifest,
) -> Result<(), SequentialWeightVerificationError> {
    if manifest.format_version != 1 {
        return Err(
            SequentialWeightVerificationError::InvalidConfirmatoryEpisodeManifest(
                "unsupported format version",
            ),
        );
    }
    if manifest.episode_ids.is_empty()
        || manifest
            .episode_ids
            .iter()
            .any(|episode_id| episode_id.trim().is_empty())
    {
        return Err(
            SequentialWeightVerificationError::InvalidConfirmatoryEpisodeManifest(
                "empty manifest or episode identifier",
            ),
        );
    }
    if manifest
        .episode_ids
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(
            SequentialWeightVerificationError::InvalidConfirmatoryEpisodeManifest(
                "episode identifiers are not strictly sorted and unique",
            ),
        );
    }
    validate_hex_digest(
        &manifest.manifest_digest,
        "confirmatory_episode_manifest_digest",
    )?;
    let expected = confirmatory_episode_manifest_digest(manifest)?;
    if expected != manifest.manifest_digest {
        return Err(SequentialWeightVerificationError::InvalidConfirmatoryEpisodeManifest(
            "manifest digest mismatch",
        ));
    }
    Ok(())
}

/// State of a candidate's anytime-valid comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequentialWeightVerificationStatus {
    /// More paired confirmatory episodes may be collected.
    Collecting,
    /// The lower confidence bound exceeds the required improvement.
    BenefitEstablished,
    /// The upper confidence bound is below the required improvement.
    BenefitNotEstablished,
    /// The observation budget ended before either conclusion was supported.
    InconclusiveAtBudget,
}

/// Replayable confidence-sequence report for one frozen candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialWeightVerificationReport {
    /// Model and protocol identities.
    pub binding: WeightVerificationBinding,
    /// Frozen sequential testing policy.
    pub config: SequentialWeightVerificationConfig,
    /// Number of unique paired episodes processed.
    pub observation_count: u64,
    /// Mean paired improvement; positive favors the candidate.
    pub mean_improvement: f64,
    /// Time-uniform lower confidence bound.
    pub lower_confidence_bound: f64,
    /// Time-uniform upper confidence bound.
    pub upper_confidence_bound: f64,
    /// Confidence radius used at the current look.
    pub confidence_radius: f64,
    /// Local alpha allocation for this candidate at this observation count.
    pub alpha_at_look: f64,
    /// Digest of ordered episode IDs and paired losses.
    pub observation_digest: String,
    /// Digest of the complete report claims.
    pub report_digest: String,
    /// Current qualification status.
    pub status: SequentialWeightVerificationStatus,
}

/// Restartable checkpoint of one in-progress candidate's ordered paired-loss stream.
///
/// It contains the episode observations needed to rebuild the running mean,
/// duplicate-ID set, and BLAKE3 stream digest. Protect the serialized content
/// as evaluation evidence; the checkpoint digest is integrity detection, not
/// authentication.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialWeightVerificationCheckpoint {
    pub format_version: u32,
    pub binding: WeightVerificationBinding,
    pub config: SequentialWeightVerificationConfig,
    pub confirmatory_episode_manifest: ConfirmatoryEpisodeManifest,
    pub observations: Vec<PairedLossObservation>,
    pub report_digest: String,
    pub checkpoint_digest: String,
}

#[derive(Serialize)]
struct VerifierCheckpointDigestClaims<'a> {
    format_version: u32,
    binding: &'a WeightVerificationBinding,
    config: SequentialWeightVerificationConfig,
    confirmatory_episode_manifest: &'a ConfirmatoryEpisodeManifest,
    observations: &'a [PairedLossObservation],
    report_digest: &'a str,
}

fn verifier_checkpoint_digest(
    checkpoint: &SequentialWeightVerificationCheckpoint,
) -> Result<String, SequentialWeightVerificationError> {
    let claims = VerifierCheckpointDigestClaims {
        format_version: checkpoint.format_version,
        binding: &checkpoint.binding,
        config: checkpoint.config,
        confirmatory_episode_manifest: &checkpoint.confirmatory_episode_manifest,
        observations: &checkpoint.observations,
        report_digest: &checkpoint.report_digest,
    };
    let encoded = serde_json::to_vec(&claims)
        .map_err(|_| SequentialWeightVerificationError::VerifierCheckpointSerialization)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-weight-verification-checkpoint-v2\0");
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

/// Failure from sequential repair verification.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SequentialWeightVerificationError {
    #[error("candidate index must be one-based and non-zero")]
    InvalidCandidateIndex,
    #[error("model or protocol digest is not a 64-character lowercase hexadecimal digest: {0}")]
    InvalidDigest(&'static str),
    #[error("campaign alpha must be in (0, 1) and minimum improvement in [0, 1]")]
    InvalidConfig,
    #[error("minimum observations must be non-zero and no greater than maximum observations")]
    InvalidObservationBudget,
    #[error("episode identifier must be non-empty")]
    EmptyEpisodeId,
    #[error("episode identifier has already been evaluated: {0}")]
    DuplicateEpisodeId(String),
    #[error("baseline or candidate loss is non-finite or outside [0, 1]")]
    InvalidLoss,
    #[error("candidate observation budget is exhausted")]
    ObservationBudgetExhausted,
    #[error("candidate evaluation is already terminal")]
    AlreadyTerminal,
    #[error("confirmatory episode manifest is empty or invalid: {0}")]
    InvalidConfirmatoryEpisodeManifest(&'static str),
    #[error("confirmatory episode manifest must contain at least {expected} IDs, got {actual}")]
    ConfirmatoryManifestTooSmall { expected: usize, actual: usize },
    #[error("one or more confirmatory episode IDs have already been reserved by this campaign")]
    ConfirmatoryEpisodeOverlap,
    #[error("episode ID is not part of this candidate's confirmatory manifest")]
    EpisodeNotInConfirmatoryManifest,
    #[error("confirmatory episode manifest must contain at least one episode")]
    EmptyConfirmatoryEpisodeManifest,
    #[error("confirmatory data shard has already been assigned in this campaign")]
    ConfirmatoryShardAlreadyUsed,
    #[error("candidate model digest has already been evaluated in this campaign")]
    CandidateModelAlreadyEvaluated,
    #[error("candidate ordinal space is exhausted")]
    CandidateIndexExhausted,
    #[error("failed to serialize sequential verification report")]
    ReportSerialization,
    #[error("statistical report has not established the required benefit")]
    ReportNotQualified,
    #[error("statistical report field failed independent recomputation: {0}")]
    ReportFieldMismatch(&'static str),
    #[error("statistical report digest does not match its claims")]
    ReportDigestMismatch,
    #[error("unsupported campaign snapshot version {0}")]
    UnsupportedCampaignSnapshotVersion(u32),
    #[error("campaign snapshot ledger is malformed or inconsistent")]
    InvalidCampaignSnapshotLedger,
    #[error("campaign snapshot digest does not match its contents")]
    CampaignSnapshotDigestMismatch,
    #[error("failed to serialize campaign snapshot")]
    CampaignSnapshotSerialization,
    #[error("unsupported verifier checkpoint version {0}")]
    UnsupportedVerifierCheckpointVersion(u32),
    #[error("verifier checkpoint digest does not match its contents")]
    VerifierCheckpointDigestMismatch,
    #[error("verifier checkpoint does not reproduce its claimed statistical report")]
    VerifierCheckpointReportMismatch,
    #[error("failed to serialize verifier checkpoint")]
    VerifierCheckpointSerialization,
}

/// Deterministic, integrity-digested checkpoint of a sequential repair campaign.
///
/// The digest detects accidental or partial checkpoint corruption. It is not a
/// signature or trust anchor; persistent storage must authenticate the checkpoint
/// and the independent verifier must still validate signed candidate receipts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialWeightVerificationCampaignSnapshot {
    pub format_version: u32,
    pub config: SequentialWeightVerificationConfig,
    pub evaluation_protocol_digest: String,
    pub next_candidate_index: u64,
    pub used_candidate_digests: Vec<String>,
    pub used_confirmatory_shard_digests: Vec<String>,
    pub used_confirmatory_episode_id_digests: Vec<String>,
    pub snapshot_digest: String,
}

#[derive(Serialize)]
struct CampaignSnapshotDigestClaims<'a> {
    format_version: u32,
    config: SequentialWeightVerificationConfig,
    evaluation_protocol_digest: &'a str,
    next_candidate_index: u64,
    used_candidate_digests: &'a [String],
    used_confirmatory_shard_digests: &'a [String],
    used_confirmatory_episode_id_digests: &'a [String],
}

fn campaign_snapshot_digest(
    snapshot: &SequentialWeightVerificationCampaignSnapshot,
) -> Result<String, SequentialWeightVerificationError> {
    let claims = CampaignSnapshotDigestClaims {
        format_version: snapshot.format_version,
        config: snapshot.config,
        evaluation_protocol_digest: &snapshot.evaluation_protocol_digest,
        next_candidate_index: snapshot.next_candidate_index,
        used_candidate_digests: &snapshot.used_candidate_digests,
        used_confirmatory_shard_digests: &snapshot.used_confirmatory_shard_digests,
        used_confirmatory_episode_id_digests: &snapshot.used_confirmatory_episode_id_digests,
    };
    let encoded = serde_json::to_vec(&claims)
        .map_err(|_| SequentialWeightVerificationError::CampaignSnapshotSerialization)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-weight-verification-campaign-snapshot-v2\0");
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

/// Campaign ledger that fixes alpha and the evaluation protocol across candidate edits.
/// It rejects reuse of candidate digests, shard digests, and episode IDs even when
/// a reused subset is presented under a different shard digest.
#[derive(Debug)]
pub struct SequentialWeightVerificationCampaign {
    config: SequentialWeightVerificationConfig,
    evaluation_protocol_digest: String,
    next_candidate_index: u64,
    used_candidate_digests: BTreeSet<String>,
    used_confirmatory_shards: BTreeSet<String>,
    used_confirmatory_episode_id_digests: BTreeSet<String>,
}

impl SequentialWeightVerificationCampaign {
    /// Start a campaign with one family-wise alpha budget and a frozen protocol.
    pub fn new(
        config: SequentialWeightVerificationConfig,
        evaluation_protocol_digest: String,
    ) -> Result<Self, SequentialWeightVerificationError> {
        validate_config(config)?;
        validate_hex_digest(&evaluation_protocol_digest, "evaluation_protocol_digest")?;
        Ok(Self {
            config,
            evaluation_protocol_digest,
            next_candidate_index: 1,
            used_candidate_digests: BTreeSet::new(),
            used_confirmatory_shards: BTreeSet::new(),
            used_confirmatory_episode_id_digests: BTreeSet::new(),
        })
    }

    /// Start the next candidate with a fresh confirmatory data shard.
    /// The campaign assigns the ordinal and enforces the shared alpha budget.
    pub fn start_candidate(
        &mut self,
        baseline_model_digest: String,
        candidate_model_digest: String,
        confirmatory_shard_digest: String,
        confirmatory_episode_manifest: ConfirmatoryEpisodeManifest,
        precondition_gate_digest: String,
    ) -> Result<SequentialWeightVerification, SequentialWeightVerificationError> {
        if self.next_candidate_index == u64::MAX {
            return Err(SequentialWeightVerificationError::CandidateIndexExhausted);
        }
        validate_confirmatory_episode_manifest(&confirmatory_episode_manifest)?;
        if confirmatory_episode_manifest.episode_count() < self.config.maximum_observations as usize
        {
            return Err(
                SequentialWeightVerificationError::ConfirmatoryManifestTooSmall {
                    expected: self.config.maximum_observations as usize,
                    actual: confirmatory_episode_manifest.episode_count(),
                },
            );
        }
        if self
            .used_candidate_digests
            .contains(&candidate_model_digest)
        {
            return Err(SequentialWeightVerificationError::CandidateModelAlreadyEvaluated);
        }
        if self
            .used_confirmatory_shards
            .contains(&confirmatory_shard_digest)
        {
            return Err(SequentialWeightVerificationError::ConfirmatoryShardAlreadyUsed);
        }
        let episode_id_digests = confirmatory_episode_manifest
            .episode_ids
            .iter()
            .map(|episode_id| confirmatory_episode_id_digest(episode_id))
            .collect::<Vec<_>>();
        if episode_id_digests
            .iter()
            .any(|digest| self.used_confirmatory_episode_id_digests.contains(digest))
        {
            return Err(SequentialWeightVerificationError::ConfirmatoryEpisodeOverlap);
        }
        let binding = WeightVerificationBinding {
            candidate_index: self.next_candidate_index,
            baseline_model_digest,
            candidate_model_digest: candidate_model_digest.clone(),
            evaluation_protocol_digest: self.evaluation_protocol_digest.clone(),
            confirmatory_shard_digest: confirmatory_shard_digest.clone(),
            confirmatory_episode_manifest_digest: confirmatory_episode_manifest
                .manifest_digest
                .clone(),
            precondition_gate_digest,
        };
        let verifier = SequentialWeightVerification::from_binding(
            binding,
            self.config,
            confirmatory_episode_manifest,
        )?;
        self.used_candidate_digests.insert(candidate_model_digest);
        self.used_confirmatory_shards
            .insert(confirmatory_shard_digest);
        self.used_confirmatory_episode_id_digests
            .extend(episode_id_digests);
        self.next_candidate_index += 1;
        Ok(verifier)
    }

    /// Return the next candidate ordinal to be assigned.
    pub fn next_candidate_index(&self) -> u64 {
        self.next_candidate_index
    }

    /// Return the number of confirmatory shards reserved by this campaign.
    pub fn used_confirmatory_shard_count(&self) -> usize {
        self.used_confirmatory_shards.len()
    }

    /// Return the number of unique confirmatory episode IDs reserved by this campaign.
    pub fn used_confirmatory_episode_id_count(&self) -> usize {
        self.used_confirmatory_episode_id_digests.len()
    }

    /// Create a deterministic checkpoint suitable for durable storage.
    pub fn snapshot(
        &self,
    ) -> Result<SequentialWeightVerificationCampaignSnapshot, SequentialWeightVerificationError>
    {
        let mut snapshot = SequentialWeightVerificationCampaignSnapshot {
            format_version: 2,
            config: self.config,
            evaluation_protocol_digest: self.evaluation_protocol_digest.clone(),
            next_candidate_index: self.next_candidate_index,
            used_candidate_digests: self.used_candidate_digests.iter().cloned().collect(),
            used_confirmatory_shard_digests: self
                .used_confirmatory_shards
                .iter()
                .cloned()
                .collect(),
            used_confirmatory_episode_id_digests: self
                .used_confirmatory_episode_id_digests
                .iter()
                .cloned()
                .collect(),
            snapshot_digest: String::new(),
        };
        snapshot.snapshot_digest = campaign_snapshot_digest(&snapshot)?;
        Ok(snapshot)
    }

    /// Restore a campaign without resetting its alpha spend or no-reuse ledger.
    ///
    /// A valid digest guarantees content consistency only. Authenticate the
    /// checkpoint through the storage/trust layer before accepting it as state.
    pub fn from_snapshot(
        snapshot: SequentialWeightVerificationCampaignSnapshot,
    ) -> Result<Self, SequentialWeightVerificationError> {
        if snapshot.format_version != 2 {
            return Err(
                SequentialWeightVerificationError::UnsupportedCampaignSnapshotVersion(
                    snapshot.format_version,
                ),
            );
        }
        validate_config(snapshot.config)?;
        validate_hex_digest(
            &snapshot.evaluation_protocol_digest,
            "evaluation_protocol_digest",
        )?;
        validate_hex_digest(&snapshot.snapshot_digest, "campaign_snapshot_digest")?;
        if snapshot.next_candidate_index == 0 {
            return Err(SequentialWeightVerificationError::InvalidCampaignSnapshotLedger);
        }

        let expected_count = snapshot.next_candidate_index - 1;
        let candidate_count = u64::try_from(snapshot.used_candidate_digests.len())
            .map_err(|_| SequentialWeightVerificationError::InvalidCampaignSnapshotLedger)?;
        let shard_count = u64::try_from(snapshot.used_confirmatory_shard_digests.len())
            .map_err(|_| SequentialWeightVerificationError::InvalidCampaignSnapshotLedger)?;
        if candidate_count != expected_count || shard_count != expected_count {
            return Err(SequentialWeightVerificationError::InvalidCampaignSnapshotLedger);
        }
        if snapshot
            .used_candidate_digests
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || snapshot
                .used_confirmatory_shard_digests
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || snapshot
                .used_confirmatory_episode_id_digests
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(SequentialWeightVerificationError::InvalidCampaignSnapshotLedger);
        }

        let mut used_candidate_digests = BTreeSet::new();
        for digest in &snapshot.used_candidate_digests {
            validate_hex_digest(digest, "candidate_model_digest")?;
            if !used_candidate_digests.insert(digest.clone()) {
                return Err(SequentialWeightVerificationError::InvalidCampaignSnapshotLedger);
            }
        }
        let mut used_confirmatory_shards = BTreeSet::new();
        for digest in &snapshot.used_confirmatory_shard_digests {
            validate_hex_digest(digest, "confirmatory_shard_digest")?;
            if !used_confirmatory_shards.insert(digest.clone()) {
                return Err(SequentialWeightVerificationError::InvalidCampaignSnapshotLedger);
            }
        }

        let expected_episode_id_count = expected_count
            .checked_mul(snapshot.config.maximum_observations)
            .ok_or(SequentialWeightVerificationError::InvalidCampaignSnapshotLedger)?;
        let episode_id_count =
            u64::try_from(snapshot.used_confirmatory_episode_id_digests.len())
                .map_err(|_| SequentialWeightVerificationError::InvalidCampaignSnapshotLedger)?;
        if episode_id_count < expected_episode_id_count {
            return Err(SequentialWeightVerificationError::InvalidCampaignSnapshotLedger);
        }
        let mut used_confirmatory_episode_id_digests = BTreeSet::new();
        for digest in &snapshot.used_confirmatory_episode_id_digests {
            validate_hex_digest(digest, "confirmatory_episode_id_digest")?;
            if !used_confirmatory_episode_id_digests.insert(digest.clone()) {
                return Err(SequentialWeightVerificationError::InvalidCampaignSnapshotLedger);
            }
        }

        let expected_digest = campaign_snapshot_digest(&snapshot)?;
        if expected_digest != snapshot.snapshot_digest {
            return Err(SequentialWeightVerificationError::CampaignSnapshotDigestMismatch);
        }

        Ok(Self {
            config: snapshot.config,
            evaluation_protocol_digest: snapshot.evaluation_protocol_digest,
            next_candidate_index: snapshot.next_candidate_index,
            used_candidate_digests,
            used_confirmatory_shards,
            used_confirmatory_episode_id_digests,
        })
    }
}

/// Incremental verifier for one frozen baseline/candidate/protocol triple.
#[derive(Debug)]
pub struct SequentialWeightVerification {
    binding: WeightVerificationBinding,
    config: SequentialWeightVerificationConfig,
    confirmatory_episode_manifest: ConfirmatoryEpisodeManifest,
    seen_episode_ids: BTreeSet<String>,
    observation_count: u64,
    mean_improvement: f64,
    observation_hasher: blake3::Hasher,
    observations: Vec<PairedLossObservation>,
    terminal: bool,
}

impl SequentialWeightVerification {
    /// Construct only after the campaign ledger assigns the binding.
    fn from_binding(
        binding: WeightVerificationBinding,
        config: SequentialWeightVerificationConfig,
        confirmatory_episode_manifest: ConfirmatoryEpisodeManifest,
    ) -> Result<Self, SequentialWeightVerificationError> {
        validate_binding(&binding)?;
        validate_config(config)?;
        validate_confirmatory_episode_manifest(&confirmatory_episode_manifest)?;
        if binding.confirmatory_episode_manifest_digest
            != confirmatory_episode_manifest.manifest_digest
        {
            return Err(
                SequentialWeightVerificationError::InvalidConfirmatoryEpisodeManifest(
                    "binding digest does not match manifest",
                ),
            );
        }
        if confirmatory_episode_manifest.episode_count() < config.maximum_observations as usize {
            return Err(
                SequentialWeightVerificationError::ConfirmatoryManifestTooSmall {
                    expected: config.maximum_observations as usize,
                    actual: confirmatory_episode_manifest.episode_count(),
                },
            );
        }

        let mut observation_hasher = blake3::Hasher::new();
        observation_hasher.update(b"symthaea-weight-verification-observations-v1\0");

        Ok(Self {
            binding,
            config,
            confirmatory_episode_manifest,
            seen_episode_ids: BTreeSet::new(),
            observation_count: 0,
            mean_improvement: 0.0,
            observation_hasher,
            observations: Vec::new(),
            terminal: false,
        })
    }

    /// Add one paired episode and return the updated confidence-sequence report.
    ///
    /// A candidate cannot be modified after the first observation because its
    /// model digest is fixed in the verifier binding. Each episode identifier
    /// may appear only once.
    pub fn observe(
        &mut self,
        observation: PairedLossObservation,
    ) -> Result<SequentialWeightVerificationReport, SequentialWeightVerificationError> {
        if self.terminal {
            return Err(SequentialWeightVerificationError::AlreadyTerminal);
        }
        if self.observation_count >= self.config.maximum_observations {
            return Err(SequentialWeightVerificationError::ObservationBudgetExhausted);
        }
        if observation.episode_id.trim().is_empty() {
            return Err(SequentialWeightVerificationError::EmptyEpisodeId);
        }
        if !self
            .confirmatory_episode_manifest
            .contains(&observation.episode_id)
        {
            return Err(SequentialWeightVerificationError::EpisodeNotInConfirmatoryManifest);
        }
        if !observation.baseline_loss.is_finite()
            || !(0.0..=1.0).contains(&observation.baseline_loss)
            || !observation.candidate_loss.is_finite()
            || !(0.0..=1.0).contains(&observation.candidate_loss)
        {
            return Err(SequentialWeightVerificationError::InvalidLoss);
        }
        if self.seen_episode_ids.contains(&observation.episode_id) {
            return Err(SequentialWeightVerificationError::DuplicateEpisodeId(
                observation.episode_id,
            ));
        }

        let improvement = observation.baseline_loss - observation.candidate_loss;
        self.seen_episode_ids.insert(observation.episode_id.clone());
        self.observation_count += 1;
        self.mean_improvement +=
            (improvement - self.mean_improvement) / self.observation_count as f64;

        let episode_id_bytes = observation.episode_id.as_bytes();
        self.observation_hasher
            .update(&(episode_id_bytes.len() as u64).to_le_bytes());
        self.observation_hasher.update(episode_id_bytes);
        self.observation_hasher
            .update(&observation.baseline_loss.to_bits().to_le_bytes());
        self.observation_hasher
            .update(&observation.candidate_loss.to_bits().to_le_bytes());
        self.observations.push(observation);

        let report = self.current_report()?;
        self.terminal = matches!(
            report.status,
            SequentialWeightVerificationStatus::BenefitEstablished
                | SequentialWeightVerificationStatus::BenefitNotEstablished
                | SequentialWeightVerificationStatus::InconclusiveAtBudget
        );
        Ok(report)
    }

    /// Return the current report without consuming another episode.
    pub fn report(
        &self,
    ) -> Result<SequentialWeightVerificationReport, SequentialWeightVerificationError> {
        self.current_report()
    }

    /// Create a restartable checkpoint of this candidate's full ordered evidence stream.
    pub fn checkpoint(
        &self,
    ) -> Result<SequentialWeightVerificationCheckpoint, SequentialWeightVerificationError> {
        let report = self.current_report()?;
        let mut checkpoint = SequentialWeightVerificationCheckpoint {
            format_version: 2,
            binding: self.binding.clone(),
            config: self.config,
            confirmatory_episode_manifest: self.confirmatory_episode_manifest.clone(),
            observations: self.observations.clone(),
            report_digest: report.report_digest,
            checkpoint_digest: String::new(),
        };
        checkpoint.checkpoint_digest = verifier_checkpoint_digest(&checkpoint)?;
        Ok(checkpoint)
    }

    /// Restore a verifier by replaying every paired observation into fresh state.
    ///
    /// Duplicate IDs, terminal-state overflow, malformed losses, changed report
    /// claims, and a checkpoint digest mismatch all fail closed.
    pub fn from_checkpoint(
        checkpoint: SequentialWeightVerificationCheckpoint,
    ) -> Result<Self, SequentialWeightVerificationError> {
        if checkpoint.format_version != 2 {
            return Err(
                SequentialWeightVerificationError::UnsupportedVerifierCheckpointVersion(
                    checkpoint.format_version,
                ),
            );
        }
        validate_binding(&checkpoint.binding)?;
        validate_confirmatory_episode_manifest(&checkpoint.confirmatory_episode_manifest)?;
        if checkpoint.binding.confirmatory_episode_manifest_digest
            != checkpoint.confirmatory_episode_manifest.manifest_digest
        {
            return Err(SequentialWeightVerificationError::VerifierCheckpointReportMismatch);
        }
        validate_config(checkpoint.config)?;
        validate_hex_digest(&checkpoint.report_digest, "report_digest")?;
        validate_hex_digest(&checkpoint.checkpoint_digest, "checkpoint_digest")?;
        let observation_count = u64::try_from(checkpoint.observations.len())
            .map_err(|_| SequentialWeightVerificationError::VerifierCheckpointReportMismatch)?;
        if observation_count > checkpoint.config.maximum_observations {
            return Err(SequentialWeightVerificationError::VerifierCheckpointReportMismatch);
        }
        let expected_digest = verifier_checkpoint_digest(&checkpoint)?;
        if expected_digest != checkpoint.checkpoint_digest {
            return Err(SequentialWeightVerificationError::VerifierCheckpointDigestMismatch);
        }

        let mut verifier = Self::from_binding(
            checkpoint.binding,
            checkpoint.config,
            checkpoint.confirmatory_episode_manifest,
        )?;
        for observation in checkpoint.observations {
            if verifier.terminal {
                return Err(SequentialWeightVerificationError::VerifierCheckpointReportMismatch);
            }
            verifier.observe(observation)?;
        }
        let restored_report = verifier.current_report()?;
        if restored_report.report_digest != checkpoint.report_digest {
            return Err(SequentialWeightVerificationError::VerifierCheckpointReportMismatch);
        }
        Ok(verifier)
    }

    fn current_report(
        &self,
    ) -> Result<SequentialWeightVerificationReport, SequentialWeightVerificationError> {
        let n = self.observation_count;
        let n_float = n as f64;
        let candidate = self.binding.candidate_index as f64;
        let candidate_alpha = self.config.campaign_alpha / (candidate * (candidate + 1.0));
        let (alpha_at_look, confidence_radius, lower_confidence_bound, upper_confidence_bound) =
            if n == 0 {
                // No confidence look has occurred yet. Return the full bounded
                // range and a finite initial candidate-level alpha allocation.
                (candidate_alpha, 1.0, -1.0, 1.0)
            } else {
                let alpha_denominator = candidate * (candidate + 1.0) * n_float * (n_float + 1.0);
                let alpha_at_look = self.config.campaign_alpha / alpha_denominator;
                let log_tail_ratio = std::f64::consts::LN_2 - self.config.campaign_alpha.ln()
                    + candidate.ln()
                    + (candidate + 1.0).ln()
                    + n_float.ln()
                    + (n_float + 1.0).ln()
                    + 1e-12;
                let confidence_radius = (2.0 * log_tail_ratio / n_float).sqrt();
                let lower = (self.mean_improvement - confidence_radius).max(-1.0);
                let upper = (self.mean_improvement + confidence_radius).min(1.0);
                (alpha_at_look, confidence_radius, lower, upper)
            };

        let status = if n < self.config.minimum_observations {
            SequentialWeightVerificationStatus::Collecting
        } else if lower_confidence_bound > self.config.minimum_improvement {
            SequentialWeightVerificationStatus::BenefitEstablished
        } else if upper_confidence_bound < self.config.minimum_improvement {
            SequentialWeightVerificationStatus::BenefitNotEstablished
        } else if n >= self.config.maximum_observations {
            SequentialWeightVerificationStatus::InconclusiveAtBudget
        } else {
            SequentialWeightVerificationStatus::Collecting
        };

        let observation_digest = self
            .observation_hasher
            .clone()
            .finalize()
            .to_hex()
            .to_string();
        let claims = ReportDigestClaims {
            binding: &self.binding,
            config: self.config,
            observation_count: n,
            mean_improvement: self.mean_improvement,
            lower_confidence_bound,
            upper_confidence_bound,
            confidence_radius,
            alpha_at_look,
            observation_digest: &observation_digest,
            status,
        };
        let encoded = serde_json::to_vec(&claims)
            .map_err(|_| SequentialWeightVerificationError::ReportSerialization)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-weight-verification-report-v1\0");
        hasher.update(&(encoded.len() as u64).to_le_bytes());
        hasher.update(&encoded);
        let report_digest = hasher.finalize().to_hex().to_string();

        Ok(SequentialWeightVerificationReport {
            binding: self.binding.clone(),
            config: self.config,
            observation_count: n,
            mean_improvement: self.mean_improvement,
            lower_confidence_bound,
            upper_confidence_bound,
            confidence_radius,
            alpha_at_look,
            observation_digest,
            report_digest,
            status,
        })
    }
}

/// Independently recompute the statistical claims and digest for a qualified report.
///
/// This validates the confidence-sequence arithmetic and report integrity. It
/// does not reproduce the raw episode stream; the independent verifier must
/// still obtain the fresh confirmatory shard and recompute its ordered loss
/// digest before signing acceptance.
pub fn validate_qualified_report(
    report: &SequentialWeightVerificationReport,
) -> Result<(), SequentialWeightVerificationError> {
    validate_binding(&report.binding)?;
    validate_config(report.config)?;
    validate_hex_digest(&report.observation_digest, "observation_digest")?;
    validate_hex_digest(&report.report_digest, "report_digest")?;

    if report.status != SequentialWeightVerificationStatus::BenefitEstablished
        || report.observation_count < report.config.minimum_observations
        || report.observation_count > report.config.maximum_observations
        || report.lower_confidence_bound <= report.config.minimum_improvement
    {
        return Err(SequentialWeightVerificationError::ReportNotQualified);
    }
    if !report.mean_improvement.is_finite() || !(-1.0..=1.0).contains(&report.mean_improvement) {
        return Err(SequentialWeightVerificationError::ReportFieldMismatch(
            "mean_improvement",
        ));
    }

    let n = report.observation_count as f64;
    let candidate = report.binding.candidate_index as f64;
    let alpha_at_look =
        report.config.campaign_alpha / (candidate * (candidate + 1.0) * n * (n + 1.0));
    let log_tail_ratio = std::f64::consts::LN_2 - report.config.campaign_alpha.ln()
        + candidate.ln()
        + (candidate + 1.0).ln()
        + n.ln()
        + (n + 1.0).ln()
        + 1e-12;
    let confidence_radius = (2.0 * log_tail_ratio / n).sqrt();
    let lower = (report.mean_improvement - confidence_radius).max(-1.0);
    let upper = (report.mean_improvement + confidence_radius).min(1.0);

    for (field, actual, expected) in [
        ("alpha_at_look", report.alpha_at_look, alpha_at_look),
        (
            "confidence_radius",
            report.confidence_radius,
            confidence_radius,
        ),
        (
            "lower_confidence_bound",
            report.lower_confidence_bound,
            lower,
        ),
        (
            "upper_confidence_bound",
            report.upper_confidence_bound,
            upper,
        ),
    ] {
        if !actual.is_finite() || actual.to_bits() != expected.to_bits() {
            return Err(SequentialWeightVerificationError::ReportFieldMismatch(
                field,
            ));
        }
    }

    let claims = ReportDigestClaims {
        binding: &report.binding,
        config: report.config,
        observation_count: report.observation_count,
        mean_improvement: report.mean_improvement,
        lower_confidence_bound: report.lower_confidence_bound,
        upper_confidence_bound: report.upper_confidence_bound,
        confidence_radius: report.confidence_radius,
        alpha_at_look: report.alpha_at_look,
        observation_digest: &report.observation_digest,
        status: report.status,
    };
    let encoded = serde_json::to_vec(&claims)
        .map_err(|_| SequentialWeightVerificationError::ReportSerialization)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-weight-verification-report-v1\0");
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    let expected_digest = hasher.finalize().to_hex().to_string();
    if expected_digest != report.report_digest {
        return Err(SequentialWeightVerificationError::ReportDigestMismatch);
    }
    Ok(())
}

#[derive(Serialize)]
struct ReportDigestClaims<'a> {
    binding: &'a WeightVerificationBinding,
    config: SequentialWeightVerificationConfig,
    observation_count: u64,
    mean_improvement: f64,
    lower_confidence_bound: f64,
    upper_confidence_bound: f64,
    confidence_radius: f64,
    alpha_at_look: f64,
    observation_digest: &'a str,
    status: SequentialWeightVerificationStatus,
}

fn validate_hex_digest(
    value: &str,
    name: &'static str,
) -> Result<(), SequentialWeightVerificationError> {
    if value.len() != DIGEST_HEX_LEN
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(SequentialWeightVerificationError::InvalidDigest(name));
    }
    Ok(())
}

fn validate_binding(
    binding: &WeightVerificationBinding,
) -> Result<(), SequentialWeightVerificationError> {
    if binding.candidate_index == 0 {
        return Err(SequentialWeightVerificationError::InvalidCandidateIndex);
    }
    for (name, value) in [
        (
            "baseline_model_digest",
            binding.baseline_model_digest.as_str(),
        ),
        (
            "candidate_model_digest",
            binding.candidate_model_digest.as_str(),
        ),
        (
            "evaluation_protocol_digest",
            binding.evaluation_protocol_digest.as_str(),
        ),
        (
            "confirmatory_shard_digest",
            binding.confirmatory_shard_digest.as_str(),
        ),
        (
            "confirmatory_episode_manifest_digest",
            binding.confirmatory_episode_manifest_digest.as_str(),
        ),
        (
            "precondition_gate_digest",
            binding.precondition_gate_digest.as_str(),
        ),
    ] {
        if value.len() != DIGEST_HEX_LEN
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(SequentialWeightVerificationError::InvalidDigest(name));
        }
    }
    if binding.baseline_model_digest == binding.candidate_model_digest {
        return Err(SequentialWeightVerificationError::InvalidDigest(
            "candidate must differ from baseline",
        ));
    }
    Ok(())
}

fn validate_config(
    config: SequentialWeightVerificationConfig,
) -> Result<(), SequentialWeightVerificationError> {
    if !config.campaign_alpha.is_finite()
        || !(0.0..1.0).contains(&config.campaign_alpha)
        || !config.minimum_improvement.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_improvement)
    {
        return Err(SequentialWeightVerificationError::InvalidConfig);
    }
    if config.minimum_observations == 0 || config.minimum_observations > config.maximum_observations
    {
        return Err(SequentialWeightVerificationError::InvalidObservationBudget);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn config(
        minimum_improvement: f64,
        maximum_observations: u64,
    ) -> SequentialWeightVerificationConfig {
        SequentialWeightVerificationConfig {
            campaign_alpha: 0.05,
            minimum_improvement,
            minimum_observations: 2,
            maximum_observations,
        }
    }

    fn campaign(
        minimum_improvement: f64,
        maximum_observations: u64,
    ) -> SequentialWeightVerificationCampaign {
        SequentialWeightVerificationCampaign::new(
            config(minimum_improvement, maximum_observations),
            digest('c'),
        )
        .expect("valid campaign")
    }

    fn manifest_for_shard(
        campaign: &SequentialWeightVerificationCampaign,
        shard_marker: char,
    ) -> ConfirmatoryEpisodeManifest {
        ConfirmatoryEpisodeManifest::new(
            (0..campaign.config.maximum_observations)
                .map(|index| format!("episode-{shard_marker}-{index}"))
                .collect(),
        )
        .expect("valid confirmatory episode manifest")
    }

    fn candidate(
        campaign: &mut SequentialWeightVerificationCampaign,
        model_marker: char,
        shard_marker: char,
    ) -> SequentialWeightVerification {
        let manifest = manifest_for_shard(campaign, shard_marker);
        campaign
            .start_candidate(
                digest('a'),
                digest(model_marker),
                digest(shard_marker),
                manifest,
                digest('e'),
            )
            .expect("fresh candidate")
    }

    fn observation_for_shard(
        shard_marker: char,
        index: u64,
        baseline_loss: f64,
        candidate_loss: f64,
    ) -> PairedLossObservation {
        PairedLossObservation {
            episode_id: format!("episode-{shard_marker}-{index}"),
            baseline_loss,
            candidate_loss,
        }
    }

    fn observation(index: u64, baseline_loss: f64, candidate_loss: f64) -> PairedLossObservation {
        observation_for_shard('d', index, baseline_loss, candidate_loss)
    }

    fn episodes_for_replay_test(shard_marker: char) -> [PairedLossObservation; 3] {
        [
            observation_for_shard(shard_marker, 1, 0.8, 0.4),
            observation_for_shard(shard_marker, 2, 0.7, 0.3),
            observation_for_shard(shard_marker, 3, 0.9, 0.5),
        ]
    }

    #[test]
    fn confirmatory_manifest_is_canonical_and_rejects_duplicate_or_empty_ids() {
        let forward =
            ConfirmatoryEpisodeManifest::new(vec!["episode-b".into(), "episode-a".into()])
                .expect("canonical manifest");
        let reverse =
            ConfirmatoryEpisodeManifest::new(vec!["episode-a".into(), "episode-b".into()])
                .expect("equivalent canonical manifest");
        assert_eq!(
            forward.episode_ids,
            vec!["episode-a".to_owned(), "episode-b".to_owned()]
        );
        assert_eq!(forward.manifest_digest, reverse.manifest_digest);
        assert!(matches!(
            ConfirmatoryEpisodeManifest::new(Vec::new()),
            Err(SequentialWeightVerificationError::EmptyConfirmatoryEpisodeManifest)
        ));
        assert!(matches!(
            ConfirmatoryEpisodeManifest::new(vec!["duplicate".into(), "duplicate".into()]),
            Err(
                SequentialWeightVerificationError::InvalidConfirmatoryEpisodeManifest(
                    "duplicate episode identifiers"
                )
            )
        ));
        assert!(matches!(
            ConfirmatoryEpisodeManifest::new(vec![" ".into()]),
            Err(SequentialWeightVerificationError::EmptyEpisodeId)
        ));
    }

    #[test]
    fn candidate_manifest_must_cover_the_preregistered_observation_budget() {
        let mut campaign = campaign(0.1, 10);
        let short_manifest = ConfirmatoryEpisodeManifest::new(vec!["episode-only-one".into()])
            .expect("valid but too-short manifest");
        assert!(matches!(
            campaign.start_candidate(
                digest('a'),
                digest('b'),
                digest('d'),
                short_manifest,
                digest('e'),
            ),
            Err(
                SequentialWeightVerificationError::ConfirmatoryManifestTooSmall {
                    expected: 10,
                    actual: 1
                }
            )
        ));
        assert_eq!(campaign.next_candidate_index(), 1);
        assert_eq!(campaign.used_confirmatory_episode_id_count(), 0);
    }

    #[test]
    fn candidate_observation_must_belong_to_its_frozen_manifest() {
        let mut campaign = campaign(0.1, 10);
        let mut verifier = candidate(&mut campaign, 'b', 'd');
        assert!(matches!(
            verifier.observe(observation_for_shard('e', 0, 0.8, 0.4)),
            Err(SequentialWeightVerificationError::EpisodeNotInConfirmatoryManifest)
        ));
        assert_eq!(
            verifier
                .report()
                .expect("unchanged report")
                .observation_count,
            0
        );
    }

    #[test]
    fn distinct_shard_digests_cannot_reuse_overlapping_episode_ids() {
        let mut campaign = campaign(0.1, 10);
        let _first = candidate(&mut campaign, 'b', 'd');
        let overlapping_manifest = manifest_for_shard(&campaign, 'd');

        assert!(matches!(
            campaign.start_candidate(
                digest('a'),
                digest('c'),
                digest('e'),
                overlapping_manifest,
                digest('e'),
            ),
            Err(SequentialWeightVerificationError::ConfirmatoryEpisodeOverlap)
        ));
        assert_eq!(campaign.next_candidate_index(), 2);
        assert_eq!(campaign.used_confirmatory_shard_count(), 1);
        assert_eq!(campaign.used_confirmatory_episode_id_count(), 10);
    }

    #[test]
    fn restored_campaign_preserves_episode_reservations_across_different_shard_digests() {
        let mut campaign = campaign(0.1, 10);
        let _first = candidate(&mut campaign, 'b', 'd');
        let snapshot = campaign.snapshot().expect("snapshot");
        let mut restored =
            SequentialWeightVerificationCampaign::from_snapshot(snapshot).expect("restore");
        let overlapping_manifest_after_restore = manifest_for_shard(&restored, 'd');
        assert!(matches!(
            restored.start_candidate(
                digest('a'),
                digest('c'),
                digest('e'),
                overlapping_manifest_after_restore,
                digest('e'),
            ),
            Err(SequentialWeightVerificationError::ConfirmatoryEpisodeOverlap)
        ));
        assert_eq!(restored.next_candidate_index(), 2);
        assert_eq!(restored.used_confirmatory_shard_count(), 1);
        assert_eq!(restored.used_confirmatory_episode_id_count(), 10);
    }

    #[test]
    fn strong_paired_improvement_is_established_with_anytime_valid_bounds() {
        let mut campaign = campaign(0.05, 2_000);
        let mut verifier = candidate(&mut campaign, 'b', 'd');
        let mut report = verifier.report().expect("empty report");
        for index in 0..600 {
            report = verifier
                .observe(observation(index, 0.9, 0.5))
                .expect("paired episode");
            if report.status == SequentialWeightVerificationStatus::BenefitEstablished {
                break;
            }
        }
        assert_eq!(
            report.status,
            SequentialWeightVerificationStatus::BenefitEstablished
        );
        assert!(report.lower_confidence_bound > report.config.minimum_improvement);
        assert!(report.observation_count >= 2);
        assert!(report.observation_count <= 600);
    }

    #[test]
    fn no_effect_is_not_misreported_as_superiority() {
        let mut campaign = campaign(0.1, 6_000);
        let mut verifier = candidate(&mut campaign, 'b', 'd');
        let mut report = verifier.report().expect("empty report");
        for index in 0..6_000 {
            report = verifier
                .observe(observation(index, 0.5, 0.5))
                .expect("paired episode");
            if report.status != SequentialWeightVerificationStatus::Collecting {
                break;
            }
        }
        assert_eq!(
            report.status,
            SequentialWeightVerificationStatus::BenefitNotEstablished
        );
        assert!(report.upper_confidence_bound < report.config.minimum_improvement);
    }

    #[test]
    fn later_candidates_receive_smaller_alpha_allocation() {
        let mut campaign = campaign(0.2, 100);
        let mut first = candidate(&mut campaign, 'b', 'd');
        let first_report = first.observe(observation(0, 0.8, 0.5)).expect("first look");
        let mut seventh = candidate(&mut campaign, 'c', 'e');
        for (model, shard) in [('e', 'f'), ('f', '1'), ('1', '2'), ('2', '3'), ('3', '4')] {
            seventh = candidate(&mut campaign, model, shard);
        }
        let later_report = seventh
            .observe(observation_for_shard('4', 0, 0.8, 0.5))
            .expect("seventh candidate look");
        assert_eq!(later_report.binding.candidate_index, 7);
        assert!(later_report.alpha_at_look < first_report.alpha_at_look);
        assert!(later_report.confidence_radius > first_report.confidence_radius);
    }

    #[test]
    fn campaign_assigns_ordinals_and_prevents_candidate_or_shard_reuse() {
        let mut campaign = campaign(0.1, 10);
        let first = candidate(&mut campaign, 'b', 'd');
        assert_eq!(first.binding.candidate_index, 1);
        assert_eq!(campaign.next_candidate_index(), 2);
        assert_eq!(campaign.used_confirmatory_shard_count(), 1);
        assert_eq!(campaign.used_confirmatory_episode_id_count(), 10);
        let reused_candidate_manifest = manifest_for_shard(&campaign, 'e');
        assert!(matches!(
            campaign.start_candidate(
                digest('a'),
                digest('b'),
                digest('e'),
                reused_candidate_manifest,
                digest('e'),
            ),
            Err(SequentialWeightVerificationError::CandidateModelAlreadyEvaluated)
        ));
        let reused_shard_manifest = manifest_for_shard(&campaign, 'd');
        assert!(matches!(
            campaign.start_candidate(
                digest('a'),
                digest('c'),
                digest('d'),
                reused_shard_manifest,
                digest('e'),
            ),
            Err(SequentialWeightVerificationError::ConfirmatoryShardAlreadyUsed)
        ));
        assert_eq!(campaign.next_candidate_index(), 2);
        assert_eq!(campaign.used_confirmatory_shard_count(), 1);
    }

    #[test]
    fn campaign_snapshot_restore_preserves_alpha_ordinal_and_no_reuse_ledger() {
        let mut campaign = campaign(0.1, 100);
        let _first = candidate(&mut campaign, 'b', 'd');
        let _second = candidate(&mut campaign, 'c', 'e');
        let snapshot = campaign.snapshot().expect("campaign snapshot");
        let encoded = serde_json::to_vec(&snapshot).expect("serialize checkpoint");
        let decoded: SequentialWeightVerificationCampaignSnapshot =
            serde_json::from_slice(&encoded).expect("deserialize checkpoint");
        let mut restored = SequentialWeightVerificationCampaign::from_snapshot(decoded.clone())
            .expect("restore checkpoint");

        assert_eq!(restored.next_candidate_index(), 3);
        assert_eq!(restored.used_confirmatory_shard_count(), 2);
        assert_eq!(restored.used_confirmatory_episode_id_count(), 200);
        assert_eq!(
            restored
                .snapshot()
                .expect("repeat snapshot")
                .snapshot_digest,
            snapshot.snapshot_digest
        );
        let reused_candidate_manifest = manifest_for_shard(&restored, 'f');
        assert!(matches!(
            restored.start_candidate(
                digest('a'),
                digest('b'),
                digest('f'),
                reused_candidate_manifest,
                digest('e'),
            ),
            Err(SequentialWeightVerificationError::CandidateModelAlreadyEvaluated)
        ));
        let reused_shard_manifest = manifest_for_shard(&restored, 'd');
        assert!(matches!(
            restored.start_candidate(
                digest('a'),
                digest('f'),
                digest('d'),
                reused_shard_manifest,
                digest('e'),
            ),
            Err(SequentialWeightVerificationError::ConfirmatoryShardAlreadyUsed)
        ));

        let next = candidate(&mut restored, 'f', '1');
        assert_eq!(next.binding.candidate_index, 3);
    }

    #[test]
    fn campaign_snapshot_rejects_corrupt_digest_and_inconsistent_ledger() {
        let mut campaign = campaign(0.1, 100);
        let _first = candidate(&mut campaign, 'b', 'd');
        let snapshot = campaign.snapshot().expect("campaign snapshot");

        let mut corrupt_digest = snapshot.clone();
        corrupt_digest.snapshot_digest = "e".repeat(64);
        assert!(matches!(
            SequentialWeightVerificationCampaign::from_snapshot(corrupt_digest),
            Err(SequentialWeightVerificationError::CampaignSnapshotDigestMismatch)
        ));

        let mut corrupt_ledger = snapshot;
        corrupt_ledger.next_candidate_index = 3;
        assert!(matches!(
            SequentialWeightVerificationCampaign::from_snapshot(corrupt_ledger),
            Err(SequentialWeightVerificationError::InvalidCampaignSnapshotLedger)
        ));
    }

    #[test]
    fn duplicate_episode_ids_are_rejected_without_consuming_budget() {
        let mut campaign = campaign(0.1, 10);
        let mut verifier = candidate(&mut campaign, 'b', 'd');
        verifier
            .observe(observation(1, 0.8, 0.4))
            .expect("first episode");
        assert!(matches!(
            verifier.observe(observation(1, 0.2, 0.1)),
            Err(SequentialWeightVerificationError::DuplicateEpisodeId(_))
        ));
        assert_eq!(verifier.report().expect("report").observation_count, 1);
    }

    #[test]
    fn malformed_digests_losses_and_budgets_fail_closed() {
        assert!(matches!(
            SequentialWeightVerificationCampaign::new(config(0.1, 10), "bad".into()),
            Err(SequentialWeightVerificationError::InvalidDigest(_))
        ));
        assert!(matches!(
            SequentialWeightVerificationCampaign::new(config(0.1, 0), digest('c')),
            Err(SequentialWeightVerificationError::InvalidObservationBudget)
        ));

        let mut campaign = campaign(0.1, 10);
        let invalid_candidate_manifest = manifest_for_shard(&campaign, 'd');
        assert!(matches!(
            campaign.start_candidate(
                digest('a'),
                "not-a-digest".into(),
                digest('d'),
                invalid_candidate_manifest,
                digest('e'),
            ),
            Err(SequentialWeightVerificationError::InvalidDigest(_))
        ));
        let mut verifier = candidate(&mut campaign, 'b', 'd');
        assert!(matches!(
            verifier.observe(observation(1, 1.1, 0.1)),
            Err(SequentialWeightVerificationError::InvalidLoss)
        ));
        assert_eq!(verifier.report().expect("report").observation_count, 0);
    }

    #[test]
    fn report_digest_is_replay_deterministic_and_binds_the_evidence_stream() {
        let mut left_campaign = campaign(0.05, 20);
        let mut right_campaign = campaign(0.05, 20);
        let mut left = candidate(&mut left_campaign, 'b', 'd');
        let mut right = candidate(&mut right_campaign, 'b', 'd');
        let mut left_report = left.report().expect("initial");
        let mut right_report = right.report().expect("initial");
        for episode in episodes_for_replay_test('d') {
            left_report = left.observe(episode.clone()).expect("left");
            right_report = right.observe(episode).expect("right");
        }
        assert_eq!(
            left_report.observation_digest,
            right_report.observation_digest
        );
        assert_eq!(left_report.report_digest, right_report.report_digest);

        let mut altered_campaign = campaign(0.05, 20);
        let mut altered = candidate(&mut altered_campaign, 'b', 'e');
        let mut altered_report = altered.report().expect("initial");
        for (index, episode) in episodes_for_replay_test('e').into_iter().enumerate() {
            let changed = if index == 2 {
                PairedLossObservation {
                    candidate_loss: 0.4,
                    ..episode
                }
            } else {
                episode
            };
            altered_report = altered.observe(changed).expect("altered evidence");
        }
        assert_ne!(left_report.report_digest, altered_report.report_digest);
    }

    fn qualified_report() -> SequentialWeightVerificationReport {
        let mut campaign = campaign(0.05, 600);
        let mut verifier = candidate(&mut campaign, 'b', 'd');
        let mut report = verifier.report().expect("initial report");
        for index in 0..600 {
            report = verifier
                .observe(observation(index, 0.9, 0.5))
                .expect("paired episode");
            if report.status == SequentialWeightVerificationStatus::BenefitEstablished {
                break;
            }
        }
        assert_eq!(
            report.status,
            SequentialWeightVerificationStatus::BenefitEstablished
        );
        report
    }

    #[test]
    fn verifier_checkpoint_round_trip_preserves_ordered_stream_and_continuation() {
        let mut campaign = campaign(0.05, 1_000);
        let mut original = candidate(&mut campaign, 'b', 'd');
        for index in 0..12 {
            original
                .observe(observation(index, 0.8, 0.6))
                .expect("initial paired observations");
        }
        let checkpoint = original.checkpoint().expect("candidate checkpoint");
        let encoded = serde_json::to_vec(&checkpoint).expect("serialize checkpoint");
        let decoded: SequentialWeightVerificationCheckpoint =
            serde_json::from_slice(&encoded).expect("deserialize checkpoint");
        let mut restored =
            SequentialWeightVerification::from_checkpoint(decoded).expect("restore checkpoint");
        assert_eq!(
            original.report().expect("original report").report_digest,
            restored.report().expect("restored report").report_digest
        );
        for index in 12..30 {
            let next = observation(index, 0.9, 0.5);
            let original_report = original
                .observe(next.clone())
                .expect("original continuation");
            let restored_report = restored.observe(next).expect("restored continuation");
            assert_eq!(original_report.report_digest, restored_report.report_digest);
        }
    }

    #[test]
    fn verifier_checkpoint_rejects_tampering_and_inconsistent_replay() {
        let mut campaign = campaign(0.05, 1_000);
        let mut verifier = candidate(&mut campaign, 'b', 'd');
        for index in 0..10 {
            verifier
                .observe(observation(index, 0.8, 0.6))
                .expect("paired observation");
        }
        let checkpoint = verifier.checkpoint().expect("checkpoint");

        let mut altered_observation = checkpoint.clone();
        altered_observation.observations[0].candidate_loss = 0.9;
        assert!(matches!(
            SequentialWeightVerification::from_checkpoint(altered_observation),
            Err(SequentialWeightVerificationError::VerifierCheckpointDigestMismatch)
        ));

        let mut altered_manifest = checkpoint.clone();
        altered_manifest.confirmatory_episode_manifest.episode_ids[0] =
            "tampered-episode-id".into();
        altered_manifest.checkpoint_digest =
            verifier_checkpoint_digest(&altered_manifest).expect("recompute checkpoint digest");
        assert!(matches!(
            SequentialWeightVerification::from_checkpoint(altered_manifest),
            Err(
                SequentialWeightVerificationError::InvalidConfirmatoryEpisodeManifest(
                    "manifest digest mismatch"
                )
            )
        ));

        let mut altered_claim = checkpoint;
        altered_claim.report_digest = digest('f');
        altered_claim.checkpoint_digest = verifier_checkpoint_digest(&altered_claim)
            .expect("recompute digest after modifying the claim");
        assert!(matches!(
            SequentialWeightVerification::from_checkpoint(altered_claim),
            Err(SequentialWeightVerificationError::VerifierCheckpointReportMismatch)
        ));
    }

    #[test]
    fn qualified_report_validation_recomputes_bounds_and_digest() {
        let report = qualified_report();
        validate_qualified_report(&report).expect("qualified report validates");

        let mut altered_bound = report.clone();
        altered_bound.lower_confidence_bound += 0.001;
        assert!(matches!(
            validate_qualified_report(&altered_bound),
            Err(SequentialWeightVerificationError::ReportFieldMismatch(
                "lower_confidence_bound"
            ))
        ));

        let mut altered_digest = report.clone();
        altered_digest.report_digest = "e".repeat(64);
        assert!(matches!(
            validate_qualified_report(&altered_digest),
            Err(SequentialWeightVerificationError::ReportDigestMismatch)
        ));

        let mut unqualified = report;
        unqualified.status = SequentialWeightVerificationStatus::BenefitNotEstablished;
        assert!(matches!(
            validate_qualified_report(&unqualified),
            Err(SequentialWeightVerificationError::ReportNotQualified)
        ));
    }

    #[test]
    fn inconclusive_budget_is_terminal_and_cannot_be_extended() {
        let mut campaign = campaign(0.5, 2);
        let mut verifier = candidate(&mut campaign, 'b', 'd');
        verifier
            .observe(observation(1, 0.5, 0.5))
            .expect("first episode");
        let report = verifier
            .observe(observation(2, 0.5, 0.5))
            .expect("budget-ending episode");
        assert_eq!(
            report.status,
            SequentialWeightVerificationStatus::InconclusiveAtBudget
        );
        assert!(matches!(
            verifier.observe(observation(3, 0.9, 0.1)),
            Err(SequentialWeightVerificationError::AlreadyTerminal)
        ));
    }
}
