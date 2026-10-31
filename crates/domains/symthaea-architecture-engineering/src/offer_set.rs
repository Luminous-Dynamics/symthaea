//! Content-addressed proof of the choices available at each policy decision boundary.
//!
//! A shared proposal-stream digest is only an identity assertion. This manifest records the
//! actual canonical direction set and NEW permission seen at each sequential decision or
//! pre-batch snapshot, plus a digest of the evaluator-owned state from which that offer set
//! was derived. Cross-policy verification requires byte-equivalent snapshots for each key.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::event_tape::{ActionIntent, EventTape};

const SCHEMA: &str = "symthaea.ar5b4.offer-set-manifest";
const VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfferSetError {
    pub path: String,
    pub message: String,
}

impl OfferSetError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for OfferSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for OfferSetError {}

/// The exact options visible to every worker sharing one decision boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OfferSetSnapshot {
    pub round_id: u64,
    pub batch_id: Option<String>,
    pub observed_through_sequence: u64,
    /// Canonical direction IDs, sorted and unique.
    pub available_direction_ids: Vec<String>,
    pub new_proposal_allowed: bool,
    /// Digest of the full evaluator-owned state from which the offer set was derived.
    pub source_state_sha256: String,
    pub snapshot_sha256: String,
}

#[derive(Serialize)]
struct UnsignedSnapshot<'a> {
    round_id: u64,
    batch_id: &'a Option<String>,
    observed_through_sequence: u64,
    available_direction_ids: &'a [String],
    new_proposal_allowed: bool,
    source_state_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OfferSetManifest {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub policy_id: String,
    pub source_commit: String,
    pub snapshots: Vec<OfferSetSnapshot>,
    pub manifest_sha256: String,
}

#[derive(Serialize)]
struct UnsignedManifest<'a> {
    schema: &'a str,
    version: &'a str,
    campaign_id: &'a str,
    policy_id: &'a str,
    source_commit: &'a str,
    snapshots: &'a [OfferSetSnapshot],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfferSetMatchSummary {
    pub campaign_id: String,
    pub policy_ids: Vec<String>,
    pub shared_decision_boundaries: usize,
    pub shared_batch_snapshots: usize,
    pub shared_sequential_snapshots: usize,
    pub offer_set_manifest_sha256: String,
}

impl OfferSetSnapshot {
    pub fn new(
        round_id: u64,
        batch_id: Option<String>,
        observed_through_sequence: u64,
        mut available_direction_ids: Vec<String>,
        new_proposal_allowed: bool,
        source_state_sha256: &str,
    ) -> Result<Self, OfferSetError> {
        if round_id == 0 {
            return Err(OfferSetError::new("round_id", "round IDs begin at one"));
        }
        if let Some(batch) = &batch_id {
            if batch.trim().is_empty() {
                return Err(OfferSetError::new("batch_id", "batch ID cannot be empty"));
            }
        }
        validate_sha256(source_state_sha256, "source_state_sha256")?;
        for (index, direction_id) in available_direction_ids.iter().enumerate() {
            if direction_id.trim().is_empty() {
                return Err(OfferSetError::new(
                    format!("available_direction_ids[{index}]"),
                    "direction ID must be non-empty",
                ));
            }
        }
        available_direction_ids.sort_unstable();
        if available_direction_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(OfferSetError::new(
                "available_direction_ids",
                "canonical direction IDs must be unique",
            ));
        }
        let mut snapshot = Self {
            round_id,
            batch_id,
            observed_through_sequence,
            available_direction_ids,
            new_proposal_allowed,
            source_state_sha256: source_state_sha256.to_owned(),
            snapshot_sha256: String::new(),
        };
        snapshot.snapshot_sha256 = snapshot.compute_digest()?;
        Ok(snapshot)
    }

    pub fn validate(&self) -> Result<(), OfferSetError> {
        if self.round_id == 0 {
            return Err(OfferSetError::new("round_id", "round IDs begin at one"));
        }
        if let Some(batch) = &self.batch_id {
            if batch.trim().is_empty() {
                return Err(OfferSetError::new("batch_id", "batch ID cannot be empty"));
            }
        }
        validate_sha256(&self.source_state_sha256, "source_state_sha256")?;
        validate_sha256(&self.snapshot_sha256, "snapshot_sha256")?;
        if self.available_direction_ids.iter().any(|value| value.trim().is_empty()) {
            return Err(OfferSetError::new(
                "available_direction_ids",
                "direction IDs must be non-empty",
            ));
        }
        let mut seen = BTreeSet::new();
        for direction in &self.available_direction_ids {
            if !seen.insert(direction) {
                return Err(OfferSetError::new(
                    "available_direction_ids",
                    "canonical direction IDs must be unique",
                ));
            }
        }
        if self.available_direction_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(OfferSetError::new(
                "available_direction_ids",
                "canonical direction IDs must be sorted lexicographically",
            ));
        }
        if self.compute_digest()? != self.snapshot_sha256 {
            return Err(OfferSetError::new(
                "snapshot_sha256",
                "offer-set snapshot digest does not match its payload",
            ));
        }
        Ok(())
    }

    fn compute_digest(&self) -> Result<String, OfferSetError> {
        digest_serializable(&UnsignedSnapshot {
            round_id: self.round_id,
            batch_id: &self.batch_id,
            observed_through_sequence: self.observed_through_sequence,
            available_direction_ids: &self.available_direction_ids,
            new_proposal_allowed: self.new_proposal_allowed,
            source_state_sha256: &self.source_state_sha256,
        })
    }
}

impl OfferSetManifest {
    /// Create a manifest for the given tape; snapshots are sorted and then frozen by digest.
    pub fn new(
        tape: &EventTape,
        mut snapshots: Vec<OfferSetSnapshot>,
    ) -> Result<Self, OfferSetError> {
        snapshots.sort_by(|left, right| {
            (left.round_id, &left.batch_id).cmp(&(right.round_id, &right.batch_id))
        });
        let mut manifest = Self {
            schema: SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            campaign_id: tape.header().campaign_id.clone(),
            policy_id: tape.header().policy.policy_id.clone(),
            source_commit: tape.header().source_commit.clone(),
            snapshots,
            manifest_sha256: String::new(),
        };
        manifest.manifest_sha256 = manifest.compute_digest()?;
        manifest.validate_for_tape(tape)?;
        Ok(manifest)
    }

    /// Verify provenance, exact decision-boundary coverage, and every action's eligibility.
    pub fn validate_for_tape(&self, tape: &EventTape) -> Result<(), OfferSetError> {
        if self.schema != SCHEMA || self.version != VERSION {
            return Err(OfferSetError::new(
                "schema",
                "unsupported offer-set manifest schema/version",
            ));
        }
        if self.campaign_id != tape.header().campaign_id
            || self.policy_id != tape.header().policy.policy_id
            || self.source_commit != tape.header().source_commit
        {
            return Err(OfferSetError::new(
                "manifest.identity",
                "campaign, policy or source commit differs from the event tape",
            ));
        }
        validate_sha256(&self.manifest_sha256, "manifest_sha256")?;
        if self.compute_digest()? != self.manifest_sha256 {
            return Err(OfferSetError::new(
                "manifest_sha256",
                "manifest digest does not match its payload",
            ));
        }

        let mut by_boundary = BTreeMap::<(u64, Option<String>), &OfferSetSnapshot>::new();
        for (index, snapshot) in self.snapshots.iter().enumerate() {
            snapshot.validate().map_err(|error| {
                OfferSetError::new(format!("snapshots[{index}]"), error.to_string())
            })?;
            let key = (snapshot.round_id, snapshot.batch_id.clone());
            if by_boundary.insert(key, snapshot).is_some() {
                return Err(OfferSetError::new(
                    format!("snapshots[{index}]"),
                    "duplicate round/batch offer-set snapshot",
                ));
            }
        }

        let mut expected_boundaries = BTreeSet::new();
        for (index, envelope) in tape.events().iter().enumerate() {
            let event = &envelope.payload;
            let key = (event.round_id, event.batch_id.clone());
            expected_boundaries.insert(key.clone());
            let snapshot = by_boundary.get(&key).ok_or_else(|| {
                OfferSetError::new(
                    format!("events[{index}]"),
                    "no offer-set snapshot exists for this decision boundary",
                )
            })?;
            if snapshot.observed_through_sequence != event.observed_through_sequence {
                return Err(OfferSetError::new(
                    format!("events[{index}].observed_through_sequence"),
                    "event visible-prefix cutoff differs from the offer-set snapshot",
                ));
            }
            match &event.intent {
                ActionIntent::EvaluateCandidate { direction_id, .. } => {
                    if !snapshot.available_direction_ids.contains(direction_id) {
                        return Err(OfferSetError::new(
                            format!("events[{index}].intent.direction_id"),
                            "policy evaluated a direction that was not in its offered choice set",
                        ));
                    }
                }
                ActionIntent::ProposeDirection { .. } => {
                    if !snapshot.new_proposal_allowed {
                        return Err(OfferSetError::new(
                            format!("events[{index}].intent"),
                            "policy proposed a new direction when NEW was not offered",
                        ));
                    }
                }
            }
        }

        let actual_boundaries: BTreeSet<_> = by_boundary.keys().cloned().collect();
        if expected_boundaries != actual_boundaries {
            return Err(OfferSetError::new(
                "snapshots",
                "manifest contains missing or unused decision-boundary snapshots",
            ));
        }
        Ok(())
    }

    fn compute_digest(&self) -> Result<String, OfferSetError> {
        digest_serializable(&UnsignedManifest {
            schema: &self.schema,
            version: &self.version,
            campaign_id: &self.campaign_id,
            policy_id: &self.policy_id,
            source_commit: &self.source_commit,
            snapshots: &self.snapshots,
        })
    }
}

/// Prove that all policies received the same content-addressed offer set at every round/batch.
///
/// This is stronger than comparing the header proposal-stream digest. It verifies a manifest
/// against each tape and then requires identical per-boundary snapshot digests across policies.
pub fn verify_matched_offer_sets(
    tapes: &[EventTape],
    manifests: &[OfferSetManifest],
) -> Result<OfferSetMatchSummary, OfferSetError> {
    if tapes.len() < 2 || tapes.len() != manifests.len() {
        return Err(OfferSetError::new(
            "inputs",
            "at least two tapes and exactly one manifest per tape are required",
        ));
    }

    let mut manifests_by_policy = BTreeMap::new();
    for manifest in manifests {
        if manifests_by_policy
            .insert(manifest.policy_id.as_str(), manifest)
            .is_some()
        {
            return Err(OfferSetError::new(
                "manifests",
                "duplicate policy offer-set manifest",
            ));
        }
    }

    let mut first_snapshots: Option<BTreeMap<(u64, Option<String>), String>> = None;
    let mut campaign_id: Option<String> = None;
    let mut source_commit: Option<String> = None;
    let mut policy_ids = Vec::new();
    let mut shared_batch_snapshots = 0usize;
    let mut shared_sequential_snapshots = 0usize;

    for (tape_index, tape) in tapes.iter().enumerate() {
        let header = tape.header();
        let policy_id = header.policy.policy_id.clone();
        let manifest = manifests_by_policy.get(policy_id.as_str()).ok_or_else(|| {
            OfferSetError::new(
                format!("tapes[{tape_index}]"),
                "no offer-set manifest matches this tape policy ID",
            )
        })?;
        manifest.validate_for_tape(tape).map_err(|error| {
            OfferSetError::new(format!("tapes[{tape_index}].offer_set"), error.to_string())
        })?;

        if let Some(expected) = &campaign_id {
            if expected != &header.campaign_id {
                return Err(OfferSetError::new(
                    format!("tapes[{tape_index}].campaign_id"),
                    "offer-set campaign IDs differ",
                ));
            }
        } else {
            campaign_id = Some(header.campaign_id.clone());
        }
        if let Some(expected) = &source_commit {
            if expected != &header.source_commit {
                return Err(OfferSetError::new(
                    format!("tapes[{tape_index}].source_commit"),
                    "offer-set source commit differs across policies",
                ));
            }
        } else {
            source_commit = Some(header.source_commit.clone());
        }

        let current: BTreeMap<_, _> = manifest
            .snapshots
            .iter()
            .map(|snapshot| {
                (
                    (snapshot.round_id, snapshot.batch_id.clone()),
                    snapshot.snapshot_sha256.clone(),
                )
            })
            .collect();
        if let Some(expected) = &first_snapshots {
            if expected != &current {
                return Err(OfferSetError::new(
                    format!("tapes[{tape_index}].offer_set"),
                    "actual offered options or source-state digest differs at one or more decision boundaries",
                ));
            }
        } else {
            first_snapshots = Some(current);
        }

        for snapshot in &manifest.snapshots {
            if snapshot.batch_id.is_some() {
                shared_batch_snapshots += 1;
            } else {
                shared_sequential_snapshots += 1;
            }
        }
        policy_ids.push(policy_id);
    }

    let mut policy_ids = policy_ids;
    policy_ids.sort_unstable();
    let snapshots = first_snapshots.unwrap_or_default();
    let mut digest_items: Vec<_> = snapshots.iter().collect();
    digest_items.sort_by(|left, right| left.0.cmp(right.0));
    let offer_set_manifest_sha256 = digest_serializable(
        &digest_items
            .iter()
            .map(|(key, digest)| (key, digest))
            .collect::<Vec<_>>(),
    )?;

    Ok(OfferSetMatchSummary {
        campaign_id: campaign_id.unwrap_or_default(),
        policy_ids,
        shared_decision_boundaries: snapshots.len(),
        shared_batch_snapshots: shared_batch_snapshots / tapes.len(),
        shared_sequential_snapshots: shared_sequential_snapshots / tapes.len(),
        offer_set_manifest_sha256,
    })
}

fn validate_sha256(value: &str, path: &str) -> Result<(), OfferSetError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(OfferSetError::new(
            path,
            "expected a lowercase 64-character SHA-256 digest",
        ));
    }
    Ok(())
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, OfferSetError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        OfferSetError::new(
            "serialization",
            format!("deterministic offer-set encoding failed: {error}"),
        )
    })?;
    let digest = Sha256::digest(&bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_tape::{
        ActionEvent, ActionOutcome, CandidateDisposition, ComparisonMode, EventTapeHeader,
        ExecutionContract, ExecutionMode, MeterValue, PolicySnapshot, ResourceReceipt,
    };
    use serde_json::json;

    const H1: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H3: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const H4: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const H5: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn digest<T: Serialize>(value: &T) -> String {
        let bytes = serde_json::to_vec(value).unwrap();
        let hash = Sha256::digest(bytes);
        hash.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn tape(policy_id: &str, config: serde_json::Value) -> EventTape {
        let measured = |value| MeterValue::Measured { value };
        let header = EventTapeHeader {
            schema: "symthaea.ar5b4.event-tape".to_owned(),
            version: "0.1.0".to_owned(),
            campaign_id: "offer-set-campaign".to_owned(),
            source_commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            execution_environment_sha256: H1.to_owned(),
            comparison_mode: ComparisonMode::MechanismMatched,
            proposal_source_id: "shared-proposal-source-v1".to_owned(),
            proposal_stream_sha256: Some(H2.to_owned()),
            task_recipe_sha256: H3.to_owned(),
            comparison_profile_sha256: H4.to_owned(),
            evaluator_sha256: H5.to_owned(),
            verifier_sha256: H1.to_owned(),
            budget_model_id: "offer-set-budget-v1".to_owned(),
            budget_model_sha256: H2.to_owned(),
            budget_limit_microunits: 100,
            initial_direction_ids: vec!["A".to_owned()],
            execution: ExecutionContract {
                mode: ExecutionMode::Sequential,
                worker_count: 1,
                seats_per_round: 1,
                first_round_claim_cap: None,
                steady_state_claim_cap: None,
            },
            policy: PolicySnapshot {
                policy_id: policy_id.to_owned(),
                algorithm_id: format!("algorithm-{policy_id}"),
                config_sha256: digest(&config),
                config,
                has_hidden_oracle_access: false,
            },
        };
        let mut tape = EventTape::new(header).unwrap();
        tape.append(ActionEvent {
            event_id: format!("event-{policy_id}"),
            policy_id: policy_id.to_owned(),
            worker_id: 0,
            round_id: 1,
            batch_id: None,
            observed_through_sequence: 0,
            visible_history_sha256: String::new(),
            retry_of_event_id: None,
            intent: ActionIntent::EvaluateCandidate {
                candidate_sha256: H5.to_owned(),
                parent_candidate_sha256: None,
                semantic_delta_sha256: H3.to_owned(),
                direction_id: "A".to_owned(),
            },
            outcome: ActionOutcome::CandidateEvaluation {
                task_recipe_sha256: H3.to_owned(),
                comparison_profile_sha256: H4.to_owned(),
                evaluator_sha256: H5.to_owned(),
                verifier_sha256: H1.to_owned(),
                verifier_receipt_sha256: H2.to_owned(),
                disposition: CandidateDisposition::Verified,
                score: Some(0.9),
                uncertainty: Some(0.0),
                evidence_sha256: Some(H3.to_owned()),
            },
            resources: ResourceReceipt {
                host_cpu_millis: measured(10),
                wall_millis: measured(5),
                accelerator_millis: measured(0),
                peak_memory_bytes: measured(1024),
                accelerator_peak_memory_bytes: measured(0),
                energy_millijoules: measured(1),
                input_tokens: measured(0),
                output_tokens: measured(0),
                tool_calls: measured(0),
                evaluator_calls: measured(1),
                verifier_calls: measured(1),
                charged_budget_microunits: measured(10),
            },
        })
        .unwrap();
        tape
    }

    fn manifest(
        tape: &EventTape,
        offered: Vec<&str>,
        state_sha256: &str,
    ) -> OfferSetManifest {
        let snapshot = OfferSetSnapshot::new(
            1,
            None,
            0,
            offered.into_iter().map(str::to_owned).collect(),
            true,
            state_sha256,
        )
        .unwrap();
        OfferSetManifest::new(tape, vec![snapshot]).unwrap()
    }

    #[test]
    fn identical_content_addressed_offer_sets_match_across_policies() {
        let left = tape("board", json!({"alpha": 0.5}));
        let right = tape("thompson", json!({"prior": 1.0}));
        let left_manifest = manifest(&left, vec!["A", "B"], H3);
        let right_manifest = manifest(&right, vec!["B", "A"], H3);
        let summary = verify_matched_offer_sets(
            &[left, right],
            &[left_manifest, right_manifest],
        )
        .unwrap();
        assert_eq!(summary.shared_decision_boundaries, 1);
        assert_eq!(summary.shared_sequential_snapshots, 1);
        assert_eq!(summary.shared_batch_snapshots, 0);
    }

    #[test]
    fn cross_policy_offer_set_mismatch_is_rejected() {
        let left = tape("board", json!({"alpha": 0.5}));
        let right = tape("thompson", json!({"prior": 1.0}));
        let left_manifest = manifest(&left, vec!["A", "B"], H3);
        let right_manifest = manifest(&right, vec!["A", "C"], H4);
        assert!(verify_matched_offer_sets(
            &[left, right],
            &[left_manifest, right_manifest],
        )
        .is_err());
    }

    #[test]
    fn unavailable_direction_cannot_be_evaluated() {
        let tape = tape("board", json!({"alpha": 0.5}));
        let snapshot = OfferSetSnapshot::new(1, None, 0, vec!["B".to_owned()], true, H3)
            .unwrap();
        assert!(OfferSetManifest::new(&tape, vec![snapshot]).is_err());
    }

    #[test]
    fn snapshot_payload_tampering_invalidates_manifest() {
        let tape = tape("board", json!({"alpha": 0.5}));
        let mut manifest = manifest(&tape, vec!["A"], H3);
        manifest.snapshots[0].new_proposal_allowed = false;
        assert!(manifest.validate_for_tape(&tape).is_err());
    }

    #[test]
    fn manifest_requires_every_decision_boundary_exactly_once() {
        let tape = tape("board", json!({"alpha": 0.5}));
        let snapshot = OfferSetSnapshot::new(1, None, 0, vec!["A".to_owned()], true, H3)
            .unwrap();
        let extra = OfferSetSnapshot::new(2, None, 1, vec!["A".to_owned()], true, H3)
            .unwrap();
        assert!(OfferSetManifest::new(&tape, vec![snapshot, extra]).is_err());
    }
}
